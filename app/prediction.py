"""Prédiction, sauvegarde et chargement sécurisé du pipeline."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from app.config import FEATURES, MAX_BATCH_RECORDS
from app.preprocessing import DataValidationError, validate_data
from app.utils import library_versions, sha256_file, utc_now, write_json

logger = logging.getLogger(__name__)


class ModelLoadError(Exception):
    """Le modèle sauvegardé ne peut pas être chargé en confiance."""


class PredictionError(RuntimeError):
    """Le modèle a produit une sortie qui viole le contrat (probabilité invalide)."""


def _check_threshold(threshold) -> float:
    valid_type = isinstance(threshold, (int, float, np.floating)) and not isinstance(threshold, bool)
    if not valid_type or math.isnan(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("Le seuil de décision doit être un nombre compris entre 0 et 1")
    return float(threshold)


def predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    """Probabilité de fraude ; vérifie le contrat de sortie."""
    proba = np.asarray(model.predict_proba(X[FEATURES].astype("float64"))[:, 1], dtype=float)
    if not np.isfinite(proba).all() or (proba < 0).any() or (proba > 1).any():
        raise PredictionError("Le modèle a produit une probabilité non finie ou hors de [0, 1]")
    return proba


def predict_fraud(model, X: pd.DataFrame, threshold: float = 0.5, strict_schema: bool = False) -> pd.DataFrame:
    """Valide l'entrée, calcule la probabilité de fraude et applique le seuil.

    strict_schema=True refuse toute colonne hors contrat (usage type API).
    Le journal ne contient que des volumes, jamais de valeurs de transaction.
    """
    threshold = _check_threshold(threshold)
    validate_data(X, require_target=False, allow_extra_columns=not strict_schema)
    proba = predict_proba(model, X)
    labels = (proba >= threshold).astype(int)
    logger.info(
        "Lot évalué : %d transactions, %d signalées (seuil %.4f)",
        len(labels),
        int(labels.sum()),
        threshold,
    )
    return pd.DataFrame({"fraud_probability": proba, "is_fraud": labels}, index=X.index)


def score_records(model, records: list[dict], threshold: float, max_records: int = MAX_BATCH_RECORDS) -> list[dict]:
    """Point d'entrée de type API : liste d'enregistrements JSON -> scores.

    Refuse la requête entière si un enregistrement a un champ manquant,
    un champ non prévu ou un type non numérique.
    """
    if not isinstance(records, list) or not records:
        raise DataValidationError("Requête attendue : liste non vide d'enregistrements")
    if len(records) > max_records:
        raise DataValidationError(f"Trop d'enregistrements ({len(records)} > {max_records})")
    expected = set(FEATURES)
    for i, record in enumerate(records):
        if not isinstance(record, dict):
            raise DataValidationError(f"Enregistrement {i} : objet attendu")
        keys = set(record)
        missing, extra = sorted(expected - keys), sorted(str(k) for k in keys - expected)
        if missing or extra:
            raise DataValidationError(f"Enregistrement {i} : champs manquants {missing}, champs non autorisés {extra}")
        bad = sorted(k for k, v in record.items() if isinstance(v, bool) or not isinstance(v, (int, float)))
        if bad:
            raise DataValidationError(f"Enregistrement {i} : type non numérique pour {bad}")
    frame = pd.DataFrame.from_records(records, columns=FEATURES)
    result = predict_fraud(model, frame, threshold, strict_schema=True)
    return [
        {"fraud_probability": float(p), "is_fraud": bool(label)}
        for p, label in zip(result["fraud_probability"], result["is_fraud"], strict=True)
    ]


# --- Persistance -----------------------------------------------------------------------
def _sidecar(path: Path) -> Path:
    return path.with_suffix(".json")


def save_model(pipeline, path: str | Path, threshold: float, metadata: dict | None = None) -> Path:
    """Sauvegarde le pipeline (joblib) et un fichier de métadonnées avec son empreinte SHA-256."""
    path = Path(path)
    if path.suffix != ".joblib":
        raise ValueError("Le modèle doit être sauvegardé avec l'extension .joblib")
    threshold = _check_threshold(threshold)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    info = {
        "threshold": threshold,
        "sha256": sha256_file(path),
        "created_at": utc_now(),
        "libraries": library_versions(),
        **(metadata or {}),
    }
    write_json(info, _sidecar(path))
    return path


def load_model(path: str | Path):
    """Recharge un pipeline après vérification de son empreinte.

    joblib s'appuie sur pickle : charger un fichier non maîtrisé peut exécuter du code.
    L'empreinte SHA-256 enregistrée à la sauvegarde est donc vérifiée AVANT le chargement.
    """
    path = Path(path)
    if path.suffix != ".joblib":
        raise ModelLoadError(f"Extension non autorisée : {path.suffix}")
    if not path.is_file():
        raise ModelLoadError(f"Modèle introuvable : {path}")
    sidecar = _sidecar(path)
    if not sidecar.is_file():
        raise ModelLoadError("Métadonnées du modèle absentes : chargement refusé")
    try:
        info = json.loads(sidecar.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ModelLoadError("Métadonnées du modèle illisibles") from exc
    if sha256_file(path) != info.get("sha256"):
        raise ModelLoadError("Empreinte SHA-256 différente : fichier modifié ou corrompu, chargement refusé")
    try:
        model = joblib.load(path)
    except Exception as exc:  # fichier tronqué, version incompatible…
        raise ModelLoadError(f"Échec du chargement : {exc.__class__.__name__}") from exc
    if not hasattr(model, "predict_proba"):
        raise ModelLoadError("L'objet chargé n'est pas un modèle de classification probabiliste")
    try:
        _check_threshold(info.get("threshold"))
    except ValueError as exc:
        raise ModelLoadError("Seuil de décision invalide dans les métadonnées") from exc
    return model, info
