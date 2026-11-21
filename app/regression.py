"""Référence (baseline) et comparaison de non-régression.

La référence est créée MANUELLEMENT (commande `baseline-create`) et n'est jamais
recalculée automatiquement : sinon une régression deviendrait la nouvelle norme.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import pandas as pd

from app.config import REGRESSION_TOLERANCES, SEED, TARGET
from app.utils import git_commit, library_versions, read_json, utc_now, write_json

TRACKED_METRICS = ("pr_auc", "roc_auc", "recall", "precision", "fpr", "f1")
DECISION_ORDER = {"accepter": 0, "investiguer": 1, "rejeter": 2}


def dataset_fingerprint(df: pd.DataFrame) -> dict:
    """Caractéristiques et empreinte du contenu d'un jeu de données."""
    row_hashes = pd.util.hash_pandas_object(df, index=False).to_numpy()
    return {
        "n_rows": len(df),
        "n_fraud": int(df[TARGET].sum()) if TARGET in df else None,
        "fraud_rate": float(df[TARGET].mean()) if TARGET in df else None,
        "n_columns": df.shape[1],
        "content_sha256": hashlib.sha256(row_hashes.tobytes()).hexdigest(),
    }


def build_run_record(
    label: str,
    config: dict,
    threshold: float,
    val_metrics: dict,
    val_df: pd.DataFrame,
    seed: int = SEED,
) -> dict:
    return {
        "label": label,
        "created_at": utc_now(),
        "code_version": git_commit(),
        "libraries": library_versions(),
        "seed": seed,
        "model": config,
        "threshold": float(threshold),
        "metrics": {name: val_metrics[name] for name in TRACKED_METRICS},
        "confusion": {k: val_metrics[k] for k in ("tp", "fp", "fn", "tn")},
        "validation_set": dataset_fingerprint(val_df),
    }


def save_baseline(record: dict, path: str | Path, force: bool = False) -> Path:
    path = Path(path)
    if path.exists() and not force:
        raise FileExistsError(
            f"Référence existante ({path}) : elle ne doit pas être écrasée sans décision explicite (--force)"
        )
    return write_json(record, path)


def load_baseline(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Référence introuvable : {path}")
    return read_json(path)


def compare_to_baseline(current: dict, baseline: dict, tolerances: dict = REGRESSION_TOLERANCES) -> dict:
    """Compare une nouvelle version à la référence et propose une décision motivée."""
    rows = []
    for metric, (direction, warn, reject) in tolerances.items():
        ref = baseline["metrics"][metric]
        new = current["metrics"][metric]
        if new is None or ref is None or (isinstance(new, float) and math.isnan(new)):
            rows.append({"metric": metric, "baseline": ref, "current": new, "degradation": None, "status": "rejeter"})
            continue
        degradation = (ref - new) if direction == "higher" else (new - ref)
        if degradation > reject:
            status = "rejeter"
        elif degradation > warn:
            status = "investiguer"
        else:
            status = "accepter"
        rows.append(
            {
                "metric": metric,
                "baseline": ref,
                "current": new,
                "delta": new - ref,
                "degradation": degradation,
                "status": status,
            }
        )

    notes = []
    decision = max((r["status"] for r in rows), key=DECISION_ORDER.get, default="accepter")
    if current["validation_set"]["content_sha256"] != baseline["validation_set"]["content_sha256"]:
        notes.append("Jeu de validation différent de celui de la référence : comparaison non probante.")
        decision = max(decision, "investiguer", key=DECISION_ORDER.get)
    if current.get("libraries") != baseline.get("libraries"):
        notes.append("Versions des bibliothèques différentes de la référence.")
    if current.get("model") != baseline.get("model"):
        notes.append("Configuration du modèle modifiée par rapport à la référence.")
    return {
        "decision": decision,
        "metrics": rows,
        "notes": notes,
        "threshold_baseline": baseline["threshold"],
        "threshold_current": current["threshold"],
    }
