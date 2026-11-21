"""Chargement, validation et préparation des données de transactions.

Contrat de données (voir data/README.md) :
- colonnes obligatoires : Time, V1…V28, Amount (et Class pour l'entraînement) ;
- types numériques (les booléens et le texte sont refusés) ;
- aucune valeur manquante ni infinie ;
- Time >= 0 et Amount >= 0 ;
- Class ∈ {0, 1}.

Politique : une donnée qui viole le contrat est REJETÉE explicitement
(DataValidationError). Aucune imputation silencieuse n'est faite, car compléter
une variable d'une transaction à scorer pourrait masquer une anomalie ou une
tentative de contournement. Seule correction documentée : la suppression des
doublons exacts du jeu d'entraînement (remove_duplicates), pour éviter qu'une
même transaction se retrouve dans deux partitions.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from app.config import FEATURES, MAX_FILE_SIZE_MB, SEED, TARGET, TEST_SIZE, VAL_SIZE

logger = logging.getLogger(__name__)

ALLOWED_COLUMNS = frozenset([*FEATURES, TARGET])


class DataValidationError(ValueError):
    """La donnée ne respecte pas le contrat d'entrée."""


class DataLoadError(Exception):
    """Le fichier de données ne peut pas être chargé."""


# --- Chargement ----------------------------------------------------------------
def load_data(path: str | Path, max_size_mb: float = MAX_FILE_SIZE_MB) -> pd.DataFrame:
    """Charge un fichier CSV de transactions en contrôlant son existence, son format et sa taille."""
    path = Path(path)
    if not path.is_file():
        raise DataLoadError(f"Fichier de données introuvable : {path}")
    if path.suffix.lower() != ".csv":
        raise DataLoadError(f"Extension non autorisée « {path.suffix} » : seul le format .csv est accepté")
    size_mb = path.stat().st_size / 1024**2
    if size_mb > max_size_mb:
        raise DataLoadError(f"Fichier trop volumineux ({size_mb:.1f} Mo > {max_size_mb} Mo)")
    try:
        df = pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise DataLoadError(f"Fichier illisible ({path.name}) : {exc.__class__.__name__}") from exc
    logger.info("Données chargées : %d lignes, %d colonnes", *df.shape)
    return df


# --- Validation ------------------------------------------------------------------
def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def validate_data(
    df: pd.DataFrame,
    require_target: bool = True,
    allow_extra_columns: bool = True,
) -> pd.DataFrame:
    """Vérifie le schéma, les types et les valeurs ; lève DataValidationError sinon.

    Les messages d'erreur citent les colonnes concernées, jamais les valeurs,
    pour ne pas exposer de données de transaction dans les journaux.
    """
    if not isinstance(df, pd.DataFrame):
        raise DataValidationError(f"Entrée attendue : pandas.DataFrame, reçu {type(df).__name__}")
    if df.shape[0] == 0 or df.shape[1] == 0:
        raise DataValidationError("Données vides : aucune ligne ou aucune colonne")

    duplicated = sorted({str(c) for c in df.columns[df.columns.duplicated()]})
    if duplicated:
        raise DataValidationError(f"Colonnes dupliquées : {duplicated}")

    required = [*FEATURES, TARGET] if require_target else list(FEATURES)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataValidationError(f"Colonnes manquantes : {missing}")

    extra = [str(c) for c in df.columns if c not in ALLOWED_COLUMNS]
    if extra and not allow_extra_columns:
        raise DataValidationError(f"Colonnes non prévues par le contrat : {extra}")

    bad_types = [c for c in required if not _is_numeric(df[c])]
    if bad_types:
        raise DataValidationError(f"Type non numérique pour : {bad_types}")

    values = df[FEATURES].to_numpy(dtype="float64", na_value=np.nan)
    nan_cols = [c for c, bad in zip(FEATURES, np.isnan(values).any(axis=0), strict=True) if bad]
    if nan_cols:
        raise DataValidationError(f"Valeurs manquantes dans : {nan_cols}")
    inf_cols = [c for c, bad in zip(FEATURES, np.isinf(values).any(axis=0), strict=True) if bad]
    if inf_cols:
        raise DataValidationError(f"Valeurs infinies dans : {inf_cols}")

    if (df["Amount"] < 0).any():
        raise DataValidationError("Montant (Amount) négatif interdit")
    if (df["Time"] < 0).any():
        raise DataValidationError("Temps (Time) négatif interdit")

    if require_target:
        target = df[TARGET]
        if target.isna().any():
            raise DataValidationError("Valeurs manquantes dans la cible Class")
        invalid = sorted(set(target.unique().tolist()) - {0, 1})
        if invalid:
            raise DataValidationError(f"Valeurs de cible non autorisées (attendu 0 ou 1) : {invalid[:5]}")
    return df


def invalid_row_mask(df: pd.DataFrame) -> pd.Series:
    """Indique, ligne par ligne, les transactions qui violent le contrat de valeurs."""
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise DataValidationError(f"Colonnes manquantes : {missing}")
    numeric = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype="float64", na_value=np.nan)
    bad = ~np.isfinite(values).all(axis=1)
    with np.errstate(invalid="ignore"):
        bad |= values[:, FEATURES.index("Amount")] < 0
        bad |= values[:, FEATURES.index("Time")] < 0
    return pd.Series(bad, index=df.index, name="invalid")


# --- Correction documentée -------------------------------------------------------
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime les doublons exacts (première occurrence conservée)."""
    deduplicated = df.drop_duplicates().reset_index(drop=True)
    removed = len(df) - len(deduplicated)
    if removed:
        logger.info("%d doublons exacts supprimés", removed)
    return deduplicated


# --- Préparation -------------------------------------------------------------------
def preprocess_data(df: pd.DataFrame, require_target: bool = True) -> tuple[pd.DataFrame, pd.Series | None]:
    """Valide puis sépare les variables explicatives (ordre du contrat, float64) et la cible."""
    validate_data(df, require_target=require_target)
    X = df[FEATURES].astype("float64")
    y = df[TARGET].astype("int64") if require_target else None
    return X, y


def build_preprocessor(features: list[str] | None = None) -> ColumnTransformer:
    """Transformations apprises, à ajuster UNIQUEMENT sur les données d'entraînement.

    - Amount : log1p (distribution très asymétrique, montants extrêmes) puis standardisation ;
    - autres variables : standardisation.
    """
    features = list(features or FEATURES)
    others = [c for c in features if c != "Amount"]
    transformers = []
    if "Amount" in features:
        amount_pipeline = Pipeline(
            [
                ("log1p", FunctionTransformer(np.log, feature_names_out="one-to-one")),
                ("scale", StandardScaler()),
            ]
        )
        transformers.append(("amount", amount_pipeline, ["Amount"]))
    if others:
        transformers.append(("scale", StandardScaler(), others))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def split_data(
    df: pd.DataFrame,
    seed: int = SEED,
    test_size: float = TEST_SIZE,
    val_size: float = VAL_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Découpage stratifié entraînement / validation / test (index d'origine conservés)."""
    if TARGET not in df.columns:
        raise DataValidationError("La cible Class est nécessaire pour un découpage stratifié")
    train_val, test = train_test_split(df, test_size=test_size, stratify=df[TARGET], random_state=seed)
    relative_val = val_size / (1 - test_size)
    train, val = train_test_split(train_val, test_size=relative_val, stratify=train_val[TARGET], random_state=seed)
    return train, val, test
