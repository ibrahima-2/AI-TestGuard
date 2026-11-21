"""Perturbations contrôlées et mesure de sensibilité du modèle.

Les perturbations sont propres à chaque type de variable :
- V1…V28 (composantes PCA) : bruit gaussien proportionnel à l'écart-type
  d'entraînement de chaque composante (ε × σ) ;
- Amount : variation relative (un montant ne se « bruite » pas en absolu) ;
- Time : décalage en secondes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.config import FEATURES, PCA_FEATURES
from app.evaluation import evaluate_model


def add_gaussian_noise(
    X: pd.DataFrame,
    stds: pd.Series,
    epsilon: float,
    rng: np.random.Generator,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    columns = columns or PCA_FEATURES
    perturbed = X.copy()
    noise = rng.normal(0.0, 1.0, size=(len(X), len(columns))) * (epsilon * stds[columns].to_numpy())
    perturbed[columns] = X[columns].to_numpy() + noise
    return perturbed


def perturb_amount_relative(X: pd.DataFrame, max_relative: float, rng: np.random.Generator) -> pd.DataFrame:
    perturbed = X.copy()
    perturbed["Amount"] = X["Amount"] * (1 + rng.uniform(-max_relative, max_relative, len(X)))
    return perturbed


def scale_amount(X: pd.DataFrame, factor: float) -> pd.DataFrame:
    perturbed = X.copy()
    perturbed["Amount"] = X["Amount"] * factor
    return perturbed


def set_amount(X: pd.DataFrame, value: float) -> pd.DataFrame:
    perturbed = X.copy()
    perturbed["Amount"] = float(value)
    return perturbed


def shift_time(X: pd.DataFrame, seconds: float) -> pd.DataFrame:
    perturbed = X.copy()
    perturbed["Time"] = (X["Time"] + seconds).clip(lower=0)
    return perturbed


def jitter_time(X: pd.DataFrame, max_seconds: float, rng: np.random.Generator) -> pd.DataFrame:
    perturbed = X.copy()
    perturbed["Time"] = (X["Time"] + rng.uniform(-max_seconds, max_seconds, len(X))).clip(lower=0)
    return perturbed


def scale_columns(X: pd.DataFrame, columns: list[str], factor: float) -> pd.DataFrame:
    perturbed = X.copy()
    perturbed[columns] = X[columns] * factor
    return perturbed


def inject_values(
    df: pd.DataFrame, columns: list[str], fraction: float, value, rng: np.random.Generator
) -> tuple[pd.DataFrame, int]:
    """Remplace `value` dans une fraction des lignes ; retourne la copie et le nombre de lignes touchées."""
    corrupted = df.copy()
    n_rows = max(1, int(round(fraction * len(df))))
    rows = rng.choice(len(df), size=n_rows, replace=False)
    for column in columns:
        if not isinstance(value, (int, float)):
            corrupted[column] = corrupted[column].astype(object)
        corrupted.iloc[rows, corrupted.columns.get_loc(column)] = value
    return corrupted, n_rows


def sensitivity(model, X_ref: pd.DataFrame, X_perturbed: pd.DataFrame, y_true, threshold: float) -> dict:
    """Mesure l'effet d'une perturbation sur les scores, les décisions et les métriques."""
    p_ref = model.predict_proba(X_ref[FEATURES])[:, 1]
    p_new = model.predict_proba(X_perturbed[FEATURES])[:, 1]
    n_non_finite = int((~np.isfinite(p_new)).sum())
    delta = np.abs(p_new - p_ref)
    label_ref, label_new = p_ref >= threshold, p_new >= threshold
    flips = label_ref != label_new
    margin = np.abs(p_ref - threshold)
    before = evaluate_model(y_true, label_ref.astype(int), p_ref)
    after = evaluate_model(y_true, label_new.astype(int), np.clip(np.nan_to_num(p_new), 0, 1))
    return {
        "mean_abs_delta": float(delta.mean()),
        "p95_abs_delta": float(np.percentile(delta, 95)),
        "max_abs_delta": float(delta.max()),
        "n_flips": int(flips.sum()),
        "flip_rate": float(flips.mean()),
        "flips_to_fraud": int((~label_ref & label_new).sum()),
        "flips_to_normal": int((label_ref & ~label_new).sum()),
        "median_margin_flipped": float(np.median(margin[flips])) if flips.any() else None,
        "median_margin_all": float(np.median(margin)),
        "n_non_finite": n_non_finite,
        "recall_before": before["recall"],
        "recall_after": after["recall"],
        "precision_before": before["precision"],
        "precision_after": after["precision"],
        "fpr_before": before["fpr"],
        "fpr_after": after["fpr"],
        "delta_recall": after["recall"] - before["recall"],
        "delta_precision": after["precision"] - before["precision"],
        "delta_fpr": after["fpr"] - before["fpr"],
    }
