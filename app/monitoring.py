"""Indicateurs de surveillance post-déploiement : dérive, plages de valeurs, taux d'alertes."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from app.config import ALERT_RATE_MAX_RATIO, FEATURES, PSI_ALERT, PSI_WARNING


def population_stability_index(reference, current, bins: int = 10, eps: float = 1e-6) -> float:
    """PSI entre deux distributions, sur des classes définies par les quantiles de la référence.

    Repères usuels : < 0,10 stable ; 0,10–0,20 à surveiller ; > 0,20 dérive significative.
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1))[1:-1])
    ref_counts = np.bincount(np.searchsorted(edges, reference, side="right"), minlength=len(edges) + 1)
    cur_counts = np.bincount(np.searchsorted(edges, current, side="right"), minlength=len(edges) + 1)
    ref_share = np.clip(ref_counts / len(reference), eps, None)
    cur_share = np.clip(cur_counts / len(current), eps, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def drift_status(psi: float, warning: float = PSI_WARNING, alert: float = PSI_ALERT) -> str:
    if psi >= alert:
        return "alerte"
    if psi >= warning:
        return "surveiller"
    return "stable"


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, features: list[str] | None = None) -> list[dict]:
    """PSI et test de Kolmogorov-Smirnov, variable par variable (dérive des données, sans label)."""
    rows = []
    for feature in features or FEATURES:
        psi = population_stability_index(reference[feature], current[feature])
        ks = ks_2samp(reference[feature], current[feature])
        rows.append(
            {
                "feature": feature,
                "psi": psi,
                "ks_statistic": float(ks.statistic),
                "ks_pvalue": float(ks.pvalue),
                "status": drift_status(psi),
            }
        )
    return rows


def fit_reference_ranges(X: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Plages [min, max] observées à l'entraînement."""
    return {c: (float(X[c].min()), float(X[c].max())) for c in FEATURES}


def out_of_range_rates(X: pd.DataFrame, ranges: dict[str, tuple[float, float]]) -> dict[str, float]:
    """Part des transactions hors de la plage d'entraînement, par variable."""
    return {c: float(((X[c] < low) | (X[c] > high)).mean()) for c, (low, high) in ranges.items()}


def alert_rate_status(labels, expected_rate: float, max_ratio: float = ALERT_RATE_MAX_RATIO) -> dict:
    """Compare le taux de transactions signalées à celui attendu (mesurable sans label)."""
    observed = float(np.mean(labels))
    ratio = observed / expected_rate if expected_rate > 0 else float("inf")
    if ratio > max_ratio:
        status = "alerte_hausse"
    elif ratio < 1 / max_ratio:
        status = "alerte_baisse"
    else:
        status = "normal"
    return {"observed_rate": observed, "expected_rate": expected_rate, "ratio": ratio, "status": status}
