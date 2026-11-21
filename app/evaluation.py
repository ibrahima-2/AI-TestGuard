"""Métriques de classification, critères d'acceptation, incertitudes et graphiques."""

from __future__ import annotations

import math
import operator
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from app.config import ACCEPTANCE_CRITERIA, SEED  # noqa: E402

OPERATORS = {">=": operator.ge, "<=": operator.le}


def _binary_vector(values, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} doit être un vecteur à une dimension")
    if not np.isin(array, [0, 1]).all():
        raise ValueError(f"{name} doit contenir uniquement les valeurs 0 et 1")
    return array.astype(int)


def _ratio(numerator: int, denominator: int, default: float) -> float:
    return float(numerator / denominator) if denominator else default


def evaluate_model(y_true, y_pred, y_proba=None) -> dict:
    """Calcule les métriques d'un classifieur binaire (classe positive = fraude).

    - precision vaut 0 si aucune transaction n'est signalée (convention zero_division=0) ;
    - recall, fnr (resp. fpr) valent NaN si le jeu ne contient aucune fraude
      (resp. aucune transaction normale), car ils ne sont alors pas définis ;
    - roc_auc et pr_auc exigent les probabilités et la présence des deux classes.
    """
    y_true = _binary_vector(y_true, "y_true")
    y_pred = _binary_vector(y_pred, "y_pred")
    if len(y_true) == 0:
        raise ValueError("Impossible d'évaluer un jeu vide")
    if len(y_true) != len(y_pred):
        raise ValueError(f"Longueurs différentes : y_true={len(y_true)}, y_pred={len(y_pred)}")

    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel())
    n = tp + tn + fp + fn
    precision = _ratio(tp, tp + fp, 0.0)
    recall = _ratio(tp, tp + fn, math.nan)
    if math.isnan(recall):
        f1 = math.nan
    else:
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    metrics = {
        "n": n,
        "n_positive": tp + fn,
        "n_negative": tn + fp,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": _ratio(fp, fp + tn, math.nan),
        "fnr": _ratio(fn, fn + tp, math.nan),
        "accuracy": (tp + tn) / n,
        "roc_auc": math.nan,
        "pr_auc": math.nan,
    }

    if y_proba is not None:
        proba = np.asarray(y_proba, dtype=float)
        if proba.shape != y_true.shape:
            raise ValueError("y_proba doit avoir la même taille que y_true")
        if not np.isfinite(proba).all() or (proba < 0).any() or (proba > 1).any():
            raise ValueError("y_proba doit contenir des probabilités finies comprises entre 0 et 1")
        if 0 < metrics["n_positive"] < n:
            metrics["roc_auc"] = float(roc_auc_score(y_true, proba))
            metrics["pr_auc"] = float(average_precision_score(y_true, proba))
    return metrics


def check_acceptance(metrics: dict, criteria: dict = ACCEPTANCE_CRITERIA) -> dict:
    """Compare les métriques aux critères ; une métrique non définie (NaN) échoue."""
    details = {}
    for name, (op, threshold) in criteria.items():
        value = metrics.get(name, math.nan)
        defined = value is not None and not (isinstance(value, float) and math.isnan(value))
        passed = bool(defined and OPERATORS[op](value, threshold))
        details[name] = {"value": value, "operator": op, "threshold": threshold, "passed": passed}
    return {"accepted": all(d["passed"] for d in details.values()), "criteria": details}


def bootstrap_ci(
    y_true,
    y_proba,
    threshold: float,
    n_boot: int = 1000,
    seed: int = SEED,
    alpha: float = 0.05,
) -> dict:
    """Intervalles de confiance par bootstrap stratifié (nombre de fraudes conservé)."""
    y_true = _binary_vector(y_true, "y_true")
    proba = np.asarray(y_proba, dtype=float)
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y_true == 1)
    neg = np.flatnonzero(y_true == 0)
    names = ("precision", "recall", "f1", "fpr", "pr_auc")
    samples = {name: [] for name in names}
    for _ in range(n_boot):
        idx = np.concatenate(
            [rng.choice(pos, size=len(pos), replace=True), rng.choice(neg, size=len(neg), replace=True)]
        )
        y_b, p_b = y_true[idx], proba[idx]
        m = evaluate_model(y_b, (p_b >= threshold).astype(int), p_b)
        for name in names:
            samples[name].append(m[name])
    low, high = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {
        name: {
            "low": float(np.nanpercentile(values, low)),
            "high": float(np.nanpercentile(values, high)),
        }
        for name, values in samples.items()
    } | {"n_boot": n_boot, "confidence": 1 - alpha}


def threshold_tradeoff(y_true, y_proba, thresholds) -> list[dict]:
    """Précision, rappel, FPR et volumes d'alertes pour une série de seuils."""
    rows = []
    for threshold in thresholds:
        m = evaluate_model(y_true, (np.asarray(y_proba) >= threshold).astype(int), y_proba)
        rows.append(
            {
                "threshold": float(threshold),
                "precision": m["precision"],
                "recall": m["recall"],
                "fpr": m["fpr"],
                "tp": m["tp"],
                "fp": m["fp"],
                "fn": m["fn"],
                "alerts": m["tp"] + m["fp"],
            }
        )
    return rows


def error_analysis(X: pd.DataFrame, y_true, y_proba, threshold: float) -> dict:
    """Caractérise les quatre cellules de la matrice de confusion (montant, score)."""
    y_true = np.asarray(y_true)
    proba = np.asarray(y_proba)
    y_pred = proba >= threshold
    groups = {
        "TP": (y_true == 1) & y_pred,
        "FN": (y_true == 1) & ~y_pred,
        "FP": (y_true == 0) & y_pred,
        "TN": (y_true == 0) & ~y_pred,
    }
    amount = X["Amount"].to_numpy()
    summary = {}
    for name, mask in groups.items():
        if mask.any():
            summary[name] = {
                "count": int(mask.sum()),
                "amount_median": float(np.median(amount[mask])),
                "amount_mean": float(amount[mask].mean()),
                "amount_total": float(amount[mask].sum()),
                "score_median": float(np.median(proba[mask])),
            }
        else:
            summary[name] = {"count": 0}
    fn_scores = proba[groups["FN"]]
    summary["FN_detail"] = {
        "score_below_0_1": int((fn_scores < 0.1).sum()),
        "score_between_0_1_and_threshold": int(((fn_scores >= 0.1) & (fn_scores < threshold)).sum()),
    }
    fp_scores = proba[groups["FP"]]
    summary["FP_detail"] = {"score_above_0_99": int((fp_scores > 0.99).sum())}
    return summary


# --- Graphiques ---------------------------------------------------------------------
def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_roc_curves(curves: dict, path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 5))
    for label, (y_true, proba) in curves.items():
        fpr, tpr, _ = roc_curve(y_true, proba)
        ax.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc_score(y_true, proba):.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Hasard")
    ax.set_xlabel("Taux de faux positifs (FPR)")
    ax.set_ylabel("Rappel (TPR)")
    ax.set_title("Courbes ROC (jeu de test)")
    ax.legend(loc="lower right")
    return _save(fig, path)


def plot_pr_curves(curves: dict, path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 5))
    prevalence = None
    for label, (y_true, proba) in curves.items():
        precision, recall, _ = precision_recall_curve(y_true, proba)
        ap = average_precision_score(y_true, proba)
        ax.step(recall, precision, where="post", label=f"{label} (AP = {ap:.3f})")
        prevalence = float(np.mean(y_true))
    if prevalence is not None:
        ax.axhline(prevalence, linestyle="--", color="grey", label=f"Hasard ({prevalence:.4f})")
    ax.set_xlabel("Rappel")
    ax.set_ylabel("Précision")
    ax.set_ylim(0, 1.02)
    ax.set_title("Courbes précision-rappel (jeu de test)")
    ax.legend(loc="lower left")
    return _save(fig, path)


def plot_confusion_matrix(metrics: dict, title: str, path: str | Path) -> Path:
    matrix = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    labels = np.array([["TN", "FP"], ["FN", "TP"]])
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(np.log1p(matrix), cmap="Blues")
    for i in range(2):
        for j in range(2):
            color = "white" if np.log1p(matrix[i, j]) > np.log1p(matrix).max() / 2 else "black"
            ax.text(
                j,
                i,
                f"{labels[i, j]}\n{matrix[i, j]:,}".replace(",", " "),
                ha="center",
                va="center",
                color=color,
                fontsize=12,
            )
    ax.set_xticks([0, 1], ["Prédit normale", "Prédit fraude"])
    ax.set_yticks([0, 1], ["Réel normale", "Réel fraude"])
    ax.set_title(title)
    return _save(fig, path)


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def plot_threshold_tradeoff(y_true, y_proba, selected: float, path: str | Path) -> Path:
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    x = _logit(thresholds)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, precision[:-1], label="Précision")
    ax.plot(x, recall[:-1], label="Rappel")
    ax.axvline(_logit(selected), color="black", linestyle="--", label=f"Seuil retenu ({selected:.6f})")
    ticks = [0.01, 0.1, 0.5, 0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999]
    ax.set_xticks(_logit(ticks), [f"{t:g}" for t in ticks], rotation=35)
    ax.set_xlim(x.min(), x.max())
    ax.set_xlabel("Seuil de décision (échelle logit)")
    ax.set_ylabel("Valeur")
    ax.set_title("Précision et rappel selon le seuil (jeu de validation)")
    ax.legend(loc="lower left")
    return _save(fig, path)


def plot_score_distribution(y_true, y_proba, threshold: float, path: str | Path) -> Path:
    y_true = np.asarray(y_true)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bins = np.linspace(-16, 16, 64)
    logit = _logit(y_proba)
    ax.hist(logit[y_true == 0], bins=bins, alpha=0.6, density=True, label="Normales")
    ax.hist(logit[y_true == 1], bins=bins, alpha=0.6, density=True, label="Fraudes")
    ax.axvline(_logit(threshold), color="black", linestyle="--", label="Seuil")
    ax.set_xlabel("Score (logit de la probabilité)")
    ax.set_ylabel("Densité")
    ax.set_title("Distribution des scores (jeu de test)")
    ax.legend()
    return _save(fig, path)
