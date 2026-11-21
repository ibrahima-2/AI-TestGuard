"""Construction, entraînement et réglage des modèles (données d'entraînement / validation uniquement)."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from app.config import FEATURES, MIN_RECALL_FOR_THRESHOLD, SEED
from app.evaluation import evaluate_model
from app.preprocessing import build_preprocessor, preprocess_data

logger = logging.getLogger(__name__)

MODEL_NAMES = ("logistic_regression", "decision_tree")

# Grilles volontairement réduites : modèles légers, réglage sur la validation.
PARAM_GRIDS = {
    "logistic_regression": [{"C": c} for c in (0.01, 0.1, 1.0)],
    "decision_tree": [{"max_depth": depth, "min_samples_leaf": leaf} for depth in (4, 6, 8) for leaf in (1, 20)],
}


def _make_estimator(model_name: str, seed: int, params: dict):
    params = dict(params)
    if model_name == "logistic_regression":
        params.setdefault("class_weight", "balanced")
        params.setdefault("max_iter", 2000)
        return LogisticRegression(random_state=seed, **params)
    if model_name == "decision_tree":
        params.setdefault("class_weight", "balanced")
        return DecisionTreeClassifier(random_state=seed, **params)
    if model_name == "dummy":
        return DummyClassifier(strategy="most_frequent")
    raise ValueError(f"Modèle inconnu : {model_name}")


def build_pipeline(
    model_name: str = "logistic_regression",
    seed: int = SEED,
    features: list[str] | None = None,
    **params,
) -> Pipeline:
    """Pipeline scikit-learn : transformations + modèle, ajustés ensemble sur l'entraînement."""
    return Pipeline(
        [
            ("preprocess", build_preprocessor(features or FEATURES)),
            ("model", _make_estimator(model_name, seed, params)),
        ]
    )


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str = "logistic_regression",
    seed: int = SEED,
    features: list[str] | None = None,
    **params,
) -> Pipeline:
    """Entraîne le pipeline sur des données d'ENTRAÎNEMENT."""
    if len(X) != len(y):
        raise ValueError(f"X et y n'ont pas la même longueur ({len(X)} != {len(y)})")
    if model_name != "dummy" and set(np.unique(y).tolist()) != {0, 1}:
        raise ValueError("L'entraînement nécessite les deux classes (0 et 1) dans y")
    pipeline = build_pipeline(model_name, seed, features, **params)
    pipeline.fit(X, y)
    return pipeline


def select_threshold(y_true, y_proba, min_recall: float = MIN_RECALL_FOR_THRESHOLD) -> float:
    """Seuil maximisant la précision sous contrainte de rappel >= min_recall.

    À appeler sur le jeu de VALIDATION. Si la contrainte est inatteignable,
    le seuil le plus bas (rappel maximal) est retourné.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    precision, recall = precision[:-1], recall[:-1]
    eligible = np.flatnonzero(recall >= min_recall)
    if eligible.size == 0:
        logger.warning("Rappel %.2f inatteignable : seuil minimal retenu", min_recall)
        return float(thresholds[0])
    best = eligible[np.argmax(precision[eligible])]
    return float(thresholds[best])


def tune_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_name: str,
    seed: int = SEED,
    grid: list[dict] | None = None,
    features: list[str] | None = None,
) -> tuple[Pipeline, dict, list[dict]]:
    """Choisit les hyperparamètres qui maximisent la PR-AUC de VALIDATION."""
    history = []
    best = (None, None, -np.inf)
    for params in grid or PARAM_GRIDS[model_name]:
        pipeline = train_model(X_train, y_train, model_name, seed, features, **params)
        score = float(average_precision_score(y_val, pipeline.predict_proba(X_val)[:, 1]))
        history.append({"params": params, "val_pr_auc": score})
        if score > best[2]:
            best = (pipeline, params, score)
    return best[0], dict(best[1]), history


def train_and_validate(train_df: pd.DataFrame, val_df: pd.DataFrame, config: dict, seed: int = SEED) -> dict:
    """Entraîne une configuration figée, fixe son seuil et la mesure sur la validation.

    Utilisé pour la référence de non-régression et l'évaluation de nouvelles versions :
    le jeu de test n'intervient pas.
    """
    X_train, y_train = preprocess_data(train_df)
    X_val, y_val = preprocess_data(val_df)
    pipeline = train_model(
        X_train,
        y_train,
        config["model_name"],
        seed,
        features=config.get("features"),
        **config.get("params", {}),
    )
    proba = pipeline.predict_proba(X_val)[:, 1]
    threshold = select_threshold(y_val, proba, config.get("min_recall", MIN_RECALL_FOR_THRESHOLD))
    metrics = evaluate_model(y_val, (proba >= threshold).astype(int), proba)
    return {"pipeline": pipeline, "threshold": threshold, "val_metrics": metrics, "val_proba": proba}
