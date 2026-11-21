"""Parties II et III : tests de l'entraînement, du seuil, de la prédiction et des métriques."""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app.config import FEATURES, PCA_FEATURES
from app.evaluation import bootstrap_ci, check_acceptance, error_analysis, evaluate_model, threshold_tradeoff
from app.prediction import PredictionError, predict_fraud, predict_proba, score_records
from app.preprocessing import DataValidationError
from app.training import build_pipeline, select_threshold, train_model, tune_model


# --- T04 : contrat de sortie ------------------------------------------------------------------
def test_probabilities_are_finite_and_between_0_and_1(trained_model, test_features):
    model, _ = trained_model
    X, _ = test_features
    proba = predict_proba(model, X)
    assert proba.shape == (len(X),)
    assert np.isfinite(proba).all()
    assert ((proba >= 0) & (proba <= 1)).all()


def test_labels_follow_threshold_exactly(trained_model, test_features):
    model, threshold = trained_model
    X, _ = test_features
    result = predict_fraud(model, X, threshold)
    expected = (result["fraud_probability"] >= threshold).astype(int)
    assert result["is_fraud"].tolist() == expected.tolist()
    assert set(result["is_fraud"].unique()) <= {0, 1}


def test_extreme_thresholds(trained_model, test_features):
    model, _ = trained_model
    X, _ = test_features
    assert predict_fraud(model, X, 0.0)["is_fraud"].all()
    assert predict_fraud(model, X, 1.0)["is_fraud"].sum() <= (predict_proba(model, X) == 1.0).sum()


@pytest.mark.parametrize("threshold", [-0.1, 1.5, float("nan"), "0.5", None, True])
def test_invalid_threshold_is_rejected(trained_model, test_features, threshold):
    model, _ = trained_model
    X, _ = test_features
    with pytest.raises(ValueError, match="seuil"):
        predict_fraud(model, X, threshold)


def test_prediction_preserves_index_and_handles_single_row(trained_model, test_features):
    model, threshold = trained_model
    X, _ = test_features
    one = X.iloc[[3]]
    result = predict_fraud(model, one, threshold)
    assert result.index.tolist() == one.index.tolist()
    assert len(result) == 1


def test_column_order_does_not_change_predictions(trained_model, test_features):
    model, threshold = trained_model
    X, _ = test_features
    reordered = X[list(reversed(FEATURES))]
    pd.testing.assert_frame_equal(predict_fraud(model, X, threshold), predict_fraud(model, reordered, threshold))


def test_predictions_are_deterministic(trained_model, test_features):
    model, threshold = trained_model
    X, _ = test_features
    np.testing.assert_array_equal(predict_proba(model, X), predict_proba(model, X))


def test_invalid_input_is_rejected_not_predicted(trained_model, test_features):
    model, threshold = trained_model
    X, _ = test_features
    corrupted = X.copy()
    corrupted.iloc[0, 0] = np.nan
    with pytest.raises(DataValidationError):
        predict_fraud(model, corrupted, threshold)


def test_invalid_model_output_is_detected(test_features):
    class BrokenModel:
        def predict_proba(self, X):
            return np.column_stack([np.zeros(len(X)), np.full(len(X), 1.7)])

    X, _ = test_features
    with pytest.raises(PredictionError):
        predict_proba(BrokenModel(), X)


def test_prediction_logs_volumes_without_transaction_values(trained_model, test_features, caplog):
    model, threshold = trained_model
    X, _ = test_features
    row = X.iloc[[0]].copy()
    row["Amount"] = 123456.78
    with caplog.at_level(logging.DEBUG):
        predict_fraud(model, row, threshold)
    assert "1 transactions" in caplog.text
    assert "123456" not in caplog.text


# --- Entraînement ----------------------------------------------------------------------------------
def test_pipeline_contains_preprocessing_and_model():
    pipeline = build_pipeline("logistic_regression")
    assert list(pipeline.named_steps) == ["preprocess", "model"]
    assert pipeline.named_steps["model"].class_weight == "balanced"


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match="inconnu"):
        build_pipeline("random_magic")


def test_training_requires_both_classes(test_features):
    X, _ = test_features
    with pytest.raises(ValueError, match="deux classes"):
        train_model(X, pd.Series(np.zeros(len(X), dtype=int)))


def test_training_rejects_mismatched_lengths(test_features):
    X, y = test_features
    with pytest.raises(ValueError, match="longueur"):
        train_model(X, y.iloc[:-1])


def test_training_is_reproducible(synthetic_splits):
    train, _, _ = synthetic_splits
    X = train.drop(columns="Class")
    y = train["Class"]
    p1 = train_model(X, y, "decision_tree", seed=3, max_depth=4).predict_proba(X)[:, 1]
    p2 = train_model(X, y, "decision_tree", seed=3, max_depth=4).predict_proba(X)[:, 1]
    np.testing.assert_array_equal(p1, p2)


def test_model_learns_better_than_chance(trained_model, test_features):
    model, _ = trained_model
    X, y = test_features
    proba = predict_proba(model, X)
    metrics = evaluate_model(y, (proba >= 0.5).astype(int), proba)
    assert metrics["roc_auc"] > 0.9


def test_tuning_selects_best_validation_score(synthetic_splits):
    train, val, _ = synthetic_splits
    X_tr, y_tr = train.drop(columns="Class"), train["Class"]
    X_val, y_val = val.drop(columns="Class"), val["Class"]
    grid = [{"max_depth": 1}, {"max_depth": 4}]
    _, params, history = tune_model(X_tr, y_tr, X_val, y_val, "decision_tree", grid=grid)
    best = max(history, key=lambda h: h["val_pr_auc"])
    assert params == best["params"]
    assert len(history) == 2


# --- Seuil de décision -------------------------------------------------------------------------------
def test_threshold_meets_recall_constraint_with_best_precision():
    y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
    proba = np.array([0.1, 0.2, 0.3, 0.4, 0.65, 0.9, 0.6, 0.7, 0.8, 0.95])
    threshold = select_threshold(y, proba, min_recall=0.75)
    metrics = evaluate_model(y, (proba >= threshold).astype(int))
    assert metrics["recall"] >= 0.75
    assert threshold == pytest.approx(0.7)
    assert metrics["precision"] == pytest.approx(0.75)


def test_unreachable_recall_falls_back_to_lowest_threshold():
    y = np.array([0, 1, 0, 1])
    proba = np.array([0.2, 0.4, 0.6, 0.8])
    assert select_threshold(y, proba, min_recall=1.01) == pytest.approx(0.2)


# --- Métriques (Partie III) ---------------------------------------------------------------------------
def test_confusion_counts_and_rates_are_exact():
    y_true = [0, 0, 0, 0, 0, 1, 1, 1]
    y_pred = [0, 0, 0, 1, 1, 1, 1, 0]
    m = evaluate_model(y_true, y_pred)
    assert (m["tp"], m["tn"], m["fp"], m["fn"]) == (2, 3, 2, 1)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 * 0.5 * (2 / 3) / (0.5 + 2 / 3))
    assert m["fpr"] == pytest.approx(0.4)
    assert m["fnr"] == pytest.approx(1 / 3)
    assert m["accuracy"] == pytest.approx(5 / 8)


def test_accuracy_is_misleading_on_imbalanced_data():
    y_true = np.array([1] * 2 + [0] * 998)
    y_pred = np.zeros(1000, dtype=int)
    m = evaluate_model(y_true, y_pred)
    assert m["accuracy"] == pytest.approx(0.998)
    assert m["recall"] == 0.0
    assert m["precision"] == 0.0


def test_auc_metrics_with_probabilities():
    y_true = [0, 0, 1, 1]
    proba = [0.1, 0.4, 0.35, 0.8]
    m = evaluate_model(y_true, [0, 1, 0, 1], proba)
    assert m["roc_auc"] == pytest.approx(0.75)
    assert m["pr_auc"] == pytest.approx(0.8333, abs=1e-4)


def test_undefined_metrics_are_nan_when_single_class():
    m = evaluate_model([0, 0, 0], [0, 1, 0], [0.1, 0.9, 0.2])
    assert math.isnan(m["recall"])
    assert math.isnan(m["roc_auc"])
    assert m["fpr"] == pytest.approx(1 / 3)


@pytest.mark.parametrize(
    ("y_true", "y_pred", "proba", "message"),
    [
        ([0, 1], [0], None, "Longueurs"),
        ([0, 2], [0, 1], None, "0 et 1"),
        ([], [], None, "vide"),
        ([0, 1], [0, 1], [0.2, 1.4], "probabilités"),
        ([0, 1], [0, 1], [0.2], "même taille"),
    ],
)
def test_evaluate_rejects_inconsistent_inputs(y_true, y_pred, proba, message):
    with pytest.raises(ValueError, match=message):
        evaluate_model(y_true, y_pred, proba)


def test_acceptance_criteria_pass_and_fail():
    criteria = {"recall": (">=", 0.75), "fpr": ("<=", 0.005)}
    assert check_acceptance({"recall": 0.8, "fpr": 0.001}, criteria)["accepted"]
    result = check_acceptance({"recall": 0.74, "fpr": 0.001}, criteria)
    assert not result["accepted"]
    assert not result["criteria"]["recall"]["passed"]
    assert not check_acceptance({"recall": float("nan"), "fpr": 0.0}, criteria)["accepted"]


def test_bootstrap_interval_contains_point_estimate_and_is_reproducible():
    rng = np.random.default_rng(0)
    y = np.array([1] * 40 + [0] * 400)
    proba = np.clip(y * 0.6 + rng.normal(0.2, 0.15, len(y)), 0, 1)
    point = evaluate_model(y, (proba >= 0.5).astype(int), proba)
    ci = bootstrap_ci(y, proba, 0.5, n_boot=200, seed=1)
    assert ci["recall"]["low"] <= point["recall"] <= ci["recall"]["high"]
    assert ci == bootstrap_ci(y, proba, 0.5, n_boot=200, seed=1)


def test_threshold_tradeoff_is_monotonic():
    rng = np.random.default_rng(0)
    y = np.array([1] * 30 + [0] * 300)
    proba = np.clip(y * 0.5 + rng.uniform(0, 0.5, len(y)), 0, 1)
    rows = threshold_tradeoff(y, proba, [0.1, 0.3, 0.5, 0.7, 0.9])
    recalls = [r["recall"] for r in rows]
    alerts = [r["alerts"] for r in rows]
    assert recalls == sorted(recalls, reverse=True)
    assert alerts == sorted(alerts, reverse=True)


def test_error_analysis_partitions_all_transactions(test_features, trained_model):
    model, threshold = trained_model
    X, y = test_features
    proba = predict_proba(model, X)
    summary = error_analysis(X, y, proba, threshold)
    assert sum(summary[k]["count"] for k in ("TP", "FN", "FP", "TN")) == len(X)


# --- Point d'entrée de type API ---------------------------------------------------------------------
def _record(X: pd.DataFrame, i: int = 0) -> dict:
    return {k: float(v) for k, v in X.iloc[i].items()}


def test_score_records_returns_one_result_per_record(trained_model, test_features):
    model, threshold = trained_model
    X, _ = test_features
    results = score_records(model, [_record(X, 0), _record(X, 1)], threshold)
    assert len(results) == 2
    assert set(results[0]) == {"fraud_probability", "is_fraud"}
    assert isinstance(results[0]["is_fraud"], bool)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda r: r.pop("V3"), "champs manquants"),
        (lambda r: r.update(card_number=1234), "non autorisés"),
        (lambda r: r.update(Amount="12.5"), "non numérique"),
        (lambda r: r.update(V1=True), "non numérique"),
        (lambda r: r.update(Amount=None), "non numérique"),
    ],
)
def test_score_records_rejects_malformed_records(trained_model, test_features, mutate, message):
    model, threshold = trained_model
    X, _ = test_features
    record = _record(X)
    mutate(record)
    with pytest.raises(DataValidationError, match=message):
        score_records(model, [record], threshold)


@pytest.mark.parametrize("payload", [[], {}, "transactions", None, [1, 2]])
def test_score_records_rejects_malformed_payload(trained_model, payload):
    model, threshold = trained_model
    with pytest.raises(DataValidationError):
        score_records(model, payload, threshold)


def test_score_records_limits_batch_size(trained_model, test_features):
    model, threshold = trained_model
    X, _ = test_features
    with pytest.raises(DataValidationError, match="Trop"):
        score_records(model, [_record(X)] * 11, threshold, max_records=10)


# --- Propriété : toute transaction valide reçoit une probabilité valide -------------------------------
@settings(max_examples=50, deadline=None)
@given(
    pca=st.lists(st.floats(-50, 50, allow_nan=False), min_size=28, max_size=28),
    time=st.floats(0, 1e6, allow_nan=False),
    amount=st.floats(0, 1e12, allow_nan=False),
)
def test_any_valid_transaction_gets_valid_probability(trained_model, pca, time, amount):
    model, threshold = trained_model
    row = pd.DataFrame([[time, *pca, amount]], columns=["Time", *PCA_FEATURES, "Amount"])
    result = predict_fraud(model, row, threshold)
    p = result["fraud_probability"].iloc[0]
    assert 0.0 <= p <= 1.0
