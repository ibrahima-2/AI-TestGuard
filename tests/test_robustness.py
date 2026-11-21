"""Partie V — Robustesse et sensibilité ; indicateurs de dérive (Partie X)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app.config import FEATURES, PCA_FEATURES
from app.monitoring import (
    alert_rate_status,
    drift_report,
    drift_status,
    fit_reference_ranges,
    out_of_range_rates,
    population_stability_index,
)
from app.prediction import predict_fraud, predict_proba
from app.preprocessing import DataValidationError, invalid_row_mask, validate_data
from app.robustness import (
    add_gaussian_noise,
    inject_values,
    jitter_time,
    perturb_amount_relative,
    scale_amount,
    scale_columns,
    sensitivity,
    set_amount,
    shift_time,
)


@pytest.fixture(scope="module")
def context(trained_model, test_features, synthetic_splits):
    model, threshold = trained_model
    X, y = test_features
    train, _, _ = synthetic_splits
    X_train = train.drop(columns="Class")
    return {
        "model": model,
        "threshold": threshold,
        "X": X,
        "y": y,
        "X_train": X_train,
        "stds": X_train[PCA_FEATURES].std(),
    }


# --- Exemple de l'énoncé (contrat de sortie) -----------------------------------------------------------
def test_prediction_probabilities_are_valid(context):
    probabilities = context["model"].predict_proba(context["X"])[:, 1]
    assert np.isfinite(probabilities).all()
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)


# --- Scénarios 1, 2 et 6 : entrées à rejeter -------------------------------------------------------------
def test_missing_values_are_rejected_and_counted(context):
    rng = np.random.default_rng(0)
    corrupted, n_rows = inject_values(context["X"], ["V14", "Amount"], 0.05, np.nan, rng)
    assert int(invalid_row_mask(corrupted).sum()) == n_rows
    with pytest.raises(DataValidationError, match="manquantes"):
        predict_fraud(context["model"], corrupted, context["threshold"])


def test_valid_rows_remain_scorable_after_filtering(context):
    rng = np.random.default_rng(1)
    corrupted, n_rows = inject_values(context["X"], ["V2"], 0.1, np.inf, rng)
    mask = invalid_row_mask(corrupted)
    result = predict_fraud(context["model"], corrupted[~mask], context["threshold"])
    assert len(result) == len(corrupted) - n_rows


@pytest.mark.parametrize("value", [np.inf, -np.inf, "abc"])
def test_infinite_or_non_numeric_values_are_rejected(context, value):
    corrupted, _ = inject_values(context["X"], ["Amount"], 0.02, value, np.random.default_rng(2))
    with pytest.raises(DataValidationError):
        validate_data(corrupted, require_target=False)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda X: X.drop(columns=["V14"]),
        lambda X: X.assign(Amount=X["Amount"].astype(str)),
        lambda X: X.rename(columns={"V1": "v1"}),
    ],
    ids=["colonne_absente", "type_texte", "nom_modifie"],
)
def test_invalid_schema_is_rejected(context, mutation):
    with pytest.raises(DataValidationError):
        predict_fraud(context["model"], mutation(context["X"]), context["threshold"])


def test_extra_column_rejected_in_strict_mode(context):
    with pytest.raises(DataValidationError, match="non prévues"):
        predict_fraud(context["model"], context["X"].assign(ip="x"), context["threshold"], strict_schema=True)


# --- Scénario 3 : montants extrêmes -> traitement + alerte ----------------------------------------------------
@pytest.mark.parametrize("amount", [1e6, 1e9, 1e12])
def test_extreme_amounts_are_scored_and_flagged_out_of_range(context, amount):
    X_extreme = set_amount(context["X"], amount)
    proba = predict_proba(context["model"], X_extreme)
    assert np.isfinite(proba).all()
    rates = out_of_range_rates(X_extreme, fit_reference_ranges(context["X_train"]))
    assert rates["Amount"] == 1.0


# --- Scénario 4 : perturbations limitées -> mesurer la sensibilité ---------------------------------------------
def test_small_noise_has_small_effect(context):
    rng = np.random.default_rng(3)
    X_noisy = add_gaussian_noise(context["X"], context["stds"], 0.01, rng)
    result = sensitivity(context["model"], context["X"], X_noisy, context["y"], context["threshold"])
    assert result["mean_abs_delta"] < 0.02
    assert result["flip_rate"] < 0.02
    assert result["n_non_finite"] == 0


def test_sensitivity_grows_with_noise_amplitude(context):
    deltas = []
    for epsilon in (0.01, 0.1, 0.5):
        X_noisy = add_gaussian_noise(context["X"], context["stds"], epsilon, np.random.default_rng(4))
        deltas.append(
            sensitivity(context["model"], context["X"], X_noisy, context["y"], context["threshold"])["mean_abs_delta"]
        )
    assert deltas == sorted(deltas)


def test_class_changes_concentrate_near_decision_boundary(context):
    X_noisy = add_gaussian_noise(context["X"], context["stds"], 0.5, np.random.default_rng(5))
    result = sensitivity(context["model"], context["X"], X_noisy, context["y"], context["threshold"])
    assert result["n_flips"] > 0
    assert result["median_margin_flipped"] < result["median_margin_all"]


def test_noise_only_touches_pca_columns(context):
    X_noisy = add_gaussian_noise(context["X"], context["stds"], 0.1, np.random.default_rng(6))
    assert X_noisy["Amount"].equals(context["X"]["Amount"])
    assert X_noisy["Time"].equals(context["X"]["Time"])
    assert not X_noisy["V1"].equals(context["X"]["V1"])


def test_relative_amount_perturbation_stays_within_bounds(context):
    X_pert = perturb_amount_relative(context["X"], 0.05, np.random.default_rng(7))
    ratio = X_pert["Amount"] / context["X"]["Amount"].replace(0, np.nan)
    assert ratio.dropna().between(0.95, 1.05).all()
    assert (X_pert["Amount"] >= 0).all()


def test_time_perturbations_never_produce_negative_time(context):
    assert (jitter_time(context["X"], 1e6, np.random.default_rng(8))["Time"] >= 0).all()
    assert (shift_time(context["X"], -1e9)["Time"] == 0).all()


def test_sensitivity_counts_flips_correctly():
    class Fixed:
        def __init__(self, p):
            self.p = np.asarray(p)

        def predict_proba(self, X):
            p = self.p if X["V1"].iloc[0] == 0 else self.p[::-1]
            return np.column_stack([1 - p, p])

    X0 = pd.DataFrame(np.zeros((4, len(FEATURES))), columns=FEATURES)
    X1 = X0.assign(V1=1.0)
    model = Fixed([0.1, 0.4, 0.6, 0.9])
    result = sensitivity(model, X0, X1, np.array([0, 0, 1, 1]), 0.5)
    assert result["n_flips"] == 4
    assert result["flips_to_fraud"] == 2
    assert result["flips_to_normal"] == 2


# --- Scénario 5 : changement de distribution -> alerte de dérive ------------------------------------------------
def test_psi_is_low_for_same_distribution_and_high_for_shift():
    rng = np.random.default_rng(9)
    reference = rng.normal(0, 1, 5000)
    assert population_stability_index(reference, rng.normal(0, 1, 5000)) < 0.1
    assert population_stability_index(reference, rng.normal(1, 1, 5000)) > 0.2


def test_psi_handles_constant_reference():
    assert population_stability_index(np.zeros(100), np.zeros(50)) == pytest.approx(0.0)


@pytest.mark.parametrize(("psi", "status"), [(0.05, "stable"), (0.15, "surveiller"), (0.5, "alerte")])
def test_drift_status_levels(psi, status):
    assert drift_status(psi) == status


def test_drift_report_flags_only_shifted_features(context):
    shifted = scale_columns(context["X"], ["V1", "V2"], 3.0)
    rows = {r["feature"]: r for r in drift_report(context["X"], shifted)}
    assert rows["V1"]["status"] == "alerte"
    assert rows["V2"]["status"] == "alerte"
    assert rows["V10"]["status"] == "stable"
    assert rows["V10"]["psi"] == pytest.approx(0.0, abs=1e-9)


def test_time_shift_is_detected_out_of_range(context):
    ranges = fit_reference_ranges(context["X_train"])
    before = out_of_range_rates(context["X"], ranges)
    after = out_of_range_rates(shift_time(context["X"], 172_800 * 2), ranges)
    assert before["Time"] == 0.0
    assert after["Time"] == 1.0
    assert after["V1"] == before["V1"]  # les autres variables ne sont pas affectées


def test_alert_rate_monitor():
    assert alert_rate_status(np.array([0] * 98 + [1] * 2), 0.02)["status"] == "normal"
    assert alert_rate_status(np.array([0] * 90 + [1] * 10), 0.02)["status"] == "alerte_hausse"
    assert alert_rate_status(np.zeros(100), 0.02)["status"] == "alerte_baisse"


def test_amount_scaling_changes_scores_but_stays_valid(context):
    proba = predict_proba(context["model"], scale_amount(context["X"], 10))
    assert np.isfinite(proba).all()


# --- Propriété : aucune perturbation bornée ne rend la sortie invalide -------------------------------------------
@settings(max_examples=30, deadline=None)
@given(epsilon=st.floats(0, 2), factor=st.floats(0, 100), seed=st.integers(0, 10_000))
def test_bounded_perturbations_never_break_output_contract(context, epsilon, factor, seed):
    rng = np.random.default_rng(seed)
    X_pert = scale_amount(add_gaussian_noise(context["X"], context["stds"], epsilon, rng), factor)
    proba = predict_proba(context["model"], X_pert)
    assert np.isfinite(proba).all()
    assert ((proba >= 0) & (proba <= 1)).all()
