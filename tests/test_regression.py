"""Partie VI — Validation hors échantillon et non-régression par rapport à une référence figée."""

from __future__ import annotations

import copy

import pytest

from app.config import BASELINE_DIR, FULL_DATA_PATH
from app.evaluation import evaluate_model
from app.preprocessing import preprocess_data, remove_duplicates, split_data
from app.regression import (
    build_run_record,
    compare_to_baseline,
    dataset_fingerprint,
    load_baseline,
    save_baseline,
)
from app.training import train_and_validate

SAMPLE_BASELINE = BASELINE_DIR / "baseline_sample.json"
FULL_BASELINE = BASELINE_DIR / "baseline_full.json"
REQUIRED_KEYS = {"code_version", "libraries", "seed", "model", "threshold", "metrics", "validation_set"}


def _record(**metric_overrides) -> dict:
    metrics = {"pr_auc": 0.80, "roc_auc": 0.98, "recall": 0.82, "precision": 0.88, "fpr": 0.0002, "f1": 0.85}
    metrics.update(metric_overrides)
    return {
        "metrics": metrics,
        "threshold": 0.9,
        "validation_set": {"content_sha256": "abc"},
        "libraries": {"scikit-learn": "1.9.1"},
        "model": {"model_name": "logistic_regression"},
    }


# --- Logique de décision ------------------------------------------------------------------------------
def test_identical_version_is_accepted():
    assert compare_to_baseline(_record(), _record())["decision"] == "accepter"


def test_improvement_is_accepted():
    assert compare_to_baseline(_record(pr_auc=0.85, recall=0.9), _record())["decision"] == "accepter"


def test_small_degradation_triggers_investigation():
    result = compare_to_baseline(_record(pr_auc=0.78), _record())
    assert result["decision"] == "investiguer"


def test_large_degradation_is_rejected():
    result = compare_to_baseline(_record(recall=0.70), _record())
    assert result["decision"] == "rejeter"
    assert next(r for r in result["metrics"] if r["metric"] == "recall")["status"] == "rejeter"


def test_fpr_increase_is_a_degradation():
    assert compare_to_baseline(_record(fpr=0.0015), _record())["decision"] == "rejeter"


def test_undefined_metric_is_rejected():
    assert compare_to_baseline(_record(pr_auc=None), _record())["decision"] == "rejeter"


def test_different_validation_set_is_not_comparable():
    current = _record()
    current["validation_set"] = {"content_sha256": "autre"}
    result = compare_to_baseline(current, _record())
    assert result["decision"] == "investiguer"
    assert any("validation différent" in note for note in result["notes"])


def test_library_and_config_changes_are_reported():
    current = copy.deepcopy(_record())
    current["libraries"] = {"scikit-learn": "2.0"}
    current["model"] = {"model_name": "decision_tree"}
    notes = compare_to_baseline(current, _record())["notes"]
    assert any("bibliothèques" in n for n in notes)
    assert any("Configuration" in n for n in notes)


# --- La référence ne s'écrase pas automatiquement ------------------------------------------------------
def test_baseline_is_never_overwritten_without_force(tmp_path):
    path = tmp_path / "baseline.json"
    save_baseline(_record(), path)
    with pytest.raises(FileExistsError):
        save_baseline(_record(pr_auc=0.1), path)
    assert load_baseline(path)["metrics"]["pr_auc"] == 0.80
    save_baseline(_record(pr_auc=0.1), path, force=True)
    assert load_baseline(path)["metrics"]["pr_auc"] == 0.1


def test_missing_baseline_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_baseline(tmp_path / "absent.json")


def test_dataset_fingerprint_detects_content_change(synthetic_df):
    original = dataset_fingerprint(synthetic_df)
    modified = synthetic_df.copy()
    modified.loc[0, "V1"] += 1e-9
    assert dataset_fingerprint(synthetic_df) == original
    assert dataset_fingerprint(modified)["content_sha256"] != original["content_sha256"]


# --- Référence versionnée -----------------------------------------------------------------------------------
@pytest.mark.parametrize("path", [SAMPLE_BASELINE, FULL_BASELINE], ids=["échantillon", "complet"])
def test_versioned_baseline_records_everything_needed(path):
    baseline = load_baseline(path)
    assert set(baseline) >= REQUIRED_KEYS
    assert {"python", "numpy", "pandas", "scikit-learn"} <= set(baseline["libraries"])
    assert 0 < baseline["threshold"] < 1
    assert {"n_rows", "n_fraud", "content_sha256"} <= set(baseline["validation_set"])


# --- Non-régression du pipeline actuel ----------------------------------------------------------------------
def _current_record(df, baseline):
    train, val, _ = split_data(remove_duplicates(df), seed=baseline["seed"])
    run = train_and_validate(train, val, baseline["model"], seed=baseline["seed"])
    return (
        build_run_record("courant", baseline["model"], run["threshold"], run["val_metrics"], val, baseline["seed"]),
        train,
        val,
    )


def test_current_code_does_not_regress_on_sample(sample_df):
    baseline = load_baseline(SAMPLE_BASELINE)
    current, _, _ = _current_record(sample_df, baseline)
    result = compare_to_baseline(current, baseline)
    assert current["validation_set"]["content_sha256"] == baseline["validation_set"]["content_sha256"]
    assert result["decision"] != "rejeter", result


def test_no_overfitting_gap_on_sample(sample_df):
    baseline = load_baseline(SAMPLE_BASELINE)
    _, train, val = _current_record(sample_df, baseline)
    run = train_and_validate(train, val, baseline["model"], seed=baseline["seed"])
    X_tr, y_tr = preprocess_data(train)
    p_tr = run["pipeline"].predict_proba(X_tr)[:, 1]
    train_metrics = evaluate_model(y_tr, (p_tr >= run["threshold"]).astype(int), p_tr)
    assert train_metrics["pr_auc"] - run["val_metrics"]["pr_auc"] <= 0.10


@pytest.mark.slow
def test_current_code_does_not_regress_on_full_dataset(full_df):
    if not FULL_DATA_PATH.exists():
        pytest.skip("Jeu complet absent")
    baseline = load_baseline(FULL_BASELINE)
    current, _, _ = _current_record(full_df, baseline)
    result = compare_to_baseline(current, baseline)
    assert result["decision"] == "accepter", result
