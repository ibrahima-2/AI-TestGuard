"""Partie IV — Tests d'intégration de bout en bout et prévention des fuites de données."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from app import cli
from app.config import FEATURES, SAMPLE_DATA_PATH, TARGET
from app.evaluation import evaluate_model
from app.prediction import ModelLoadError, load_model, predict_fraud, save_model
from app.preprocessing import load_data, preprocess_data, remove_duplicates, split_data, validate_data
from app.training import select_threshold, train_and_validate, train_model
from app.utils import sha256_file

CONFIG = {"model_name": "logistic_regression", "params": {"C": 0.1}, "features": FEATURES, "min_recall": 0.8}


@pytest.fixture(scope="module")
def pipeline_run(sample_splits, tmp_path_factory):
    """Scénario 4.1 complet sur l'échantillon réel versionné."""
    train, val, test = sample_splits
    X_tr, y_tr = preprocess_data(train)
    X_val, y_val = preprocess_data(val)
    X_te, y_te = preprocess_data(test)
    model = train_model(X_tr, y_tr, "logistic_regression", C=0.1)
    threshold = select_threshold(y_val, model.predict_proba(X_val)[:, 1])
    path = save_model(model, tmp_path_factory.mktemp("models") / "model.joblib", threshold, {"config": CONFIG})
    return {"model": model, "threshold": threshold, "path": path, "X_tr": X_tr, "X_te": X_te, "y_te": y_te}


def test_end_to_end_pipeline_produces_valid_metrics(pipeline_run):
    loaded, info = load_model(pipeline_run["path"])
    result = predict_fraud(loaded, pipeline_run["X_te"], info["threshold"])
    metrics = evaluate_model(pipeline_run["y_te"], result["is_fraud"], result["fraud_probability"])
    assert metrics["n"] == len(pipeline_run["X_te"])
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert metrics["roc_auc"] > 0.9


def test_predictions_identical_after_save_and_reload(pipeline_run):
    loaded, info = load_model(pipeline_run["path"])
    before = pipeline_run["model"].predict_proba(pipeline_run["X_te"])[:, 1]
    after = loaded.predict_proba(pipeline_run["X_te"])[:, 1]
    np.testing.assert_allclose(after, before, rtol=0, atol=1e-12)
    assert info["threshold"] == pytest.approx(pipeline_run["threshold"])


def test_saved_metadata_documents_the_model(pipeline_run):
    _, info = load_model(pipeline_run["path"])
    assert {"threshold", "sha256", "created_at", "libraries", "config"} <= set(info)
    assert {"python", "scikit-learn", "numpy", "pandas"} <= set(info["libraries"])


def test_no_non_finite_values_after_preprocessing(pipeline_run, sample_splits):
    preprocessor = pipeline_run["model"].named_steps["preprocess"]
    for part in sample_splits:
        X, _ = preprocess_data(part)
        transformed = np.asarray(preprocessor.transform(X))
        assert transformed.shape == (len(part), len(FEATURES))
        assert np.isfinite(transformed).all()


def test_scaler_is_fitted_on_training_data_only(pipeline_run):
    scaler = pipeline_run["model"].named_steps["preprocess"].named_transformers_["scale"]
    columns = [c for c in FEATURES if c != "Amount"]
    np.testing.assert_allclose(scaler.mean_, pipeline_run["X_tr"][columns].mean().to_numpy())
    assert not np.allclose(scaler.mean_, pipeline_run["X_te"][columns].mean().to_numpy())
    amount_scaler = pipeline_run["model"].named_steps["preprocess"].named_transformers_["amount"]["scale"]
    assert amount_scaler.mean_[0] == pytest.approx(np.log1p(pipeline_run["X_tr"]["Amount"]).mean())


def test_partitions_are_disjoint_and_free_of_shared_duplicates(sample_splits):
    train, val, test = sample_splits
    assert set(train.index).isdisjoint(test.index)
    assert set(val.index).isdisjoint(test.index)
    shared = pd.merge(train, test, how="inner", on=[*FEATURES, TARGET])
    assert shared.empty


def test_threshold_is_derived_from_validation_only_and_reproducible(sample_splits):
    train, val, _ = sample_splits
    first = train_and_validate(train, val, CONFIG)
    second = train_and_validate(train, val, CONFIG)
    _, y_val = preprocess_data(val)
    assert first["threshold"] == second["threshold"]
    assert first["threshold"] == select_threshold(y_val, first["val_proba"])
    assert first["val_metrics"]["n"] == len(val)


# --- Gestion explicite des erreurs de chargement ---------------------------------------------------------
def test_missing_model_file_raises(tmp_path):
    with pytest.raises(ModelLoadError, match="introuvable"):
        load_model(tmp_path / "absent.joblib")


def test_wrong_extension_is_refused(tmp_path):
    with pytest.raises(ModelLoadError, match="Extension"):
        load_model(tmp_path / "model.pkl")


def test_tampered_model_is_refused(pipeline_run, tmp_path):
    copy = tmp_path / "model.joblib"
    copy.write_bytes(pipeline_run["path"].read_bytes() + b"\x00")
    copy.with_suffix(".json").write_text(pipeline_run["path"].with_suffix(".json").read_text())
    with pytest.raises(ModelLoadError, match="SHA-256"):
        load_model(copy)


def test_missing_metadata_is_refused(pipeline_run, tmp_path):
    copy = tmp_path / "model.joblib"
    copy.write_bytes(pipeline_run["path"].read_bytes())
    with pytest.raises(ModelLoadError, match="Métadonnées"):
        load_model(copy)


def test_corrupted_metadata_is_refused(pipeline_run, tmp_path):
    copy = tmp_path / "model.joblib"
    copy.write_bytes(pipeline_run["path"].read_bytes())
    copy.with_suffix(".json").write_text("{not json")
    with pytest.raises(ModelLoadError, match="illisibles"):
        load_model(copy)


def test_object_without_predict_proba_is_refused(tmp_path):
    path = tmp_path / "model.joblib"
    joblib.dump({"not": "a model"}, path)
    path.with_suffix(".json").write_text(json.dumps({"sha256": sha256_file(path), "threshold": 0.5}))
    with pytest.raises(ModelLoadError, match="probabiliste"):
        load_model(path)


def test_invalid_threshold_in_metadata_is_refused(pipeline_run, tmp_path):
    copy = tmp_path / "model.joblib"
    copy.write_bytes(pipeline_run["path"].read_bytes())
    meta = json.loads(pipeline_run["path"].with_suffix(".json").read_text())
    meta["threshold"] = 3.0
    copy.with_suffix(".json").write_text(json.dumps(meta))
    with pytest.raises(ModelLoadError, match="Seuil"):
        load_model(copy)


def test_save_requires_joblib_extension(pipeline_run, tmp_path):
    with pytest.raises(ValueError, match="joblib"):
        save_model(pipeline_run["model"], tmp_path / "model.pkl", 0.5)


# --- Chaîne complète en ligne de commande (échantillon) ----------------------------------------------------
def test_cli_full_chain_on_sample(tmp_path):
    out, models = tmp_path / "reports", tmp_path / "models"
    common = ["--data", str(SAMPLE_DATA_PATH), "--output", str(out)]
    cli.main(["train", *common, "--models-dir", str(models), "--n-boot", "50"])
    model = ["--model", str(models / "fraud_model.joblib")]
    baseline = ["--baseline", str(tmp_path / "baseline.json")]
    cli.main(["robustness", *common, *model])
    cli.main(["performance", *common, *model, "--repeats", "50"])
    cli.main(["drift", *common, *model])
    cli.main(["baseline-create", *common, *model, *baseline])
    with pytest.raises(FileExistsError):
        cli.main(["baseline-create", *common, *model, *baseline])
    cli.main(["regression", *common, *model, *baseline])

    for name in ("evaluation", "robustness", "performance", "drift", "regression"):
        assert (out / "results" / f"{name}.json").is_file()
    for name in ("roc_curves", "pr_curves", "threshold_tradeoff", "score_distribution"):
        assert (out / "figures" / f"{name}.png").stat().st_size > 0

    evaluation = json.loads((out / "results" / "evaluation.json").read_text())
    assert evaluation["reload_max_abs_diff"] == 0.0
    assert evaluation["data"]["n_duplicates_removed"] >= 0
    robustness = json.loads((out / "results" / "robustness.json").read_text())
    assert all(r["rejected"] for r in robustness["rejection_scenarios"])
    regression = json.loads((out / "results" / "regression.json").read_text())
    assert regression["candidates"][0]["comparison"]["decision"] == "accepter"


def test_cli_make_sample(tmp_path, sample_df):
    source = tmp_path / "source.csv"
    sample_df.to_csv(source, index=False)
    output = tmp_path / "mini.csv"
    cli.main(["make-sample", "--data", str(source), "--output", str(output), "--n-normal", "50", "--n-fraud", "10"])
    mini = load_data(output)
    validate_data(mini)
    assert len(mini) == 60
    assert mini[TARGET].sum() == 10


# --- Jeu complet (tests lents) ---------------------------------------------------------------------------
@pytest.mark.slow
def test_full_dataset_matches_documented_characteristics(full_df):
    validate_data(full_df)
    assert full_df.shape == (284_807, 31)
    assert int(full_df[TARGET].sum()) == 492
    assert int(full_df.duplicated().sum()) == 1081


@pytest.mark.slow
def test_without_deduplication_duplicates_leak_across_splits(full_df):
    """Justifie la déduplication : sans elle, des transactions identiques sont dans train ET test."""
    train, _, test = split_data(full_df)
    leaked = pd.merge(train, test, how="inner", on=[*FEATURES, TARGET])
    assert len(leaked) > 0
    train_d, _, test_d = split_data(remove_duplicates(full_df))
    assert pd.merge(train_d, test_d, how="inner", on=[*FEATURES, TARGET]).empty
