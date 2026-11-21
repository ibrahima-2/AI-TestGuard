"""Partie II : tests unitaires des fonctions de chargement, validation et préparation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from sklearn.preprocessing import StandardScaler

from app.config import FEATURES, PCA_FEATURES, TARGET
from app.preprocessing import (
    DataLoadError,
    DataValidationError,
    build_preprocessor,
    invalid_row_mask,
    load_data,
    preprocess_data,
    remove_duplicates,
    split_data,
    validate_data,
)


# --- Exemple de l'énoncé --------------------------------------------------------------
def test_standard_scaler():
    data = pd.DataFrame({"Amount": [100.0, 200.0, 300.0]})
    scaled = StandardScaler().fit_transform(data)
    assert np.isfinite(scaled).all()
    assert abs(scaled.mean()) < 1e-10
    assert np.isclose(scaled.std(), 1.0)


# --- T01 : colonnes requises --------------------------------------------------------------
def test_valid_data_is_accepted_unchanged(synthetic_df):
    original = synthetic_df.copy()
    result = validate_data(synthetic_df)
    pd.testing.assert_frame_equal(result, original)


@pytest.mark.parametrize("column", ["Time", "V1", "V14", "V28", "Amount", TARGET])
def test_missing_column_is_rejected_and_named(synthetic_df, column):
    with pytest.raises(DataValidationError, match=f"Colonnes manquantes.*'{column}'"):
        validate_data(synthetic_df.drop(columns=[column]))


def test_target_not_required_for_inference(synthetic_df):
    validate_data(synthetic_df.drop(columns=[TARGET]), require_target=False)


def test_duplicated_column_is_rejected(synthetic_df):
    duplicated = pd.concat([synthetic_df, synthetic_df[["V3"]]], axis=1)
    with pytest.raises(DataValidationError, match="Colonnes dupliquées.*V3"):
        validate_data(duplicated)


def test_extra_column_allowed_by_default_but_rejected_in_strict_mode(synthetic_df):
    with_extra = synthetic_df.assign(merchant_id=1)
    validate_data(with_extra)
    with pytest.raises(DataValidationError, match="non prévues.*merchant_id"):
        validate_data(with_extra, allow_extra_columns=False)


# --- Données vides et objets invalides ----------------------------------------------------------
def test_empty_dataframe_with_columns_is_rejected(synthetic_df):
    with pytest.raises(DataValidationError, match="Données vides"):
        validate_data(synthetic_df.iloc[0:0])


def test_dataframe_without_columns_is_rejected():
    with pytest.raises(DataValidationError, match="Données vides"):
        validate_data(pd.DataFrame())


@pytest.mark.parametrize("bad_input", [None, [1, 2, 3], {"Time": [0]}, np.zeros((3, 31))])
def test_non_dataframe_input_is_rejected(bad_input):
    with pytest.raises(DataValidationError, match="pandas.DataFrame"):
        validate_data(bad_input)


# --- T02 : types -----------------------------------------------------------------------------
def test_text_column_is_rejected(synthetic_df):
    df = synthetic_df.assign(Amount=synthetic_df["Amount"].astype(str))
    with pytest.raises(DataValidationError, match="Type non numérique.*Amount"):
        validate_data(df)


def test_boolean_column_is_rejected(synthetic_df):
    df = synthetic_df.assign(V5=synthetic_df["V5"] > 0)
    with pytest.raises(DataValidationError, match="Type non numérique.*V5"):
        validate_data(df)


def test_integer_columns_are_accepted(synthetic_df):
    df = synthetic_df.assign(Time=synthetic_df["Time"].round().astype(int))
    validate_data(df)


# --- T03 : valeurs manquantes / infinies -------------------------------------------------------------
@pytest.mark.parametrize("column", ["Time", "V7", "Amount"])
def test_missing_value_is_rejected_and_column_named(synthetic_df, column):
    df = synthetic_df.copy()
    df.loc[3, column] = np.nan
    with pytest.raises(DataValidationError, match=f"Valeurs manquantes.*{column}"):
        validate_data(df)


@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_infinite_value_is_rejected(synthetic_df, value):
    df = synthetic_df.copy()
    df.loc[0, "V2"] = value
    with pytest.raises(DataValidationError, match="Valeurs infinies.*V2"):
        validate_data(df)


def test_nullable_dtype_with_missing_value_is_rejected(synthetic_df):
    df = synthetic_df.copy()
    df["V9"] = pd.array([None, *df["V9"].iloc[1:]], dtype="Float64")
    with pytest.raises(DataValidationError, match="Valeurs manquantes.*V9"):
        validate_data(df)


@pytest.mark.parametrize(("column", "message"), [("Amount", "Montant"), ("Time", "Temps")])
def test_negative_amount_or_time_is_rejected(synthetic_df, column, message):
    df = synthetic_df.copy()
    df.loc[1, column] = -1.0
    with pytest.raises(DataValidationError, match=message):
        validate_data(df)


def test_error_message_does_not_leak_values(synthetic_df):
    df = synthetic_df.copy()
    df.loc[0, "Amount"] = -98765.43
    with pytest.raises(DataValidationError) as excinfo:
        validate_data(df)
    assert "98765" not in str(excinfo.value)


# --- Cible -------------------------------------------------------------------------------------
@pytest.mark.parametrize("bad_value", [2, -1, 0.5])
def test_target_values_other_than_0_and_1_are_rejected(synthetic_df, bad_value):
    df = synthetic_df.copy()
    df[TARGET] = df[TARGET].astype(float)
    df.loc[0, TARGET] = bad_value
    with pytest.raises(DataValidationError, match="cible non autorisées"):
        validate_data(df)


def test_missing_target_is_rejected(synthetic_df):
    df = synthetic_df.copy()
    df[TARGET] = df[TARGET].astype(float)
    df.loc[0, TARGET] = np.nan
    with pytest.raises(DataValidationError, match="cible"):
        validate_data(df)


def test_float_target_with_binary_values_is_accepted(synthetic_df):
    validate_data(synthetic_df.assign(**{TARGET: synthetic_df[TARGET].astype(float)}))


# --- invalid_row_mask : comptage ligne à ligne ---------------------------------------------------------
def test_invalid_row_mask_flags_exactly_the_corrupted_rows(synthetic_df):
    df = synthetic_df.copy()
    df["Amount"] = df["Amount"].astype(object)
    df.loc[2, "V1"] = np.nan
    df.loc[5, "V3"] = np.inf
    df.loc[7, "Amount"] = "abc"
    df.loc[9, "Amount"] = -5.0
    mask = invalid_row_mask(df)
    assert mask.dtype == bool
    assert sorted(df.index[mask]) == [2, 5, 7, 9]


def test_invalid_row_mask_requires_schema(synthetic_df):
    with pytest.raises(DataValidationError):
        invalid_row_mask(synthetic_df.drop(columns=["V1"]))


# --- Préparation et cohérence des dimensions ------------------------------------------------------------
def test_preprocess_returns_features_in_contract_order(synthetic_df):
    shuffled = synthetic_df[list(reversed(synthetic_df.columns))]
    X, y = preprocess_data(shuffled)
    assert list(X.columns) == FEATURES
    assert X.shape == (len(synthetic_df), 30)
    assert (X.dtypes == "float64").all()
    assert y.tolist() == synthetic_df[TARGET].tolist()


def test_preprocess_without_target_returns_none(synthetic_df):
    X, y = preprocess_data(synthetic_df.drop(columns=[TARGET]), require_target=False)
    assert y is None
    assert X.shape[1] == len(FEATURES)


def test_preprocess_rejects_invalid_data(synthetic_df):
    df = synthetic_df.copy()
    df.loc[0, "V4"] = np.nan
    with pytest.raises(DataValidationError):
        preprocess_data(df)


def test_preprocessor_output_is_finite_and_keeps_dimensions(synthetic_df):
    X, _ = preprocess_data(synthetic_df)
    transformed = build_preprocessor().fit_transform(X)
    assert transformed.shape == (len(X), 30)
    assert np.isfinite(np.asarray(transformed)).all()


def test_preprocessor_handles_zero_and_extreme_amounts(synthetic_df):
    X, _ = preprocess_data(synthetic_df)
    preprocessor = build_preprocessor().fit(X)
    extreme = X.iloc[:3].copy()
    extreme["Amount"] = [0.0, 1e9, 1e15]
    assert np.isfinite(np.asarray(preprocessor.transform(extreme))).all()


def test_preprocessor_without_time_drops_the_column(synthetic_df):
    X, _ = preprocess_data(synthetic_df)
    features = [f for f in FEATURES if f != "Time"]
    preprocessor = build_preprocessor(features).fit(X)
    assert "Time" not in preprocessor.get_feature_names_out()
    assert preprocessor.transform(X).shape == (len(X), 29)


def test_single_observation_is_processed(synthetic_df):
    X_train, _ = preprocess_data(synthetic_df)
    preprocessor = build_preprocessor().fit(X_train)
    X_one, y_one = preprocess_data(synthetic_df.iloc[[0]])
    out = np.asarray(preprocessor.transform(X_one))
    assert X_one.shape == (1, 30)
    assert len(y_one) == 1
    assert out.shape == (1, 30)
    assert np.isfinite(out).all()


def test_scaler_uses_training_statistics_only(synthetic_df):
    X, _ = preprocess_data(synthetic_df)
    train, other = X.iloc[:400], X.iloc[400:] + 3.0
    preprocessor = build_preprocessor().fit(train)
    scaler = preprocessor.named_transformers_["scale"]
    np.testing.assert_allclose(scaler.mean_, train[[c for c in FEATURES if c != "Amount"]].mean())
    shifted = np.asarray(preprocessor.transform(other))
    # Les données décalées ne sont pas recentrées : la moyenne transformée reste loin de 0.
    assert abs(shifted[:, 2:].mean()) > 1.0


# --- Doublons ------------------------------------------------------------------------------------
def test_remove_duplicates_keeps_first_occurrence(synthetic_df):
    with_dups = pd.concat([synthetic_df, synthetic_df.iloc[:5]], ignore_index=True)
    deduplicated = remove_duplicates(with_dups)
    assert len(deduplicated) == len(synthetic_df)
    assert not deduplicated.duplicated().any()
    pd.testing.assert_frame_equal(deduplicated, synthetic_df.reset_index(drop=True))


# --- Découpage ------------------------------------------------------------------------------------------
def test_split_sizes_and_disjoint_indices(synthetic_df):
    train, val, test = split_data(synthetic_df)
    n = len(synthetic_df)
    assert len(train) + len(val) + len(test) == n
    assert abs(len(test) - 0.2 * n) <= 1
    assert abs(len(val) - 0.2 * n) <= 1
    assert set(train.index).isdisjoint(val.index)
    assert set(train.index).isdisjoint(test.index)
    assert set(val.index).isdisjoint(test.index)


def test_split_preserves_class_proportion(synthetic_df):
    rate = synthetic_df[TARGET].mean()
    for part in split_data(synthetic_df):
        assert part[TARGET].mean() == pytest.approx(rate, abs=0.015)


def test_split_is_reproducible_with_same_seed(synthetic_df):
    first = split_data(synthetic_df, seed=7)
    second = split_data(synthetic_df, seed=7)
    other = split_data(synthetic_df, seed=8)
    for a, b in zip(first, second, strict=True):
        assert a.index.equals(b.index)
    assert not first[2].index.equals(other[2].index)


def test_split_requires_target(synthetic_df):
    with pytest.raises(DataValidationError):
        split_data(synthetic_df.drop(columns=[TARGET]))


# --- Chargement de fichiers ---------------------------------------------------------------------------
def test_load_data_roundtrip(tmp_path, synthetic_df):
    path = tmp_path / "transactions.csv"
    synthetic_df.to_csv(path, index=False)
    loaded = load_data(path)
    pd.testing.assert_frame_equal(loaded, synthetic_df, check_exact=False)


def test_load_missing_file_raises_explicit_error(tmp_path):
    with pytest.raises(DataLoadError, match="introuvable"):
        load_data(tmp_path / "absent.csv")


def test_load_rejects_unexpected_extension(tmp_path):
    path = tmp_path / "transactions.pkl"
    path.write_bytes(b"\x80\x04")
    with pytest.raises(DataLoadError, match="Extension"):
        load_data(path)


def test_load_rejects_empty_file(tmp_path):
    path = tmp_path / "vide.csv"
    path.write_text("")
    with pytest.raises(DataLoadError, match="illisible"):
        load_data(path)


def test_load_rejects_oversized_file(tmp_path, synthetic_df):
    path = tmp_path / "gros.csv"
    synthetic_df.to_csv(path, index=False)
    with pytest.raises(DataLoadError, match="volumineux"):
        load_data(path, max_size_mb=0.001)


# --- Tests par propriétés (Hypothesis) ---------------------------------------------------------------------
finite_value = st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    pca=st.lists(st.lists(finite_value, min_size=28, max_size=28), min_size=1, max_size=20),
    data=st.data(),
)
def test_any_valid_batch_passes_validation_and_stays_finite(pca, data):
    n = len(pca)
    times = data.draw(st.lists(st.floats(0, 1e6, allow_nan=False), min_size=n, max_size=n))
    amounts = data.draw(st.lists(st.floats(0, 1e9, allow_nan=False), min_size=n, max_size=n))
    df = pd.DataFrame(pca, columns=PCA_FEATURES)
    df.insert(0, "Time", times)
    df["Amount"] = amounts
    X, _ = preprocess_data(df, require_target=False)
    assert X.shape == (n, 30)
    assert np.isfinite(X.to_numpy()).all()
