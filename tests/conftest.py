"""Jeux de données et modèles partagés par les tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.config import FEATURES, FULL_DATA_PATH, PCA_FEATURES, SAMPLE_DATA_PATH, SEED, TARGET
from app.preprocessing import load_data, preprocess_data, remove_duplicates, split_data
from app.training import select_threshold, train_model


def make_synthetic_df(n: int = 800, fraud_rate: float = 0.1, seed: int = SEED) -> pd.DataFrame:
    """Données synthétiques au format du contrat ; les fraudes diffèrent sur V1 à V4."""
    rng = np.random.default_rng(seed)
    n_fraud = max(2, round(n * fraud_rate))
    y = np.zeros(n, dtype=int)
    y[:n_fraud] = 1
    rng.shuffle(y)
    data = {"Time": np.sort(rng.uniform(0, 172_792, n))}
    for i, column in enumerate(PCA_FEATURES):
        shift = (2.5 if i % 2 == 0 else -2.5) if i < 4 else 0.0
        data[column] = rng.normal(0.0, 1.0, n) + shift * y
    data["Amount"] = np.round(rng.lognormal(3.0, 1.2, n), 2)
    df = pd.DataFrame(data)[FEATURES]
    df[TARGET] = y
    return df


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    return make_synthetic_df()


@pytest.fixture(scope="session")
def synthetic_splits():
    return split_data(make_synthetic_df(n=2000, fraud_rate=0.08))


@pytest.fixture(scope="session")
def trained_model(synthetic_splits):
    """Régression logistique entraînée sur des données synthétiques, seuil réglé sur la validation."""
    train, val, _ = synthetic_splits
    X_train, y_train = preprocess_data(train)
    X_val, y_val = preprocess_data(val)
    model = train_model(X_train, y_train, "logistic_regression")
    threshold = select_threshold(y_val, model.predict_proba(X_val)[:, 1])
    return model, threshold


@pytest.fixture(scope="session")
def test_features(synthetic_splits):
    _, _, test = synthetic_splits
    return preprocess_data(test)


@pytest.fixture(scope="session")
def sample_df() -> pd.DataFrame:
    """Échantillon réel versionné (data/sample_creditcard.csv) : il doit exister."""
    if not SAMPLE_DATA_PATH.exists():
        pytest.fail(f"Échantillon versionné absent : {SAMPLE_DATA_PATH}")
    return load_data(SAMPLE_DATA_PATH)


@pytest.fixture(scope="session")
def full_df() -> pd.DataFrame:
    if not FULL_DATA_PATH.exists():
        pytest.skip("Jeu complet absent (data/creditcard.csv) : voir data/README.md")
    return load_data(FULL_DATA_PATH)


@pytest.fixture(scope="session")
def sample_splits(sample_df):
    return split_data(remove_duplicates(sample_df))
