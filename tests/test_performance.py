"""Partie VIII : coût d'exécution (seuils volontairement larges pour rester stables en CI)."""

from __future__ import annotations

import pytest

from app.config import PERFORMANCE_CRITERIA
from app.evaluation import check_acceptance
from app.performance import (
    environment_info,
    measure_latency,
    measure_memory,
    measure_model_latency,
    measure_throughput,
    model_size_mb,
)
from app.prediction import save_model

pytestmark = pytest.mark.performance


def test_single_prediction_latency_p95_within_budget(trained_model, test_features):
    model, threshold = trained_model
    X, _ = test_features
    latency = measure_latency(model, X, threshold, n_repeats=200, warmup=20)
    assert latency["n"] == 200
    assert latency["p50_ms"] <= latency["p95_ms"] <= latency["p99_ms"] <= latency["max_ms"]
    assert latency["p95_ms"] <= PERFORMANCE_CRITERIA["latency_p95_ms"][1]


def test_validation_overhead_is_measured_separately(trained_model, test_features):
    model, threshold = trained_model
    X, _ = test_features
    full = measure_latency(model, X, threshold, n_repeats=200, warmup=20)
    model_only = measure_model_latency(model, X, n_repeats=200, warmup=20)
    assert model_only["p50_ms"] <= full["p50_ms"]


def test_batch_throughput(trained_model, test_features):
    model, threshold = trained_model
    X, _ = test_features
    result = measure_throughput(model, X, threshold, n_repeats=3)
    assert result["batch_size"] == len(X)
    assert result["throughput_tps"] >= PERFORMANCE_CRITERIA["throughput_tps"][1]


def test_model_size(trained_model, tmp_path):
    model, threshold = trained_model
    path = save_model(model, tmp_path / "model.joblib", threshold)
    assert model_size_mb(path) <= PERFORMANCE_CRITERIA["model_size_mb"][1]


def test_memory_increase(trained_model, test_features):
    model, threshold = trained_model
    X, _ = test_features
    memory = measure_memory(model, X, threshold)
    assert memory["memory_increase_mb"] <= PERFORMANCE_CRITERIA["memory_increase_mb"][1]


def test_performance_acceptance_logic():
    good = {"latency_p95_ms": 2.0, "throughput_tps": 5000, "model_size_mb": 0.1, "memory_increase_mb": 20}
    assert check_acceptance(good, PERFORMANCE_CRITERIA)["accepted"]
    assert not check_acceptance({**good, "latency_p95_ms": 80.0}, PERFORMANCE_CRITERIA)["accepted"]


def test_environment_is_documented():
    info = environment_info()
    assert {"platform", "python", "cpu_count_logical", "total_memory_gb"} <= set(info)
