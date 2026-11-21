"""Mesure du coût d'exécution : latence, débit, taille et mémoire."""

from __future__ import annotations

import os
import platform
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

from app.config import FEATURES
from app.prediction import predict_fraud


def _summary_ms(timings_s: list[float]) -> dict:
    ms = np.asarray(timings_s) * 1000
    return {
        "n": int(ms.size),
        "mean_ms": float(ms.mean()),
        "p50_ms": float(np.percentile(ms, 50)),
        "p95_ms": float(np.percentile(ms, 95)),
        "p99_ms": float(np.percentile(ms, 99)),
        "max_ms": float(ms.max()),
    }


def measure_latency(model, X: pd.DataFrame, threshold: float, n_repeats: int = 500, warmup: int = 50) -> dict:
    """Latence d'une prédiction isolée, validation des entrées comprise (predict_fraud)."""
    timings = []
    for i in range(warmup + n_repeats):
        row = X.iloc[[i % len(X)]]
        start = time.perf_counter()
        predict_fraud(model, row, threshold)
        elapsed = time.perf_counter() - start
        if i >= warmup:
            timings.append(elapsed)
    return _summary_ms(timings)


def measure_model_latency(model, X: pd.DataFrame, n_repeats: int = 500, warmup: int = 50) -> dict:
    """Latence du seul appel predict_proba (sans validation), pour isoler le coût du modèle."""
    timings = []
    features = X[FEATURES]
    for i in range(warmup + n_repeats):
        row = features.iloc[[i % len(features)]]
        start = time.perf_counter()
        model.predict_proba(row)
        elapsed = time.perf_counter() - start
        if i >= warmup:
            timings.append(elapsed)
    return _summary_ms(timings)


def measure_throughput(model, X: pd.DataFrame, threshold: float, n_repeats: int = 5) -> dict:
    """Débit (transactions/s) sur un lot, médiane de n_repeats mesures."""
    durations = []
    for _ in range(n_repeats):
        start = time.perf_counter()
        predict_fraud(model, X, threshold)
        durations.append(time.perf_counter() - start)
    median = float(np.median(durations))
    return {"batch_size": len(X), "median_s": median, "throughput_tps": len(X) / median}


def measure_memory(model, X: pd.DataFrame, threshold: float) -> dict:
    """Mémoire résidente du processus et pic d'allocations Python pendant un lot."""
    process = psutil.Process(os.getpid())
    rss_before = process.memory_info().rss
    tracemalloc.start()
    predict_fraud(model, X, threshold)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after = process.memory_info().rss
    return {
        "batch_size": len(X),
        "rss_before_mb": rss_before / 1024**2,
        "rss_after_mb": rss_after / 1024**2,
        "tracemalloc_peak_mb": peak / 1024**2,
        "memory_increase_mb": max(peak, rss_after - rss_before) / 1024**2,
    }


def model_size_mb(path: str | Path) -> float:
    return Path(path).stat().st_size / 1024**2


def environment_info() -> dict:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "inconnu",
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "total_memory_gb": round(psutil.virtual_memory().total / 1024**3, 1),
        "python": platform.python_version(),
    }
