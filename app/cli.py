"""Ligne de commande du projet : python -m app.cli <commande> [options].

Commandes :
  make-sample      extrait l'échantillon versionné utilisé par la CI
  train            réglage sur la validation, sélection, évaluation finale unique sur le test
  robustness       scénarios de données imparfaites et de perturbations
  performance      latence, débit, taille et mémoire
  drift            indicateurs de surveillance (dérive, taux d'alertes)
  baseline-create  crée la référence de non-régression (action manuelle)
  regression       compare des versions candidates à la référence
  all              enchaîne train, robustness, performance, drift et regression
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from app.config import (
    BASELINE_DIR,
    FEATURES,
    FULL_DATA_PATH,
    MAX_OVERFIT_GAP_PR_AUC,
    MIN_RECALL_FOR_THRESHOLD,
    MODEL_PATH,
    PCA_FEATURES,
    PERFORMANCE_CRITERIA,
    REPORTS_DIR,
    SEED,
    TARGET,
)
from app.evaluation import (
    bootstrap_ci,
    check_acceptance,
    error_analysis,
    evaluate_model,
    plot_confusion_matrix,
    plot_pr_curves,
    plot_roc_curves,
    plot_score_distribution,
    plot_threshold_tradeoff,
    threshold_tradeoff,
)
from app.monitoring import (
    alert_rate_status,
    drift_report,
    fit_reference_ranges,
    out_of_range_rates,
)
from app.performance import (
    environment_info,
    measure_latency,
    measure_memory,
    measure_model_latency,
    measure_throughput,
    model_size_mb,
)
from app.prediction import load_model, predict_fraud, save_model
from app.preprocessing import (
    DataValidationError,
    invalid_row_mask,
    load_data,
    preprocess_data,
    remove_duplicates,
    split_data,
    validate_data,
)
from app.regression import (
    build_run_record,
    compare_to_baseline,
    dataset_fingerprint,
    load_baseline,
    save_baseline,
)
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
from app.training import (
    MODEL_NAMES,
    select_threshold,
    train_and_validate,
    train_model,
    tune_model,
)
from app.utils import git_commit, read_json, write_json

logger = logging.getLogger("app.cli")
LABELS = {"logistic_regression": "Régression logistique", "decision_tree": "Arbre de décision"}


# --- Préparation commune ---------------------------------------------------------------
def prepare_splits(data_path: str | Path, seed: int = SEED):
    """Charge, valide, déduplique puis découpe (toujours dans cet ordre)."""
    raw = load_data(data_path)
    validate_data(raw)
    df = remove_duplicates(raw)
    train, val, test = split_data(df, seed)
    info = {
        "file": Path(data_path).name,
        "n_raw": len(raw),
        "n_fraud_raw": int(raw[TARGET].sum()),
        "n_duplicates_removed": len(raw) - len(df),
        "n_after_dedup": len(df),
        "n_fraud_after_dedup": int(df[TARGET].sum()),
        "splits": {
            name: {"n": len(part), "n_fraud": int(part[TARGET].sum()), "fraud_rate": float(part[TARGET].mean())}
            for name, part in (("train", train), ("validation", val), ("test", test))
        },
        "seed": seed,
    }
    return df, train, val, test, info


def _fmt(value, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/d"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def _print_table(rows: list[dict], columns: list[str]) -> None:
    widths = {c: max(len(c), *(len(_fmt(r.get(c))) for r in rows)) for c in columns}
    print("  ".join(c.ljust(widths[c]) for c in columns))
    for row in rows:
        print("  ".join(_fmt(row.get(c)).ljust(widths[c]) for c in columns))


# --- make-sample ---------------------------------------------------------------------
def cmd_make_sample(args) -> None:
    df = remove_duplicates(load_data(args.data))
    validate_data(df)
    fraud = df[df[TARGET] == 1].sample(n=args.n_fraud, random_state=args.seed)
    normal = df[df[TARGET] == 0].sample(n=args.n_normal, random_state=args.seed)
    sample = pd.concat([normal, fraud]).sort_values("Time", kind="stable").reset_index(drop=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(args.output, index=False, float_format="%.8g")
    print(f"Échantillon écrit : {args.output} ({len(sample)} lignes, {args.n_fraud} fraudes)")


# --- train -----------------------------------------------------------------------------
def cmd_train(args) -> dict:
    output = Path(args.output)
    figures = output / "figures"
    df, train, val, test, data_info = prepare_splits(args.data, args.seed)
    X_tr, y_tr = preprocess_data(train)
    X_val, y_val = preprocess_data(val)
    X_te, y_te = preprocess_data(test)

    # 1. Réglage et seuil : entraînement + validation uniquement.
    candidates = {}
    for name in MODEL_NAMES:
        pipeline, params, history = tune_model(X_tr, y_tr, X_val, y_val, name, args.seed)
        p_val = pipeline.predict_proba(X_val)[:, 1]
        threshold = select_threshold(y_val, p_val, MIN_RECALL_FOR_THRESHOLD)
        p_tr = pipeline.predict_proba(X_tr)[:, 1]
        candidates[name] = {
            "pipeline": pipeline,
            "params": params,
            "tuning_history": history,
            "threshold": threshold,
            "p_val": p_val,
            "train_metrics": evaluate_model(y_tr, (p_tr >= threshold).astype(int), p_tr),
            "val_metrics": evaluate_model(y_val, (p_val >= threshold).astype(int), p_val),
        }
        candidates[name]["val_acceptance"] = check_acceptance(candidates[name]["val_metrics"])

    dummy = train_model(X_tr, y_tr, "dummy")
    p_dummy = dummy.predict_proba(X_val)[:, 1]
    dummy_val = evaluate_model(y_val, (p_dummy >= 0.5).astype(int), p_dummy)

    # 2. Sélection sur la validation : critères respectés, puis meilleure PR-AUC.
    eligible = [n for n, c in candidates.items() if c["val_acceptance"]["accepted"]]
    pool = eligible or list(candidates)
    selected = max(pool, key=lambda n: candidates[n]["val_metrics"]["pr_auc"])
    chosen = candidates[selected]
    config = {
        "model_name": selected,
        "params": chosen["params"],
        "features": FEATURES,
        "min_recall": MIN_RECALL_FOR_THRESHOLD,
    }

    # 3. Gel du modèle avant toute utilisation du jeu de test.
    model_path = save_model(
        chosen["pipeline"],
        Path(args.models_dir) / "fraud_model.joblib",
        chosen["threshold"],
        metadata={
            "config": config,
            "seed": args.seed,
            "selected_on": "validation",
            "code_version": git_commit(),
            "training_set": dataset_fingerprint(train),
            "validation_set": dataset_fingerprint(val),
        },
    )
    loaded, info = load_model(model_path)

    # 4. Évaluation finale unique sur le test.
    p_te_loaded = loaded.predict_proba(X_te)[:, 1]
    reload_max_diff = float(np.abs(p_te_loaded - chosen["pipeline"].predict_proba(X_te)[:, 1]).max())
    threshold = info["threshold"]
    test_results = {}
    curves = {}
    for name, cand in candidates.items():
        p_te = p_te_loaded if name == selected else cand["pipeline"].predict_proba(X_te)[:, 1]
        metrics = evaluate_model(y_te, (p_te >= cand["threshold"]).astype(int), p_te)
        test_results[name] = metrics
        curves[LABELS[name]] = (y_te.to_numpy(), p_te)
        plot_confusion_matrix(metrics, f"{LABELS[name]} (test)", figures / f"confusion_{name}.png")

    test_metrics = test_results[selected]
    acceptance = check_acceptance(test_metrics)
    gap = chosen["train_metrics"]["pr_auc"] - test_metrics["pr_auc"]
    acceptance["criteria"]["overfit_gap_pr_auc"] = {
        "value": gap,
        "operator": "<=",
        "threshold": MAX_OVERFIT_GAP_PR_AUC,
        "passed": bool(gap <= MAX_OVERFIT_GAP_PR_AUC),
    }
    acceptance["accepted"] = all(c["passed"] for c in acceptance["criteria"].values())

    plot_roc_curves(curves, figures / "roc_curves.png")
    plot_pr_curves(curves, figures / "pr_curves.png")
    plot_threshold_tradeoff(y_val, chosen["p_val"], threshold, figures / "threshold_tradeoff.png")
    plot_score_distribution(y_te, p_te_loaded, threshold, figures / "score_distribution.png")

    report = {
        "data": data_info,
        "selection_rule": "Critères d'acceptation respectés sur la validation, puis PR-AUC de validation maximale",
        "threshold_rule": f"Précision maximale sous contrainte de rappel >= {MIN_RECALL_FOR_THRESHOLD} (validation)",
        "selected_model": selected,
        "config": config,
        "threshold": threshold,
        "model_file": {"path": model_path.name, "sha256": info["sha256"], "size_mb": model_size_mb(model_path)},
        "reload_max_abs_diff": reload_max_diff,
        "naive_baseline_validation": dummy_val,
        "candidates": {
            name: {
                "params": c["params"],
                "tuning_history": c["tuning_history"],
                "threshold": c["threshold"],
                "train": c["train_metrics"],
                "validation": c["val_metrics"],
                "validation_acceptance": c["val_acceptance"],
                "test": test_results[name],
            }
            for name, c in candidates.items()
        },
        "test_acceptance": acceptance,
        "test_confidence_intervals": bootstrap_ci(y_te, p_te_loaded, threshold, n_boot=args.n_boot, seed=args.seed),
        "threshold_tradeoff_validation": threshold_tradeoff(
            y_val, chosen["p_val"], sorted({0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 0.999, threshold})
        ),
        "error_analysis_test": error_analysis(X_te, y_te, p_te_loaded, threshold),
    }
    write_json(report, output / "results" / "evaluation.json")

    rows = [{"modèle": "Référence naïve (val)", **dummy_val}]
    for name, c in candidates.items():
        rows.append({"modèle": f"{LABELS[name]} (val)", **c["val_metrics"]})
        rows.append({"modèle": f"{LABELS[name]} (test)", **test_results[name]})
    _print_table(
        rows, ["modèle", "accuracy", "precision", "recall", "f1", "fpr", "roc_auc", "pr_auc", "tp", "fp", "fn"]
    )
    print(f"\nModèle retenu : {selected}, seuil {threshold:.6f}, accepté sur le test : {acceptance['accepted']}")
    return report


# --- robustness -------------------------------------------------------------------------
def _rejection_row(
    scenario: str,
    corrupted: pd.DataFrame,
    n_injected: int | None,
    count_rows: bool = True,
    **validation_kwargs,
) -> dict:
    """Vérifie qu'une entrée invalide est rejetée ; compte les lignes invalides (scénarios de valeurs)."""
    n_invalid = int(invalid_row_mask(corrupted).sum()) if count_rows else None
    try:
        validate_data(corrupted, require_target=False, **validation_kwargs)
        rejected, message = False, "accepté"
    except DataValidationError as exc:
        rejected, message = True, str(exc)
    return {
        "scenario": scenario,
        "expected": "rejet",
        "rejected": rejected,
        "message": message,
        "n_injected_rows": n_injected,
        "n_invalid_rows_detected": n_invalid,
        "result": "conforme" if rejected else "ANOMALIE",
    }


def cmd_robustness(args) -> dict:
    model, info = load_model(args.model)
    threshold = info["threshold"]
    config = info["config"]
    df, train, val, test, _ = prepare_splits(args.data, args.seed)
    X_tr, _ = preprocess_data(train)
    X_te, y_te = preprocess_data(test)
    rng = np.random.default_rng(args.seed)
    stds = X_tr[PCA_FEATURES].std()
    ranges = fit_reference_ranges(X_tr)

    rejections = []
    corrupted, n = inject_values(test, ["V14", "Amount"], 0.01, np.nan, rng)
    rejections.append(_rejection_row("1. Valeurs manquantes (1 % des lignes, V14 et Amount)", corrupted, n))
    corrupted, n = inject_values(test, ["V4"], 0.005, np.inf, rng)
    rejections.append(_rejection_row("2a. Valeurs infinies (0,5 % des lignes, V4)", corrupted, n))
    corrupted, n = inject_values(test, ["Amount"], 10 / len(test), "abc", rng)
    rejections.append(_rejection_row("2b. Valeurs non numériques (10 lignes, Amount)", corrupted, n))
    rejections.append(_rejection_row("6a. Colonne V14 absente", test.drop(columns=["V14"]), None, count_rows=False))
    rejections.append(
        _rejection_row(
            "6b. Colonne supplémentaire (mode strict API)",
            test.assign(card_number=0),
            None,
            count_rows=False,
            allow_extra_columns=False,
        )
    )
    rejections.append(
        _rejection_row(
            "6c. Amount transmis en texte", test.assign(Amount=test["Amount"].astype(str)), None, count_rows=False
        )
    )
    duplicated = pd.concat([test, test[["V1"]]], axis=1)
    rejections.append(_rejection_row("6d. Colonne V1 dupliquée", duplicated, None, count_rows=False))
    rejections.append(_rejection_row("6e. Montant négatif", test.assign(Amount=-test["Amount"] - 1), None))

    perturbations = [
        ("3a. Amount × 10", scale_amount(X_te, 10), "traitement + alerte hors plage"),
        ("3b. Amount = 10^9 pour toutes les transactions", set_amount(X_te, 1e9), "traitement + alerte hors plage"),
        ("4a. Bruit gaussien V1–V28, ε = 1 % σ", add_gaussian_noise(X_te, stds, 0.01, rng), "traitement"),
        ("4b. Bruit gaussien V1–V28, ε = 5 % σ", add_gaussian_noise(X_te, stds, 0.05, rng), "traitement"),
        ("4c. Bruit gaussien V1–V28, ε = 10 % σ", add_gaussian_noise(X_te, stds, 0.10, rng), "traitement"),
        ("4d. Amount ± 5 % (uniforme)", perturb_amount_relative(X_te, 0.05, rng), "traitement"),
        ("4e. Time ± 60 s (uniforme)", jitter_time(X_te, 60, rng), "traitement"),
        ("5a. Time + 48 h (période future)", shift_time(X_te, 172_800), "traitement + alerte dérive"),
        (
            "5b. V1–V28 × 1,5 (changement d'échelle)",
            scale_columns(X_te, PCA_FEATURES, 1.5),
            "traitement + alerte dérive",
        ),
    ]
    sensitivity_rows = []
    for scenario, X_pert, expected in perturbations:
        row = {"scenario": scenario, "expected": expected, **sensitivity(model, X_te, X_pert, y_te, threshold)}
        oor = out_of_range_rates(X_pert, ranges)
        row["max_out_of_range_rate"] = max(oor.values())
        row["out_of_range_features"] = sorted(c for c, r in oor.items() if r > 0.01)
        row["n_invalid_rows"] = int(invalid_row_mask(X_pert).sum())
        sensitivity_rows.append(row)

    # 5c. Validation temporelle : entraînement sur le jour 1, évaluation sur le jour 2.
    temporal = []
    day1, day2 = df[df["Time"] < 86_400], df[df["Time"] >= 86_400]
    tr1, va1 = train_test_split(day1, test_size=0.25, stratify=day1[TARGET], random_state=args.seed)
    for label, features in (("avec Time", FEATURES), ("sans Time", [f for f in FEATURES if f != "Time"])):
        cfg = {**config, "features": features}
        result = train_and_validate(tr1, va1, cfg, args.seed)
        X2, y2 = preprocess_data(day2)
        p2 = result["pipeline"].predict_proba(X2)[:, 1]
        metrics = evaluate_model(y2, (p2 >= result["threshold"]).astype(int), p2)
        temporal.append(
            {
                "variant": label,
                "threshold": result["threshold"],
                "day2": metrics,
                "day1_validation": result["val_metrics"],
            }
        )

    report = {
        "model": config,
        "threshold": threshold,
        "evaluation_set": "test (aucun réglage n'est fait à partir de ces résultats)",
        "rejection_scenarios": rejections,
        "sensitivity_scenarios": sensitivity_rows,
        "temporal_validation": temporal,
        "baseline_test_metrics": evaluate_model(
            y_te, (model.predict_proba(X_te)[:, 1] >= threshold).astype(int), model.predict_proba(X_te)[:, 1]
        ),
    }
    write_json(report, Path(args.output) / "results" / "robustness.json")
    _print_table(rejections, ["scenario", "rejected", "n_injected_rows", "n_invalid_rows_detected", "result"])
    print()
    _print_table(
        sensitivity_rows,
        [
            "scenario",
            "mean_abs_delta",
            "flip_rate",
            "flips_to_fraud",
            "flips_to_normal",
            "delta_recall",
            "delta_precision",
            "delta_fpr",
            "max_out_of_range_rate",
        ],
    )
    print()
    for t in temporal:
        m = t["day2"]
        print(
            f"Temporel {t['variant']}: recall={m['recall']:.3f} precision={m['precision']:.3f} "
            f"fpr={m['fpr']:.5f} pr_auc={m['pr_auc']:.3f}"
        )
    return report


# --- performance ---------------------------------------------------------------------------
def cmd_performance(args) -> dict:
    model, info = load_model(args.model)
    threshold = info["threshold"]
    _, _, _, test, _ = prepare_splits(args.data, args.seed)
    X_te, _ = preprocess_data(test)
    latency = measure_latency(model, X_te, threshold, n_repeats=args.repeats)
    model_only = measure_model_latency(model, X_te, n_repeats=args.repeats)
    throughput = [
        measure_throughput(model, X_te.iloc[:size], threshold) for size in (100, 1_000, 10_000) if size <= len(X_te)
    ]
    throughput.append(measure_throughput(model, X_te, threshold, n_repeats=3))
    memory = measure_memory(model, X_te, threshold)
    size = model_size_mb(args.model)
    observed = {
        "latency_p95_ms": latency["p95_ms"],
        "throughput_tps": max(t["throughput_tps"] for t in throughput),
        "model_size_mb": size,
        "memory_increase_mb": memory["memory_increase_mb"],
    }
    report = {
        "protocol": {
            "environment": environment_info(),
            "latency_repeats": args.repeats,
            "warmup": 50,
            "latency_scope": "predict_fraud sur 1 transaction : validation du schéma + prétraitement + modèle + seuil",
            "model_latency_scope": "model.predict_proba sur 1 transaction (sans validation)",
            "throughput_scope": "predict_fraud sur des lots de taille croissante, médiane de plusieurs passes",
            "timer": "time.perf_counter",
            "api": "aucune API HTTP déployée : la latence HTTP (sérialisation, réseau) n'est pas mesurée",
        },
        "latency_single": latency,
        "latency_model_only": model_only,
        "throughput": throughput,
        "memory": memory,
        "model_size_mb": size,
        "acceptance": check_acceptance(observed, PERFORMANCE_CRITERIA),
    }
    write_json(report, Path(args.output) / "results" / "performance.json")
    print(
        f"Latence P95 (validation incluse) : {latency['p95_ms']:.2f} ms ; modèle seul : {model_only['p95_ms']:.2f} ms"
    )
    for t in throughput:
        print(f"Débit lot {t['batch_size']} : {t['throughput_tps']:.0f} tx/s")
    print(
        f"Taille : {size:.4f} Mo ; mémoire : +{memory['memory_increase_mb']:.1f} Mo ; "
        f"accepté : {report['acceptance']['accepted']}"
    )
    return report


# --- drift -------------------------------------------------------------------------------
def cmd_drift(args) -> dict:
    model, info = load_model(args.model)
    threshold = info["threshold"]
    df, train, val, test, _ = prepare_splits(args.data, args.seed)
    X_tr, _ = preprocess_data(train)
    X_val, _ = preprocess_data(val)
    X_te, _ = preprocess_data(test)
    expected_rate = float(predict_fraud(model, X_val, threshold)["is_fraud"].mean())

    def summarize(name, current):
        rows = drift_report(X_tr, current)
        labels = predict_fraud(model, current, threshold)["is_fraud"]
        return {
            "batch": name,
            "n": len(current),
            "n_alert": sum(r["status"] == "alerte" for r in rows),
            "n_warning": sum(r["status"] == "surveiller" for r in rows),
            "top_features": sorted(rows, key=lambda r: r["psi"], reverse=True)[:5],
            "alert_rate": alert_rate_status(labels, expected_rate),
        }

    day1 = preprocess_data(df[df["Time"] < 86_400])[0]
    day2 = preprocess_data(df[df["Time"] >= 86_400])[0]
    batches = [
        summarize("Jeu de test (même distribution)", X_te),
        summarize("Transactions du jour 1", day1),
        summarize("Transactions du jour 2", day2),
        summarize("Simulation : Amount × 3", scale_amount(X_te, 3)),
        summarize("Simulation : V1–V5 × 1,5", scale_columns(X_te, [f"V{i}" for i in range(1, 6)], 1.5)),
        summarize("Simulation : Amount × 10", scale_amount(X_te, 10)),
    ]
    report = {"reference": "jeu d'entraînement", "expected_alert_rate_validation": expected_rate, "batches": batches}
    write_json(report, Path(args.output) / "results" / "drift.json")
    for b in batches:
        top = ", ".join(f"{r['feature']}={r['psi']:.2f}" for r in b["top_features"][:3])
        print(
            f"{b['batch']:<36} alertes PSI={b['n_alert']:>2}  surveiller={b['n_warning']:>2}  "
            f"taux signalé={b['alert_rate']['observed_rate']:.4%} ({b['alert_rate']['status']})  top: {top}"
        )
    return report


# --- baseline / regression ---------------------------------------------------------------
def cmd_baseline_create(args) -> dict:
    info = read_json(Path(args.model).with_suffix(".json"))
    config = info["config"]
    _, train, val, _, _ = prepare_splits(args.data, args.seed)
    result = train_and_validate(train, val, config, args.seed)
    record = build_run_record("référence", config, result["threshold"], result["val_metrics"], val, args.seed)
    save_baseline(record, args.baseline, force=args.force)
    print(f"Référence écrite : {args.baseline}")
    return record


def cmd_regression(args) -> dict:
    baseline = load_baseline(args.baseline)
    base_config = baseline["model"]
    _, train, val, _, _ = prepare_splits(args.data, args.seed)
    variants = {
        "v1.0.1 : réentraînement à l'identique": base_config,
        "v1.1.0 : sans la variable Time": {
            **base_config,
            "features": [f for f in base_config["features"] if f != "Time"],
        },
        "v1.1.1 : sans pondération des classes": {
            **base_config,
            "params": {**base_config["params"], "class_weight": None},
        },
    }
    results = []
    for label, config in variants.items():
        run = train_and_validate(train, val, config, args.seed)
        record = build_run_record(label, config, run["threshold"], run["val_metrics"], val, args.seed)
        comparison = compare_to_baseline(record, baseline)
        results.append({"candidate": record, "comparison": comparison})
        print(f"\n{label} -> décision : {comparison['decision'].upper()}")
        _print_table(comparison["metrics"], ["metric", "baseline", "current", "degradation", "status"])
        for note in comparison["notes"]:
            print(f"  note : {note}")
    report = {"baseline": baseline, "candidates": results}
    write_json(report, Path(args.output) / "results" / "regression.json")
    return report


def cmd_all(args) -> None:
    cmd_train(args)
    args.model = str(Path(args.models_dir) / "fraud_model.joblib")
    cmd_robustness(args)
    cmd_performance(args)
    cmd_drift(args)
    if Path(args.baseline).exists():
        cmd_regression(args)
    else:
        print(f"\nRéférence absente ({args.baseline}) : lancer `baseline-create` pour activer la non-régression.")


# --- Analyse des arguments ------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.cli", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p, model=True):
        p.add_argument("--data", default=str(FULL_DATA_PATH))
        p.add_argument("--output", default=str(REPORTS_DIR))
        p.add_argument("--seed", type=int, default=SEED)
        if model:
            p.add_argument("--model", default=str(MODEL_PATH))
        return p

    p = sub.add_parser("make-sample")
    p.add_argument("--data", default=str(FULL_DATA_PATH))
    p.add_argument("--output", default="data/sample_creditcard.csv")
    p.add_argument("--n-normal", type=int, default=3000)
    p.add_argument("--n-fraud", type=int, default=150)
    p.add_argument("--seed", type=int, default=SEED)
    p.set_defaults(func=cmd_make_sample)

    p = common(sub.add_parser("train"), model=False)
    p.add_argument("--models-dir", default=str(MODEL_PATH.parent))
    p.add_argument("--n-boot", type=int, default=1000)
    p.set_defaults(func=cmd_train)

    common(sub.add_parser("robustness")).set_defaults(func=cmd_robustness)

    p = common(sub.add_parser("performance"))
    p.add_argument("--repeats", type=int, default=1000)
    p.set_defaults(func=cmd_performance)

    common(sub.add_parser("drift")).set_defaults(func=cmd_drift)

    p = common(sub.add_parser("baseline-create"))
    p.add_argument("--baseline", default=str(BASELINE_DIR / "baseline_full.json"))
    p.add_argument("--force", action="store_true", help="écrase une référence existante (décision explicite)")
    p.set_defaults(func=cmd_baseline_create)

    p = common(sub.add_parser("regression"))
    p.add_argument("--baseline", default=str(BASELINE_DIR / "baseline_full.json"))
    p.set_defaults(func=cmd_regression)

    p = common(sub.add_parser("all"), model=False)
    p.add_argument("--models-dir", default=str(MODEL_PATH.parent))
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--repeats", type=int, default=1000)
    p.add_argument("--baseline", default=str(BASELINE_DIR / "baseline_full.json"))
    p.set_defaults(func=cmd_all)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s : %(message)s")
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
