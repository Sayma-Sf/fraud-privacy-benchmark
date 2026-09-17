"""Run the whole benchmark and write ``results/`` and ``reports/figures/``.

    python -m src.pipeline           # full run, ~30 min on a laptop CPU (CTGAN is most of it)
    python -m src.pipeline --quick   # small smoke run in ~2 min, written to results_quick/

Steps:
1. LightGBM on the real data: the accuracy ceiling.
2. Differentially private logistic regression across the epsilon grid.
3. Synthetic data (Gaussian copula, CTGAN): train on synthetic, test on real,
   plus fidelity and memorisation checks.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config
from src.features.transform import engineer, scale_to_unit
from src.ingest.load import TARGET, load_transactions, split
from src.models.baseline import train_lightgbm
from src.models.metrics import evaluate, pr_curve
from src.privacy.dp_models import epsilon_sweep, summarize_sweep
from src.privacy.leakage import dcr_report
from src.privacy.synthetic import ClassConditionalSynthesizer, fidelity_scores
from src.reporting.figures import render_all

LIBRARIES = ["numpy", "pandas", "scikit-learn", "lightgbm", "diffprivlib", "sdv", "torch"]


def log(message: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)


def arrays(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    return engineer(df), df[TARGET].to_numpy()


def run_synthetic(kind: str, splits, args) -> tuple[dict, dict]:
    """Fit a generator on real train rows, then play the data recipient.

    The recipient only ever sees synthetic rows: they split them for early stopping
    and threshold choice. Real rows are used once, to score the finished model.
    """
    started = time.time()
    generator = ClassConditionalSynthesizer.fit(
        splits.train,
        kind,
        max_rows_per_class=args.synth_max_rows if kind == "ctgan" else None,
        ctgan_steps=args.ctgan_steps,
        seed=config.SEED,
    )
    fit_seconds = time.time() - started
    synthetic = generator.sample(len(splits.train), seed=config.SEED)

    synth_train, synth_val = train_test_split(
        synthetic, test_size=0.25, stratify=synthetic[TARGET], random_state=config.SEED
    )
    X_tr, y_tr = arrays(synth_train)
    X_va, y_va = arrays(synth_val)
    X_te, y_te = arrays(splits.test)
    model = train_lightgbm(X_tr, y_tr, X_va, y_va, config.SEED)
    scores_test = model.predict_proba(X_te)[:, 1]
    tstr = evaluate(y_va, model.predict_proba(X_va)[:, 1], y_te, scores_test)

    real_fraud = splits.train[splits.train[TARGET] == 1]
    holdout = pd.concat([splits.val, splits.test])
    scaled = {
        "train": scale_to_unit(engineer(splits.train)),
        "holdout": scale_to_unit(engineer(splits.test)),
        "synthetic": scale_to_unit(engineer(synthetic)),
        "train_fraud": scale_to_unit(engineer(real_fraud)),
        "holdout_fraud": scale_to_unit(engineer(holdout[holdout[TARGET] == 1])),
        "synthetic_fraud": scale_to_unit(engineer(synthetic[synthetic[TARGET] == 1])),
    }
    result = {
        "fit_seconds": round(fit_seconds, 1),
        "rows_generated": len(synthetic),
        "frauds_generated": int(synthetic[TARGET].sum()),
        "tstr": {**tstr, "best_iteration": int(model.best_iteration_)},
        "fidelity": fidelity_scores(splits.train, synthetic, seed=config.SEED),
        "dcr_all": dcr_report(scaled["train"], scaled["holdout"], scaled["synthetic"],
                              seed=config.SEED),
        "dcr_fraud": dcr_report(scaled["train_fraud"], scaled["holdout_fraud"],
                                scaled["synthetic_fraud"], seed=config.SEED),
    }
    return result, pr_curve(y_te, scores_test)


def main() -> None:
    parser = argparse.ArgumentParser(description="Privacy-utility benchmark for fraud detection")
    parser.add_argument("--data", type=Path, default=config.DATA_PATH)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--quick", action="store_true", help="small smoke run")
    parser.add_argument("--repeats", type=int, default=config.DP_REPEATS)
    parser.add_argument("--data-norm", type=float, default=config.DP_DATA_NORM)
    parser.add_argument("--ctgan-steps", type=int, default=config.CTGAN_STEPS)
    parser.add_argument("--synth-max-rows", type=int, default=config.SYNTH_MAX_ROWS_PER_CLASS)
    parser.add_argument("--skip-synthetic", action="store_true")
    args = parser.parse_args()

    epsilons = config.EPSILONS
    df = load_transactions(args.data)
    if args.quick:
        legit = df[df[TARGET] == 0].sample(30_000, random_state=config.SEED)
        df = pd.concat([legit, df[df[TARGET] == 1]]).reset_index(drop=True)
        epsilons = [0.1, 1.0, 10.0, math.inf]
        args.repeats, args.ctgan_steps, args.synth_max_rows = 3, 60, 3_000
    out = args.out or (config.ROOT / "results_quick" if args.quick else config.RESULTS_DIR)
    figures_dir = out / "figures" if args.quick else config.FIGURES_DIR
    out.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}

    splits = split(df, config.VAL_SIZE, config.TEST_SIZE, config.SEED)
    X_train, y_train = arrays(splits.train)
    X_val, y_val = arrays(splits.val)
    X_test, y_test = arrays(splits.test)
    sizes = "/".join(f"{len(X):,}" for X in (X_train, X_val, X_test))
    log(f"Loaded {len(df):,} rows; train/val/test = {sizes}")

    log("1/3 LightGBM baseline on real data")
    started = time.time()
    lgbm = train_lightgbm(X_train, y_train, X_val, y_val, config.SEED)
    s_test = lgbm.predict_proba(X_test)[:, 1]
    baseline = {
        **evaluate(y_val, lgbm.predict_proba(X_val)[:, 1], y_test, s_test),
        "best_iteration": int(lgbm.best_iteration_),
    }
    curves: dict = {"lightgbm": pr_curve(y_test, s_test)}
    timings["baseline"] = time.time() - started
    log(f"    AUPRC {baseline['auprc']:.3f}")

    log(f"2/3 DP logistic regression: {len(epsilons)} epsilons x {args.repeats} runs")
    started = time.time()
    runs, curves["dp"] = epsilon_sweep(
        scale_to_unit(X_train), y_train, scale_to_unit(X_val), y_val,
        scale_to_unit(X_test), y_test,
        epsilons=epsilons, repeats=args.repeats, data_norm=args.data_norm, seed=config.SEED,
    )
    summary = summarize_sweep(runs)
    runs.round(6).to_csv(out / "dp_runs.csv", index=False)
    timings["dp_sweep"] = time.time() - started
    for row in summary.itertuples():
        log(f"    eps={row.epsilon_label:>5}  AUPRC {row.auprc_mean:.3f} ± {row.auprc_std:.3f}")

    synthetic, curves["synthetic"] = {}, {}
    if not args.skip_synthetic:
        for kind in config.SYNTHESIZERS:
            log(f"3/3 Synthetic data: {kind}")
            started = time.time()
            synthetic[kind], curves["synthetic"][kind] = run_synthetic(kind, splits, args)
            timings[f"synthetic_{kind}"] = time.time() - started
            s = synthetic[kind]
            log(f"    TSTR AUPRC {s['tstr']['auprc']:.3f}, "
                f"fidelity {s['fidelity']['overall']:.3f}, "
                f"closer-to-train {s['dcr_all']['share_closer_to_train']:.3f} "
                f"(fraud {s['dcr_fraud']['share_closer_to_train']:.3f})")

    sweep_records = summary.to_dict(orient="records")
    for record in sweep_records:
        record["epsilon"] = None if math.isinf(record["epsilon"]) else record["epsilon"]
    results = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset": {
            "rows": len(df),
            "frauds": int(df[TARGET].sum()),
            "fraud_rate": float(df[TARGET].mean()),
            "train_rows": len(X_train),
            "val_rows": len(X_val),
            "test_rows": len(X_test),
            "test_frauds": int(y_test.sum()),
        },
        "config": {
            "quick": args.quick,
            "seed": config.SEED,
            "epsilons": [None if math.isinf(e) else e for e in epsilons],
            "dp_repeats": args.repeats,
            "dp_data_norm": args.data_norm,
            "ctgan_steps": args.ctgan_steps,
            "ctgan_max_rows_per_class": args.synth_max_rows,
        },
        "versions": {lib: version(lib) for lib in LIBRARIES},
        "baseline": {"lightgbm": baseline},
        "dp_sweep": sweep_records,
        "synthetic": synthetic,
        "timings_seconds": {k: round(v, 1) for k, v in timings.items()},
    }
    (out / "summary.json").write_text(json.dumps(results, indent=2))
    (out / "pr_curves.json").write_text(json.dumps(curves))
    written = render_all(results, figures_dir)
    log(f"Wrote {out / 'summary.json'} and {len(written)} figures to {figures_dir}")


if __name__ == "__main__":
    main()
