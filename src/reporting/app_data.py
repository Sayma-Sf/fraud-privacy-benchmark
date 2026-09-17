"""Data helpers for the Streamlit app.

Uses only json, math and pandas, so the deployed demo installs a few small
packages instead of the whole training stack. It reads results/, never raw data.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
INF_LABEL = "∞"

MODEL_NAMES = {
    "lightgbm": "LightGBM on real data (no privacy)",
    "gaussian_copula": "Gaussian copula synthetic",
    "ctgan": "CTGAN synthetic",
}


def load_results(results_dir: Path = RESULTS_DIR) -> tuple[dict, dict]:
    summary = json.loads((results_dir / "summary.json").read_text(encoding="utf-8"))
    curves = json.loads((results_dir / "pr_curves.json").read_text(encoding="utf-8"))
    return summary, curves


def sweep_frame(summary: dict) -> pd.DataFrame:
    """One row per epsilon, ascending, with ``inf`` for the no-noise model."""
    frame = pd.DataFrame(summary["dp_sweep"])
    frame["epsilon"] = frame["epsilon"].fillna(math.inf).astype(float)
    frame["label"] = frame["epsilon"].map(lambda e: INF_LABEL if math.isinf(e) else f"{e:g}")
    no_noise = frame.loc[frame["epsilon"].map(math.isinf), "auprc_mean"].iloc[0]
    frame["share_of_no_noise"] = frame["auprc_mean"] / no_noise
    return frame.sort_values("epsilon").reset_index(drop=True)


def odds_multiplier(epsilon: float) -> str:
    """e^epsilon, written for people: how much more likely any inference can become."""
    if math.isinf(epsilon):
        return "unbounded"
    if epsilon >= 6 * math.log(10):  # a million or more: the exponent is the readable part
        return f"about 10^{int(epsilon / math.log(10))}×"
    value = math.exp(epsilon)
    return f"{value:.2f}×" if value < 10 else f"{value:,.0f}×"


def plain_language(epsilon: float) -> str:
    if math.isinf(epsilon):
        return (
            "**No privacy guarantee.** The model is trained normally, so in principle it can "
            "leak details of the transactions it was trained on."
        )
    if epsilon <= 1:
        strength = "Strong privacy."
    elif epsilon <= 10:
        strength = "Moderate privacy."
    else:
        strength = "Weak privacy: the guarantee exists on paper but promises little."
    return (
        f"**{strength}** Anything someone concludes about one transaction from this model "
        f"becomes at most **{odds_multiplier(epsilon)}** more likely because that "
        "transaction was in the training data."
    )


def curves_frame(curves: dict[str, dict[str, list[float]]]) -> pd.DataFrame:
    """Long-format precision-recall points: one row per (model, recall)."""
    frames = [
        pd.DataFrame({"model": name, "recall": c["recall"], "precision": c["precision"]})
        for name, c in curves.items()
    ]
    return pd.concat(frames, ignore_index=True)
