"""Feature engineering shared by every model, real or synthetic."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.ingest.load import V_COLUMNS

FEATURES = [*V_COLUMNS, "hour", "log_amount"]

# Fixed clipping bounds, set up front rather than measured on the training data.
# A differentially private model is only private if its preprocessing is too:
# scaling by the training min/max would itself leak the most extreme
# transactions. In production these would come from the schema owner.
FEATURE_BOUNDS: dict[str, tuple[float, float]] = {
    **{c: (-10.0, 10.0) for c in V_COLUMNS},
    "hour": (0.0, 24.0),
    "log_amount": (0.0, 11.0),  # log1p(60,000), well above any single card payment
}


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw columns into model features.

    ``Time`` counts seconds from the first transaction in the file, so ``hour`` is
    the position in a 24-hour cycle (offset from the real clock, but still daily).
    """
    out = df[V_COLUMNS].copy()
    out["hour"] = (df["Time"] % 86_400) / 3_600
    out["log_amount"] = np.log1p(df["Amount"].clip(lower=0))
    return out[FEATURES]


def scale_to_unit(features: pd.DataFrame) -> np.ndarray:
    """Clip each feature to its fixed bounds and map it linearly onto [-1, 1]."""
    lo = np.array([FEATURE_BOUNDS[c][0] for c in features.columns])
    hi = np.array([FEATURE_BOUNDS[c][1] for c in features.columns])
    clipped = np.clip(features.to_numpy(dtype=float), lo, hi)
    return 2.0 * (clipped - lo) / (hi - lo) - 1.0
