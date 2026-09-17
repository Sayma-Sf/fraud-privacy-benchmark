"""Non-private reference model trained on the real data.

LightGBM is the accuracy ceiling. The no-noise end of the privacy sweep
(logistic regression at epsilon = inf) is the second reference point, so the
gap to LightGBM is the cost of a simpler model and the drop below epsilon = inf
is the cost of privacy.
"""

from __future__ import annotations

import numpy as np
from lightgbm import LGBMClassifier, early_stopping


def train_lightgbm(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, seed: int
) -> LGBMClassifier:
    """Gradient-boosted trees, early-stopped on validation AUPRC. The best-case ceiling."""
    model = LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        metric="average_precision",
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[early_stopping(100, verbose=False)],
    )
    return model
