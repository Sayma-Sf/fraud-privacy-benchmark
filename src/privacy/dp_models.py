"""Differentially private logistic regression, swept across privacy budgets.

Epsilon in one paragraph: differential privacy promises that the trained model
would come out almost the same whether or not any single transaction was in the
training data. Epsilon is how much "almost" allows. Any conclusion someone draws
from the model about one transaction can become at most e^epsilon times more
likely because that transaction was used: about 1.1x at epsilon = 0.1, 2.7x at
1, and 22,000x at 10, which is barely a promise. Smaller epsilon means more
noise is added during training, so the model gets more private and less accurate.

diffprivlib implements this with objective perturbation (Chaudhuri et al., 2011):
each row is clipped to a maximum L2 norm (``data_norm``), which caps how much
one transaction can move the model, and calibrated random noise is added to the
training objective.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
from diffprivlib.accountant import BudgetAccountant
from diffprivlib.models import LogisticRegression as DPLogisticRegression
from sklearn.linear_model import LogisticRegression

from src.models.metrics import evaluate, pr_curve


def epsilon_label(epsilon: float) -> str:
    return "inf" if math.isinf(epsilon) else f"{epsilon:g}"


def train_dp_logreg(
    X: np.ndarray, y: np.ndarray, epsilon: float, data_norm: float, seed: int
) -> DPLogisticRegression | LogisticRegression:
    """Fit an epsilon-DP logistic regression. epsilon = inf returns the noiseless equivalent."""
    if math.isinf(epsilon):
        # Same clipping, no noise: the non-private end of the curve on equal terms.
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X = X * np.minimum(1.0, data_norm / np.clip(norms, 1e-12, None))
        return LogisticRegression(max_iter=2000, random_state=seed).fit(X, y)
    model = DPLogisticRegression(
        epsilon=epsilon,
        data_norm=data_norm,
        max_iter=2000,
        random_state=seed,
        accountant=BudgetAccountant(),
    )
    with warnings.catch_warnings():
        # Very small epsilons can stop the optimiser early; the noise dominates anyway.
        warnings.simplefilter("ignore", category=UserWarning)
        model.fit(X, y)
    return model


def epsilon_sweep(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    epsilons: list[float],
    repeats: int,
    data_norm: float,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, dict[str, list[float]]]]:
    """Train ``repeats`` models per epsilon.

    Returns one row per (epsilon, repeat) with test metrics, plus the test PR curve
    of the median-AUPRC run at each epsilon.
    """
    rows, curves = [], {}
    for epsilon in epsilons:
        label = epsilon_label(epsilon)
        # Without noise every repeat is identical, so one fit is enough.
        n_runs = 1 if math.isinf(epsilon) else repeats
        runs = []
        for r in range(n_runs):
            model = train_dp_logreg(X_train, y_train, epsilon, data_norm, seed + r)
            s_val = model.predict_proba(X_val)[:, 1]
            s_test = model.predict_proba(X_test)[:, 1]
            metrics = evaluate(y_val, s_val, y_test, s_test)
            runs.append((metrics["auprc"], s_test))
            rows.append({"epsilon": epsilon, "epsilon_label": label, "repeat": r, **metrics})
        median_run = sorted(runs, key=lambda run: run[0])[len(runs) // 2]
        curves[label] = pr_curve(y_test, median_run[1])
    return pd.DataFrame(rows), curves


def summarize_sweep(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean, spread and a 10th-90th percentile band of each metric per epsilon."""
    grouped = runs.groupby("epsilon", sort=True)
    summary = grouped.agg(
        epsilon_label=("epsilon_label", "first"),
        runs=("auprc", "size"),
        auprc_mean=("auprc", "mean"),
        auprc_std=("auprc", "std"),
        auprc_p10=("auprc", lambda s: s.quantile(0.1)),
        auprc_p90=("auprc", lambda s: s.quantile(0.9)),
        roc_auc_mean=("roc_auc", "mean"),
        precision_mean=("precision", "mean"),
        recall_mean=("recall", "mean"),
        f1_mean=("f1", "mean"),
    )
    return summary.reset_index().fillna({"auprc_std": 0.0})
