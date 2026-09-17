"""Evaluation for a ~577:1 imbalanced problem.

Accuracy is useless here (predicting "not fraud" for everything scores 99.83%),
so the headline metric is AUPRC: the area under the precision-recall curve. A
random scorer gets roughly the fraud rate (~0.0017); a perfect one gets 1.0.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def pick_threshold(y_val: np.ndarray, scores_val: np.ndarray) -> float:
    """Alert threshold that maximises F1 on the validation split."""
    precision, recall, thresholds = precision_recall_curve(y_val, scores_val)
    # The last precision/recall pair has no threshold attached.
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    return float(thresholds[int(np.argmax(f1))])


def metrics_at(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    flagged = scores >= threshold
    tp = int(np.sum(flagged & (y == 1)))
    precision = tp / flagged.sum() if flagged.sum() else 0.0
    recall = tp / (y == 1).sum() if (y == 1).sum() else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "alerts": int(flagged.sum()),
        "frauds_caught": tp,
    }


def pr_curve(y: np.ndarray, scores: np.ndarray, n_points: int = 101) -> dict[str, list[float]]:
    """Precision-recall curve resampled onto a fixed recall grid, small enough for the app.

    Uses interpolated precision: the best precision achievable at that recall or higher.
    """
    precision, recall, _ = precision_recall_curve(y, scores)
    grid = np.linspace(0.0, 1.0, n_points)
    interpolated = [precision[recall >= r].max() if (recall >= r).any() else 0.0 for r in grid]
    return {"recall": grid.round(4).tolist(), "precision": np.round(interpolated, 4).tolist()}


def evaluate(
    y_val: np.ndarray, scores_val: np.ndarray, y_test: np.ndarray, scores_test: np.ndarray
) -> dict[str, float]:
    """AUPRC and ROC-AUC on test, plus precision/recall at a threshold chosen on validation."""
    threshold = pick_threshold(y_val, scores_val)
    return {
        "auprc": float(average_precision_score(y_test, scores_test)),
        "roc_auc": float(roc_auc_score(y_test, scores_test)),
        **metrics_at(y_test, scores_test, threshold),
    }
