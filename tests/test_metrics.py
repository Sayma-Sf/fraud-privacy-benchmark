import numpy as np
import pytest

from src.models.metrics import evaluate, metrics_at, pick_threshold, pr_curve

Y = np.array([0, 0, 0, 0, 1, 0, 1, 1])
PERFECT = np.array([0.1, 0.2, 0.1, 0.3, 0.9, 0.2, 0.8, 0.95])


def test_perfect_scores_give_perfect_metrics():
    result = evaluate(Y, PERFECT, Y, PERFECT)
    assert result["auprc"] == pytest.approx(1.0)
    assert result["precision"] == 1.0 and result["recall"] == 1.0


def test_pick_threshold_separates_classes():
    assert 0.3 < pick_threshold(Y, PERFECT) <= 0.8


def test_metrics_at_counts_alerts():
    result = metrics_at(Y, PERFECT, threshold=0.85)
    assert result["alerts"] == 2
    assert result["frauds_caught"] == 2
    assert result["precision"] == 1.0
    assert result["recall"] == pytest.approx(2 / 3)


def test_metrics_at_with_no_alerts_is_zero_not_nan():
    result = metrics_at(Y, PERFECT, threshold=2.0)
    assert result["precision"] == 0.0 and result["f1"] == 0.0


def test_pr_curve_is_resampled_and_non_increasing():
    rng = np.random.default_rng(0)
    y = rng.random(500) < 0.1
    scores = y * 0.5 + rng.random(500)
    curve = pr_curve(y, scores, n_points=21)
    assert len(curve["recall"]) == len(curve["precision"]) == 21
    assert np.all(np.diff(curve["precision"]) <= 1e-9)
