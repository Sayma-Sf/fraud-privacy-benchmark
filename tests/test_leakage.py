import numpy as np
import pytest

from src.privacy.leakage import dcr_report


@pytest.fixture(scope="module")
def real():
    rng = np.random.default_rng(0)
    return rng.normal(size=(600, 5)), rng.normal(size=(600, 5))


def test_copied_rows_are_flagged(real):
    train, holdout = real
    report = dcr_report(train, holdout, synthetic=train.copy(), max_rows=600)
    assert report["share_closer_to_train"] == 1.0
    assert report["exact_copies"] == 600
    assert report["median_dcr_synthetic"] == 0.0


def test_fresh_rows_from_same_distribution_look_like_holdout(real):
    train, holdout = real
    fresh = np.random.default_rng(1).normal(size=(600, 5))
    report = dcr_report(train, holdout, synthetic=fresh, max_rows=600)
    assert 0.4 < report["share_closer_to_train"] < 0.6
    assert report["exact_copies"] == 0
