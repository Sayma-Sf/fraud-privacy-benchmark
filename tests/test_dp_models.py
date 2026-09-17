import math

import numpy as np
import pytest
from diffprivlib.models import LogisticRegression as DPLogisticRegression

from src.features.transform import engineer, scale_to_unit
from src.ingest.load import split
from src.privacy.dp_models import epsilon_label, epsilon_sweep, summarize_sweep, train_dp_logreg


@pytest.fixture(scope="module")
def arrays(transactions):
    parts = split(transactions, val_size=0.2, test_size=0.2, seed=0)
    return {
        name: (scale_to_unit(engineer(df)), df["Class"].to_numpy())
        for name, df in (("train", parts.train), ("val", parts.val), ("test", parts.test))
    }


def test_epsilon_label():
    assert epsilon_label(math.inf) == "inf"
    assert epsilon_label(0.5) == "0.5"
    assert epsilon_label(10.0) == "10"


def test_finite_epsilon_uses_diffprivlib(arrays):
    X, y = arrays["train"]
    private = train_dp_logreg(X, y, 1.0, data_norm=1.0, seed=0)
    no_noise = train_dp_logreg(X, y, math.inf, data_norm=1.0, seed=0)
    assert isinstance(private, DPLogisticRegression)
    assert not isinstance(no_noise, DPLogisticRegression)


def test_smaller_epsilon_adds_more_noise(arrays):
    X, y = arrays["train"]
    reference = train_dp_logreg(X, y, math.inf, data_norm=1.0, seed=0).coef_

    def mean_distance(epsilon):
        return np.mean([
            np.linalg.norm(train_dp_logreg(X, y, epsilon, data_norm=1.0, seed=s).coef_ - reference)
            for s in range(8)
        ])

    assert mean_distance(0.1) > mean_distance(100.0)


def test_sweep_shapes_and_privacy_cost(arrays):
    runs, curves = epsilon_sweep(
        *arrays["train"], *arrays["val"], *arrays["test"],
        epsilons=[0.01, math.inf], repeats=3, data_norm=1.0, seed=0,
    )
    assert len(runs) == 3 + 1  # repeats for finite epsilon, a single fit without noise
    assert set(curves) == {"0.01", "inf"}

    summary = summarize_sweep(runs).set_index("epsilon_label")
    assert summary.loc["inf", "auprc_mean"] > summary.loc["0.01", "auprc_mean"]
    assert summary.loc["inf", "auprc_std"] == 0.0
