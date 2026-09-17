import numpy as np
import pandas as pd

from src.features.transform import FEATURES, engineer, scale_to_unit


def test_engineer_builds_expected_features(transactions):
    features = engineer(transactions)
    assert list(features.columns) == FEATURES
    assert features["hour"].between(0, 24, inclusive="left").all()
    np.testing.assert_allclose(features["log_amount"], np.log1p(transactions["Amount"]))


def test_hour_wraps_every_24_hours():
    raw = pd.DataFrame({"Time": [0.0, 3_600.0, 86_400.0 + 7_200.0], "Amount": [1.0, 1.0, 1.0]})
    for i in range(1, 29):
        raw[f"V{i}"] = 0.0
    assert engineer(raw)["hour"].tolist() == [0.0, 1.0, 2.0]


def test_scale_to_unit_clips_to_fixed_bounds():
    features = pd.DataFrame(
        {
            **{f"V{i}": [-50.0, 0.0, 50.0] for i in range(1, 29)},
            "hour": [0.0, 12.0, 24.0],
            "log_amount": [0.0, 5.5, 99.0],
        }
    )
    scaled = scale_to_unit(features)
    assert scaled.min() == -1.0 and scaled.max() == 1.0
    np.testing.assert_allclose(scaled[1], 0.0)
