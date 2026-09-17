import math

import pytest

from src.reporting.app_data import (
    INF_LABEL,
    RESULTS_DIR,
    curves_frame,
    odds_multiplier,
    plain_language,
    sweep_frame,
)
from src.reporting.figures import render_all

CURVE = {"recall": [0.0, 0.5, 1.0], "precision": [1.0, 0.6, 0.1]}


def sweep_row(epsilon, auprc):
    return {
        "epsilon": epsilon, "epsilon_label": "inf" if epsilon is None else f"{epsilon:g}",
        "runs": 1 if epsilon is None else 5, "auprc_mean": auprc, "auprc_std": 0.01,
        "auprc_p10": auprc - 0.02, "auprc_p90": auprc + 0.02, "roc_auc_mean": 0.9,
        "precision_mean": 0.5, "recall_mean": 0.5, "f1_mean": 0.5,
    }


@pytest.fixture
def summary():
    tstr = {"auprc": 0.7}
    dcr = {"share_closer_to_train": 0.5, "exact_copies": 0}
    return {
        "dataset": {"rows": 1000, "frauds": 10, "fraud_rate": 0.01, "test_rows": 200,
                    "test_frauds": 2},
        "config": {"dp_repeats": 5, "seed": 0},
        "baseline": {"lightgbm": {"auprc": 0.8}},
        "dp_sweep": [sweep_row(None, 0.6), sweep_row(0.1, 0.2), sweep_row(1.0, 0.4)],
        "synthetic": {"ctgan": {"tstr": tstr, "fidelity": {"overall": 0.8},
                                "dcr_all": dcr, "dcr_fraud": dcr}},
    }


def test_sweep_frame_orders_epsilon_and_puts_no_noise_last(summary):
    frame = sweep_frame(summary)
    assert frame["label"].tolist() == ["0.1", "1", INF_LABEL]
    assert frame["share_of_no_noise"].iloc[-1] == 1.0
    assert frame["share_of_no_noise"].iloc[0] == pytest.approx(0.2 / 0.6)


@pytest.mark.parametrize(
    ("epsilon", "expected"),
    [(1.0, "2.72×"), (10.0, "22,026×"), (100.0, "about 10^43×"), (math.inf, "unbounded")],
)
def test_odds_multiplier(epsilon, expected):
    assert odds_multiplier(epsilon) == expected


def test_plain_language_grades_privacy():
    assert "Strong privacy" in plain_language(0.5)
    assert "Weak privacy" in plain_language(50)
    assert "No privacy guarantee" in plain_language(math.inf)


def test_curves_frame_is_long_format():
    frame = curves_frame({"a": CURVE, "b": CURVE})
    assert len(frame) == 6
    assert set(frame["model"]) == {"a", "b"}


def test_render_all_writes_light_and_dark_figures(summary, tmp_path):
    written = render_all(summary, tmp_path)
    assert {p.name for p in written} == {
        f"{name}_{mode}.png" for name in ("privacy_utility", "synthetic_utility")
        for mode in ("light", "dark")
    }
    assert all(p.stat().st_size > 10_000 for p in written)


@pytest.mark.skipif(not (RESULTS_DIR / "summary.json").exists(), reason="no committed results")
def test_app_runs_on_committed_results():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file("../app.py", default_timeout=60).run()
    assert not app.exception
    assert app.select_slider[0].value == "1"
    app.select_slider[0].set_value(INF_LABEL).run()
    assert not app.exception
