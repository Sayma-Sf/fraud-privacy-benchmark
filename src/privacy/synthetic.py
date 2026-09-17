"""Synthetic transactions generated with SDV, one generator per class.

Fraud is 0.17% of rows. A single generator fitted to the whole table spends
almost all its capacity on legitimate transactions and produces unconvincing
fraud, so each class gets its own generator. They are then sampled at the real
fraud rate, which keeps the ratio exact.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pandas as pd
import torch
from sdv.evaluation.single_table import evaluate_quality
from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer

from src.ingest.load import RAW_COLUMNS, TARGET


def transactions_metadata(include_target: bool = False) -> Metadata:
    columns = {c: {"sdtype": "numerical", "computer_representation": "Float"} for c in RAW_COLUMNS}
    if include_target:
        columns[TARGET] = {"sdtype": "categorical"}
    return Metadata.load_from_dict({"columns": columns}, single_table_name="transactions")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _make_synthesizer(kind: str, n_rows: int, ctgan_steps: int):
    metadata = transactions_metadata()
    if kind == "gaussian_copula":
        return GaussianCopulaSynthesizer(metadata, default_distribution="truncnorm")
    if kind == "ctgan":
        # Epochs mean very different things for 295 fraud rows and 20,000 legitimate
        # ones, so both generators get the same number of optimiser steps instead.
        batch_size = 500 if n_rows >= 5_000 else 50
        steps_per_epoch = max(n_rows // batch_size, 1)
        epochs = math.ceil(ctgan_steps / steps_per_epoch)
        return CTGANSynthesizer(
            metadata, epochs=epochs, batch_size=batch_size, pac=10, enable_gpu=False
        )
    raise ValueError(f"Unknown synthesizer: {kind}")


class ClassConditionalSynthesizer:
    """One fitted SDV generator per class, sampled at a fixed fraud rate."""

    def __init__(self, kind: str, fraud_rate: float, generators: dict[int, object]):
        self.kind = kind
        self.fraud_rate = fraud_rate
        self.generators = generators

    @classmethod
    def fit(
        cls,
        train: pd.DataFrame,
        kind: str,
        *,
        max_rows_per_class: int | None = None,
        ctgan_steps: int = 4000,
        seed: int = 0,
    ) -> ClassConditionalSynthesizer:
        """Fit on real training rows (raw columns plus ``Class``).

        ``max_rows_per_class`` caps how many rows each generator sees. It keeps CTGAN
        on legitimate transactions to a CPU-friendly size and never drops fraud rows,
        of which there are only a few hundred.
        """
        generators = {}
        for label in (0, 1):
            rows = train.loc[train[TARGET] == label, RAW_COLUMNS]
            if max_rows_per_class and len(rows) > max_rows_per_class:
                rows = rows.sample(max_rows_per_class, random_state=seed)
            _seed_everything(seed + label)
            generator = _make_synthesizer(kind, len(rows), ctgan_steps)
            generator.fit(rows)
            generators[label] = generator
        return cls(kind, float(train[TARGET].mean()), generators)

    def sample(self, n_rows: int, seed: int = 0) -> pd.DataFrame:
        n_fraud = max(1, round(n_rows * self.fraud_rate))
        _seed_everything(seed)
        parts = [
            self.generators[0].sample(n_rows - n_fraud).assign(**{TARGET: 0}),
            self.generators[1].sample(n_fraud).assign(**{TARGET: 1}),
        ]
        synthetic = pd.concat(parts, ignore_index=True)
        return synthetic.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def fidelity_scores(
    real: pd.DataFrame, synthetic: pd.DataFrame, *, max_rows: int = 20_000, seed: int = 0
) -> dict[str, float]:
    """SDMetrics quality report: how closely columns and column pairs match the real data.

    Scores run from 0 to 1. "Column shapes" compares each column's distribution;
    "column pair trends" compares correlations between every pair of columns.
    """
    real = real.sample(min(len(real), max_rows), random_state=seed)
    synthetic = synthetic.sample(min(len(synthetic), max_rows), random_state=seed)
    report = evaluate_quality(
        real, synthetic, transactions_metadata(include_target=True), verbose=False
    )
    properties = report.get_properties().set_index("Property")["Score"]
    return {
        "overall": float(report.get_score()),
        "column_shapes": float(properties["Column Shapes"]),
        "column_pair_trends": float(properties["Column Pair Trends"]),
    }
