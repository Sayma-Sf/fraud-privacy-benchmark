"""Load the transactions file, check its schema, and split it."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

V_COLUMNS = [f"V{i}" for i in range(1, 29)]
RAW_COLUMNS = ["Time", *V_COLUMNS, "Amount"]
TARGET = "Class"


@dataclass(frozen=True)
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Return the expected columns in canonical order, or raise if the file is off."""
    missing = [c for c in [*RAW_COLUMNS, TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df = df[[*RAW_COLUMNS, TARGET]]
    if df.isna().any().any():
        raise ValueError("Transactions contain missing values")
    if not set(df[TARGET].unique()) <= {0, 1}:
        raise ValueError(f"{TARGET} must be 0/1, got {sorted(df[TARGET].unique())}")
    if (df["Amount"] < 0).any():
        raise ValueError("Amount must be non-negative")
    return df.astype({TARGET: int})


def load_transactions(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python data/download.py` first.")
    return validate_schema(pd.read_csv(path))


def split(df: pd.DataFrame, val_size: float, test_size: float, seed: int) -> Splits:
    """Stratified train/val/test split that keeps the ~0.17% fraud rate in every part."""
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df[TARGET], random_state=seed
    )
    train, val = train_test_split(
        train_val,
        test_size=val_size / (1 - test_size),
        stratify=train_val[TARGET],
        random_state=seed,
    )
    return Splits(*(part.reset_index(drop=True) for part in (train, val, test)))
