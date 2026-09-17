import pandas as pd
import pytest

from src.ingest.load import RAW_COLUMNS, TARGET, load_transactions, split, validate_schema


def test_validate_schema_returns_canonical_column_order(transactions):
    shuffled = transactions[list(reversed(transactions.columns))]
    assert list(validate_schema(shuffled).columns) == [*RAW_COLUMNS, TARGET]


def test_validate_schema_rejects_missing_column(transactions):
    with pytest.raises(ValueError, match="Missing columns"):
        validate_schema(transactions.drop(columns="V14"))


def test_validate_schema_rejects_non_binary_label(transactions):
    bad = transactions.assign(Class=transactions["Class"] * 2)
    with pytest.raises(ValueError, match="must be 0/1"):
        validate_schema(bad)


def test_load_transactions_points_to_download_script(tmp_path):
    with pytest.raises(FileNotFoundError, match="data/download.py"):
        load_transactions(tmp_path / "creditcard.csv")


def test_split_is_stratified_and_disjoint(transactions):
    parts = split(transactions, val_size=0.2, test_size=0.2, seed=0)
    sizes = [len(parts.train), len(parts.val), len(parts.test)]
    assert sum(sizes) == len(transactions)
    assert sizes[0] == pytest.approx(0.6 * len(transactions), abs=2)

    overall = transactions[TARGET].mean()
    for part in (parts.train, parts.val, parts.test):
        assert part[TARGET].mean() == pytest.approx(overall, abs=0.005)

    # Time is a continuous random draw in the fixture, so it identifies rows.
    times = pd.concat([parts.train["Time"], parts.val["Time"], parts.test["Time"]])
    assert times.is_unique
