"""Download the ULB credit card fraud dataset to ``data/creditcard.csv``.

The canonical source is Kaggle (``mlg-ulb/creditcardfraud``), which needs an API
token in ``~/.kaggle/kaggle.json``. Without one, the script falls back to the
identical OpenML mirror (dataset 1597), which needs no login. Both produce the
same 284,807 rows, and the file is checked before it is written.

Raw data is gitignored and must never be committed.

    python data/download.py            # Kaggle if configured, else OpenML
    python data/download.py --source openml
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
TARGET = DATA_DIR / "creditcard.csv"

KAGGLE_SLUG = "mlg-ulb/creditcardfraud"
OPENML_PARQUET = "https://data.openml.org/datasets/0000/1597/dataset_1597.pq"

COLUMNS = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount", "Class"]
EXPECTED_ROWS = 284_807
EXPECTED_FRAUD = 492


def kaggle_configured() -> bool:
    token = Path.home() / ".kaggle" / "kaggle.json"
    return importlib.util.find_spec("kaggle") is not None and token.exists()


def from_kaggle() -> pd.DataFrame:
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            [sys.executable, "-m", "kaggle", "datasets", "download",
             "-d", KAGGLE_SLUG, "-p", tmp, "--unzip"],
            check=True,
        )
        return pd.read_csv(Path(tmp) / "creditcard.csv")


def from_openml() -> pd.DataFrame:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "creditcard.pq"
        print(f"Downloading {OPENML_PARQUET}")
        urllib.request.urlretrieve(OPENML_PARQUET, path)
        df = pd.read_parquet(path)
    # OpenML stores the label as a nominal attribute ("0"/"1"); Kaggle stores an int.
    df["Class"] = df["Class"].astype(str).str.strip("'\"").astype(int)
    return df


def validate(df: pd.DataFrame) -> pd.DataFrame:
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Downloaded file is missing columns: {sorted(missing)}")
    df = df[COLUMNS]
    if len(df) != EXPECTED_ROWS or int(df["Class"].sum()) != EXPECTED_FRAUD:
        raise ValueError(
            f"Expected {EXPECTED_ROWS} rows with {EXPECTED_FRAUD} frauds, "
            f"got {len(df)} rows with {int(df['Class'].sum())} frauds"
        )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", choices=["auto", "kaggle", "openml"], default="auto")
    args = parser.parse_args()

    use_kaggle = args.source == "kaggle" or (args.source == "auto" and kaggle_configured())
    df = validate(from_kaggle() if use_kaggle else from_openml())
    df.to_csv(TARGET, index=False)
    print(f"Wrote {len(df):,} rows ({int(df['Class'].sum())} frauds) to {TARGET}")


if __name__ == "__main__":
    main()
