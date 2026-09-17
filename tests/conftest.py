import numpy as np
import pandas as pd
import pytest

from src.ingest.load import V_COLUMNS


@pytest.fixture(scope="session")
def transactions() -> pd.DataFrame:
    """Small stand-in with the real schema and a learnable fraud signal. No real data in CI."""
    rng = np.random.default_rng(0)
    n = 4000
    fraud = rng.random(n) < 0.05
    v = rng.normal(size=(n, len(V_COLUMNS)))
    v[fraud, :3] += 3.0
    df = pd.DataFrame(v, columns=V_COLUMNS)
    df.insert(0, "Time", rng.uniform(0, 172_800, n))
    df["Amount"] = rng.exponential(80, n).round(2)
    df["Class"] = fraud.astype(int)
    return df
