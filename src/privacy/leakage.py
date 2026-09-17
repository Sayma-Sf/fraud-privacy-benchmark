"""Check whether synthetic rows are copies of real training rows.

The synthesizers here (Gaussian copula, CTGAN) have no formal privacy guarantee.
Synthetic data is only safer to share if the generator learned the distribution
rather than memorising rows, and this is an empirical check of that.

Distance to closest record (DCR): take a sample of real training rows and an
equally sized set of real rows the generator never saw (the holdout). For each
synthetic row, find its nearest neighbour in both. If the generator generalises,
a synthetic row is as likely to sit near a holdout row as near a training row,
so about 50% land closer to training. Well above 50% means the generator is
reproducing the rows it was trained on.
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def nearest_distance(reference: np.ndarray, queries: np.ndarray) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=1).fit(reference)
    distances, _ = nn.kneighbors(queries)
    return distances[:, 0]


def dcr_report(
    train: np.ndarray,
    holdout: np.ndarray,
    synthetic: np.ndarray,
    *,
    max_rows: int = 5000,
    seed: int = 0,
) -> dict[str, float]:
    """DCR summary for synthetic rows against real train and holdout rows.

    All three arrays must already share one feature scaling.
    """
    rng = np.random.default_rng(seed)
    n = min(len(train), len(holdout), max_rows)
    train_s = train[rng.choice(len(train), n, replace=False)]
    holdout_s = holdout[rng.choice(len(holdout), n, replace=False)]
    synth_s = synthetic[rng.choice(len(synthetic), min(len(synthetic), max_rows), replace=False)]

    to_train = nearest_distance(train_s, synth_s)
    to_holdout = nearest_distance(holdout_s, synth_s)
    closer_to_train = np.mean(to_train < to_holdout) + 0.5 * np.mean(to_train == to_holdout)

    return {
        "rows_compared": int(n),
        "share_closer_to_train": float(closer_to_train),
        "median_dcr_synthetic": float(np.median(to_train)),
        # Reference point: how close genuinely unseen real rows sit to the training rows.
        "median_dcr_holdout": float(np.median(nearest_distance(train_s, holdout_s))),
        "exact_copies": int(np.sum(to_train < 1e-9)),
    }
