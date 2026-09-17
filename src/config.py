"""Project-wide settings: paths, seeds, split sizes and experiment grids."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "creditcard.csv"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "reports" / "figures"

SEED = 42

# Stratified 60/20/20 split. Validation picks early-stopping rounds and alert
# thresholds; the test split is touched once, for the numbers in the README.
VAL_SIZE = 0.2
TEST_SIZE = 0.2

# Privacy budgets for the differentially private classifier. inf = no noise.
EPSILONS = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, float("inf")]
# DP training is randomised, so every epsilon is trained this many times.
DP_REPEATS = 20
# Each row is clipped to this L2 norm before DP training, which caps how far one
# transaction can move the model. Picked from {0.5, 1, 1.5, 2, sqrt(30)} on the
# validation split. See "Limitations" in the README for what that choice costs.
DP_DATA_NORM = 1.0

# Synthetic data generators, each fitted once per class (see src/privacy/synthetic.py).
SYNTHESIZERS = ["gaussian_copula", "ctgan"]
# CTGAN sees at most this many legitimate rows (all fraud rows are always used)
# and trains for this many optimiser steps per class. Sized for a laptop CPU.
SYNTH_MAX_ROWS_PER_CLASS = 20_000
CTGAN_STEPS = 4000
