"""Project-wide constants. No logic, no I/O."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Training
RANDOM_STATE = 42
SAMPLE_FRACTION = 0.1  # take 10% of the balanced data so the from-scratch LR can fit it in memory
TRAIN_TEST_SPLIT = (0.8, 0.2)
TARGET_COL = "CANCELLED"

# Schema
COLUMNS_TO_KEEP = [
    "FL_DATE",
    "OP_CARRIER",
    "ORIGIN",
    "DEST",
    "CRS_DEP_TIME",
    "CRS_ARR_TIME",
    "CANCELLED",
    "CRS_ELAPSED_TIME",
    "DISTANCE",
]
# Dropped during feature engineering because they correlate strongly with kept ones
# (CRS_ARR_TIME ↔ CRS_DEP_TIME, DISTANCE ↔ CRS_ELAPSED_TIME).
REDUNDANT_COLS_TO_DROP = ("CRS_ARR_TIME", "DISTANCE")

CATEGORICAL_COLS = ("OP_CARRIER", "ORIGIN", "DEST")
NUMERIC_COLS = (
    "CRS_DEP_TIME",
    "CRS_ELAPSED_TIME",
    "FL_DATE_WEEKDAY",
    "FL_DATE_MONTH",
)
# Final feature order fed into VectorAssembler.
FEATURE_COLS = (
    "CRS_DEP_TIME",
    "CRS_ELAPSED_TIME",
    "FL_DATE_WEEKDAY",
    "FL_DATE_MONTH",
    "AIRLINE_ID",
    "ORIGIN_ID",
    "DEST_ID",
)

# Grid-search ranges for the from-scratch LR. Two phases: first learning_rate + n_iterations,
# then lambda while holding the first two fixed (matches the original notebook).
LEARNING_RATES = (0.1, 0.2, 0.4, 0.01, 0.02, 0.04)
N_ITERATIONS_GRID = (100, 200, 400, 500, 1000)
LAMBDA_GRID = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)

# MLlib LR comparison model: maxIter matches the upper bound of N_ITERATIONS_GRID.
MLLIB_MAX_ITER = 1000
