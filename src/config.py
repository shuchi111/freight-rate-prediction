from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

TRAIN_PATH = DATA_DIR / "train_test.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"
TEMPLATE_PATH = DATA_DIR / "validation_predictions_template.csv"
DECEMBER_PATH = DATA_DIR / "december_chart_inputs.csv"
PREDICTIONS_PATH = ROOT / "validation_predictions.csv"

TARGET = "posted_rate"
RANDOM_SEED = 42

# Labels whose rate per mile is this many times above/below the expected
# value for their equipment and distance band are treated as data-entry errors.
OUTLIER_HIGH_RATIO = 2.0
OUTLIER_LOW_RATIO = 0.5

# Time-based validation: train on everything before HOLDOUT_START,
# evaluate on HOLDOUT_START onward (mirrors predicting the next two months).
HOLDOUT_START = "2025-09-01"
ROLLING_TEST_MONTHS = ["2025-06", "2025-07", "2025-08", "2025-09", "2025-10"]
