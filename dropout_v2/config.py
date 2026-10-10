"""All constants of the v2 pipeline. Change behaviour here, not in the modules."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

# --------------------------------------------------------------------------- source
TABLE = "aiml_usecase_curated.final_student_dropout_info_new"

IMPALA_HOST = os.environ.get("DROPOUT_IMPALA_HOST", "dl-dev-cl-dn01.datalake-dev.local")
IMPALA_PORT = int(os.environ.get("DROPOUT_IMPALA_PORT", "21050"))
IMPALA_AUTH = os.environ.get("DROPOUT_IMPALA_AUTH", "GSSAPI")
IMPALA_KERBEROS_SERVICE = os.environ.get("DROPOUT_IMPALA_KERBEROS_SERVICE", "impala")

# --------------------------------------------------------------------------- years
ACADEMIC_YEARS: List[str] = ["2022-23", "2023-24", "2024-25", "2025-26", "2026-27"]
LIVE_YEAR = "2026-27"
TRAIN_YEARS: List[str] = ["2023-24"]
VALID_YEAR = "2024-25"
TEST_YEAR = "2025-26"
# 2022-23 labels are unreliable (74.9% of its "not found" students reappear in 2024-25).
HISTORY_ONLY_YEARS: List[str] = ["2022-23"]

# Rows that are known to be missing from the source table: year -> classes.
# A student whose expected previous class falls in a gap gets prev_status PREV_YEAR_DATA_GAP.
KNOWN_GAPS: Dict[str, List[int]] = {"2023-24": [-3, -2, -1, 1]}

# --------------------------------------------------------------------------- calendar
# Academic-year order. Jan..May belong to the second calendar year of the academic year.
MONTHS: List[str] = ["jun", "jul", "aug", "sep", "oct", "nov", "dec", "jan", "feb", "mar", "apr", "may"]
QUARTERS: Dict[str, List[str]] = {
    "q1": ["jun", "jul", "aug"],
    "q2": ["sep", "oct", "nov"],
    "q3": ["dec", "jan", "feb"],
    "q4": ["mar", "apr", "may"],
}
PREV_YEAR_LAST_QUARTER: List[str] = ["mar", "apr"]

# Snapshot name -> number of visible months of the current year, counted from June.
SNAPSHOTS: Dict[str, int] = {"start": 0, "jul": 2, "aug": 3, "nov": 6, "feb": 9, "apr": 11}
DEFAULT_SNAPSHOTS: List[str] = ["start", "jul", "aug", "nov", "feb", "apr"]
# Current-year marks are year totals; only the end-of-year snapshot may use them.
END_OF_YEAR_SNAPSHOTS = {"apr"}

# --------------------------------------------------------------------------- population and label
BASE_CLASSES = tuple(range(1, 10))          # classes 1-9; class 10 is excluded by decision
KEEP_CLASS_MIN = -3                          # cleaned data keeps -3..12 for history and labels
KEEP_CLASS_MAX = 12
MAX_ROWS_PER_PERSON_YEAR = 3                 # more rows than this = placeholder person_id
DECEASED_VALUES = {"DECEASED"}               # "REPORTED DECEASED" students mostly still attend
LABEL_WINDOW: List[str] = ["jun", "jul"]     # months of T+1 that count as evidence of study
SCHOOL_REPORTING_MIN = 0.5                   # share of a school's rows with working days
CLASS_CONTINUED_MIN = 11                     # a next-year row in class >= 11 counts as continuation
USE_SOCIAL_CATEGORY = True                   # religion and caste are never used

LABEL_REASONS: List[str] = [
    "CONTINUED_ATTENDING",
    "CONTINUED_CLASS_11_PLUS",
    "DROPOUT_NOT_FOUND",
    "DROPOUT_NOT_ATTENDING",
    "UNKNOWN_SCHOOL_NOT_REPORTING",
]

# --------------------------------------------------------------------------- cleaning
NULL_TOKENS = {"", "NULL", "NONE", "NAN", "NA", "N/A", "-", "--", "?", "NIL", "UNKNOWN"}
UNKNOWN = "UNKNOWN"
OTHER = "OTHER"
MAX_WORKING_DAYS = 31
MARKS_OUTLIER_FACTOR = 3.0                   # value > factor * p99 within year x class -> missing
AGE_MIN, AGE_MAX = 3, 25
LAT_RANGE = (12.0, 20.0)
LON_RANGE = (76.0, 85.0)
MAX_HOME_SCHOOL_KM = 50.0

# --------------------------------------------------------------------------- features
RATE_SMOOTHING_WEIGHT = 50.0
RARE_CATEGORY_FRACTION = 0.0002              # categories rarer than this in training -> OTHER
RARE_CATEGORY_MIN_COUNT = 20

# --------------------------------------------------------------------------- model
NEG_PER_POS = 5.0
NEG_RATE_MIN = 0.05
EARLY_STOP_VALID_ROWS = 1_500_000
MAX_ROUNDS = 3000
EARLY_STOP_PATIENCE = 100
REFIT_ROUNDS_FACTOR = 1.15
RANDOM_SEED = 42
LGB_PARAMS: Dict[str, object] = {
    "objective": "binary",
    "metric": ["average_precision", "auc"],
    "learning_rate": 0.05,
    "num_leaves": 127,
    "min_data_in_leaf": 500,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "max_bin": 255,
    "min_data_per_group": 200,
    "cat_smooth": 20.0,
    "verbosity": -1,
    "seed": RANDOM_SEED,
    "deterministic": True,
    "num_threads": max(1, (os.cpu_count() or 2)),
}
TUNE_GRID: Dict[str, list] = {
    "num_leaves": [63, 127, 255],
    "min_data_in_leaf": [200, 500, 1000, 2000],
    "feature_fraction": [0.6, 0.8, 1.0],
    "lambda_l2": [0.0, 1.0, 5.0],
}
TUNE_SAMPLE_FRACTION = 0.3

# --------------------------------------------------------------------------- evaluation and tiers
TOP_K: List[float] = [0.01, 0.02, 0.05, 0.10, 0.20]
BOOTSTRAP_REPS = 100
BOOTSTRAP_MAX_ROWS = 2_000_000
LOGISTIC_SAMPLE_ROWS = 1_000_000
SINGLE_FEATURE_AUC_ALERT = 0.95
# Cumulative statewide rank cut-offs among ATTENDING/UNKNOWN students.
TIER_CUTS: List[tuple] = [("CRITICAL", 0.02), ("HIGH", 0.07), ("MEDIUM", 0.17)]
TIER_REST = "LOW"
LIVE_MONTH_MIN_COVERAGE = 0.5
TOP_LIST_PER_DISTRICT = 500
REASONS_PER_STUDENT = 3

# --------------------------------------------------------------------------- paths
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("DROPOUT_DATA_ROOT", str(REPO_ROOT / "data" / "v2")))
ARTIFACT_ROOT = Path(os.environ.get("DROPOUT_ARTIFACT_ROOT", str(REPO_ROOT / "artifacts" / "v2")))
LEGACY_SCORES_PATH = REPO_ROOT / "artifacts" / "statewide_scores.parquet"


# --------------------------------------------------------------------------- helpers
def visible_months(snapshot: str) -> List[str]:
    """Months of the current academic year that a snapshot model may see."""
    if snapshot not in SNAPSHOTS:
        raise ValueError("Unknown snapshot '{}'. Known: {}".format(snapshot, ", ".join(SNAPSHOTS)))
    return MONTHS[: SNAPSHOTS[snapshot]]


def year_start(year: str) -> int:
    """Calendar year in which the academic year starts ('2025-26' -> 2025)."""
    return int(year[:4])


def _shift(year: str, n: int) -> Optional[str]:
    if year not in ACADEMIC_YEARS:
        raise ValueError("Unknown academic year '{}'".format(year))
    i = ACADEMIC_YEARS.index(year) + n
    return ACADEMIC_YEARS[i] if 0 <= i < len(ACADEMIC_YEARS) else None


def prev_year(year: str, n: int = 1) -> Optional[str]:
    return _shift(year, -n)


def next_year(year: str) -> Optional[str]:
    return _shift(year, 1)


def data_tag(sample_pct: int = 0, synthetic: bool = False, tag: Optional[str] = None) -> str:
    """Name of the data cache folder."""
    if tag:
        return tag
    if synthetic:
        return "synthetic"
    return "sample{}".format(int(sample_pct)) if sample_pct and sample_pct > 0 else "full"
