"""
Centralized path and constant configuration.

All file paths are derived from this module using pathlib.
No hardcoded strings or os.path calls elsewhere in the codebase.
"""
from pathlib import Path

# ── Repo Root ──────────────────────────────────────────────────────────────────
# Resolves correctly whether called from the repo root, a subdirectory, or
# inside a Docker container, as long as this file stays at core/config.py.
REPO_ROOT: Path = Path(__file__).resolve().parent.parent

# ── Data Directories ───────────────────────────────────────────────────────────
DATA_DIR: Path = REPO_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
UFCSTATS_DATA_DIR: Path = RAW_DATA_DIR / "ufcstats"

# ── Model Artifacts ────────────────────────────────────────────────────────────
MODELS_DIR: Path = REPO_ROOT / "models"
MODEL_PATH: Path = MODELS_DIR / "xgb_prefight_model.json"
CALIBRATOR_PATH: Path = MODELS_DIR / "xgb_calibrator.pkl"
FEATURE_COLS_PATH: Path = MODELS_DIR / "xgb_feature_cols.json"
CLIP_BOUNDS_PATH: Path = MODELS_DIR / "clip_bounds.json"
FIGHTERS_PATH: Path = MODELS_DIR / "fighters_latest.csv"

# ── Event Data ─────────────────────────────────────────────────────────────────
CARDS_DIR: Path = REPO_ROOT / "cards"
ODDS_DIR: Path = REPO_ROOT / "odds"

# ── Preprocessing Constants (must match training pipeline exactly) ─────────────
LOG_COLS: list[str] = [
    "days_since_last_fight",
    "opp_days_since_last_fight",
    "fights_before",
    "opp_fights_before",
]

# Columns for which a binary missingness indicator is added before fillna(0)
MISSING_COLS: list[str] = [
    "SLpM", "SApM", "TD_Avg", "Sub_Avg", "Str_Acc", "TD_Acc",
    "opp_SLpM", "opp_SApM", "opp_TD_Avg", "opp_Sub_Avg",
    "opp_Str_Acc", "opp_TD_Acc",
]
