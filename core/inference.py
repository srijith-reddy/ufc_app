"""
Model artifact loading and inference.

Enforces prefight-only feature construction.
All preprocessing steps match the training pipeline exactly.
"""
from __future__ import annotations
import json

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from .config import (
    CALIBRATOR_PATH,
    CLIP_BOUNDS_PATH,
    FEATURE_COLS_PATH,
    FIGHTERS_PATH,
    LOG_COLS,
    MISSING_COLS,
    MODEL_PATH,
)
from .names import normalize_name


# ── Artifact Loading ───────────────────────────────────────────────────────────

def load_artifacts() -> tuple:
    """
    Load all model artifacts from disk.

    Returns:
        (booster, calibrator, feature_cols, clip_bounds, fighters_df, fighter_lookup)

    Raises:
        FileNotFoundError if any required artifact is missing, with a list of
        all missing files so the caller can diagnose in one shot.
    """
    required = {
        "XGBoost model": MODEL_PATH,
        "isotonic calibrator": CALIBRATOR_PATH,
        "feature column list": FEATURE_COLS_PATH,
        "clip bounds": CLIP_BOUNDS_PATH,
        "fighter snapshot CSV": FIGHTERS_PATH,
    }
    missing = [f"{label}: {path}" for label, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required model artifacts:\n" + "\n".join(f"  - {m}" for m in missing)
        )

    booster = xgb.Booster()
    booster.load_model(str(MODEL_PATH))

    calibrator = joblib.load(str(CALIBRATOR_PATH))

    with open(FEATURE_COLS_PATH) as f:
        feature_cols: list[str] = json.load(f)

    with open(CLIP_BOUNDS_PATH) as f:
        clip_bounds: dict = json.load(f)

    fighters_df = pd.read_csv(str(FIGHTERS_PATH))
    # Consolidate the frame before adding the derived column to avoid
    # pandas PerformanceWarning on highly fragmented DataFrames.
    fighters_df = fighters_df.copy()
    fighters_df["fighter_norm"] = fighters_df["fighter"].apply(normalize_name)

    # Canonical lookup: normalized name → original name in CSV
    fighter_lookup: dict[str, str] = {
        row["fighter_norm"]: row["fighter"]
        for _, row in fighters_df.iterrows()
    }

    return booster, calibrator, feature_cols, clip_bounds, fighters_df, fighter_lookup


# ── Feature Construction ───────────────────────────────────────────────────────

def build_feature_row(A: pd.Series, B: pd.Series) -> dict:
    """
    Construct the raw (pre-preprocessing) feature dict for a fight.

    A is Fighter A's row from fighters_latest.csv.
    B is Fighter B's row.

    Strictly prefight: only uses stats available before the fight.
    Does not touch result, method, KD, or any post-fight columns.
    """
    return {
        # Glicko rating differential
        "rating_diff": A["g_rating_before"] - B["g_rating_before"],
        "RD_diff": A["g_RD_before"] - B["g_RD_before"],

        # Physical
        "height_diff": A["fighter_height_inches"] - B["fighter_height_inches"],
        "reach_diff": A["fighter_reach_inches"] - B["fighter_reach_inches"],
        "age_diff": A["fighter_age"] - B["fighter_age"],

        # Fighter A striking & grappling
        "SLpM": A["SLpM"],
        "SApM": A["SApM"],
        "Str_Acc": A["Str_Acc"],
        "Str_Def": A["Str_Def"],
        "TD_Avg": A["TD_Avg"],
        "TD_Acc": A["TD_Acc"],
        "TD_Def": A["TD_Def"],
        "Sub_Avg": A["Sub_Avg"],

        # Fighter B striking & grappling (opponent perspective)
        "opp_SLpM": B["SLpM"],
        "opp_SApM": B["SApM"],
        "opp_Str_Acc": B["Str_Acc"],
        "opp_Str_Def": B["Str_Def"],
        "opp_TD_Avg": B["TD_Avg"],
        "opp_TD_Acc": B["TD_Acc"],
        "opp_TD_Def": B["TD_Def"],
        "opp_Sub_Avg": B["Sub_Avg"],

        # Fighter A temporal / form
        "fights_before": A["fights_before"],
        "days_since_last_fight": A["days_since_last_fight"],
        "win_rate_before": A["win_rate_before"],
        "recent_win_rate_3": A["recent_win_rate_3"],
        "recent_win_rate_5": A["recent_win_rate_5"],

        # Fighter B temporal / form
        "opp_fights_before": B["fights_before"],
        "opp_days_since_last_fight": B["days_since_last_fight"],
        "opp_win_rate_before": B["win_rate_before"],
        "opp_recent_win_rate_3": B["recent_win_rate_3"],
        "opp_recent_win_rate_5": B["recent_win_rate_5"],

        # Debut flags
        "is_debut": int(A["fights_before"] == 0),
        "opp_is_debut": int(B["fights_before"] == 0),
    }


def preprocess_row(
    row_df: pd.DataFrame,
    feature_cols: list[str],
    clip_bounds: dict,
) -> np.ndarray:
    """
    Apply training-identical preprocessing to a single feature row DataFrame.

    Steps (must mirror the training notebook exactly):
      1. Add binary missingness indicators for MISSING_COLS
      2. fillna(0)
      3. Quantile clip heavy-tailed temporal features
      4. Log-transform clipped temporal features (log1p)
      5. Select and order by feature_cols
    """
    df = row_df.copy()

    for col in MISSING_COLS:
        df[f"{col}_missing"] = df[col].isna().astype(int)

    df = df.fillna(0)

    for col, bounds in clip_bounds.items():
        if col in df.columns:
            df[col] = df[col].clip(bounds["lo"], bounds["hi"])

    for col in LOG_COLS:
        df[f"log_{col}"] = np.log1p(df[col])

    return df[feature_cols].values


# ── Inference ──────────────────────────────────────────────────────────────────

def predict_win_prob(
    booster: xgb.Booster,
    calibrator,
    X: np.ndarray,
) -> float:
    """
    Run XGBoost inference + isotonic calibration.

    Returns calibrated P(Fighter A wins), clipped to [0.01, 0.99].
    The clip prevents downstream log/Kelly math from hitting 0 or 1.
    """
    dmat = xgb.DMatrix(X)
    raw = booster.predict(dmat)[0]
    cal = calibrator.transform([raw])[0]
    return float(np.clip(cal, 0.01, 0.99))
