"""
pipelines/refresh_fighters.py
═══════════════════════════════════════════════════════════════════
Rebuild models/fighters_latest.csv from raw ufcstats CSV files.

This script is the extracted, runnable version of the feature
engineering pipeline from ufc_pipeline.ipynb (cells 9–42 minus
model training).  Run it after new fight data is available to
keep the fighter snapshot current.

What it does
────────────
  1. Load raw CSVs (fights, fight_totals, fighter profiles, history)
  2. Clean fight history (dates, outcomes)
  3. Compute Glicko-2 ratings chronologically over all fights
  4. Merge fight stats and fighter profiles
  5. Compute temporal prefight features (shifts, no leakage)
  6. Apply quantile clipping and log transforms
  7. Validate the output
  8. Write models/fighters_latest.csv

What it does NOT do
───────────────────
  - Re-train the XGBoost model (run ufc_pipeline.ipynb for that)
  - Scrape raw data from ufcstats.com (provide --data-dir yourself)
  - Update clip_bounds.json (those are fit on training data only
    and must be re-saved when the model is retrained)

Usage
─────
  python -m pipelines.refresh_fighters --data-dir /path/to/raw/csvs
  python -m pipelines.refresh_fighters --data-dir /path/to/raw/csvs --dry-run

Raw CSV files expected in --data-dir
──────────────────────────────────────
  events.csv                  — UFC event list
  fights.csv                  — all fights (fighter_A/B, WL_label, ...)
  fight_totals.csv            — per-fight strike / TD stats
  fighters_advanced.csv       — fighter profiles (height, reach, SLpM, ...)
  fighters_fight_history.csv  — fight-by-fight history per fighter (for dates)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from math import pi, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from dateutil import parser as dateutil_parser

# ── Repo paths ─────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
CLIP_BOUNDS_PATH = MODELS_DIR / "clip_bounds.json"
OUTPUT_PATH = MODELS_DIR / "fighters_latest.csv"


# ══════════════════════════════════════════════════════════════════════════════
# 1.  FIGHT HISTORY CLEANING
# ══════════════════════════════════════════════════════════════════════════════

def _clean_opponent(row: pd.Series) -> str:
    fighter = str(row["fighter_name"]).strip()
    opp = str(row["opponent"]).strip()
    opp_clean = opp.replace(fighter, "").strip()
    return " ".join(opp_clean.split())


def _extract_date(event_str: str):
    if pd.isna(event_str):
        return None
    m = re.search(r"[A-Za-z]{3,}\.?\s*\d{1,2},\s*\d{4}", str(event_str))
    if m:
        try:
            return dateutil_parser.parse(m.group(0))
        except Exception:
            return None
    return None


def _wl_to_score(wl) -> float | None:
    if wl is None:
        return None
    wl = str(wl).lower()
    if "win" in wl:
        return 1.0
    if "loss" in wl:
        return 0.0
    if "draw" in wl:
        return 0.5
    return None


def clean_fight_history(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw fighters_fight_history.csv.

    - Fix opponent names
    - Parse dates from event strings
    - Convert WL → numeric score
    - Drop rows without dates or outcomes
    - Sort chronologically
    """
    df = df_raw.copy()
    df["opponent_clean"] = df.apply(_clean_opponent, axis=1)
    df["date"] = df["event"].apply(_extract_date)
    df["score"] = df["wl"].apply(_wl_to_score)
    df = df.dropna(subset=["date", "score"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2.  GLICKO-2 ENGINE  (constant volatility)
# ══════════════════════════════════════════════════════════════════════════════

def _g(RD: float) -> float:
    return 1.0 / sqrt(1.0 + 3.0 * (RD ** 2) / (pi ** 2))


def _E(mu: float, mu_j: float, phi_j: float) -> float:
    return 1.0 / (1.0 + np.exp(-_g(phi_j) * (mu - mu_j)))


class _Fighter:
    """Mutable Glicko-2 fighter state (constant volatility variant)."""

    __slots__ = ("name", "rating", "RD", "vol")

    def __init__(self, name: str, rating: float = 1500.0, RD: float = 350.0, vol: float = 0.06):
        self.name = name
        self.rating = rating
        self.RD = RD
        self.vol = vol

    def update(self, opp_ratings: list[float], opp_RDs: list[float], scores: list[float]) -> None:
        if not opp_ratings:
            return

        mu = (self.rating - 1500.0) / 173.7178
        phi = self.RD / 173.7178
        sigma = self.vol

        v_inv = 0.0
        delta_sum = 0.0

        for r_j, RD_j, s_j in zip(opp_ratings, opp_RDs, scores):
            mu_j = (r_j - 1500.0) / 173.7178
            phi_j = RD_j / 173.7178
            g = _g(phi_j)
            E = _E(mu, mu_j, phi_j)
            v_inv += (g * g) * E * (1.0 - E)
            delta_sum += g * (s_j - E)

        v = 1.0 / v_inv
        mu_prime = mu + (phi * phi / (phi * phi + sigma * sigma + v)) * delta_sum
        phi_star = sqrt(phi * phi + sigma * sigma)
        phi_prime = 1.0 / sqrt(1.0 / (phi_star * phi_star) + 1.0 / v)

        self.rating = 173.7178 * mu_prime + 1500.0
        self.RD = 173.7178 * phi_prime


def _compute_advanced_score(
    base_score: float,
    opp_rating: float,
    avg_rating: float,
    importance: float,
    kd_self: float,
    kd_opp: float,
    sig_self: float,
    sig_opp: float,
    td_self: float,
    td_opp: float,
) -> float:
    """Performance-adjusted Glicko score (matches notebook exactly)."""
    difficulty = opp_rating / avg_rating
    performance = 1.0
    if base_score == 1:
        if kd_self > kd_opp:
            performance *= 1.10
        if sig_self > sig_opp:
            performance *= 1.05
        if td_self > td_opp:
            performance *= 1.05
        if sig_opp > sig_self and kd_opp > kd_self:
            performance *= 0.85
    return base_score * difficulty * importance * performance


def compute_glicko_ratings(
    df_fights: pd.DataFrame,
    df_totals: pd.DataFrame,
    df_history_clean: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run Glicko-2 over all historical fights in chronological order.

    Returns a DataFrame with columns:
        fight_url, fighter, rating_before, RD_before

    The *_before values are the ratings captured BEFORE each fight is processed —
    i.e., strictly prefight information.
    """
    # Attach dates from fight history
    df_hist_ufc = df_history_clean[
        df_history_clean["event"].str.contains("UFC", na=False)
    ].copy()
    df_hist_ufc["date"] = pd.to_datetime(df_hist_ufc["date"])

    df_dates = (
        df_hist_ufc.groupby(["event_url", "fighter_name"])["date"]
        .first()
        .reset_index()
    )

    fights = df_fights.copy()
    fights = fights.merge(
        df_dates.rename(columns={"fighter_name": "fighter_A", "date": "date_A"}),
        on=["event_url", "fighter_A"],
        how="left",
    )
    fights = fights.merge(
        df_dates.rename(columns={"fighter_name": "fighter_B", "date": "date_B"}),
        on=["event_url", "fighter_B"],
        how="left",
    )
    fights["date"] = fights["date_A"].fillna(fights["date_B"])
    fights = fights.drop(columns=["date_A", "date_B"])
    fights["date"] = pd.to_datetime(fights["date"])
    fights = fights.sort_values(["date", "fight_order"]).reset_index(drop=True)

    # Initialise fighter objects
    all_names = pd.concat([fights["fighter_A"], fights["fighter_B"]]).unique()
    fighters: dict[str, _Fighter] = {n: _Fighter(n) for n in all_names}
    avg_rating = 1500.0

    rows: list[dict] = []
    start = time.time()

    for i, row in fights.iterrows():
        A_name = row["fighter_A"]
        B_name = row["fighter_B"]

        if A_name not in fighters:
            fighters[A_name] = _Fighter(A_name)
        if B_name not in fighters:
            fighters[B_name] = _Fighter(B_name)

        fA = fighters[A_name]
        fB = fighters[B_name]

        # Capture ratings BEFORE updating
        rows.append({"fight_url": row["fight_url"], "fighter": A_name,
                     "rating_before": fA.rating, "RD_before": fA.RD})
        rows.append({"fight_url": row["fight_url"], "fighter": B_name,
                     "rating_before": fB.rating, "RD_before": fB.RD})

        wl = str(row["WL_label"]).lower()
        sA, sB = (1.0, 0.0) if wl == "win" else (0.0, 1.0) if wl == "loss" else (0.5, 0.5)

        # Importance weighting (matches notebook)
        fo = int(row["fight_order"])
        en = str(row["event_name"])
        if "UFC Fight Night" not in en and "UFC" in en:
            importance = {1: 1.15, 2: 1.10}.get(fo, 1.05)
        else:
            importance = {1: 1.10, 2: 1.05}.get(fo, 1.0)

        subset = df_totals[df_totals["fight_url"] == row["fight_url"]]
        if subset.empty:
            continue  # skip early events without totals data
        t = subset.iloc[0]

        scoreA = _compute_advanced_score(
            sA, fB.rating, avg_rating, importance,
            t["KD_A"], t["KD_B"], t["Sig Str_A"], t["Sig Str_B"],
            t["Td_A"], t["Td_B"],
        )
        scoreB = _compute_advanced_score(
            sB, fA.rating, avg_rating, importance,
            t["KD_B"], t["KD_A"], t["Sig Str_B"], t["Sig Str_A"],
            t["Td_B"], t["Td_A"],
        )

        fA.update([fB.rating], [fB.RD], [scoreA])
        fB.update([fA.rating], [fA.RD], [scoreB])

        if (i + 1) % 500 == 0:
            print(f"  Glicko [{i+1}/{len(fights)}] — {time.time()-start:.1f}s elapsed")

    print(f"  Glicko complete in {time.time()-start:.1f}s")
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  STAT PARSING HELPERS  (match notebook exactly)
# ══════════════════════════════════════════════════════════════════════════════

def _parse_pair(x) -> tuple[float, float]:
    if pd.isna(x):
        return (np.nan, np.nan)
    x = str(x).strip()
    if "of" in x:
        a, b = x.split("of")
        return (int(a.strip()), int(b.strip()))
    nums = x.split()
    if len(nums) == 2:
        return (int(nums[0]), int(nums[1]))
    return (np.nan, np.nan)


def _parse_int(x) -> float:
    try:
        return int(str(x).strip())
    except Exception:
        return np.nan


def _parse_float(x) -> float:
    try:
        return float(str(x).strip().replace("%", ""))
    except Exception:
        return np.nan


def _parse_time(t) -> float:
    if isinstance(t, str) and ":" in t:
        m, s = t.split(":")
        return int(m) * 60 + int(s)
    return np.nan


def _safe_split(pair: str) -> tuple[int, int]:
    a, b = str(pair).split()
    a = 0 if a == "--" else int(a)
    b = 0 if b == "--" else int(b)
    return a, b


def _height_to_inches(h) -> float:
    if pd.isna(h):
        return np.nan
    h = str(h).strip()
    if h in ("--", "", "nan", "None"):
        return np.nan
    m = re.match(r"(\d+)'\s*(\d+)", h)
    if m:
        return int(m.group(1)) * 12 + int(m.group(2))
    return np.nan


def _reach_to_inches(r) -> float:
    if pd.isna(r):
        return np.nan
    r = str(r).replace('"', "").replace(" ", "").strip()
    if r in ("--", "", "nan", "None"):
        return np.nan
    return pd.to_numeric(r, errors="coerce")


# ══════════════════════════════════════════════════════════════════════════════
# 4.  FIGHT STATS PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def build_fight_rows(df_fights_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot fights.csv from one-row-per-fight to two-rows-per-fight
    (one for each fighter), mirroring notebook cell 15.
    """
    fight_cols = [
        "event_id", "event_name", "event_url",
        "fight_order", "fight_url",
        "weight_class", "method",
        "round_end", "time_end", "scheduled_rounds",
    ]
    rows = []
    for _, r in df_fights_raw.iterrows():
        KD_A, KD_B = _safe_split(r.KD)
        STR_A, STR_B = _safe_split(r.STR)
        TD_A, TD_B = _safe_split(r.TD)
        SUB_A, SUB_B = _safe_split(r.SUB)

        base = {c: r[c] for c in fight_cols if c in r.index}

        rows.append({
            "fight_url": r.fight_url, "fighter": r.fighter_A, "opponent": r.fighter_B,
            "result": "win" if r.WL_label == "win" else ("draw" if "draw" in str(r.WL_label).lower() else "nc"),
            **base, "KD": KD_A, "STR": STR_A, "TD": TD_A, "SUB": SUB_A,
        })
        rows.append({
            "fight_url": r.fight_url, "fighter": r.fighter_B, "opponent": r.fighter_A,
            "result": "loss" if r.WL_label == "win" else ("draw" if "draw" in str(r.WL_label).lower() else "nc"),
            **base, "KD": KD_B, "STR": STR_B, "TD": TD_B, "SUB": SUB_B,
        })
    return pd.DataFrame(rows)


def parse_totals(df_totals_raw: pd.DataFrame) -> pd.DataFrame:
    """Parse all the 'X of Y' and 'mm:ss' columns in fight_totals.csv."""
    df = df_totals_raw.copy()

    df["KD_A"] = df["KD_A"].apply(_parse_int)
    df["KD_B"] = df["KD_B"].apply(_parse_int)

    for side in ("A", "B"):
        df[f"SigStr_{side}_L"], df[f"SigStr_{side}_Att"] = zip(*df[f"Sig Str_{side}"].apply(_parse_pair))
        df[f"SigStrPct_{side}"] = df[f"Sig Str %_{side}"].apply(_parse_float)
        df[f"TotStr_{side}_L"], df[f"TotStr_{side}_Att"] = zip(*df[f"Total Str_{side}"].apply(_parse_pair))
        df[f"Td_{side}_L"], df[f"Td_{side}_Att"] = zip(*df[f"Td_{side}"].apply(_parse_pair))
        df[f"TdPct_{side}"] = df[f"Td %_{side}"].apply(_parse_float)
        df[f"SubAtt_{side}"] = df[f"Sub Att_{side}"].apply(_parse_int)
        df[f"Rev_{side}"] = df[f"Rev_{side}"].apply(_parse_int)
        df[f"Ctrl_{side}"] = df[f"Ctrl_{side}"].apply(_parse_time)
        df[f"Head_{side}_L"], df[f"Head_{side}_Att"] = zip(*df[f"Head_{side}"].apply(_parse_pair))
        df[f"Body_{side}_L"], df[f"Body_{side}_Att"] = zip(*df[f"Body_{side}"].apply(_parse_pair))
        df[f"Leg_{side}_L"], df[f"Leg_{side}_Att"] = zip(*df[f"Leg_{side}"].apply(_parse_pair))
        df[f"Dist_{side}_L"], df[f"Dist_{side}_Att"] = zip(*df[f"Distance_{side}"].apply(_parse_pair))
        df[f"Clinch_{side}_L"], df[f"Clinch_{side}_Att"] = zip(*df[f"Clinch_{side}"].apply(_parse_pair))
        df[f"Ground_{side}_L"], df[f"Ground_{side}_Att"] = zip(*df[f"Ground_{side}"].apply(_parse_pair))

    return df


def merge_stats(df_fight_clean: pd.DataFrame, df_totals: pd.DataFrame) -> pd.DataFrame:
    """Merge parsed totals onto fighter rows, pick fighter vs opponent values."""
    df = df_fight_clean.merge(df_totals.copy(), on="fight_url", how="left")
    df["is_A"] = df["fighter"] == df["fighter_A"]

    def pick(col_a: str, col_b: str) -> pd.Series:
        return np.where(df["is_A"], df[col_a], df[col_b])

    df["KD_f"]     = pick("KD_A",        "KD_B")
    df["SigStr_f"] = pick("SigStr_A_L",  "SigStr_B_L")
    df["TD_f"]     = pick("Td_A_L",      "Td_B_L")
    df["Ctrl_f"]   = pick("Ctrl_A",      "Ctrl_B")
    df["Head_f"]   = pick("Head_A_L",    "Head_B_L")
    df["Body_f"]   = pick("Body_A_L",    "Body_B_L")
    df["Leg_f"]    = pick("Leg_A_L",     "Leg_B_L")

    df["KD_o"]     = pick("KD_B",        "KD_A")
    df["SigStr_o"] = pick("SigStr_B_L",  "SigStr_A_L")
    df["TD_o"]     = pick("Td_B_L",      "Td_A_L")
    df["Ctrl_o"]   = pick("Ctrl_B",      "Ctrl_A")
    df["Head_o"]   = pick("Head_B_L",    "Head_A_L")
    df["Body_o"]   = pick("Body_B_L",    "Body_A_L")
    df["Leg_o"]    = pick("Leg_B_L",     "Leg_A_L")

    for stat in ("KD", "SigStr", "TD", "Ctrl", "Head", "Body", "Leg"):
        df[f"{stat}_diff"] = df[f"{stat}_f"] - df[f"{stat}_o"]

    return df.sort_values(["event_id", "fight_order"]).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  PROFILE + PHYSICAL FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def merge_profiles(df: pd.DataFrame, df_profiles: pd.DataFrame) -> pd.DataFrame:
    """Merge fighter and opponent profiles (height, reach, SLpM, etc.)."""
    profile_cols = [
        "name", "height", "weight", "reach", "stance", "dob",
        "SLpM", "Str_Acc", "SApM", "Str_Def",
        "TD_Avg", "TD_Acc", "TD_Def", "Sub_Avg",
    ]
    prof = df_profiles[profile_cols].rename(columns={"name": "fighter"})
    opp_prof = df_profiles[profile_cols].rename(columns={
        "name": "opponent", "height": "opp_height", "weight": "opp_weight",
        "reach": "opp_reach", "stance": "opp_stance", "dob": "opp_dob",
        "SLpM": "opp_SLpM", "Str_Acc": "opp_Str_Acc", "SApM": "opp_SApM",
        "Str_Def": "opp_Str_Def", "TD_Avg": "opp_TD_Avg", "TD_Acc": "opp_TD_Acc",
        "TD_Def": "opp_TD_Def", "Sub_Avg": "opp_Sub_Avg",
    })
    df = df.merge(prof, on="fighter", how="left")
    df = df.merge(opp_prof, on="opponent", how="left")
    return df


def add_physical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convert height/reach to inches, compute differences."""
    df = df.copy()
    df["fighter_height_inches"] = df["height"].apply(_height_to_inches)
    df["opponent_height_inches"] = df["opp_height"].apply(_height_to_inches)
    df["fighter_reach_inches"] = df["reach"].apply(_reach_to_inches)
    df["opponent_reach_inches"] = df["opp_reach"].apply(_reach_to_inches)
    df["height_diff"] = df["fighter_height_inches"] - df["opponent_height_inches"]
    df["reach_diff"] = df["fighter_reach_inches"] - df["opponent_reach_inches"]
    return df


def add_age_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute fighter and opponent ages at fight date."""
    df = df.copy()
    df["dob"] = pd.to_datetime(df["dob"], errors="coerce")
    df["opp_dob"] = pd.to_datetime(df["opp_dob"], errors="coerce")
    df["fighter_age"] = (df["date"] - df["dob"]).dt.days / 365.25
    df["opponent_age"] = (df["date"] - df["opp_dob"]).dt.days / 365.25
    df["age_diff"] = df["fighter_age"] - df["opponent_age"]
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 6.  TEMPORAL FEATURES  (prefight-only, no leakage)
# ══════════════════════════════════════════════════════════════════════════════

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute prefight temporal features using groupby + shift.

    All values are computed from fights BEFORE the current fight — the shift
    ensures no leakage of the current fight's result into any feature.
    """
    df = df.copy()

    # Keep only win/loss rows for target
    df = df[df["result"].isin(["win", "loss"])].copy()
    df["WL_label"] = df["result"]
    df["target"] = (df["WL_label"] == "win").astype(int)

    # Drop rows missing Glicko or date
    df = df.dropna(subset=["g_rating_before", "opp_rating_before"])
    df = df.dropna(subset=["date"])
    df["date"] = pd.to_datetime(df["date"])

    # Remove pre-modern era
    df = df[df["date"] >= "2005-01-01"]

    # Remove fighters with only one fight in the dataset
    fight_counts = df.groupby("fighter")["fight_url"].transform("count")
    df = df[fight_counts > 1]

    df = df.sort_values(["fighter", "date"])

    df["fights_before"] = df.groupby("fighter").cumcount()
    df["days_since_last_fight"] = df.groupby("fighter")["date"].diff().dt.days

    df["wins_before"] = df.groupby("fighter")["target"].cumsum().shift(1)
    df["win_rate_before"] = np.where(
        df["fights_before"] > 0,
        df["wins_before"] / df["fights_before"],
        0.0,
    )

    df["recent_win_rate_3"] = (
        df.groupby("fighter")["target"].shift(1).rolling(3).mean()
    )
    df["recent_win_rate_5"] = (
        df.groupby("fighter")["target"].shift(1).rolling(5).mean()
    )

    for col in ["days_since_last_fight", "recent_win_rate_3", "recent_win_rate_5"]:
        df[col] = df[col].fillna(0)

    df["is_debut"] = (df["fights_before"] == 0).astype(int)

    # Merge opponent temporal features by fight_url
    temporal_cols = [
        "fights_before", "days_since_last_fight", "win_rate_before",
        "recent_win_rate_3", "recent_win_rate_5", "is_debut",
    ]
    opp_temp = (
        df[["fight_url", "fighter"] + temporal_cols]
        .rename(columns={"fighter": "opponent"})
        .rename(columns={c: f"opp_{c}" for c in temporal_cols})
    )
    df = df.merge(opp_temp, on=["fight_url", "opponent"], how="left")

    # Clean percentage columns
    pct_cols = [
        "Str_Acc", "Str_Def", "TD_Acc", "TD_Def",
        "opp_Str_Acc", "opp_Str_Def", "opp_TD_Acc", "opp_TD_Def",
    ]
    for col in pct_cols:
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                .str.replace("%", "", regex=False)
                .replace(["None", ""], np.nan)
                .astype(float) / 100.0
            )

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 7.  CLIPPING + LOG TRANSFORMS
# ══════════════════════════════════════════════════════════════════════════════

def apply_clip_and_log(df: pd.DataFrame, clip_bounds: dict) -> pd.DataFrame:
    """
    Apply existing clip bounds (from training) and log-transform temporal
    features.  Does NOT refit the clip bounds — those are fixed at training time.
    """
    df = df.copy()

    for col, bounds in clip_bounds.items():
        if col in df.columns:
            df[col] = df[col].clip(bounds["lo"], bounds["hi"])

    log_cols = [
        "days_since_last_fight", "opp_days_since_last_fight",
        "fights_before", "opp_fights_before",
    ]
    for col in log_cols:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col])

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 8.  OUTPUT VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

REQUIRED_COLS = [
    "fighter", "g_rating_before", "g_RD_before",
    "fighter_height_inches", "fighter_reach_inches", "fighter_age",
    "SLpM", "SApM", "Str_Acc", "Str_Def",
    "TD_Avg", "TD_Acc", "TD_Def", "Sub_Avg",
    "fights_before", "days_since_last_fight",
    "win_rate_before", "recent_win_rate_3", "recent_win_rate_5",
]

MIN_FIGHTERS = 500  # sanity floor — real data has 3000+


def validate_output(df: pd.DataFrame) -> list[str]:
    """
    Return a list of validation error strings.
    Empty list means the output looks sane.
    """
    errors: list[str] = []

    n = len(df)
    if n < MIN_FIGHTERS:
        errors.append(f"Only {n} fighters in output — expected at least {MIN_FIGHTERS}")

    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    if "g_rating_before" in df.columns:
        lo, hi = df["g_rating_before"].min(), df["g_rating_before"].max()
        if lo < 500 or hi > 3500:
            errors.append(f"Glicko ratings out of expected range: min={lo:.0f}, max={hi:.0f}")

    if "fighter" in df.columns:
        dupes = df["fighter"].duplicated().sum()
        if dupes > 0:
            errors.append(f"{dupes} duplicate fighter names in output")

    return errors


# ══════════════════════════════════════════════════════════════════════════════
# 9.  RAW LOADERS + SHARED FEATURE FRAME
# ══════════════════════════════════════════════════════════════════════════════

def load_raw_csvs(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Load the raw UFCStats CSV bundle required for feature engineering."""
    required_files = {
        "events.csv": "events",
        "fights.csv": "fights",
        "fight_totals.csv": "totals",
        "fighters_advanced.csv": "profiles",
        "fighters_fight_history.csv": "history",
    }
    missing = [f for f in required_files if not (data_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required raw CSV files in {data_dir}:\n"
            + "\n".join(f"  - {f}" for f in missing)
        )

    raw = {
        "events": pd.read_csv(data_dir / "events.csv"),
        "fights": pd.read_csv(data_dir / "fights.csv"),
        "totals": pd.read_csv(data_dir / "fight_totals.csv"),
        "profiles": pd.read_csv(data_dir / "fighters_advanced.csv"),
        "history": pd.read_csv(data_dir / "fighters_fight_history.csv"),
    }
    return raw


def build_feature_frame(data_dir: Path) -> pd.DataFrame:
    """
    Build the full fight-level prefight feature frame from raw UFCStats CSVs.

    Returns one row per fighter-fight with raw temporal columns intact.
    Clip bounds and log transforms are intentionally not applied here so the
    training pipeline can fit them on the training split only.
    """
    raw = load_raw_csvs(data_dir)
    df_fights = raw["fights"]
    df_totals = raw["totals"]
    df_profiles = raw["profiles"]
    df_history = raw["history"]

    print(f"  fights: {len(df_fights):,} rows")
    print(f"  fight_totals: {len(df_totals):,} rows")
    print(f"  profiles: {len(df_profiles):,} fighters")
    print(f"  history: {len(df_history):,} rows")

    # ── Step 1: Clean history ──────────────────────────────────────
    print("\n[2/7] Cleaning fight history...")
    df_hist_clean = clean_fight_history(df_history)
    print(f"  {len(df_hist_clean):,} clean history rows")

    # ── Step 2: Glicko ratings ─────────────────────────────────────
    print("\n[3/7] Computing Glicko-2 ratings...")
    df_glicko = compute_glicko_ratings(df_fights, df_totals, df_hist_clean)
    print(f"  {len(df_glicko):,} rating rows computed")

    # ── Step 3: Fight stats ────────────────────────────────────────
    print("\n[4/7] Parsing fight stats...")
    df_fight_clean = build_fight_rows(df_fights)
    df_totals_parsed = parse_totals(df_totals)
    df_merged = merge_stats(df_fight_clean, df_totals_parsed)

    # ── Step 4: Profiles + physical features ──────────────────────
    print("\n[5/7] Merging profiles and physical features...")
    df_merged = merge_profiles(df_merged, df_profiles)
    df_merged = add_physical_features(df_merged)

    # Attach dates from history
    df_hist_ufc = df_hist_clean[df_hist_clean["event"].str.contains("UFC", na=False)].copy()
    df_hist_ufc["date"] = pd.to_datetime(df_hist_ufc["date"])
    df_dates = (
        df_hist_ufc.groupby(["event_url", "fighter_name"])["date"]
        .first()
        .reset_index()
        .rename(columns={"fighter_name": "fighter"})
    )
    df_merged = df_merged.merge(df_dates, on=["fighter", "event_url"], how="left")
    df_merged = add_age_features(df_merged)

    # Merge Glicko ratings (fighter side then opponent side)
    df_glicko_clean = df_glicko.drop(columns=["opponent"], errors="ignore")
    df_merged = df_merged.merge(
        df_glicko_clean.rename(columns={"rating_before": "g_rating_before", "RD_before": "g_RD_before"}),
        on=["fight_url", "fighter"],
        how="left",
    )
    df_merged = df_merged.drop(
        columns=[c for c in ["opponent_y", "event_url_y", "date_y"] if c in df_merged.columns],
        errors="ignore",
    )
    df_merged = df_merged.rename(columns={
        c: c.rstrip("_x") for c in df_merged.columns if c.endswith("_x")
    })
    df_merged = df_merged.loc[:, ~df_merged.columns.duplicated()]

    df_merged = df_merged.merge(
        df_glicko_clean.rename(columns={
            "fighter": "opponent",
            "rating_before": "opp_rating_before",
            "RD_before": "opp_RD_before",
        }),
        on=["fight_url", "opponent"],
        how="left",
    )
    df_merged["rating_diff"] = df_merged["g_rating_before"] - df_merged["opp_rating_before"]
    df_merged["RD_diff"] = df_merged["g_RD_before"] - df_merged["opp_RD_before"]

    # ── Step 5: Temporal features ─────────────────────────────────
    print("\n[6/7] Computing temporal prefight features...")
    df_final = add_temporal_features(df_merged)
    print(f"  {len(df_final):,} fight rows after temporal feature pass")

    return df_final


# ══════════════════════════════════════════════════════════════════════════════
# 10.  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run(data_dir: Path, dry_run: bool = False) -> pd.DataFrame:
    """
    Execute the full refresh pipeline.

    Parameters
    ----------
    data_dir : Path
        Directory containing the raw CSV files from ufcstats scraping.
    dry_run : bool
        If True, run the entire pipeline but do NOT write any output files.

    Returns
    -------
    fighters_latest : pd.DataFrame
        One row per fighter with all prefight features populated.
    """
    print(f"\n{'='*60}")
    print(f"UFC Fighters Data Refresh")
    print(f"data_dir : {data_dir}")
    print(f"dry_run  : {dry_run}")
    print(f"{'='*60}\n")

    # ── Load raw CSVs ─────────────────────────────────────────────
    print("[1/7] Loading raw CSVs...")
    df_final = build_feature_frame(data_dir)

    # ── Step 6: Clip + log ────────────────────────────────────────
    if CLIP_BOUNDS_PATH.exists():
        with open(CLIP_BOUNDS_PATH) as f:
            clip_bounds = json.load(f)
        df_final = apply_clip_and_log(df_final, clip_bounds)
        print(f"  Applied clip bounds from {CLIP_BOUNDS_PATH}")
    else:
        print(f"  WARNING: {CLIP_BOUNDS_PATH} not found — skipping clip+log transform")

    # ── Step 7: Build snapshot (last fight per fighter) ───────────
    print("\n[7/7] Building fighter snapshot...")
    fighters_latest = (
        df_final.sort_values("date")
        .groupby("fighter", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    print(f"  {len(fighters_latest):,} unique fighters in snapshot")

    # ── Validate ──────────────────────────────────────────────────
    errors = validate_output(fighters_latest)
    if errors:
        print("\n⚠️  VALIDATION WARNINGS:")
        for e in errors:
            print(f"  - {e}")
    else:
        print("  Validation passed ✓")

    # ── Write output ──────────────────────────────────────────────
    if dry_run:
        print(f"\n[dry-run] Would write {len(fighters_latest):,} rows → {OUTPUT_PATH}")
    else:
        MODELS_DIR.mkdir(exist_ok=True)
        fighters_latest.to_csv(OUTPUT_PATH, index=False)
        print(f"\n✅ Saved → {OUTPUT_PATH}  ({len(fighters_latest):,} fighters)")

    return fighters_latest


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rebuild models/fighters_latest.csv from raw ufcstats CSV files."
    )
    p.add_argument(
        "--data-dir",
        required=True,
        type=Path,
        help="Directory containing events.csv, fights.csv, fight_totals.csv, "
             "fighters_advanced.csv, fighters_fight_history.csv",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the full pipeline but do not write any output files.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    try:
        run(data_dir=args.data_dir, dry_run=args.dry_run)
    except FileNotFoundError as exc:
        print(f"\n❌ {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"\n❌ Pipeline failed: {exc}", file=sys.stderr)
        raise
