import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
from xgboost import XGBClassifier

# ===============================================================
# STREAMLIT CONFIG
# ===============================================================
st.set_page_config(page_title="UFC Fight Predictor", layout="wide")

# ===============================================================
# CONSTANTS
# ===============================================================
LOG_COLS = [
    "days_since_last_fight",
    "opp_days_since_last_fight",
    "fights_before",
    "opp_fights_before",
]

# ===============================================================
# MODEL + ASSET LOADER
# ===============================================================
@st.cache_resource
def load_assets():
    model = XGBClassifier()
    model.load_model("models/xgb_prefight_model.json")

    calibrator = joblib.load("models/xgb_calibrator.pkl")

    with open("models/xgb_feature_cols.json") as f:
        feature_cols = json.load(f)

    with open("models/clip_bounds.json") as f:
        clip_bounds = json.load(f)

    fighters_df = pd.read_csv("models/fighters_latest.csv")

    return model, calibrator, feature_cols, clip_bounds, fighters_df


# ===============================================================
# PREPROCESSING (MATCHES TRAINING)
# ===============================================================
def preprocess_row(row_df, feature_cols, clip_bounds):
    df = row_df.copy()

    # Fill NA (same as training)
    df = df.fillna(0)

    # Quantile clipping (TRAIN-FIT bounds)
    for col, bounds in clip_bounds.items():
        if col in df.columns:
            df[col] = df[col].clip(bounds["lo"], bounds["hi"])

    # Log transforms
    for col in LOG_COLS:
        df[f"log_{col}"] = np.log1p(df[col])

    # Final feature ordering
    X = df[feature_cols].values
    return X


# ===============================================================
# INFERENCE (CALIBRATED)
# ===============================================================
def predict_win_prob(model, calibrator, X):
    raw_prob = model.predict_proba(X)[0, 1]
    cal_prob = calibrator.transform([raw_prob])[0]
    return float(np.clip(cal_prob, 0.01, 0.99))


# ===============================================================
# LOAD ASSETS
# ===============================================================
model, calibrator, feature_cols, clip_bounds, fighters_df = load_assets()

# ===============================================================
# UI HEADER
# ===============================================================
st.title("🥊 UFC Main Card Predictor")
st.caption("XGBoost · Glicko + Physical + Temporal · Calibrated Probabilities")

st.subheader("Main Card Matchups")
num_fights = st.slider("Number of fights", 1, 8, 5)

fighters = sorted(fighters_df["fighter"].unique())
predictions = []

# ===============================================================
# FIGHT LOOP
# ===============================================================
for i in range(num_fights):
    st.markdown(f"### Fight {i + 1}")
    col1, col2 = st.columns(2)

    with col1:
        fighter_a = st.selectbox(
            f"Fighter A ({i + 1})",
            fighters,
            key=f"a_{i}"
        )

    with col2:
        fighter_b = st.selectbox(
            f"Fighter B ({i + 1})",
            fighters,
            key=f"b_{i}"
        )

    if fighter_a == fighter_b:
        st.warning("Choose two different fighters")
        continue

    A = fighters_df[fighters_df["fighter"] == fighter_a].iloc[0]
    B = fighters_df[fighters_df["fighter"] == fighter_b].iloc[0]

    # ===============================================================
    # PREFIGHT FEATURE ROW (MODEL-ALIGNED)
    # ===============================================================
    row = {
        "rating_diff": A["g_rating_before"] - B["g_rating_before"],
        "RD_diff": A["g_RD_before"] - B["g_RD_before"],

        "height_diff": A["fighter_height_inches"] - B["fighter_height_inches"],
        "reach_diff": A["fighter_reach_inches"] - B["fighter_reach_inches"],
        "age_diff": A["fighter_age"] - B["fighter_age"],

        "SLpM": A["SLpM"],
        "SApM": A["SApM"],
        "Str_Acc": A["Str_Acc"],
        "Str_Def": A["Str_Def"],
        "TD_Avg": A["TD_Avg"],
        "TD_Acc": A["TD_Acc"],
        "TD_Def": A["TD_Def"],
        "Sub_Avg": A["Sub_Avg"],

        "opp_SLpM": B["SLpM"],
        "opp_SApM": B["SApM"],
        "opp_Str_Acc": B["Str_Acc"],
        "opp_Str_Def": B["Str_Def"],
        "opp_TD_Avg": B["TD_Avg"],
        "opp_TD_Acc": B["TD_Acc"],
        "opp_TD_Def": B["TD_Def"],
        "opp_Sub_Avg": B["Sub_Avg"],

        "fights_before": A["fights_before"],
        "days_since_last_fight": A["days_since_last_fight"],
        "win_rate_before": A["win_rate_before"],
        "recent_win_rate_3": A["recent_win_rate_3"],
        "recent_win_rate_5": A["recent_win_rate_5"],

        "opp_fights_before": B["fights_before"],
        "opp_days_since_last_fight": B["days_since_last_fight"],
        "opp_win_rate_before": B["win_rate_before"],
        "opp_recent_win_rate_3": B["recent_win_rate_3"],
        "opp_recent_win_rate_5": B["recent_win_rate_5"],
    }

    X = preprocess_row(pd.DataFrame([row]), feature_cols, clip_bounds)
    prob = predict_win_prob(model, calibrator, X)

    st.progress(int(prob * 100))
    st.metric(f"{fighter_a} win probability", f"{prob:.1%}")

    predictions.append((fighter_a, fighter_b, prob))


# ===============================================================
# SUMMARY
# ===============================================================
if predictions:
    st.subheader("📊 Main Card Summary")
    summary_df = pd.DataFrame(
        predictions,
        columns=["Fighter A", "Fighter B", "Win Prob (A)"]
    )
    st.dataframe(summary_df, use_container_width=True)
