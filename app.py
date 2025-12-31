import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import re
import unicodedata
import xgboost as xgb
from pathlib import Path

# ===============================================================
# STREAMLIT CONFIG
# ===============================================================
st.set_page_config(page_title="UFC Event Predictor", layout="wide")

# ===============================================================
# PAGE NAVIGATION
# ===============================================================
page = st.sidebar.radio(
    "📘 Navigation",
    ["Event Predictor", "Betting Guide"],
)

# ===============================================================
# CONSTANTS
# ===============================================================
LOG_COLS = [
    "days_since_last_fight",
    "opp_days_since_last_fight",
    "fights_before",
    "opp_fights_before",
]

MISSING_COLS = [
    "SLpM", "SApM", "TD_Avg", "Sub_Avg", "Str_Acc", "TD_Acc",
    "opp_SLpM", "opp_SApM", "opp_TD_Avg", "opp_Sub_Avg",
    "opp_Str_Acc", "opp_TD_Acc",
]

# ===============================================================
# NAME NORMALIZATION (FIXED)
# ===============================================================
def normalize_name(name: str) -> str:
    name = unicodedata.normalize("NFKD", name)
    name = name.encode("ascii", "ignore").decode("ascii")
    name = name.lower()
    name = re.sub(r"[^a-z\s]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def name_aliases(name: str) -> set[str]:
    base = normalize_name(name)
    parts = base.split()

    aliases = {base}

    if len(parts) >= 2:
        first = parts[0]
        last = parts[-1]
        middle = parts[1:-1]

        # Basic swaps
        aliases.add(f"{last} {first}")
        aliases.add(f"{first} {last}")

        # Handle compound surnames (Cortes Acosta)
        if middle:
            compound_spaced = " ".join(middle + [last])      # cortes acosta
            compound_nospace = "".join(middle + [last])      # cortesacosta

            aliases.update({
                f"{first} {compound_spaced}",
                f"{first} {compound_nospace}",
                compound_spaced,
                compound_nospace,
                f"{compound_spaced} {first}",
                f"{compound_nospace} {first}",
            })

        # Full reverse
        aliases.add(" ".join(reversed(parts)))

    return {a.strip() for a in aliases if a.strip()}


# ===============================================================
# ODDS / BETTING UTILS
# ===============================================================
def american_to_decimal(odds: int) -> float:
    if odds == 0:
        return np.nan
    if odds > 0:
        return 1.0 + odds / 100.0
    return 1.0 + 100.0 / abs(odds)

def implied_prob_from_decimal(decimal_odds: float) -> float:
    if decimal_odds is None or np.isnan(decimal_odds) or decimal_odds <= 1.0:
        return np.nan
    return 1.0 / decimal_odds

def ev_per_dollar(p: float, d: float) -> float:
    if d is None or np.isnan(d) or d <= 1.0:
        return np.nan
    return float(p * d - 1.0)

def kelly_fraction(p: float, d: float) -> float:
    if d is None or np.isnan(d) or d <= 1.0:
        return 0.0
    b = d - 1.0
    return float(max(0.0, (b * p - (1 - p)) / b))

def kelly_note(fk: float) -> str:
    if fk >= 0.05:
        return "Large sizing signal"
    if fk >= 0.02:
        return "Moderate sizing signal"
    if fk > 0:
        return "Small sizing signal"
    return "No sizing signal"

def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default

# ===============================================================
# LOAD MODEL + ASSETS
# ===============================================================
@st.cache_resource
def load_assets():
    booster = xgb.Booster()
    booster.load_model("models/xgb_prefight_model.json")

    calibrator = joblib.load("models/xgb_calibrator.pkl")

    with open("models/xgb_feature_cols.json") as f:
        feature_cols = json.load(f)

    with open("models/clip_bounds.json") as f:
        clip_bounds = json.load(f)

    fighters_df = pd.read_csv("models/fighters_latest.csv")
    fighters_df["fighter_norm"] = fighters_df["fighter"].apply(normalize_name)

    fighter_lookup = {
        row["fighter_norm"]: row["fighter"]
        for _, row in fighters_df.iterrows()
    }

    return booster, calibrator, feature_cols, clip_bounds, fighters_df, fighter_lookup
    
def betting_guide():
    st.title("📘 Betting Logic & Decision Guide")
    st.caption("How the app thinks, why bets are filtered, and what every label actually means")

    st.divider()

    # ==========================================================
    # WHAT THIS APP IS REALLY DOING
    # ==========================================================
    st.subheader("🧠 What the App Is Optimizing")

    st.markdown("""
Its nice to predict the winner of a fight but is the betting market mispricing this fight enough to justify risking capital?

As a result:
- You can have a strong pick and still no bet
- You can have high confidence and negative EV
- Most fights should be filtered out

This behavior is intentional.
""")

    st.divider()

    # ==========================================================
    # CORE TERMS USED EVERYWHERE
    # ==========================================================
    st.subheader("📚 Core Terms")

    st.markdown("""
Model Probability  
The model’s estimated probability that the fighter wins.

Market Probability  
The implied probability from betting odds.

Edge  
Difference between model probability and market probability.

Expected Value (EV)  
Expected profit per $1 bet in the long run.

Kelly (Full)  
Optimal fraction of bankroll to wager if probabilities are correct.

Kelly Multiplier  
Risk control applied to Kelly sizing.
""")

    st.divider()

    # ==========================================================
    # BETTING MATH
    # ==========================================================
    st.subheader("📐 Betting Math (Exact Formulas Used)")

    st.markdown(r"""
Implied Market Probability

$$
P_{market} = \frac{1}{d}
$$

Expected Value per $1

$$
EV = p \cdot d - 1
$$

Edge

$$
Edge = p_{model} - P_{market}
$$

Kelly Criterion (full Kelly)

Let

$$
b = d - 1
$$

Then

$$
Kelly = \max\left(0,\; \frac{b p - (1 - p)}{b} \right)
$$

If Kelly ≤ 0, the optimal bet size is zero.
""")

    st.divider()
    
    st.subheader("🔢 How Market Probability & Best Odds Are Calculated")

    st.markdown("""
This app aggregates **many sportsbook prices** and uses them in two
different ways — intentionally.

---

### 📈 Raw Odds (Example)

Waldo Cortes-Acosta moneyline odds from fightodds.io:

- -335  
- -330  
- -310  
- -300  
- -290  

### 🔁 Convert to Decimal Odds

Negative American odds are converted as:

Decimal = 1 + 100 / |odds|

Result:

[1.30, 1.30, 1.32, 1.33, 1.34]


---

### 🏆 Best Odds (Used for EV & Kelly)

Best Odds = max(decimal_odds) = 1.34

This answers:
**“What is the best price I can personally bet?”**

Used for:
- Expected Value (EV)
- Kelly sizing

---

### 📊 Market Probability (Used for EDGE)

Market probability is computed from the **median** price:

Median decimal = 1.32
Market Prob = 1 / 1.32 ≈ 75.8%


Why median?
- Filters out outlier books
- Represents market consensus
- More robust than best odds

Used only for:
- EDGE calculation

---

### 🧠 Why EV and EDGE Can Disagree

Example:

- Model Prob = **75.1%**
- Market Prob = **75.8%**
- Best Odds = **1.34**

EDGE = -0.7%
EV ≈ +0.006
Kelly = 0

Result:
🚫 **No Edge**

High confidence ≠ profitable bet.

---

### ⚠️ Key Rule

- **Best Odds** → what you can get  
- **Market Probability** → what the market believes  

""")
    
    st.divider()
    
    # ==========================================================
    # SCENARIO 1 — CLEAR +EV BET
    # ==========================================================
    st.subheader("✅ Scenario 1: Clear +EV Bet")

    st.markdown("""
Model Prob: 65%  
Best Odds: 2.00  

Market Prob = 50%  
Edge = +15%  
EV = +0.30  
Kelly (full) = 30%

Interpretation:
- Large disagreement between model and market
- Positive EV is meaningful
- Optimal bet size is non-trivial

App label:
+EV Bet
""")

    st.divider()

    # ==========================================================
    # SCENARIO 2 — STRONG PICK BUT NO BET
    # ==========================================================
    st.subheader("🚫 Scenario 2: Strong Pick, Bad Bet")

    st.markdown("""
Model Prob: 75%  
Best Odds: 1.30  

Market Prob = 76.9%  
Edge = -1.9%  
EV = -0.025  
Kelly = 0%

Interpretation:
- Market already agrees or is sharper
- Odds are too short
- Confidence does not imply profitability

App label:
No Edge
""")

    st.divider()

    # ==========================================================
    # SCENARIO 3 — SLIGHT EDGE
    # ==========================================================
    st.subheader("⚠️ Scenario 3: Slight Edge")

    st.markdown("""
Model Prob: 56%  
Best Odds: 1.80  

Market Prob = 55.6%  
Edge = +0.4%  
EV = +0.008  
Kelly (full) ≈ 0.8%

Interpretation:
- Positive EV exists
- Edge is extremely small
- Kelly stake is tiny
- Variance dominates outcomes

App label:
Slight Edge
""")

    st.divider()

    # ==========================================================
    # SCENARIO 4 — SLIGHT EDGE VS AGGRESSIVE KELLY
    # ==========================================================
    st.subheader("⚠️ Why You See Slight Edge but Aggressive Kelly")

    st.markdown("""
This confuses many users.

Edge and Kelly measure different things.

Edge measures disagreement between model and market.

Kelly measures optimal bet size given:
- probability
- payout size

You can have:
- small Edge
- large odds payout
- high confidence

Which produces a large Kelly value.

Example:

Model Prob = 70%  
Odds = 2.40  

Edge may be small if market agrees, but payoff is large.

Result:
- Slight Edge label
- Kelly (full) = 10%+
- App says Aggressive Edge

This is mathematically correct.
""")

    st.divider()

    # ==========================================================
    # SCENARIO 5 — EV > 0 BUT KELLY = 0
    # ==========================================================
    st.subheader("🧮 Scenario 5: EV Positive but Kelly = 0")

    st.markdown("""
This happens at short odds.

Even with positive EV:
- payout is too small
- downside risk dominates

Kelly math returns zero.

Interpretation:
- Profitable in theory
- Not worth risking bankroll

App behavior:
Stake = $0
""")

    st.divider()

    # ==========================================================
    # SCENARIO 6 — ODDS BUT EV = N/A
    # ==========================================================
    st.subheader("❓ Scenario 6: Odds Shown but EV = N/A")

    st.markdown("""
Reasons:
- too few sportsbooks
- stale or inconsistent prices
- unreliable market median

App behavior:
- EV suppressed
- Edge suppressed
- Bet excluded from simulation

This is a safety feature.
""")

    st.divider()

    # ==========================================================
    # SCENARIO 7 — NO ODDS
    # ==========================================================
    st.subheader("📭 Scenario 7: Prediction Without Betting")

    st.markdown("""
Model predictions can exist without odds.

Betting logic cannot.

Reasons:
- market not open
- late fight addition
- name scrape mismatch

Prediction is shown.
Bet recommendation is not.
""")

    st.divider()

    # ==========================================================
    # WHY MOST FIGHTS ARE NO EDGE
    # ==========================================================
    st.subheader("📉 Why Most Fights Are No Edge")

    st.markdown("""
This is normal.

Markets are:
- competitive
- fast
- efficient for popular fights

If many bets qualify, something is wrong.
""")

    st.divider()

    # ==========================================================
    # BETTING SETTINGS — HOW TO THINK ABOUT THE SLIDERS
    # ==========================================================
    st.subheader("🎛 Betting Settings — How Decisions Are Actually Made")

    st.markdown(r"""
This section explains **how the sliders interact with the math**, using
real examples from the app output.

These settings are **hard filters**, not suggestions.
If a bet fails *any* filter, it is treated as **non-actionable**.

---

## 💰 Starting Bankroll

This is your **total risk capital**.

All stakes are computed as:

$$
\text{Stake} = \text{Bankroll} \times f_{\text{Kelly}} \times \text{Kelly Multiplier}
$$

Changing the bankroll **scales bet size**, but does **not** change:
- EV
- Edge
- Kelly fraction
- Whether a bet qualifies

---

## 🧠 Kelly Multiplier (Risk Dial)

Kelly gives the *theoretically optimal* fraction of bankroll:

$$
f_{\text{Kelly}} = \frac{(d-1)p - (1-p)}{d-1}
$$

However, **full Kelly is extremely aggressive**.

That’s why the app uses:

- 0.25 → quarter Kelly (very conservative)
- 0.50 → half Kelly (balanced)
- 1.00 → full Kelly (high drawdown risk)

### Interpretation
- High Kelly % ≠ bet all-in
- High Kelly % = **strong disagreement with the market**
- The multiplier controls **survivability**, not edge quality

---

## 📐 Min EDGE (Model vs Market Disagreement)

Defined as:

$$
\text{EDGE} = p_{\text{model}} - p_{\text{market}}
$$

This answers:
> *How much does my model disagree with the betting market?*

### Why this matters
- Small edges are dominated by noise
- Markets are often sharper than models
- Filtering edge protects against false positives

### Typical values
- 0.01 → too permissive
- 0.02 → reasonable default
- 0.05+ → very selective

---

## 📈 Min EV (Long-Run Profit Filter)

Expected value per dollar:

$$
\text{EV} = p \cdot d - 1
$$

This answers:
> *If I placed this bet infinitely many times, would I make money?*

### Interpretation
- EV > 0 → profitable in expectation
- EV ≈ 0 → noise
- EV < 0 → losing bet

EV alone is **not sufficient** — it must agree with EDGE and Kelly.

---

## 🧮 How the App Classifies Bets

### 💰 +EV Bet
All conditions satisfied:
- EV ≥ Min EV
- EDGE ≥ Min EDGE
- Kelly > 0

These are included in:
- stake recommendations
- bankroll simulation

---

### ⚠️ Slight Edge
Conditions:
- EV > 0
- Kelly > 0
- **but fails Min EV or Min EDGE**

Example:

Model Prob = 64.5%  
Market Prob = 60.0%  
Best Odds = 1.66  

$$
\text{EV} = 0.645 \times 1.66 - 1 = +0.009
$$

Kelly (full) ≈ 1.6%

Stake ≈ \$8 on \$1000 bankroll

### Why this is NOT simulated
- Stake is economically negligible
- Variance overwhelms signal
- Repeated tiny bets inflate drawdown risk

👉 **Displayed for transparency, excluded for discipline**

---

### 🚫 No Edge
Occurs when:
- EV ≤ 0, or
- Kelly = 0, or
- Market already prices outcome efficiently

High confidence **does not matter** if the price is bad.

---

## 🔥 Why Kelly Can Look "Aggressive"

Example:

Model Prob = 79.6%  
Market Prob = 66.7%  
Best Odds = 1.68  

$$
\text{EV} = 0.796 \times 1.68 - 1 = +0.333
$$

$$
f_{\text{Kelly}} \approx 49.3\%
$$

This does **not** mean:
- bet half your bankroll blindly

It means:
- the model sees **massive mispricing**
- sizing signal is strong
- you still control risk via the multiplier

---

## 🧠 Mental Checklist (Use This)

Before trusting a bet, ask:

1. Is EV meaningfully positive?
2. Is EDGE large enough to beat noise?
3. Is Kelly non-trivial?
4. Does stake size justify variance?

If any answer is **no** → the app blocks it.

---

### Final Rule of Thumb

> **Good models reject far more bets than they place.**  
> If everything looks bettable, something is wrong.
""")

    st.divider()

    st.subheader("📊 Odds Data Source")

    st.markdown("""
    All betting odds shown in this app are sourced from **fightodds.io**.

    fightodds.io aggregates live moneyline prices across many major sportsbooks, including (but not limited to):

    - BetOnline
    - Bovada
    - Cloudbet
    - Pinnacle
    - DraftKings
    - FanDuel
    - BetMGM
    - Caesars
    - Circa
    - Betway
    - Polymarket
    - Stake
    - Bookmaker
    - BetRivers
    - Hard Rock Bet

    The app does **not** scrape sportsbooks directly.
    It relies on fightodds.io as a public odds aggregation layer.

    For each fighter:
    - All available sportsbook prices are collected
    - Prices are deduplicated
    - The **best available odds** are used for EV and Kelly sizing
    - The **median market price** is used to estimate market-implied probability
    """)
    st.divider()
    # ==========================================================
    # FINAL WARNING
    # ==========================================================
    st.warning("""
This app optimizes decision quality.

Variance is unavoidable.
Kelly assumes your probabilities are accurate.
Overbetting destroys bankrolls faster than bad picks.
""")


# ===============================================================
# ROUTING
# ===============================================================
if page == "Betting Guide":
    betting_guide()
    st.stop()


# ===============================================================
# PREPROCESSING (EXACT TRAINING PARITY)
# ===============================================================
def preprocess_row(row_df, feature_cols, clip_bounds):
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

# ===============================================================
# INFERENCE
# ===============================================================
def predict_win_prob(booster, calibrator, X):
    dmat = xgb.DMatrix(X)
    raw = booster.predict(dmat)[0]   # probability
    cal = calibrator.transform([raw])[0]
    return float(np.clip(cal, 0.01, 0.99))

# ===============================================================
# BANKROLL SIMULATION
# ===============================================================
def simulate_bankroll_paths(bets, initial_bankroll, n_sims, kelly_mult):
    """
    bets: list of dicts with keys:
      - fight
      - pick (string)
      - p (float)
      - decimal_odds (float)
      - stake_frac_full_kelly (float)
    Stake used = bankroll * stake_frac_full_kelly * kelly_mult
    """
    if not bets:
        return np.array([])

    final_bankrolls = np.zeros(n_sims, dtype=float)

    for s in range(n_sims):
        bankroll = float(initial_bankroll)
        for b in bets:
            p = float(b["p"])
            d = float(b["decimal_odds"])
            fk = float(b["stake_frac_full_kelly"])
            stake = bankroll * fk * float(kelly_mult)

            # skip tiny stakes
            if stake <= 1e-9:
                continue

            if np.random.rand() < p:
                bankroll += stake * (d - 1.0)
            else:
                bankroll -= stake

            if bankroll <= 0:
                bankroll = 0.0
                break

        final_bankrolls[s] = bankroll

    return final_bankrolls

# ===============================================================
# LOAD EVERYTHING
# ===============================================================
model, calibrator, feature_cols, clip_bounds, fighters_df, fighter_lookup = load_assets()

# ===============================================================
# SIDEBAR CONTROLS
# ===============================================================
st.sidebar.header("💰 Betting Settings")

bankroll = st.sidebar.number_input(
    "Starting Bankroll ($)",
    min_value=50, max_value=500_000, value=1000, step=50
)
kelly_mult = st.sidebar.slider(
    "Kelly Multiplier (0.25 = quarter Kelly)",
    min_value=0.05, max_value=1.0, value=0.5, step=0.05
)
min_edge = st.sidebar.slider(
    "Min EDGE to consider a bet",
    min_value=-0.10, max_value=0.30, value=0.02, step=0.01,
    help="EDGE = ModelProb - MarketImpliedProb. Higher means more disagreement vs market."
)
min_ev = st.sidebar.slider(
    "Min EV to consider a bet",
    min_value=-0.20, max_value=0.50, value=0.01, step=0.01,
    help="EV is expected profit per $1 bet. Positive EV suggests long-run advantage."
)
n_sims = st.sidebar.number_input(
    "Bankroll Simulations",
    min_value=500, max_value=50_000, value=5000, step=500
)
show_raw_odds = st.sidebar.checkbox(
    "📉 Show raw sportsbook odds",
    value=False,
    help="Reveal individual sportsbook odds used to compute market median and best price"
)


# ===============================================================
# UI
# ===============================================================
st.title("🥊 UFC Event Predictor")

event_number = st.text_input("Enter UFC Event Number (e.g., 324)")

if not event_number.isdigit():
    st.warning("Please enter a valid UFC event number (e.g. 324)")
    st.stop()

event_number = int(event_number)

with st.spinner(f"Fetching UFC {event_number} fight card..."):
    try:
        fights = json.load(open(f"cards/ufc_{event_number}.json"))
    except Exception as e:
        st.error(f"Failed to fetch UFC event card. ({e})")
        st.stop()

if not fights:
    st.warning("No fights found — event may not be published yet.")
    st.stop()

@st.cache_data(show_spinner=False)
def load_odds(event_number: int):
    path = Path("odds") / f"ufc_{event_number}.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f).get("odds", {})

with st.spinner("Fetching betting odds from fightodds.io..."):
    try:
        odds_map = load_odds(event_number)
    except Exception as e:
        st.warning(f"Odds scrape failed — continuing without odds. ({e})")
        odds_map = {}

# ===============================================================
# HELPERS (odds lookup + formatting)
# ===============================================================
def get_odds_for_name(name: str, odds_map: dict) -> list[int]:
    """
    Robust odds resolver:
    - checks ALL aliases
    - unions odds across matches (instead of taking first hit)
    - returns sorted unique american odds
    """
    found = set()
    for k in name_aliases(name):
        if k in odds_map and odds_map[k]:
            # odds_map[k] might be list or set depending on your scrape version
            try:
                found.update(list(odds_map[k]))
            except TypeError:
                # fallback (shouldn't happen, but safe)
                for v in odds_map[k]:
                    found.add(v)
    # keep only sane ints
    cleaned = [int(o) for o in found if isinstance(o, (int, np.integer)) and -5000 < int(o) < 5000 and int(o) != 0]
    return sorted(set(cleaned))

def format_american(odds: int) -> str:
    return f"+{odds}" if odds > 0 else str(odds)


# ===============================================================
# RUN PREDICTIONS + ODDS + EV + KELLY
# ===============================================================
results = []
bets_for_sim = []

st.subheader(f"🥊 UFC {event_number} Predictions")

for fighter_a, fighter_b in fights:
    a_norm = normalize_name(fighter_a)
    b_norm = normalize_name(fighter_b)

    # DATA AVAILABILITY CHECK (fighters snapshot)
    if a_norm not in fighter_lookup:
        results.append({
            "Fight": f"{fighter_a} vs {fighter_b}",
            "Winner": "Unknown",
            "Prob_A": None,
            "Status": "⚠️ No data on Fighter A (recent debut / name mismatch) — RISKY BET",
        })
        continue

    if b_norm not in fighter_lookup:
        results.append({
            "Fight": f"{fighter_a} vs {fighter_b}",
            "Winner": "Unknown",
            "Prob_A": None,
            "Status": "⚠️ No data on Fighter B (recent debut / name mismatch) — RISKY BET",
        })
        continue

    # Pull fighter rows
    A = fighters_df[fighters_df["fighter"] == fighter_lookup[a_norm]].iloc[0]
    B = fighters_df[fighters_df["fighter"] == fighter_lookup[b_norm]].iloc[0]

    # Build feature row (same as your training pipeline)
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

        "is_debut": int(A["fights_before"] == 0),
        "opp_is_debut": int(B["fights_before"] == 0),
    }

    X = preprocess_row(pd.DataFrame([row]), feature_cols, clip_bounds)
    prob_a = predict_win_prob(model, calibrator, X)

    # Pick + confidence
    winner = fighter_a if prob_a >= 0.5 else fighter_b
    confidence = abs(prob_a - 0.5)
    status = "🔥 Strong pick" if confidence >= 0.15 else "⚖️ Close fight"

    # Odds lists (ROBUST, UNION across aliases)
    odds_list_a = get_odds_for_name(fighter_a, odds_map)
    odds_list_b = get_odds_for_name(fighter_b, odds_map)

    dec_a = [american_to_decimal(o) for o in odds_list_a] if odds_list_a else []
    dec_b = [american_to_decimal(o) for o in odds_list_b] if odds_list_b else []

    # Market median prob for A (used for EDGE(A))
    market_prob_a = np.nan
    if dec_a:
        median_dec_a = float(np.median(dec_a))
        market_prob_a = float(implied_prob_from_decimal(median_dec_a))

    pick_side = "A" if winner == fighter_a else "B"
    p_pick = float(prob_a if pick_side == "A" else (1.0 - prob_a))

    # Best odds for the PICK side
    best_odds_dec = np.nan
    odds_count_pick = 0
    if pick_side == "A" and dec_a:
        best_odds_dec = float(np.max(dec_a))
        odds_count_pick = len(dec_a)
    elif pick_side == "B" and dec_b:
        best_odds_dec = float(np.max(dec_b))
        odds_count_pick = len(dec_b)

    # EV for pick side
    ev = ev_per_dollar(p_pick, best_odds_dec)

    # EDGE(A) and Market Prob (Pick)
    edge_a = np.nan
    market_prob_pick = np.nan
    if not np.isnan(market_prob_a):
        edge_a = float(prob_a - market_prob_a)
        market_prob_pick = float(market_prob_a if pick_side == "A" else (1.0 - market_prob_a))

    # Kelly % (full kelly) for pick side based on best odds
    fk = 0.0
    stake_dollars = 0.0
    if not np.isnan(best_odds_dec):
        fk = kelly_fraction(p_pick, best_odds_dec)
        stake_dollars = bankroll * fk * kelly_mult

    include_for_sim = (
        (not np.isnan(ev)) and (not np.isnan(edge_a)) and
        (ev >= float(min_ev)) and (edge_a >= float(min_edge)) and
        (fk > 0)
    )

    if include_for_sim:
        bets_for_sim.append({
            "fight": f"{fighter_a} vs {fighter_b}",
            "pick": winner,
            "p": p_pick,
            "decimal_odds": float(best_odds_dec),
            "stake_frac_full_kelly": float(fk),
        })

    results.append({
        "Fight": f"{fighter_a} vs {fighter_b}",
        "Fighter A Name": fighter_a,
        "Fighter B Name": fighter_b,
        "Winner": winner,
        "Prob_A": prob_a,
        "Pick Side": pick_side,
        "Status": status,

        "Model Prob (Pick)": p_pick,
        "Market Prob (Pick)": safe_float(market_prob_pick),
        "Market Median Prob (A)": safe_float(market_prob_a),
        "EDGE (A)": safe_float(edge_a),

        "Best Odds (Dec)": safe_float(best_odds_dec),
        "EV (per $1)": safe_float(ev),

        "Kelly % (full)": fk,
        "Kelly Note": kelly_note(fk),
        "Stake $": stake_dollars,
        "Odds Count (Pick)": odds_count_pick,
        "Odds List A": odds_list_a,  # already sorted unique
        "Odds List B": odds_list_b,  # already sorted unique
    })


# ===============================================================
# DISPLAY
# ===============================================================
st.subheader("📊 Fight-by-Fight Breakdown")

for r in results:
    st.markdown(f"### {r['Fight']}")

    if r["Prob_A"] is None:
        st.warning(r["Status"])
        st.divider()
        continue

    p_pick = r["Model Prob (Pick)"]
    mp_pick = r["Market Prob (Pick)"]
    edge_a = r["EDGE (A)"]
    ev = r["EV (per $1)"]
    fk = r["Kelly % (full)"]
    stake = r["Stake $"]
    best_odds = r["Best Odds (Dec)"]
    odds_ct = r["Odds Count (Pick)"]

    # recommendation badge
    if (not np.isnan(ev)) and (not np.isnan(edge_a)) and fk > 0 and ev >= float(min_ev) and edge_a >= float(min_edge):
        rec = ("💰 +EV Bet", "success")
    elif (not np.isnan(ev)) and ev > 0 and fk > 0:
        rec = ("⚠️ Slight Edge", "warning")
    else:
        rec = ("🚫 No Edge", "info")

    getattr(st, rec[1])(f"**{rec[0]}** — based on your EV/EDGE thresholds + Kelly>0")

    left, right = st.columns([4, 2])

    with left:
        st.progress(int(p_pick * 100))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Model Prob (Pick)", f"{p_pick:.1%}")
        c2.metric("Market Prob (Pick)", "N/A" if mp_pick is None or np.isnan(mp_pick) else f"{mp_pick:.1%}")
        c3.metric("EV (per $1)", "N/A" if ev is None or np.isnan(ev) else f"{ev:+.3f}")
        c4.metric("Stake ($)", f"${stake:,.0f}")

    with right:
        st.markdown("**Pick**")
        st.markdown(f"## {r['Winner']}")
        st.caption(r["Status"])

        odds_txt = "N/A" if best_odds is None or np.isnan(best_odds) else f"{best_odds:.2f}"
        st.markdown(
            f"""
            **Best Odds (Dec):** {odds_txt}  
            **Kelly (full):** {fk*100:.2f}% ({r['Kelly Note']})  
            **Odds samples:** {odds_ct if odds_ct else "N/A"}  
            """
        )

    if show_raw_odds:
        with st.expander("📉 View raw sportsbook odds"):
            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown(f"**{r['Fighter A Name']}**")
                if r.get("Odds List A"):
                    for o in r["Odds List A"]:
                        st.markdown(f"- {format_american(o)}")
                else:
                    st.caption("No odds found")

            with col_b:
                st.markdown(f"**{r['Fighter B Name']}**")
                if r.get("Odds List B"):
                    for o in r["Odds List B"]:
                        st.markdown(f"- {format_american(o)}")
                else:
                    st.caption("No odds found")

    st.divider()

# ===============================================================
# SUMMARY TABLE
# ===============================================================
summary_df = pd.DataFrame([{
    "Fight": r["Fight"],
    "Pick": r["Winner"],
    "Model Prob (Pick)": "N/A" if r["Prob_A"] is None else f"{r['Model Prob (Pick)']:.1%}",
    "Market Prob (Pick)": "N/A" if (r.get("Market Prob (Pick)") is None or np.isnan(r.get("Market Prob (Pick)"))) else f"{r['Market Prob (Pick)']:.1%}",
    "EV (per $1)": "N/A" if (r.get("EV (per $1)") is None or np.isnan(r.get("EV (per $1)"))) else f"{r['EV (per $1)']:+.3f}",
    "Best Odds (Dec)": "N/A" if (r.get("Best Odds (Dec)") is None or np.isnan(r.get("Best Odds (Dec)"))) else f"{r['Best Odds (Dec)']:.2f}",
    "Kelly % (full)": "N/A" if r["Prob_A"] is None else f"{r['Kelly % (full)']*100:.2f}%",
    "Stake ($)": "N/A" if r["Prob_A"] is None else f"${r['Stake $']:,.0f}",
    "Notes": r["Status"],
} for r in results])

st.subheader("📋 Card Summary")
st.dataframe(summary_df, use_container_width=True)

# ===============================================================
# BANKROLL SIM
# ===============================================================
st.subheader("💰 Bankroll Simulation")

st.markdown("""
### 🧪 What this simulation shows
We simulate this fight card **thousands of times** using:
- your model probabilities  
- real betting odds  
- conservative Kelly sizing  

This answers: **“If I repeat this strategy, what usually happens?”**
""")

if not bets_for_sim:
    st.warning("No bets qualified for simulation (based on your EV/EDGE thresholds and available odds).")
else:
    st.caption(
        f"Simulating {len(bets_for_sim)} bets | "
        f"Starting bankroll = ${bankroll:,} | Kelly multiplier = {kelly_mult:.2f} | sims = {int(n_sims):,}"
    )

    with st.spinner("Running Monte Carlo simulation..."):
        sims = simulate_bankroll_paths(
            bets=bets_for_sim,
            initial_bankroll=bankroll,
            n_sims=int(n_sims),
            kelly_mult=kelly_mult
        )

    if sims.size > 0:
        st.metric("Median Final Bankroll", f"${np.median(sims):,.0f}")
        st.metric("5% Worst Case", f"${np.percentile(sims, 5):,.0f}")
        st.metric("95% Best Case", f"${np.percentile(sims, 95):,.0f}")
        st.metric("Chance to Grow Bankroll", f"{np.mean(sims > bankroll):.1%}")

        st.caption(
            f"Mean: ${np.mean(sims):,.0f} | "
            f"Std: ${np.std(sims):,.0f} | "
            f"Prob of drawdown >50%: {np.mean(sims < bankroll*0.5):.1%}"
        )

        st.markdown("#### ✅ Bets simulated")
        st.dataframe(pd.DataFrame([{
            "Fight": b["fight"],
            "Pick": b["pick"],
            "p_model": f"{b['p']:.1%}",
            "best_odds_dec": f"{b['decimal_odds']:.2f}",
            "full_kelly_%": f"{b['stake_frac_full_kelly']*100:.2f}%",
            "stake_$": f"${(bankroll*b['stake_frac_full_kelly']*kelly_mult):,.0f}",
        } for b in bets_for_sim]), use_container_width=True)

st.markdown("---")
st.caption("""
⚠️ **Disclaimer**

This application is for **educational and research purposes only**.

It does **not** constitute financial or betting advice.  
All probabilities are model-based estimates and may be incorrect.

Betting involves risk. **You are solely responsible for any decisions you make.**
""")

