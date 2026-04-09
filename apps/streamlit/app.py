"""
UFC Event Predictor — Streamlit interface.

This is the Streamlit entrypoint.
All prediction, feature engineering, betting math, eligibility checking,
and data loading come from core/.

To run:
    streamlit run apps/streamlit/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import streamlit as st

from core import (
    american_to_decimal,
    check_event_coverage,
    ev_per_dollar,
    get_odds_for_fighter,
    implied_prob_from_decimal,
    kelly_fraction,
    kelly_note,
    list_available_events,
    load_artifacts,
    load_fight_card,
    load_odds_map,
    safe_float,
    simulate_bankroll_paths,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="UFC Event Predictor", layout="wide")

# ── Navigation ─────────────────────────────────────────────────────────────────
page = st.sidebar.radio("📘 Navigation", ["Event Predictor", "Betting Guide"])


# ── Asset loading (cached for the Streamlit session) ──────────────────────────
@st.cache_resource
def _load_assets():
    return load_artifacts()


try:
    model, calibrator, feature_cols, clip_bounds, fighters_df, fighter_lookup = _load_assets()
except FileNotFoundError as exc:
    st.error(f"**Model artifacts missing.** {exc}")
    st.info("Run `python -m pipelines.train_model --data-dir data/raw/ufcstats` to regenerate artifacts, then restart the app.")
    st.stop()


# ──────────────────────────────────────────────────────────────────────────────
# BETTING GUIDE PAGE
# ──────────────────────────────────────────────────────────────────────────────
def betting_guide():
    st.title("📘 Betting Logic & Decision Guide")
    st.caption("How the app thinks, why bets are filtered, and what every label actually means")
    st.divider()

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

    st.subheader("📚 Core Terms")
    st.markdown("""
Model Probability
The model's estimated probability that the fighter wins.

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

    st.subheader("📐 Betting Math (Exact Formulas Used)")
    st.markdown(r"""
Implied Market Probability

$$P_{market} = \frac{1}{d}$$

Expected Value per $1

$$EV = p \cdot d - 1$$

Edge

$$Edge = p_{model} - P_{market}$$

Kelly Criterion (full Kelly) — let $b = d - 1$

$$Kelly = \max\left(0,\; \frac{b p - (1 - p)}{b} \right)$$

If Kelly ≤ 0, the optimal bet size is zero.
""")
    st.divider()

    st.subheader("🔢 How Market Probability & Best Odds Are Calculated")
    st.markdown("""
This app aggregates **many sportsbook prices** and uses them in two
different ways — intentionally.

**Best Odds** = max(decimal_odds)
- What is the best price you can personally get?
- Used for EV and Kelly sizing

**Market Probability** = 1 / median(decimal_odds)
- Represents market consensus (median filters outlier books)
- Used only for EDGE calculation

---

### 🧠 Why EV and EDGE Can Disagree

Example:

- Model Prob = **75.1%**
- Market Prob = **75.8%**
- Best Odds = **1.34**

EDGE = -0.7%
EV ≈ +0.006
Kelly = 0

Result: **No Edge**

High confidence ≠ profitable bet.
""")
    st.divider()

    st.subheader("✅ Scenario 1: Clear +EV Bet")
    st.markdown("""
Model Prob: 65% | Best Odds: 2.00

Market Prob = 50% | Edge = +15% | EV = +0.30 | Kelly (full) = 30%

App label: **+EV Bet**
""")
    st.divider()

    st.subheader("🚫 Scenario 2: Strong Pick, Bad Bet")
    st.markdown("""
Model Prob: 75% | Best Odds: 1.30

Market Prob = 76.9% | Edge = -1.9% | EV = -0.025 | Kelly = 0%

App label: **No Edge** — confidence does not imply profitability.
""")
    st.divider()

    st.subheader("⚠️ Scenario 3: Slight Edge")
    st.markdown("""
Model Prob: 56% | Best Odds: 1.80

Market Prob = 55.6% | Edge = +0.4% | EV = +0.008 | Kelly (full) ≈ 0.8%

App label: **Slight Edge** — positive EV exists but variance dominates.
""")
    st.divider()

    st.subheader("🎛 Betting Settings — How Decisions Are Actually Made")
    st.markdown(r"""
All settings are **hard filters**. If a bet fails any filter, it is **non-actionable**.

## 💰 Starting Bankroll

$$\text{Stake} = \text{Bankroll} \times f_{\text{Kelly}} \times \text{Kelly Multiplier}$$

## 🧠 Kelly Multiplier (Risk Dial)

- 0.25 → quarter Kelly (very conservative)
- 0.50 → half Kelly (balanced)
- 1.00 → full Kelly (high drawdown risk)

## 📐 Min EDGE

$$\text{EDGE} = p_{\text{model}} - p_{\text{market}}$$

Typical values: 0.01 (permissive) → 0.05+ (very selective)

## 📈 Min EV

$$\text{EV} = p \cdot d - 1$$

EV > 0 → profitable in expectation. EV alone is **not sufficient**.

## 🧮 Bet Classification

**💰 +EV Bet** — all satisfied: EV ≥ MinEV, EDGE ≥ MinEDGE, Kelly > 0
**⚠️ Slight Edge** — EV > 0, Kelly > 0, but fails MinEV or MinEDGE
**🚫 No Edge** — EV ≤ 0, or Kelly = 0
""")
    st.divider()

    st.subheader("📊 Odds Data Source")
    st.markdown("""
All betting odds are sourced from **fightodds.io**, which aggregates
moneyline prices across major sportsbooks including Pinnacle, DraftKings,
FanDuel, BetMGM, Bovada, Circa, and others.

For each fighter:
- All available sportsbook prices are collected and deduplicated
- **Best available odds** are used for EV and Kelly sizing
- **Median market price** estimates market-implied probability
""")
    st.divider()

    st.warning("""
This app optimizes decision quality.

Variance is unavoidable.
Kelly assumes your probabilities are accurate.
Overbetting destroys bankrolls faster than bad picks.
""")


# ── Routing ────────────────────────────────────────────────────────────────────
if page == "Betting Guide":
    betting_guide()
    st.stop()


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR CONTROLS
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.header("💰 Betting Settings")

bankroll = st.sidebar.number_input(
    "Starting Bankroll ($)", min_value=50, max_value=500_000, value=1000, step=50
)
kelly_mult = st.sidebar.slider(
    "Kelly Multiplier (0.25 = quarter Kelly)",
    min_value=0.05, max_value=1.0, value=0.5, step=0.05,
)
min_edge = st.sidebar.slider(
    "Min EDGE to consider a bet",
    min_value=-0.10, max_value=0.30, value=0.02, step=0.01,
    help="EDGE = ModelProb - MarketImpliedProb",
)
min_ev = st.sidebar.slider(
    "Min EV to consider a bet",
    min_value=-0.20, max_value=0.50, value=0.01, step=0.01,
    help="Expected profit per $1 bet",
)
n_sims = st.sidebar.number_input(
    "Bankroll Simulations", min_value=500, max_value=50_000, value=5000, step=500
)
show_raw_odds = st.sidebar.checkbox(
    "📉 Show raw sportsbook odds", value=False
)


# ──────────────────────────────────────────────────────────────────────────────
# EVENT PREDICTOR PAGE
# ──────────────────────────────────────────────────────────────────────────────
st.title("🥊 UFC Event Predictor")

# Show available events to help users
available = list_available_events()
if available:
    st.caption(f"Cards available locally: UFC {', '.join(str(n) for n in available)}")

event_input = st.text_input("Enter UFC Event Number (e.g., 324)")

if not event_input.isdigit():
    st.warning("Please enter a valid UFC event number (e.g. 324)")
    st.stop()

event_number = int(event_input)

# ── Load fight card ────────────────────────────────────────────────────────────
with st.spinner(f"Loading UFC {event_number} fight card..."):
    try:
        fights = load_fight_card(event_number)
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

if not fights:
    st.warning("No fights found — event card may be empty.")
    st.stop()

# ── Load odds (optional — no odds is fine) ─────────────────────────────────────
@st.cache_data(show_spinner=False)
def _load_odds(event_number: int) -> dict:
    return load_odds_map(event_number)


odds_map = _load_odds(event_number)
if not odds_map:
    st.info(f"No odds file found for UFC {event_number}. Predictions will be shown without betting metrics.")


# ──────────────────────────────────────────────────────────────────────────────
# RUN ELIGIBILITY + PREDICTIONS
# ──────────────────────────────────────────────────────────────────────────────
coverage = check_event_coverage(
    event_number, fights,
    fighters_df, fighter_lookup,
    feature_cols, clip_bounds,
    model, calibrator,
)

if coverage.status == "not_predictable":
    st.error(
        f"UFC {event_number} cannot be predicted: no fighters found in the snapshot. "
        "This is expected for very recent debuting fighters or significant name mismatches."
    )
elif coverage.status == "partially_predictable":
    n_pred = len(coverage.predictable_fights)
    n_total = len(coverage.fights)
    st.warning(
        f"UFC {event_number}: {n_pred} of {n_total} fights are predictable. "
        f"{n_total - n_pred} fight(s) are unsupported — see details below."
    )


# ──────────────────────────────────────────────────────────────────────────────
# DISPLAY FIGHT-BY-FIGHT RESULTS
# ──────────────────────────────────────────────────────────────────────────────
st.subheader(f"🥊 UFC {event_number} Predictions")

results = []
bets_for_sim = []

for f in coverage.fights:
    fighter_a, fighter_b = f.fighter_a, f.fighter_b
    fight_label = f"{fighter_a} vs {fighter_b}"

    st.markdown(f"### {fight_label}")

    # ── Unsupported fight ──────────────────────────────────────────────────────
    if not f.is_predictable:
        st.warning(f"⚠️ **Unsupported:** {f.reason}")
        results.append({
            "Fight": fight_label,
            "Winner": "—",
            "Model Prob (Pick)": None,
            "Status": f.reason,
        })
        st.divider()
        continue

    # ── Predictable fight ──────────────────────────────────────────────────────
    prob_a = f.prob_a
    winner = fighter_a if prob_a >= 0.5 else fighter_b
    confidence = abs(prob_a - 0.5)
    pick_status = "🔥 Strong pick" if confidence >= 0.15 else "⚖️ Close fight"

    pick_side = "A" if winner == fighter_a else "B"
    p_pick = float(prob_a if pick_side == "A" else 1.0 - prob_a)

    # Odds
    odds_list_a = get_odds_for_fighter(fighter_a, odds_map)
    odds_list_b = get_odds_for_fighter(fighter_b, odds_map)
    dec_a = [american_to_decimal(o) for o in odds_list_a]
    dec_b = [american_to_decimal(o) for o in odds_list_b]

    market_prob_a = np.nan
    if dec_a:
        market_prob_a = float(implied_prob_from_decimal(float(np.median(dec_a))))

    best_odds_dec = np.nan
    odds_count_pick = 0
    if pick_side == "A" and dec_a:
        best_odds_dec = float(np.max(dec_a))
        odds_count_pick = len(dec_a)
    elif pick_side == "B" and dec_b:
        best_odds_dec = float(np.max(dec_b))
        odds_count_pick = len(dec_b)

    ev = ev_per_dollar(p_pick, best_odds_dec)

    edge_a = np.nan
    market_prob_pick = np.nan
    if not np.isnan(market_prob_a):
        edge_a = float(prob_a - market_prob_a)
        market_prob_pick = float(market_prob_a if pick_side == "A" else 1.0 - market_prob_a)

    fk = 0.0
    stake_dollars = 0.0
    if not np.isnan(best_odds_dec):
        fk = kelly_fraction(p_pick, best_odds_dec)
        stake_dollars = bankroll * fk * kelly_mult

    include_for_sim = (
        not np.isnan(ev) and not np.isnan(edge_a)
        and ev >= float(min_ev)
        and edge_a >= float(min_edge)
        and fk > 0
    )

    if include_for_sim:
        bets_for_sim.append({
            "fight": fight_label,
            "pick": winner,
            "p": p_pick,
            "decimal_odds": float(best_odds_dec),
            "stake_frac_full_kelly": float(fk),
        })

    # Recommendation badge
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
        c2.metric("Market Prob (Pick)", "N/A" if np.isnan(market_prob_pick) else f"{market_prob_pick:.1%}")
        c3.metric("EV (per $1)", "N/A" if np.isnan(ev) else f"{ev:+.3f}")
        c4.metric("Stake ($)", f"${stake_dollars:,.0f}")

    with right:
        st.markdown("**Pick**")
        st.markdown(f"## {winner}")
        st.caption(pick_status)
        odds_txt = "N/A" if np.isnan(best_odds_dec) else f"{best_odds_dec:.2f}"
        st.markdown(
            f"**Best Odds (Dec):** {odds_txt}  \n"
            f"**Kelly (full):** {fk*100:.2f}% ({kelly_note(fk)})  \n"
            f"**Odds samples:** {odds_count_pick if odds_count_pick else 'N/A'}"
        )

    if show_raw_odds:
        with st.expander("📉 View raw sportsbook odds"):
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"**{fighter_a}**")
                for o in odds_list_a:
                    st.markdown(f"- {f'+{o}' if o > 0 else str(o)}")
                if not odds_list_a:
                    st.caption("No odds found")
            with col_b:
                st.markdown(f"**{fighter_b}**")
                for o in odds_list_b:
                    st.markdown(f"- {f'+{o}' if o > 0 else str(o)}")
                if not odds_list_b:
                    st.caption("No odds found")

    results.append({
        "Fight": fight_label,
        "Fighter A": fighter_a,
        "Fighter B": fighter_b,
        "Winner": winner,
        "Model Prob (Pick)": p_pick,
        "Market Prob (Pick)": safe_float(market_prob_pick),
        "EV (per $1)": safe_float(ev),
        "Best Odds (Dec)": safe_float(best_odds_dec),
        "Kelly % (full)": fk,
        "Kelly Note": kelly_note(fk),
        "Stake $": stake_dollars,
        "Status": pick_status,
    })

    st.divider()


# ──────────────────────────────────────────────────────────────────────────────
# SUMMARY TABLE
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("📋 Card Summary")

summary_df = pd.DataFrame([{
    "Fight": r["Fight"],
    "Pick": r["Winner"],
    "Model Prob": "—" if r["Model Prob (Pick)"] is None else f"{r['Model Prob (Pick)']:.1%}",
    "Market Prob": "—" if (r.get("Market Prob (Pick)") is None or np.isnan(r.get("Market Prob (Pick)"))) else f"{r['Market Prob (Pick)']:.1%}",
    "EV (per $1)": "—" if (r.get("EV (per $1)") is None or np.isnan(r.get("EV (per $1)"))) else f"{r['EV (per $1)']:+.3f}",
    "Best Odds": "—" if (r.get("Best Odds (Dec)") is None or np.isnan(r.get("Best Odds (Dec)"))) else f"{r['Best Odds (Dec)']:.2f}",
    "Kelly %": "—" if r["Model Prob (Pick)"] is None else f"{r['Kelly % (full)']*100:.2f}%",
    "Stake $": "—" if r["Model Prob (Pick)"] is None else f"${r['Stake $']:,.0f}",
    "Notes": r["Status"],
} for r in results])

st.dataframe(summary_df, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# BANKROLL SIMULATION
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("💰 Bankroll Simulation")

st.markdown("""
### 🧪 What this simulation shows
We simulate this fight card **thousands of times** using:
- your model probabilities
- real betting odds
- conservative Kelly sizing

This answers: **"If I repeat this strategy, what usually happens?"**
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
            kelly_mult=kelly_mult,
        )

    if sims.size > 0:
        st.metric("Median Final Bankroll", f"${np.median(sims):,.0f}")
        st.metric("5% Worst Case", f"${np.percentile(sims, 5):,.0f}")
        st.metric("95% Best Case", f"${np.percentile(sims, 95):,.0f}")
        st.metric("Chance to Grow Bankroll", f"{np.mean(sims > bankroll):.1%}")
        st.caption(
            f"Mean: ${np.mean(sims):,.0f} | "
            f"Std: ${np.std(sims):,.0f} | "
            f"Prob of drawdown >50%: {np.mean(sims < bankroll * 0.5):.1%}"
        )

        st.markdown("#### ✅ Bets simulated")
        st.dataframe(pd.DataFrame([{
            "Fight": b["fight"],
            "Pick": b["pick"],
            "p_model": f"{b['p']:.1%}",
            "best_odds_dec": f"{b['decimal_odds']:.2f}",
            "full_kelly_%": f"{b['stake_frac_full_kelly']*100:.2f}%",
            "stake_$": f"${(bankroll * b['stake_frac_full_kelly'] * kelly_mult):,.0f}",
        } for b in bets_for_sim]), use_container_width=True)

st.markdown("---")
st.caption("""
⚠️ **Disclaimer**

This application is for **educational and research purposes only**.
It does **not** constitute financial or betting advice.
All probabilities are model-based estimates and may be incorrect.
Betting involves risk. **You are solely responsible for any decisions you make.**
""")
