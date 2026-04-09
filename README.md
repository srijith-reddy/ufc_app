# Octagon Intel

Octagon Intel is a UFC fight intelligence platform built around calibrated prefight-only predictions. It turns model probability, live market odds, fighter feature comparisons, and coverage discipline into one clean event-level experience.

## What Octagon Intel Does

- surfaces upcoming UFC events with fight-level coverage status
- separates supportable bouts from unavailable ones rather than forcing predictions
- compares calibrated model probability against live market odds where they exist
- shows grounded prefight feature differences for each supported matchup
- surfaces kelly fraction, edge, and value state alongside each pick
- explains why unsupported fights cannot be predicted instead of hiding them

## Product Direction

This repository is built around event-level decision quality rather than a notebook wrapper or a raw odds feed.

The current product is:

- strictly prefight-only — no post-fight leakage, no retroactive features
- coverage-honest — unsupported fights are shown explicitly with reasons
- calibrated — probabilities reflect measured confidence, not raw model output
- designed to feel like a real product rather than a demo surface

## Key Views

### Event Index

The events page shows:

- upcoming UFC cards the platform is currently tracking
- coverage status per event showing how many bouts are supportable
- featured matchup and odds freshness at a glance

### Event Detail

Each event page shows:

- a coverage summary for the full card
- supported fights with calibrated probability, edge, kelly fraction, and value state
- prefight feature comparisons across striking, grappling, form, and physical attributes
- insight chips grounded in the current prefight snapshot
- unsupported fights with explicit reasons for unavailability
- per-book odds quotes where available

## Tech Stack

- Next.js 15
- React 19
- TypeScript
- Tailwind CSS
- Framer Motion
- FastAPI
- XGBoost
- Scikit-learn

## Repository Structure

```text
frontend/               Next.js product UI for landing, events, and event detail
apps/vercel/api/        FastAPI inference API with product-friendly /v1 routes
backend/                product service layer that assembles event and fight payloads
core/                   shared model, eligibility, odds, and inference logic
pipelines/              UFCStats scrape, fighter snapshot refresh, and training scripts
cards/                  local fight card artifacts per event
odds/                   local odds snapshots per event
data/                   raw UFCStats CSVs and processed fighter snapshots
models/                 exported XGBoost booster and calibration artifacts
```

## Local Development

### Backend

```bash
uvicorn apps.vercel.api.main:app --reload
```

### Frontend

```bash
cd frontend && npm install && npm run dev
```

## API Routes

Primary frontend-facing routes:

- `GET /v1/health` — product API health and available event list
- `GET /v1/events` — event summaries with coverage status
- `GET /v1/events/{event_id}` — full event detail payload with supported and unsupported fights

Low-level debug routes:

- `GET /health`
- `GET /events`
- `GET /events/{event_number}/coverage`
- `GET /events/{event_number}/predictions`

## Automation

Fighter data and model artifacts are refreshed via GitHub Actions on a monthly schedule. Fight cards and odds are updated daily.

Raw UFCStats data lives under `data/raw/ufcstats/`. Common commands:

```bash
# Incrementally refresh the raw UFCStats CSV bundle
python -m pipelines.scrape_ufcstats all

# Rebuild fighters_latest.csv from raw CSVs
python -m pipelines.refresh_fighters --data-dir data/raw/ufcstats

# Retrain and export all production model artifacts
python -m pipelines.train_model --data-dir data/raw/ufcstats

# Run the full workflow in one shot
python -m pipelines.refresh_all --card-events 324 --odds-events 324 325
```

Useful flags:

- `--full` on `scrape_ufcstats` for a from-scratch raw rebuild
- `--tune-trials 0` on `train_model` to skip Optuna and use baked-in defaults
- `--skip-train` on `refresh_all` to refresh raw data and snapshot only

## Current Status

What is working today:

- polished Next.js frontend deployed on Vercel
- FastAPI inference backend deployed on Render
- calibrated XGBoost predictions with betting metrics
- daily automated card and odds scraping via GitHub Actions
- monthly automated fighter snapshot and model refresh via GitHub Actions

What still depends on external data or future hardening:

- odds availability varies by event and provider
- missing fighters on new cards until the next snapshot refresh runs
- no historical event archive in the product UI

## Why This Repo Exists

UFC fight prediction tools are usually either overfit notebooks, raw odds scrapers, or apps that fake confidence on every bout.

Octagon Intel exists to build something more honest: a platform that knows what it can support, shows calibrated probability where it can, and says pass where it cannot.

---

## 1️⃣ Premodeling & Data Engineering (THE MOST IMPORTANT PART)

Before any model is trained, the training pipeline enforces hard constraints that define what is legally usable at prediction time.

### 1.1 Canonical fight universe
- Only win / loss outcomes kept  
- NC / draws removed  
- Dates parsed and validated  
- All rows sorted chronologically  

```

Master rows:                17,006
After removing NC/draw:     16,704
After dropping missing date:16,114
After filtering pre-2005:   15,254
After removing 1-fight profiles: 14,758

```

This avoids:
- Early-era UFC noise  
- Fighters with no historical signal  
- Artificial inflation from debut fights  

---

### 1.2 Rating integrity (Glicko)
- Fights missing pre-fight Glicko ratings are removed  
- Only g_rating_before and g_RD_before are used  
- Ensures ratings are never updated using current fight  

---

### 1.3 Temporal feature construction (prefight-only)

All temporal features are computed using groupby + shift, never cumulative leakage.

For each fighter:
- fights_before  
- wins_before  
- win_rate_before  
- recent_win_rate_3  
- recent_win_rate_5  
- days_since_last_fight  

Opponent versions are merged by fight_url, not by future stats.

---

### 1.4 Skew diagnostics (WHY transformations exist)

Measured skew on training data:
```

days_since_last_fight        5.05
opp_days_since_last_fight    5.05
fights_before                1.61
opp_fights_before            1.61

```

This justifies:
- Quantile clipping  
- Log transforms  
- Separate treatment of temporal vs skill features  

---

### 1.5 Quantile clipping (train-fit only)

For heavy-tailed features:
```

[0.1%, 99.9%] quantiles

```

Applied only using training data, then reused for:
- test set  
- Streamlit inference  

---

### 1.6 Log transforms (after clipping)

Applied to:
- fights_before  
- days_since_last_fight  
- opponent equivalents  

This stabilizes:
- Logistic Regression  
- XGBoost splits  
- Calibration behavior  

---

## 🧠 Feature Set Overview (Prefight-Only)

All models operate on a strictly prefight-available feature space.

Feature categories:
- **Rating-based:** Glicko rating and RD differentials  
- **Physical:** height, reach, age differences  
- **Striking:** SLpM, SApM, accuracy, defense  
- **Grappling:** takedown metrics, submissions  
- **Form:** win rates and recent form windows  
- **Temporal:** log-transformed experience and layoffs  

<details>
<summary><strong>Exact feature list</strong></summary>

rating_diff, RD_diff,
height_diff, reach_diff, age_diff,
SLpM, SApM, Str_Acc, Str_Def,
TD_Avg, TD_Acc, TD_Def, Sub_Avg,
opp_SLpM, opp_SApM, opp_Str_Acc, opp_Str_Def,
opp_TD_Avg, opp_TD_Acc, opp_TD_Def, opp_Sub_Avg,
log_fights_before, log_days_since_last_fight,
win_rate_before, recent_win_rate_3, recent_win_rate_5,
log_opp_fights_before, log_opp_days_since_last_fight,
opp_win_rate_before, opp_recent_win_rate_3, opp_recent_win_rate_5


</details>

---

## 2️⃣ Models Compared

All models use the same prefight data rules.

---

### Model A — Logistic Regression (Baseline)

Purpose  
Establish a strong linear baseline under perfect data hygiene.

CV (TimeSeriesSplit)
```

Fold AUCs: 0.89 – 0.91
OOF AUC:   0.7799

```

Test (Post-2021)
```

Accuracy: 0.7744
AUC:      0.8707

```

Why it matters
- Confirms signal quality  
- Provides an interpretable reference  
- Shows data > model complexity  

---

### Model B — XGBoost (Tabular, Uncalibrated)

Purpose  
Measure nonlinear lift over Logistic Regression.

CV (Optuna-tuned)
```

Fold AUCs: 0.87 – 0.94
OOF AUC:   0.7891

```

Test
```

Accuracy: 0.8078
AUC:      0.8999

```

Observation
- Strong ranking  
- Over-confident probabilities  
- Needs calibration for real usage  

---

### Model C — XGBoost + Isotonic Calibration

Calibration trained only on OOF predictions.

Training Brier
```

Uncalibrated: 0.1806
Calibrated:   0.1577

```

Test (Calibrated)
```

Accuracy: 0.8074
AUC:      0.8928
Brier:    0.1380

```

Interpretation
- AUC stable (expected)  
- Probability quality improves significantly  
- This is the deployment-grade tabular model  

---

### Model D — GRU Fighter Style Encoder (Representation Only)

Does not predict wins.

Purpose  
Learn latent fighter style embeddings from fight sequences.

- Sequence length: 5 past fights  
- 8 per-fight stats  
- Targets: style regression (SLpM, TD_Avg, Sub_Avg, Str_Acc)  

Training curve
```

MSE: 1.3758 → 0.2203

```

Use
- Generates 32-dim embeddings  
- Used downstream as features  
- Encodes stylistic evolution  

---

### Model E — GRU Style Differences + XGBoost + Calibration

Purpose  
Test whether learned style mismatches add predictive power.

CV
```

Fold AUCs: 0.85 – 0.93

```

Test
```

Accuracy: 0.7390
AUC:      0.8143
Brier:    0.1769

```

Conclusion
- Competitive but not dominant  
- Style info overlaps with tabular stats  
- Valuable analytically, not strictly superior  

---

## 3️⃣ Model Comparison (Test Set)

| Model | Calibrated | AUC | Accuracy | Brier |
|------|-----------|-----|----------|-------|
| Logistic Regression | No | 0.8707 | 0.7744 | — |
| XGBoost | No | 0.8999 | 0.8078 | 0.1806 |
| XGBoost | Yes | 0.8928 | 0.8074 | 0.1380 |
| GRU + XGBoost | Yes | 0.8143 | 0.7390 | 0.1769 |

---
