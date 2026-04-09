# Octagon Intel

Premium UFC betting intelligence built around calibrated prefight-only prediction support.

This repository is no longer organized as a notebook wrapper or a Streamlit-first demo. The flagship product surface is now a Vercel-friendly Next.js frontend backed by a clean FastAPI inference layer that preserves the original modeling discipline:

- strictly prefight-only features
- deterministic inference
- calibrated probabilities
- no leakage
- explicit event and fight support checks
- graceful unsupported-fight handling

The product is designed around real event consumption:

- browse UFC event coverage
- see which bouts are supportable from current artifacts
- compare calibrated model probability to market price when odds exist
- inspect grounded prefight feature differences in a premium fight breakdown
- surface unavailable predictions clearly instead of faking support

Streamlit remains available only as a secondary local debugging surface.

---

## Repo Layout

The repo is separated by product role:

- `frontend/` — Next.js premium product UI for landing, events, and event detail pages
- `apps/vercel/api/main.py` — FastAPI inference API with product-friendly `/v1` routes
- `backend/` — product service layer that assembles event and fight payloads
- `core/` — shared model, eligibility, odds, and inference logic
- `pipelines/` — raw UFCStats scrape, fighter snapshot refresh, training, and orchestration scripts
- `apps/streamlit/app.py` — local-only secondary Streamlit surface
- `notebooks/ufc_pipeline.ipynb` — archived research notebook, no longer the operational path

Preferred run commands:

```bash
uvicorn apps.vercel.api.main:app --reload
cd frontend && npm install && npm run dev
```

Optional local debug surface:

```bash
streamlit run apps/streamlit/app.py
```

---

## Product API

Primary frontend-facing routes:

- `GET /v1/health` — product API health and available event list
- `GET /v1/events` — event summaries with coverage status
- `GET /v1/events/{event_number}` — premium event detail payload with supported and unsupported fights

Legacy low-level routes remain available for direct debugging:

- `GET /health`
- `GET /events`
- `GET /events/{event_number}/coverage`
- `GET /events/{event_number}/predictions`

---

## Automation Workflow

The notebook is no longer the operational path. The repo now includes CLI scripts for the raw UFCStats scrape, model training, and an end-to-end refresh entrypoint.

Raw UFCStats data now lives under:

`data/raw/ufcstats/`

Common commands:

```bash
# 1) Incrementally refresh the raw UFCStats CSV bundle
python -m pipelines.scrape_ufcstats all

# 2) Rebuild just fighters_latest.csv from raw CSVs
python -m pipelines.refresh_fighters --data-dir data/raw/ufcstats

# 3) Retrain and export all production model artifacts
python -m pipelines.train_model --data-dir data/raw/ufcstats

# 4) Run the full workflow in one shot
python -m pipelines.refresh_all --card-events 324 --odds-events 324 325
```

Useful flags:

- `python -m pipelines.scrape_ufcstats all --full` for a from-scratch raw rebuild
- `python -m pipelines.train_model --data-dir data/raw/ufcstats --tune-trials 0` to skip Optuna and use baked-in XGBoost defaults
- `python -m pipelines.refresh_all --skip-train` to refresh raw data and snapshot only

Recommended cadence:

- After each completed UFC event: `scrape_ufcstats` + `train_model` (or at minimum `refresh_fighters`)
- Before upcoming events: refresh `scrape_cards.py` and `scrape_odds.py` data for the target event numbers
- For deployment automation: run these jobs in GitHub Actions or another scheduled runner, then deploy the frontend/API using the refreshed artifacts

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
