# 🥊 UFC Fight Prediction 

[https://github.com/srijith-reddy/ufc_app/issues/1#issue-3770714357](https://github.com/user-attachments/assets/a0e067f2-a68d-4e16-b25c-2c34af5dd2f8
)

⏩ **Short on time?**  
Jump to **~3:30** in the walkthrough to see the final fight card predictions.

This repository documents a full UFC fight prediction pipeline, starting from raw data construction and pre-modeling hygiene, through multiple modeling approaches, and ending with a production-safe Streamlit inference app.

---

## 1️⃣ Premodeling & Data Engineering (THE MOST IMPORTANT PART)

Before any model is trained, the notebook enforces hard constraints that define what is legally usable at prediction time.

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

Saved as:
```

models/clip_bounds.json

```

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

## 2️⃣ Models Compared

All models use the same prefight data rules.

---

### Model A — Logistic Regression (Baseline)

Purpose  
Establish a strong linear baseline under perfect data hygiene.

CV (TimeSeriesSplit)
```

Fold AUCs: 0.87 – 0.89
OOF AUC:   0.7639

```

Test (Post-2021)
```

Accuracy: 0.7524
AUC:      0.8359

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

Fold AUCs: 0.87 – 0.93
OOF AUC:   0.7826

```

Test
```

Accuracy: 0.7939
AUC:      0.8894

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

Uncalibrated: 0.1861
Calibrated:   0.1625

```

Test (Calibrated)
```

Accuracy: 0.8070
AUC:      0.8863
Brier:    0.1433

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

MSE: 1.3277 → 0.1769

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

Fold AUCs: 0.84 – 0.92

```

Test
```

Accuracy: 0.7460
AUC:      0.8254
Brier:    0.1746

```

Conclusion
- Competitive but not dominant  
- Style info overlaps with tabular stats  
- Valuable analytically, not strictly superior  

---

## 3️⃣ Model Comparison (Test Set)

| Model | Calibrated | AUC | Accuracy | Brier |
|------|-----------|-----|----------|-------|
| Logistic Regression | No | 0.8359 | 0.7524 | — |
| XGBoost (tabular) | No | 0.8894 | 0.7939 | 0.1861 |
| XGBoost (tabular) | Yes | 0.8863 | 0.8070 | 0.1433 |
| GRU + XGBoost | Yes | 0.8254 | 0.7460 | 0.1746 |

---

## 4️⃣ Streamlit Inference App (PRODUCTION-SAFE)

Design goals
- Exact parity with training preprocessing  
- Zero leakage  
- Deterministic predictions  
- Calibrated probabilities only  

---

### Assets loaded
```

models/
├─ xgb_prefight_model.json
├─ xgb_calibrator.pkl
├─ xgb_feature_cols.json
├─ clip_bounds.json
├─ fighters_latest.csv

```

---

### Preprocessing at inference

Matches training exactly:
1. NA fill  
2. Quantile clipping (train-fit bounds)  
3. Log transforms  
4. Feature reordering  

No recomputation. No shortcuts.

---

### Prediction logic
- Raw XGBoost probability  
- Isotonic calibration  
- Defensive clipping to [0.01, 0.99]  

---

### UI behavior
- Select fighters from latest snapshot  
- Build prefight feature row  
- Display:
  - Progress bar  
  - Win probability  
  - Main card summary table  
---

## 5️⃣ Dependencies
```

streamlit
pandas
numpy
scikit-learn
xgboost
optuna
torch
joblib
```
