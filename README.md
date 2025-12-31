# 🥊 UFC Fight Prediction 

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
| XGBoost (tabular) | No | 0.8999 | 0.8078 | 0.1806 |
| XGBoost (tabular) | Yes | 0.8928 | 0.8074 | 0.1380 |
| GRU + XGBoost | Yes | 0.8143 | 0.7390 | 0.1769 |

---
