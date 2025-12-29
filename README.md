# 🥊 UFC Fight Outcome Prediction  
**Leakage-Free, Time-Aware Glicko + XGBoost + GRU Style Embeddings**

This repository contains a **single end-to-end Jupyter notebook** that builds a production-grade UFC fight prediction system.  
The pipeline strictly uses **prefight-only information**, enforces **chronological integrity**, and saves **deployment-ready artifacts**.  
Everything documented below is derived **directly from notebook code and printed outputs** — no assumptions, no post-hoc edits.

---

## 1️⃣ Imports, Settings, and Global Configuration

The notebook uses:
- **PyTorch** for GRU sequence modeling
- **XGBoost** for tabular classification
- **Optuna** (earlier stage) for tuning
- **Scikit-learn** for metrics and calibration

Device selection is automatic:
- CUDA → MPS → CPU fallback

### Key constants
```

SEQ_LEN = 5
EMBED_DIM = 32
HIDDEN_DIM = 64
BATCH_SIZE = 64
EPOCHS = 25
LR = 1e-3

```

### Sequential (per-fight) features
```

SLpM, SApM,
Str_Acc, Str_Def,
TD_Avg, TD_Acc,
TD_Def, Sub_Avg

```

### Style regression targets
```

SLpM, TD_Avg, Sub_Avg, Str_Acc

```

---

## 2️⃣ Fighter Sequence Construction

Each fighter’s career is treated as a **time series**.

For every fight:
- Past fights are sorted chronologically
- Up to the **last 5 fights** are used
- Left-padded with zeros if fewer than 5
- Target is the fighter’s *current-fight style stats*

Each row represents:
- `(fighter, fight_url, date)`
- `sequence`: `(5 × 8)` tensor
- `target`: `(4,)` style vector

This produces a sequence-aligned dataset where **each embedding is conditioned only on past fights**.

---

## 3️⃣ GRU Style Encoder

### Architecture
- GRU(input=8, hidden=64)
- Mean pooling across time
- Linear projection to:
  - 32-dim **style embedding**
  - 4-dim **style regression head**

### Training setup
- Loss: MSE
- Optimizer: Adam
- Epochs: 25
- Train split: fights **before 2021-01-01**

### Training signal (printed)
```

Epoch 1/25  | Style MSE: 1.3277
Epoch 5/25  | Style MSE: 0.2885
Epoch 10/25 | Style MSE: 0.2137
Epoch 15/25 | Style MSE: 0.1896
Epoch 20/25 | Style MSE: 0.1806
Epoch 25/25 | Style MSE: 0.1769

```

The monotonic loss decay confirms the GRU learns **stable latent fighter style representations**.

---

## 4️⃣ Style Embedding Extraction

After training:
- The GRU encodes **every fight for every fighter**
- Produces a `(32,)` vector per row

Output:
- `emb_df` with columns:
```

fight_url, fighter, gru_style_0 ... gru_style_31

```

These embeddings are later merged for **both fighter and opponent**.

---

## 5️⃣ Feature Assembly for Final Model

### Embedding alignment
For each bout:
- Merge fighter embedding
- Merge opponent embedding
- Compute **32-dimensional style difference**:
```

gru_style_diff_i = fighter_style_i − opponent_style_i

```

### Final feature set
Includes:
- Glicko:
  - `rating_diff`, `RD_diff`
- Physical:
  - `height_diff`, `reach_diff`, `age_diff`
- Fighter stats:
  - SLpM, SApM, Str_Acc, Str_Def, TD_Avg, TD_Acc, TD_Def, Sub_Avg
- Opponent stats (mirrored)
- Temporal prefight features:
  - win_rate_before
  - recent_win_rate_3 / 5
  - opponent equivalents
- + 32 GRU style diff features

All missing values are filled with zero **after splitting**.

---

## 6️⃣ Time-Aware Train / Test Split

```

Train: date < 2021-01-01
Test:  date ≥ 2021-01-01

```

This ensures **no future leakage**.

---

## 7️⃣ XGBoost + OOF Calibration Pipeline

### XGBoost parameters
```

n_estimators: 600
learning_rate: 0.12
max_depth: 3
subsample: 0.9
colsample_bytree: 0.7
tree_method: hist
eval_metric: logloss
random_state: 42

```

### Cross-validation
- 5-fold **TimeSeriesSplit**
- Out-of-fold predictions collected

### Fold AUCs (printed)
```

Fold 1 AUC: 0.8399
Fold 2 AUC: 0.8786
Fold 3 AUC: 0.8987
Fold 4 AUC: 0.8885
Fold 5 AUC: 0.9161

```

---

## 8️⃣ Isotonic Calibration

- Calibrator trained on **OOF probabilities**
- Prevents test leakage
- Improves probability reliability (Brier)

---

## 9️⃣ Final Test Results (Printed)

```

===== FINAL RESULTS =====
Accuracy: 0.745958751393534
AUC:      0.8253891020358957
Brier:   0.1745996419962172

```

This reflects the **GRU + XGBoost + calibration** hybrid.

---

## 🔟 Saved Artifacts (Deployment-Ready)

Saved to `/models`:

```

gru_xgb_prefight_model.json        # final XGBoost model
gru_xgb_isotonic_calibrator.pkl    # probability calibrator
gru_xgb_feature_cols.json          # exact feature order (CRITICAL)
gru_style_encoder.pt               # trained GRU weights
gru_style_config.json              # GRU architecture + features

```

Printed confirmation:
```

✅ Saved GRU encoder, XGB model, calibrator, and metadata

```

These files are sufficient to reproduce inference **without retraining**.

---

## 🧠 Why This Pipeline Is Legit

- ✅ Chronological modeling everywhere
- ✅ Prefight-only data (no post-fight leakage)
- ✅ Fighter/opponent symmetry
- ✅ OOF-based calibration
- ✅ Sequence-aware latent representations
- ❌ No random splits
- ❌ No Kaggle-style shortcuts

This is **forecasting-grade**, not retrospective curve fitting.

---

## ▶️ How to Use

1. Open the notebook
2. Run top-to-bottom
3. Trained models appear in `/models`
4. Load GRU → build embeddings → apply XGB → calibrate → predict

---

## 📊 Final Snapshot

| Model | Accuracy | AUC | Brier |
|-----|---------|-----|-------|
GRU + XGB (calibrated) | 0.746 | 0.825 | 0.175 |

---

## 🔮 Future Extensions
- Supervise GRU on outcomes instead of style regression
- Transformer-based fighter encoders
- Betting ROI evaluation
- Weight-class or era-specific models

---
