"""
Train and export the production XGBoost prefight model artifacts.

This replaces the final notebook training cell with a reproducible CLI script.
It consumes the same raw UFCStats CSV bundle as pipelines.refresh_fighters and
writes all deployment artifacts into models/.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

from core.config import (
    CALIBRATOR_PATH,
    CLIP_BOUNDS_PATH,
    FEATURE_COLS_PATH,
    LOG_COLS,
    MISSING_COLS,
    MODEL_PATH,
    MODELS_DIR,
)
from pipelines.refresh_fighters import build_feature_frame

warnings.filterwarnings("ignore")


RAW_FEATURE_COLS = [
    "rating_diff", "RD_diff",
    "height_diff", "reach_diff", "age_diff",
    "SLpM", "SApM", "Str_Acc", "Str_Def",
    "TD_Avg", "TD_Acc", "TD_Def", "Sub_Avg",
    "opp_SLpM", "opp_SApM", "opp_Str_Acc", "opp_Str_Def",
    "opp_TD_Avg", "opp_TD_Acc", "opp_TD_Def", "opp_Sub_Avg",
    "fights_before",
    "days_since_last_fight",
    "win_rate_before",
    "recent_win_rate_3",
    "recent_win_rate_5",
    "opp_fights_before",
    "opp_days_since_last_fight",
    "opp_win_rate_before",
    "opp_recent_win_rate_3",
    "opp_recent_win_rate_5",
    "is_debut",
    "opp_is_debut",
]


FINAL_FEATURE_COLS = [
    "rating_diff", "RD_diff",
    "height_diff", "reach_diff", "age_diff",
    "SLpM", "SApM", "Str_Acc", "Str_Def",
    "TD_Avg", "TD_Acc", "TD_Def", "Sub_Avg",
    "opp_SLpM", "opp_SApM", "opp_Str_Acc", "opp_Str_Def",
    "opp_TD_Avg", "opp_TD_Acc", "opp_TD_Def", "opp_Sub_Avg",
    "log_fights_before",
    "log_days_since_last_fight",
    "win_rate_before",
    "recent_win_rate_3",
    "recent_win_rate_5",
    "log_opp_fights_before",
    "log_opp_days_since_last_fight",
    "opp_win_rate_before",
    "opp_recent_win_rate_3",
    "opp_recent_win_rate_5",
    "is_debut",
    "opp_is_debut",
] + [f"{col}_missing" for col in MISSING_COLS]


DEFAULT_PARAMS = {
    "n_estimators": 650,
    "learning_rate": 0.12,
    "max_depth": 3,
    "min_child_weight": 3,
    "gamma": 1.0,
    "subsample": 0.9,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.4,
    "reg_lambda": 0.4,
    "eval_metric": "logloss",
    "tree_method": "hist",
    "random_state": 42,
}


def prepare_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Add missingness indicators and keep only training-relevant columns."""
    df_model = df.copy()
    for col in MISSING_COLS:
        df_model[f"{col}_missing"] = df_model[col].isna().astype(int)
    df_model = df_model[RAW_FEATURE_COLS + [f"{c}_missing" for c in MISSING_COLS] + ["target", "date", "fighter"]].copy()
    df_model[RAW_FEATURE_COLS] = df_model[RAW_FEATURE_COLS].fillna(0)
    df_model["date"] = pd.to_datetime(df_model["date"])
    return df_model


def apply_train_fit_preprocessing(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, float]]]:
    """
    Fit clip bounds on the training split only, then apply clip + log to both.
    """
    train = train.copy()
    test = test.copy()

    clip_config = {
        "days_since_last_fight": (0.001, 0.999),
        "opp_days_since_last_fight": (0.001, 0.999),
        "fights_before": (0.001, 0.999),
        "opp_fights_before": (0.001, 0.999),
    }
    clip_bounds: dict[str, dict[str, float]] = {}

    for col, (lo_q, hi_q) in clip_config.items():
        lo, hi = train[col].quantile([lo_q, hi_q])
        train[col] = train[col].clip(lo, hi)
        test[col] = test[col].clip(lo, hi)
        clip_bounds[col] = {"lo": float(lo), "hi": float(hi)}

    for col in LOG_COLS:
        train[f"log_{col}"] = np.log1p(train[col])
        test[f"log_{col}"] = np.log1p(test[col])

    return train, test, clip_bounds


def tune_xgb_params(X_train: np.ndarray, y_train: np.ndarray, n_trials: int) -> dict:
    """Run the notebook-equivalent Optuna search with time-series CV."""
    if n_trials <= 0:
        return DEFAULT_PARAMS.copy()

    try:
        import optuna
    except ImportError as exc:
        raise ImportError(
            "Optuna is required for tuning. Install dev dependencies or rerun "
            "with --tune-trials 0 to use the baked-in defaults."
        ) from exc

    def objective(trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 500, 750),
            "learning_rate": trial.suggest_float("learning_rate", 0.10, 0.16),
            "max_depth": trial.suggest_int("max_depth", 3, 4),
            "min_child_weight": trial.suggest_int("min_child_weight", 2, 4),
            "gamma": trial.suggest_float("gamma", 0.6, 1.4),
            "subsample": trial.suggest_float("subsample", 0.85, 0.98),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 0.80),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.20, 0.60),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.20, 0.70),
            "eval_metric": "logloss",
            "tree_method": "hist",
            "random_state": 42,
        }

        tscv = TimeSeriesSplit(n_splits=3)
        aucs: list[float] = []

        for tr_idx, va_idx in tscv.split(X_train):
            model = XGBClassifier(**params, early_stopping_rounds=50)
            model.fit(
                X_train[tr_idx],
                y_train[tr_idx],
                eval_set=[(X_train[va_idx], y_train[va_idx])],
                verbose=False,
            )
            preds = model.predict_proba(X_train[va_idx])[:, 1]
            aucs.append(roc_auc_score(y_train[va_idx], preds))

        return float(np.mean(aucs))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_params.update({
        "eval_metric": "logloss",
        "tree_method": "hist",
        "random_state": 42,
    })
    return best_params


def save_training_summary(path: Path, summary: dict) -> None:
    """Persist training metadata for later inspection/debugging."""
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)


def run(
    data_dir: Path,
    split_date: str = "2021-01-01",
    tune_trials: int = 25,
    dry_run: bool = False,
) -> dict:
    """
    Train the XGBoost model from raw UFCStats CSVs and export artifacts.
    """
    print(f"\n{'='*60}")
    print("UFC Model Training")
    print(f"data_dir    : {data_dir}")
    print(f"split_date  : {split_date}")
    print(f"tune_trials : {tune_trials}")
    print(f"dry_run     : {dry_run}")
    print(f"{'='*60}\n")

    print("[1/5] Building feature frame from raw UFCStats data...")
    df = build_feature_frame(data_dir)
    fighters_latest = (
        df.sort_values("date")
        .groupby("fighter", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )

    print("\n[2/5] Preparing train/test splits...")
    df_model = prepare_model_frame(df)
    train = df_model[df_model["date"] < split_date].copy()
    test = df_model[df_model["date"] >= split_date].copy()
    train, test, clip_bounds = apply_train_fit_preprocessing(train, test)

    X_train = train[FINAL_FEATURE_COLS].values
    y_train = train["target"].values
    X_test = test[FINAL_FEATURE_COLS].values
    y_test = test["target"].values

    print(f"  train rows: {len(train):,}")
    print(f"  test rows : {len(test):,}")

    print("\n[3/5] Selecting XGBoost hyperparameters...")
    best_params = tune_xgb_params(X_train, y_train, tune_trials)
    print(f"  params: {best_params}")

    print("\n[4/5] Generating OOF predictions and calibrator...")
    tscv = TimeSeriesSplit(n_splits=5)
    oof_preds = np.zeros(len(X_train))

    for i, (tr_idx, va_idx) in enumerate(tscv.split(X_train), start=1):
        model = XGBClassifier(**best_params, early_stopping_rounds=50)
        model.fit(
            X_train[tr_idx],
            y_train[tr_idx],
            eval_set=[(X_train[va_idx], y_train[va_idx])],
            verbose=False,
        )
        oof_preds[va_idx] = model.predict_proba(X_train[va_idx])[:, 1]
        print(f"  fold {i} AUC: {roc_auc_score(y_train[va_idx], oof_preds[va_idx]):.4f}")

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(oof_preds, y_train)

    print("\n[5/5] Fitting final model and evaluating holdout...")
    final_xgb = XGBClassifier(**best_params)
    final_xgb.fit(X_train, y_train)

    test_probs_raw = final_xgb.predict_proba(X_test)[:, 1]
    test_probs_cal = calibrator.transform(test_probs_raw)
    test_preds = (test_probs_cal >= 0.5).astype(int)

    summary = {
        "split_date": split_date,
        "tune_trials": tune_trials,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "oof_auc": float(roc_auc_score(y_train, oof_preds)),
        "brier_uncalibrated": float(brier_score_loss(y_train, oof_preds)),
        "brier_calibrated": float(brier_score_loss(y_train, calibrator.transform(oof_preds))),
        "test_accuracy": float(accuracy_score(y_test, test_preds)),
        "test_auc": float(roc_auc_score(y_test, test_probs_cal)),
        "test_brier": float(brier_score_loss(y_test, test_probs_cal)),
        "best_params": best_params,
        "feature_cols": FINAL_FEATURE_COLS,
    }

    for key in ["oof_auc", "test_accuracy", "test_auc", "test_brier"]:
        print(f"  {key}: {summary[key]:.4f}")

    if dry_run:
        print("\n[dry-run] Skipping artifact writes.")
        return summary

    MODELS_DIR.mkdir(exist_ok=True)
    with open(CLIP_BOUNDS_PATH, "w") as f:
        json.dump(clip_bounds, f, indent=2)
    with open(FEATURE_COLS_PATH, "w") as f:
        json.dump(FINAL_FEATURE_COLS, f, indent=2)
    joblib.dump(calibrator, CALIBRATOR_PATH)
    final_xgb.save_model(MODEL_PATH)

    pd.DataFrame({
        "oof_xgb_raw": oof_preds,
        "oof_xgb_cal": calibrator.transform(oof_preds),
        "y": y_train,
    }).to_csv(MODELS_DIR / "stack_train.csv", index=False)

    pd.DataFrame({
        "xgb_test_raw": test_probs_raw,
        "xgb_test_cal": test_probs_cal,
        "y": y_test,
    }).to_csv(MODELS_DIR / "stack_test.csv", index=False)

    fighters_latest.to_csv(MODELS_DIR / "fighters_latest.csv", index=False)
    save_training_summary(MODELS_DIR / "xgb_training_summary.json", summary)
    print(f"\nSaved model artifacts → {MODELS_DIR}")
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and export the production XGBoost UFC model artifacts."
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        type=Path,
        help="Directory containing events.csv, fights.csv, fight_totals.csv, fighters_advanced.csv, fighters_fight_history.csv",
    )
    parser.add_argument(
        "--split-date",
        default="2021-01-01",
        help="Time-aware split boundary used for train/test evaluation.",
    )
    parser.add_argument(
        "--tune-trials",
        type=int,
        default=25,
        help="Number of Optuna trials. Set to 0 to use the baked-in defaults.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the training pipeline but skip writing artifact files.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    try:
        run(
            data_dir=args.data_dir,
            split_date=args.split_date,
            tune_trials=args.tune_trials,
            dry_run=args.dry_run,
        )
    except Exception as exc:
        print(f"\n❌ Training failed: {exc}", file=sys.stderr)
        raise
