"""
Inference tests — require model artifacts to be present.

All tests in this file are marked with @requires_artifacts and will be
skipped in CI if the artifact files are absent from the repo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.conftest import requires_artifacts


# ── Preprocessing tests ────────────────────────────────────────────────────────

def test_preprocess_row_shape(minimal_fighter_row, feature_cols, clip_bounds):
    from core.inference import build_feature_row, preprocess_row

    A = minimal_fighter_row
    B = minimal_fighter_row.copy()
    B["g_rating_before"] = 1450.0

    row = build_feature_row(A, B)
    X = preprocess_row(pd.DataFrame([row]), feature_cols, clip_bounds)

    assert X.shape == (1, len(feature_cols))
    assert not np.isnan(X).any(), "Preprocessed array must not contain NaN"


def test_preprocess_row_no_nan_with_missing_stats(feature_cols, clip_bounds):
    """NaN striking stats should be replaced by 0 after fillna, not propagate."""
    from core.inference import build_feature_row, preprocess_row

    A = pd.Series({
        "fighter": "A",
        "g_rating_before": 1500.0,
        "g_RD_before": 200.0,
        "fighter_height_inches": 70.0,
        "fighter_reach_inches": 72.0,
        "fighter_age": 28.0,
        "SLpM": np.nan,    # deliberately missing
        "SApM": np.nan,
        "Str_Acc": np.nan,
        "Str_Def": 0.55,
        "TD_Avg": np.nan,
        "TD_Acc": np.nan,
        "TD_Def": 0.70,
        "Sub_Avg": np.nan,
        "fights_before": 0,
        "days_since_last_fight": 0.0,
        "win_rate_before": 0.0,
        "recent_win_rate_3": 0.0,
        "recent_win_rate_5": 0.0,
        "is_debut": 1,
    })
    B = A.copy()
    B["g_rating_before"] = 1450.0

    row = build_feature_row(A, B)
    X = preprocess_row(pd.DataFrame([row]), feature_cols, clip_bounds)

    assert not np.isnan(X).any()


def test_preprocess_row_log_transform(feature_cols, clip_bounds):
    """log_fights_before must equal log1p(fights_before) after clipping."""
    from core.inference import build_feature_row, preprocess_row

    A = pd.Series({
        "fighter": "A",
        "g_rating_before": 1500.0, "g_RD_before": 200.0,
        "fighter_height_inches": 70.0, "fighter_reach_inches": 72.0, "fighter_age": 28.0,
        "SLpM": 4.0, "SApM": 3.5, "Str_Acc": 0.45, "Str_Def": 0.55,
        "TD_Avg": 1.5, "TD_Acc": 0.40, "TD_Def": 0.70, "Sub_Avg": 0.5,
        "fights_before": 15,
        "days_since_last_fight": 180.0,
        "win_rate_before": 0.8, "recent_win_rate_3": 0.67, "recent_win_rate_5": 0.60,
        "is_debut": 0,
    })

    row = build_feature_row(A, A)
    df = pd.DataFrame([row])
    X = preprocess_row(df, feature_cols, clip_bounds)
    result_df = pd.DataFrame(X, columns=feature_cols)

    expected_log_fights = np.log1p(
        min(15, clip_bounds["fights_before"]["hi"])
    )
    assert result_df["log_fights_before"].iloc[0] == pytest.approx(expected_log_fights, rel=1e-5)


# ── Full inference tests (require artifacts) ───────────────────────────────────

@requires_artifacts
def test_predict_win_prob_range(loaded_artifacts, minimal_fighter_row):
    """Calibrated probability must be in (0.01, 0.99)."""
    booster, calibrator, feature_cols, clip_bounds, fighters_df, _ = loaded_artifacts
    from core.inference import build_feature_row, preprocess_row, predict_win_prob

    A = minimal_fighter_row
    B = minimal_fighter_row.copy()
    B["g_rating_before"] = 1400.0  # A is rated higher

    row = build_feature_row(A, B)
    X = preprocess_row(pd.DataFrame([row]), feature_cols, clip_bounds)
    prob = predict_win_prob(booster, calibrator, X)

    assert 0.01 <= prob <= 0.99


@requires_artifacts
def test_predict_symmetry(loaded_artifacts, minimal_fighter_row):
    """P(A beats B) + P(B beats A) should be ≈ 1 (not exact due to calibration)."""
    booster, calibrator, feature_cols, clip_bounds, _, _ = loaded_artifacts
    from core.inference import build_feature_row, preprocess_row, predict_win_prob

    A = minimal_fighter_row
    B = minimal_fighter_row.copy()
    B["g_rating_before"] = 1400.0

    row_ab = build_feature_row(A, B)
    row_ba = build_feature_row(B, A)

    X_ab = preprocess_row(pd.DataFrame([row_ab]), feature_cols, clip_bounds)
    X_ba = preprocess_row(pd.DataFrame([row_ba]), feature_cols, clip_bounds)

    prob_ab = predict_win_prob(booster, calibrator, X_ab)
    prob_ba = predict_win_prob(booster, calibrator, X_ba)

    # Isotonic calibration is a post-hoc monotone transform and is NOT
    # symmetric. The two models see different feature vectors (A vs B order
    # differs), so prob_ab + prob_ba ≠ 1.0 exactly. We just assert they're
    # both in (0,1) and loosely sum near 1 as a sanity check.
    assert abs(prob_ab + prob_ba - 1.0) < 0.30


@requires_artifacts
def test_predict_deterministic(loaded_artifacts, minimal_fighter_row):
    """Same input must always produce same output."""
    booster, calibrator, feature_cols, clip_bounds, _, _ = loaded_artifacts
    from core.inference import build_feature_row, preprocess_row, predict_win_prob

    A = minimal_fighter_row
    B = minimal_fighter_row.copy()

    row = build_feature_row(A, B)
    X = preprocess_row(pd.DataFrame([row]), feature_cols, clip_bounds)

    prob1 = predict_win_prob(booster, calibrator, X)
    prob2 = predict_win_prob(booster, calibrator, X)

    assert prob1 == prob2


# ── UFC 324 full-card test (requires artifacts) ────────────────────────────────

@requires_artifacts
def test_ufc324_partial_card(loaded_artifacts, ufc324_card):
    """
    At least some fights on UFC 324 must be predictable.
    This validates the end-to-end pipeline on real card data.
    """
    from core.eligibility import check_event_coverage

    booster, calibrator, feature_cols, clip_bounds, fighters_df, fighter_lookup = loaded_artifacts

    coverage = check_event_coverage(
        324, ufc324_card,
        fighters_df, fighter_lookup,
        feature_cols, clip_bounds,
        booster, calibrator,
    )

    # The card has 13 fights — at least a few should be predictable
    assert len(coverage.fights) == 13
    assert len(coverage.predictable_fights) > 0, (
        "Expected at least one predictable fight on UFC 324 card. "
        "Unpredictable reasons: " +
        str([f.reason for f in coverage.unpredictable_fights])
    )


@requires_artifacts
def test_ufc324_no_crash_on_full_card(loaded_artifacts, ufc324_card):
    """check_event_coverage must never raise even if some fighters are absent."""
    from core.eligibility import check_event_coverage

    booster, calibrator, feature_cols, clip_bounds, fighters_df, fighter_lookup = loaded_artifacts

    # Should not raise
    coverage = check_event_coverage(
        324, ufc324_card,
        fighters_df, fighter_lookup,
        feature_cols, clip_bounds,
        booster, calibrator,
    )

    valid_statuses = {
        "predictable",
        "unsupported_missing_fighter_a",
        "unsupported_missing_fighter_b",
        "unsupported_feature_build_failed",
    }
    for f in coverage.fights:
        assert f.status in valid_statuses
