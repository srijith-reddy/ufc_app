"""
Eligibility checker tests.

These use mock fighters_df + mock booster/calibrator so they can run
without real model artifacts in CI.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.eligibility import (
    EventCoverage,
    FightEligibility,
    check_event_coverage,
    check_fight_eligibility,
)
from core.names import normalize_name


# ── Mock artifacts ─────────────────────────────────────────────────────────────

def _make_fighter_row(name: str, fights_before: int = 10) -> dict:
    """Return a minimal fighter row for the mock DataFrame."""
    return {
        "fighter": name,
        "fighter_norm": normalize_name(name),
        "g_rating_before": 1500.0,
        "g_RD_before": 200.0,
        "fighter_height_inches": 70.0,
        "fighter_reach_inches": 72.0,
        "fighter_age": 28.0,
        "SLpM": 4.0, "SApM": 3.5,
        "Str_Acc": 0.45, "Str_Def": 0.55,
        "TD_Avg": 1.5, "TD_Acc": 0.40, "TD_Def": 0.70, "Sub_Avg": 0.5,
        "fights_before": fights_before,
        "days_since_last_fight": 200.0,
        "win_rate_before": 0.7,
        "recent_win_rate_3": 0.67,
        "recent_win_rate_5": 0.60,
        "is_debut": int(fights_before == 0),
    }


@pytest.fixture
def mock_fighters_df() -> pd.DataFrame:
    rows = [
        _make_fighter_row("Jon Jones"),
        _make_fighter_row("Stipe Miocic"),
        _make_fighter_row("Islam Makhachev"),
        _make_fighter_row("Dustin Poirier"),
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def mock_fighter_lookup(mock_fighters_df) -> dict[str, str]:
    return {row["fighter_norm"]: row["fighter"] for _, row in mock_fighters_df.iterrows()}


@pytest.fixture
def mock_feature_cols() -> list[str]:
    # Import the real feature cols list from config (just the names, no artifacts needed)
    import json
    from core.config import FEATURE_COLS_PATH
    if FEATURE_COLS_PATH.exists():
        with open(FEATURE_COLS_PATH) as f:
            return json.load(f)
    # Minimal fallback so tests don't fail if artifacts absent
    return []


@pytest.fixture
def mock_clip_bounds() -> dict:
    return {
        "days_since_last_fight": {"lo": 0.0, "hi": 2318.93},
        "opp_days_since_last_fight": {"lo": 0.0, "hi": 2318.93},
        "fights_before": {"lo": 0.0, "hi": 29.82},
        "opp_fights_before": {"lo": 0.0, "hi": 29.82},
    }


class MockBooster:
    """Returns a fixed raw probability for any input."""
    def predict(self, dmat):
        return np.array([0.65])


class MockCalibrator:
    """Returns the value unchanged (identity calibration)."""
    def transform(self, x):
        return x


# ── Fighter lookup tests ───────────────────────────────────────────────────────

def test_missing_fighter_a(mock_fighters_df, mock_fighter_lookup, mock_feature_cols, mock_clip_bounds):
    result = check_fight_eligibility(
        "Unknown Fighter X", "Jon Jones",
        mock_fighters_df, mock_fighter_lookup,
        mock_feature_cols, mock_clip_bounds,
        MockBooster(), MockCalibrator(),
    )
    assert result.status == "unsupported_missing_fighter_a"
    assert not result.is_predictable
    assert result.prob_a is None


def test_missing_fighter_b(mock_fighters_df, mock_fighter_lookup, mock_feature_cols, mock_clip_bounds):
    result = check_fight_eligibility(
        "Jon Jones", "Unknown Fighter Y",
        mock_fighters_df, mock_fighter_lookup,
        mock_feature_cols, mock_clip_bounds,
        MockBooster(), MockCalibrator(),
    )
    assert result.status == "unsupported_missing_fighter_b"
    assert not result.is_predictable


def test_both_missing(mock_fighters_df, mock_fighter_lookup, mock_feature_cols, mock_clip_bounds):
    result = check_fight_eligibility(
        "Ghost Fighter A", "Ghost Fighter B",
        mock_fighters_df, mock_fighter_lookup,
        mock_feature_cols, mock_clip_bounds,
        MockBooster(), MockCalibrator(),
    )
    # Only fighter A is checked first — returns missing_fighter_a
    assert result.status == "unsupported_missing_fighter_a"


def test_predictable_fight(mock_fighters_df, mock_fighter_lookup, mock_feature_cols, mock_clip_bounds):
    if not mock_feature_cols:
        pytest.skip("No feature_cols available (artifact absent)")
    result = check_fight_eligibility(
        "Jon Jones", "Stipe Miocic",
        mock_fighters_df, mock_fighter_lookup,
        mock_feature_cols, mock_clip_bounds,
        MockBooster(), MockCalibrator(),
    )
    assert result.status == "predictable"
    assert result.is_predictable
    assert result.prob_a is not None
    assert 0.0 < result.prob_a < 1.0


# ── Event coverage tests ───────────────────────────────────────────────────────

def test_fully_predictable_event(mock_fighters_df, mock_fighter_lookup, mock_feature_cols, mock_clip_bounds):
    if not mock_feature_cols:
        pytest.skip("No feature_cols available")
    fights = [
        ["Jon Jones", "Stipe Miocic"],
        ["Islam Makhachev", "Dustin Poirier"],
    ]
    coverage = check_event_coverage(
        999, fights,
        mock_fighters_df, mock_fighter_lookup,
        mock_feature_cols, mock_clip_bounds,
        MockBooster(), MockCalibrator(),
    )
    assert coverage.status == "fully_predictable"
    assert len(coverage.predictable_fights) == 2
    assert len(coverage.unpredictable_fights) == 0


def test_partially_predictable_event(mock_fighters_df, mock_fighter_lookup, mock_feature_cols, mock_clip_bounds):
    if not mock_feature_cols:
        pytest.skip("No feature_cols available")
    fights = [
        ["Jon Jones", "Stipe Miocic"],       # predictable
        ["Unknown A", "Unknown B"],           # not predictable
    ]
    coverage = check_event_coverage(
        998, fights,
        mock_fighters_df, mock_fighter_lookup,
        mock_feature_cols, mock_clip_bounds,
        MockBooster(), MockCalibrator(),
    )
    assert coverage.status == "partially_predictable"
    assert len(coverage.predictable_fights) == 1
    assert len(coverage.unpredictable_fights) == 1


def test_not_predictable_event(mock_fighters_df, mock_fighter_lookup, mock_feature_cols, mock_clip_bounds):
    fights = [
        ["Ghost A", "Ghost B"],
        ["Ghost C", "Ghost D"],
    ]
    coverage = check_event_coverage(
        997, fights,
        mock_fighters_df, mock_fighter_lookup,
        mock_feature_cols, mock_clip_bounds,
        MockBooster(), MockCalibrator(),
    )
    assert coverage.status == "not_predictable"
    assert len(coverage.predictable_fights) == 0


def test_empty_card(mock_fighters_df, mock_fighter_lookup, mock_feature_cols, mock_clip_bounds):
    coverage = check_event_coverage(
        996, [],
        mock_fighters_df, mock_fighter_lookup,
        mock_feature_cols, mock_clip_bounds,
        MockBooster(), MockCalibrator(),
    )
    assert coverage.status == "not_predictable"
    assert coverage.fights == []


def test_event_coverage_dataclass_helpers(mock_fighters_df, mock_fighter_lookup, mock_feature_cols, mock_clip_bounds):
    if not mock_feature_cols:
        pytest.skip("No feature_cols available")
    fights = [
        ["Jon Jones", "Stipe Miocic"],
        ["Unknown Ghost", "Jon Jones"],
    ]
    coverage = check_event_coverage(
        995, fights,
        mock_fighters_df, mock_fighter_lookup,
        mock_feature_cols, mock_clip_bounds,
        MockBooster(), MockCalibrator(),
    )
    assert isinstance(coverage.predictable_fights, list)
    assert isinstance(coverage.unpredictable_fights, list)
    total = len(coverage.predictable_fights) + len(coverage.unpredictable_fights)
    assert total == len(coverage.fights)
