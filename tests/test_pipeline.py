"""
Tests for pipelines/refresh_fighters.py

Covers the pure functions (parsers, validators, feature builders)
without needing any raw CSV files on disk.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipelines.refresh_fighters import (
    _height_to_inches,
    _parse_float,
    _parse_int,
    _parse_pair,
    _parse_time,
    _reach_to_inches,
    _wl_to_score,
    add_physical_features,
    clean_fight_history,
    validate_output,
)


# ── Parser unit tests ──────────────────────────────────────────────────────────

class TestParsePair:
    def test_of_format(self):
        assert _parse_pair("35 of 67") == (35, 67)

    def test_space_format(self):
        assert _parse_pair("35 67") == (35, 67)

    def test_nan_input(self):
        l, r = _parse_pair(np.nan)
        assert np.isnan(l) and np.isnan(r)

    def test_zero_values(self):
        assert _parse_pair("0 of 0") == (0, 0)


class TestParseInt:
    def test_basic(self):
        assert _parse_int("3") == 3

    def test_garbage(self):
        assert np.isnan(_parse_int("--"))

    def test_float_string(self):
        # int("3.0") raises in Python, so _parse_int returns nan for float strings
        assert np.isnan(_parse_int("3.0"))


class TestParseFloat:
    def test_basic(self):
        assert _parse_float("67.5") == pytest.approx(67.5)

    def test_percent_stripped(self):
        assert _parse_float("67%") == pytest.approx(67.0)

    def test_garbage(self):
        assert np.isnan(_parse_float("--"))


class TestParseTime:
    def test_basic(self):
        assert _parse_time("3:55") == 3 * 60 + 55

    def test_zero(self):
        assert _parse_time("0:00") == 0

    def test_non_string(self):
        assert np.isnan(_parse_time(123))


class TestHeightToInches:
    def test_feet_inches(self):
        assert _height_to_inches("5' 11\"") == 71
        assert _height_to_inches("6' 2\"") == 74

    def test_dash(self):
        assert np.isnan(_height_to_inches("--"))

    def test_nan(self):
        assert np.isnan(_height_to_inches(np.nan))


class TestReachToInches:
    def test_numeric_string(self):
        assert _reach_to_inches('72"') == pytest.approx(72.0)
        assert _reach_to_inches("72") == pytest.approx(72.0)

    def test_dash(self):
        assert np.isnan(_reach_to_inches("--"))

    def test_nan(self):
        assert np.isnan(_reach_to_inches(np.nan))


class TestWlToScore:
    def test_win(self):
        assert _wl_to_score("Win") == 1.0

    def test_loss(self):
        assert _wl_to_score("Loss") == 0.0

    def test_draw(self):
        assert _wl_to_score("Draw") == 0.5

    def test_nc(self):
        assert _wl_to_score("NC") is None

    def test_none(self):
        assert _wl_to_score(None) is None


# ── clean_fight_history ────────────────────────────────────────────────────────

def _make_history_df() -> pd.DataFrame:
    return pd.DataFrame({
        "fighter_name": ["Jon Jones", "Jon Jones", "Stipe Miocic"],
        "opponent": [
            "Jon Jones Daniel Cormier",
            "Jon Jones Alexander Gustafsson",
            "Stipe Miocic Francis Ngannou",
        ],
        "event": [
            "UFC 182: Jones vs. Cormier Jan. 3, 2015",
            "UFC 165: Jones vs. Gustafsson Sept. 21, 2013",
            "UFC 220: Stipe vs. Ngannou Jan. 20, 2018",
        ],
        "wl": ["Win", "Win", "Win"],
        "event_url": ["url1", "url2", "url3"],
    })


def test_clean_fight_history_drops_no_date():
    df = _make_history_df()
    # Add a row with no parseable date
    bad = pd.DataFrame([{
        "fighter_name": "Ghost", "opponent": "Ghost Nobody",
        "event": "Some Unknown Event", "wl": "Win", "event_url": "url4",
    }])
    df = pd.concat([df, bad], ignore_index=True)
    clean = clean_fight_history(df)
    assert "Ghost" not in clean["fighter_name"].values


def test_clean_fight_history_sorted():
    df = _make_history_df()
    clean = clean_fight_history(df)
    dates = pd.to_datetime(clean["date"])
    assert (dates.diff().dropna() >= pd.Timedelta(0)).all()


def test_clean_fight_history_scores():
    df = _make_history_df()
    clean = clean_fight_history(df)
    assert (clean["score"] == 1.0).all()


# ── add_physical_features ─────────────────────────────────────────────────────

def test_add_physical_features_basic():
    df = pd.DataFrame({
        "height": ["6' 0\"", "5' 10\""],
        "opp_height": ["5' 10\"", "6' 0\""],
        "reach": ["72", "70"],
        "opp_reach": ["70", "72"],
    })
    result = add_physical_features(df)
    assert result["fighter_height_inches"].iloc[0] == 72
    assert result["height_diff"].iloc[0] == 2
    assert result["reach_diff"].iloc[0] == 2


def test_add_physical_features_nan_propagation():
    df = pd.DataFrame({
        "height": ["--", "5' 10\""],
        "opp_height": ["5' 10\"", "--"],
        "reach": [np.nan, "70"],
        "opp_reach": ["70", np.nan],
    })
    result = add_physical_features(df)
    assert np.isnan(result["height_diff"].iloc[0])
    assert np.isnan(result["reach_diff"].iloc[0])


# ── validate_output ────────────────────────────────────────────────────────────

def _make_valid_snapshot(n: int = 600) -> pd.DataFrame:
    from pipelines.refresh_fighters import REQUIRED_COLS
    data = {col: np.random.randn(n) for col in REQUIRED_COLS}
    data["fighter"] = [f"Fighter {i}" for i in range(n)]
    data["g_rating_before"] = np.random.uniform(1200, 1800, n)
    return pd.DataFrame(data)


def test_validate_output_passes_clean_data():
    df = _make_valid_snapshot(600)
    errors = validate_output(df)
    assert errors == []


def test_validate_output_too_few_fighters():
    df = _make_valid_snapshot(10)
    errors = validate_output(df)
    assert any("Only" in e for e in errors)


def test_validate_output_missing_column():
    df = _make_valid_snapshot(600).drop(columns=["g_rating_before"])
    errors = validate_output(df)
    assert any("g_rating_before" in e for e in errors)


def test_validate_output_insane_ratings():
    df = _make_valid_snapshot(600)
    df.loc[0, "g_rating_before"] = 9999
    errors = validate_output(df)
    assert any("rating" in e.lower() for e in errors)


def test_validate_output_duplicate_fighters():
    df = _make_valid_snapshot(600)
    df.loc[1, "fighter"] = df.loc[0, "fighter"]  # introduce a duplicate
    errors = validate_output(df)
    assert any("duplicate" in e.lower() for e in errors)
