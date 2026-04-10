"""
Tests for the new automation helpers.

These stay pure/unit-level so they can run without network access or model
artifacts.
"""
from __future__ import annotations

import pandas as pd

from pipelines.refresh_fighters import add_temporal_features
from pipelines.scrape_ufcstats import merge_records, parse_scheduled_rounds
from pipelines.train_model import apply_train_fit_preprocessing, prepare_model_frame


def test_parse_scheduled_rounds_matches_notebook_behavior():
    assert parse_scheduled_rounds("5") == 5
    assert parse_scheduled_rounds("3") == 3
    assert parse_scheduled_rounds("Round 5") == 5
    assert parse_scheduled_rounds("Round 2") == 3


def test_merge_records_prefers_new_rows_on_duplicate_key():
    existing = pd.DataFrame([
        {"fight_url": "u1", "fighter_A": "Old A"},
        {"fight_url": "u2", "fighter_A": "Old B"},
    ])
    new = pd.DataFrame([
        {"fight_url": "u2", "fighter_A": "New B"},
        {"fight_url": "u3", "fighter_A": "New C"},
    ])

    merged = merge_records(existing, new, ["fight_url"])

    assert len(merged) == 3
    assert merged.loc[merged["fight_url"] == "u2", "fighter_A"].iloc[0] == "New B"


def test_prepare_model_frame_adds_missingness_flags():
    df = pd.DataFrame([{
        "rating_diff": 10.0,
        "RD_diff": 5.0,
        "height_diff": 1.0,
        "reach_diff": 2.0,
        "age_diff": -1.0,
        "SLpM": None,
        "SApM": 3.0,
        "Str_Acc": None,
        "Str_Def": 0.55,
        "TD_Avg": None,
        "TD_Acc": None,
        "TD_Def": 0.70,
        "Sub_Avg": None,
        "opp_SLpM": 4.0,
        "opp_SApM": 2.0,
        "opp_Str_Acc": None,
        "opp_Str_Def": 0.60,
        "opp_TD_Avg": None,
        "opp_TD_Acc": None,
        "opp_TD_Def": 0.80,
        "opp_Sub_Avg": None,
        "fights_before": 10,
        "days_since_last_fight": 150.0,
        "win_rate_before": 0.7,
        "recent_win_rate_3": 0.66,
        "recent_win_rate_5": 0.60,
        "opp_fights_before": 8,
        "opp_days_since_last_fight": 120.0,
        "opp_win_rate_before": 0.5,
        "opp_recent_win_rate_3": 0.33,
        "opp_recent_win_rate_5": 0.40,
        "is_debut": 0,
        "opp_is_debut": 0,
        "target": 1,
        "date": "2024-01-01",
        "fighter": "Test Fighter",
    }])

    prepared = prepare_model_frame(df)

    assert prepared["SLpM_missing"].iloc[0] == 1
    assert prepared["Str_Acc_missing"].iloc[0] == 1
    assert prepared["opp_SLpM_missing"].iloc[0] == 0
    assert prepared["TD_Avg_missing"].iloc[0] == 1
    assert prepared["SLpM"].iloc[0] == 0


def test_apply_train_fit_preprocessing_adds_log_columns_and_bounds():
    train = pd.DataFrame([{
        "days_since_last_fight": 100.0,
        "opp_days_since_last_fight": 50.0,
        "fights_before": 10,
        "opp_fights_before": 8,
    }, {
        "days_since_last_fight": 200.0,
        "opp_days_since_last_fight": 80.0,
        "fights_before": 20,
        "opp_fights_before": 12,
    }])
    test = pd.DataFrame([{
        "days_since_last_fight": 1000.0,
        "opp_days_since_last_fight": 900.0,
        "fights_before": 100,
        "opp_fights_before": 90,
    }])

    train_out, test_out, clip_bounds = apply_train_fit_preprocessing(train, test)

    assert "log_days_since_last_fight" in train_out.columns
    assert "log_opp_fights_before" in test_out.columns
    assert set(clip_bounds) == {
        "days_since_last_fight",
        "opp_days_since_last_fight",
        "fights_before",
        "opp_fights_before",
    }
    assert test_out["days_since_last_fight"].iloc[0] <= clip_bounds["days_since_last_fight"]["hi"]


def test_add_temporal_features_keeps_wins_before_fighter_local():
    df = pd.DataFrame([
        {
            "fight_url": "f1",
            "fighter": "Alpha",
            "opponent": "Opp 1",
            "result": "win",
            "date": "2020-01-01",
            "g_rating_before": 1500.0,
            "opp_rating_before": 1500.0,
        },
        {
            "fight_url": "f2",
            "fighter": "Alpha",
            "opponent": "Opp 2",
            "result": "loss",
            "date": "2020-02-01",
            "g_rating_before": 1510.0,
            "opp_rating_before": 1490.0,
        },
        {
            "fight_url": "f3",
            "fighter": "Bravo",
            "opponent": "Opp 3",
            "result": "loss",
            "date": "2020-01-05",
            "g_rating_before": 1490.0,
            "opp_rating_before": 1510.0,
        },
        {
            "fight_url": "f4",
            "fighter": "Bravo",
            "opponent": "Opp 4",
            "result": "win",
            "date": "2020-02-05",
            "g_rating_before": 1505.0,
            "opp_rating_before": 1495.0,
        },
    ])

    enriched = add_temporal_features(df)
    bravo_rows = enriched[enriched["fighter"] == "Bravo"].reset_index(drop=True)

    assert pd.isna(bravo_rows.loc[0, "wins_before"])
    assert bravo_rows.loc[1, "wins_before"] == 0.0
    assert bravo_rows.loc[0, "win_rate_before"] == 0.0
    assert bravo_rows.loc[1, "win_rate_before"] == 0.0
