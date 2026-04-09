"""
Shared fixtures for all tests.

Model artifacts are loaded once per session using pytest.fixture(scope="session")
to avoid the ~3–5 second startup cost on every test function.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
CARDS_DIR = REPO_ROOT / "cards"


# ── Artifact availability guard ────────────────────────────────────────────────

def artifacts_available() -> bool:
    """Return True only if all required model files are present on disk."""
    required = [
        MODELS_DIR / "xgb_prefight_model.json",
        MODELS_DIR / "xgb_calibrator.pkl",
        MODELS_DIR / "xgb_feature_cols.json",
        MODELS_DIR / "clip_bounds.json",
        MODELS_DIR / "fighters_latest.csv",
    ]
    return all(p.exists() for p in required)


ARTIFACTS_PRESENT = artifacts_available()

requires_artifacts = pytest.mark.skipif(
    not ARTIFACTS_PRESENT,
    reason="Model artifacts not present — skipping inference tests",
)


# ── Session-scoped fixtures ────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def loaded_artifacts():
    """
    Load model artifacts once for the entire test session.
    Tests that need the full model should use this fixture.
    """
    if not ARTIFACTS_PRESENT:
        pytest.skip("Model artifacts not present")
    from core import load_artifacts
    return load_artifacts()


@pytest.fixture(scope="session")
def feature_cols():
    path = MODELS_DIR / "xgb_feature_cols.json"
    if not path.exists():
        pytest.skip("xgb_feature_cols.json not present")
    with open(path) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def clip_bounds():
    path = MODELS_DIR / "clip_bounds.json"
    if not path.exists():
        pytest.skip("clip_bounds.json not present")
    with open(path) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def ufc324_card():
    """Return the fight card for UFC 324 (always present in repo)."""
    path = CARDS_DIR / "ufc_324.json"
    if not path.exists():
        pytest.skip("ufc_324.json not present")
    with open(path) as f:
        return json.load(f)


@pytest.fixture
def minimal_fighter_row() -> pd.Series:
    """
    A minimal synthetic fighter row with all required inference columns.
    Does NOT come from the real CSV — safe to use without artifacts.
    """
    return pd.Series({
        "fighter": "Test Fighter",
        "g_rating_before": 1500.0,
        "g_RD_before": 200.0,
        "fighter_height_inches": 70.0,
        "fighter_reach_inches": 72.0,
        "fighter_age": 28.0,
        "SLpM": 4.0,
        "SApM": 3.5,
        "Str_Acc": 0.45,
        "Str_Def": 0.55,
        "TD_Avg": 1.5,
        "TD_Acc": 0.40,
        "TD_Def": 0.70,
        "Sub_Avg": 0.5,
        "fights_before": 10,
        "days_since_last_fight": 200.0,
        "win_rate_before": 0.7,
        "recent_win_rate_3": 0.67,
        "recent_win_rate_5": 0.60,
        "is_debut": 0,
    })
