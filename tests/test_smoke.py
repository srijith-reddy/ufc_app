"""
Smoke tests — import safety and basic sanity.

These tests run without any model artifacts and cover the pure logic
modules that can be tested in CI even without the 4.5 MB fighter CSV.
"""
from __future__ import annotations

import math
import pytest


# ── Import smoke ───────────────────────────────────────────────────────────────

def test_core_names_import():
    from core.names import normalize_name, name_aliases
    assert callable(normalize_name)
    assert callable(name_aliases)


def test_core_betting_import():
    from core.betting import (
        american_to_decimal,
        ev_per_dollar,
        implied_prob_from_decimal,
        kelly_fraction,
        kelly_note,
        safe_float,
        simulate_bankroll_paths,
    )
    for fn in [american_to_decimal, ev_per_dollar, implied_prob_from_decimal,
               kelly_fraction, kelly_note, safe_float, simulate_bankroll_paths]:
        assert callable(fn)


def test_core_config_import():
    from core.config import (
        CARDS_DIR, CLIP_BOUNDS_PATH, FEATURE_COLS_PATH,
        FIGHTERS_PATH, LOG_COLS, MISSING_COLS, MODEL_PATH, MODELS_DIR,
        ODDS_DIR, REPO_ROOT,
    )
    assert REPO_ROOT.is_dir()
    assert len(LOG_COLS) == 4
    assert len(MISSING_COLS) == 12


def test_core_event_import():
    from core.event import (
        get_odds_for_fighter,
        list_available_events,
        load_fight_card,
        load_odds_map,
    )
    assert callable(load_fight_card)
    assert callable(load_odds_map)


def test_core_eligibility_import():
    from core.eligibility import (
        EventCoverage,
        FightEligibility,
        check_event_coverage,
        check_fight_eligibility,
    )
    assert callable(check_fight_eligibility)
    assert callable(check_event_coverage)


def test_core_package_import():
    """Verify the top-level core package exports everything it claims to."""
    import core
    for name in [
        "normalize_name", "name_aliases",
        "american_to_decimal", "ev_per_dollar", "kelly_fraction",
        "load_fight_card", "load_odds_map", "get_odds_for_fighter",
        "check_fight_eligibility", "check_event_coverage",
        "FightEligibility", "EventCoverage",
    ]:
        assert hasattr(core, name), f"core.{name} missing"


# ── Name normalization ─────────────────────────────────────────────────────────

def test_normalize_name_basic():
    from core.names import normalize_name
    assert normalize_name("JUSTIN GAETHJE") == "justin gaethje"
    assert normalize_name("Paddy Pimblett") == "paddy pimblett"


def test_normalize_name_unicode():
    from core.names import normalize_name
    assert normalize_name("José Aldo") == "jose aldo"
    assert normalize_name("Ilia Topuria") == "ilia topuria"


def test_normalize_name_strips_special():
    from core.names import normalize_name
    assert normalize_name("O'Malley") == "omalley"
    assert normalize_name("Cortés-Acosta") == "cortesacosta"


def test_name_aliases_contains_base():
    from core.names import name_aliases
    aliases = name_aliases("Justin Gaethje")
    assert "justin gaethje" in aliases


def test_name_aliases_compound_surname():
    from core.names import name_aliases
    aliases = name_aliases("WALDO CORTES ACOSTA")
    assert "waldo cortes acosta" in aliases
    assert "waldo cortesacosta" in aliases
    assert "cortes acosta waldo" in aliases


def test_name_aliases_two_part():
    from core.names import name_aliases
    aliases = name_aliases("Sean O'Malley")
    assert "sean omalley" in aliases
    assert "omalley sean" in aliases


# ── Betting math ───────────────────────────────────────────────────────────────

def test_american_to_decimal_positive():
    from core.betting import american_to_decimal
    assert american_to_decimal(200) == pytest.approx(3.0)


def test_american_to_decimal_negative():
    from core.betting import american_to_decimal
    assert american_to_decimal(-200) == pytest.approx(1.5)


def test_american_to_decimal_zero():
    from core.betting import american_to_decimal
    import math
    assert math.isnan(american_to_decimal(0))


def test_ev_positive():
    from core.betting import ev_per_dollar
    # p=0.65, d=2.0 → EV = 0.65*2 - 1 = 0.30
    assert ev_per_dollar(0.65, 2.0) == pytest.approx(0.30)


def test_ev_negative():
    from core.betting import ev_per_dollar
    assert ev_per_dollar(0.4, 2.0) == pytest.approx(-0.20)


def test_kelly_fraction_basic():
    from core.betting import kelly_fraction
    # p=0.6, d=2.0 → b=1, Kelly = (1*0.6 - 0.4)/1 = 0.2
    assert kelly_fraction(0.6, 2.0) == pytest.approx(0.2)


def test_kelly_fraction_no_edge():
    from core.betting import kelly_fraction
    # Negative edge → Kelly = 0
    assert kelly_fraction(0.3, 2.0) == 0.0


def test_kelly_fraction_zero_odds():
    from core.betting import kelly_fraction
    assert kelly_fraction(0.6, 0.0) == 0.0


def test_simulate_bankroll_empty():
    from core.betting import simulate_bankroll_paths
    result = simulate_bankroll_paths([], 1000, 100, 0.5)
    assert result.size == 0


def test_simulate_bankroll_shape():
    from core.betting import simulate_bankroll_paths
    bets = [{"p": 0.6, "decimal_odds": 2.0, "stake_frac_full_kelly": 0.1}]
    result = simulate_bankroll_paths(bets, 1000, 200, 0.5)
    assert result.shape == (200,)
    assert (result >= 0).all()


# ── Event loading (no artifacts needed) ───────────────────────────────────────

def test_load_ufc324_card():
    from core.event import load_fight_card
    fights = load_fight_card(324)
    assert len(fights) > 0
    assert all(len(f) == 2 for f in fights)


def test_load_nonexistent_event():
    from core.event import load_fight_card
    import pytest
    with pytest.raises(FileNotFoundError):
        load_fight_card(99999)


def test_list_available_events():
    from core.event import list_available_events
    events = list_available_events()
    assert isinstance(events, list)
    assert 324 in events


def test_load_odds_map_missing_returns_empty():
    from core.event import load_odds_map
    result = load_odds_map(99999)
    assert result == {}


def test_get_odds_for_fighter_with_alias():
    from core.event import get_odds_for_fighter
    fake_odds = {
        "justin gaethje": [-335, -300],
        "gaethje justin": [-335],
    }
    result = get_odds_for_fighter("JUSTIN GAETHJE", fake_odds)
    assert -335 in result
    assert -300 in result
    assert result == sorted(set(result))


import pytest
