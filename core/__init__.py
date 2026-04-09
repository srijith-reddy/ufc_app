"""
core/ — shared UFC prediction logic.

Importable by the Streamlit app, FastAPI backend, CLI scripts, and tests.
Nothing in this package imports from streamlit or any UI framework.
"""
from .names import normalize_name, name_aliases
from .betting import (
    american_to_decimal,
    implied_prob_from_decimal,
    ev_per_dollar,
    kelly_fraction,
    kelly_note,
    safe_float,
    simulate_bankroll_paths,
)
from .inference import (
    load_artifacts,
    preprocess_row,
    predict_win_prob,
    build_feature_row,
)
from .eligibility import (
    check_fight_eligibility,
    check_event_coverage,
    FightEligibility,
    EventCoverage,
)
from .event import (
    load_fight_card,
    load_odds_map,
    get_odds_for_fighter,
    list_available_events,
)
