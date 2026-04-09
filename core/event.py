"""
Event card and odds loading utilities.

All I/O paths go through core.config — no hardcoded strings anywhere.
"""
from __future__ import annotations
import json

import numpy as np

from .config import CARDS_DIR, ODDS_DIR
from .names import name_aliases


# ── Fight Card ─────────────────────────────────────────────────────────────────

def load_fight_card(event_number: int) -> list[list[str]]:
    """
    Load the fight card JSON for a given event.

    Returns list of [fighter_a, fighter_b] pairs (uppercase names as scraped).

    Raises:
        FileNotFoundError with a helpful message if the card file is absent.
        Run `python scrape_cards.py <event_number>` to populate it.
    """
    path = CARDS_DIR / f"ufc_{event_number}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No fight card for UFC {event_number} at {path}.\n"
            f"Run: python scrape_cards.py {event_number}"
        )
    with open(path) as f:
        return json.load(f)


def list_available_events() -> list[int]:
    """
    Return sorted list of event numbers for which a local fight card exists.
    Only events with a cards/ufc_NNN.json file are included.
    """
    events: list[int] = []
    for p in CARDS_DIR.glob("ufc_*.json"):
        try:
            n = int(p.stem.split("_")[1])
            events.append(n)
        except (IndexError, ValueError):
            continue
    return sorted(events)


# ── Odds ───────────────────────────────────────────────────────────────────────

def load_odds_map(event_number: int) -> dict[str, list[int]]:
    """
    Load the odds JSON for a given event.

    Returns the `odds` dict (fighter alias → list of American odds values).
    Returns an empty dict if the file doesn't exist — odds are optional;
    predictions can still be made without them.
    """
    path = ODDS_DIR / f"ufc_{event_number}.json"
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    return data.get("odds", {})


def get_odds_for_fighter(name: str, odds_map: dict) -> list[int]:
    """
    Union odds across all name aliases for robust lookup.

    Checks every variant produced by name_aliases() and merges any matching
    entries. Returns a sorted, deduplicated list of valid American odds
    (-5000 < odds < 5000, odds != 0).

    Returns empty list if no odds are found.
    """
    found: set[int] = set()
    for alias in name_aliases(name):
        if alias in odds_map and odds_map[alias]:
            try:
                found.update(list(odds_map[alias]))
            except TypeError:
                for v in odds_map[alias]:
                    found.add(v)

    cleaned = [
        int(o)
        for o in found
        if isinstance(o, (int, np.integer)) and -5000 < int(o) < 5000 and int(o) != 0
    ]
    return sorted(set(cleaned))
