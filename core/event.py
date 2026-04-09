"""
Event card and odds loading utilities.

All I/O paths go through core.config — no hardcoded strings anywhere.
"""
from __future__ import annotations
import json
import re
from pathlib import Path

import numpy as np

from .config import CARDS_DIR, ODDS_DIR
from .names import name_aliases


def normalize_event_id(event_ref: int | str) -> str:
    """
    Normalize an event reference into the canonical product event id.

    Examples:
      324                             -> ufc-324
      "324"                           -> ufc-324
      "ufc_324"                       -> ufc-324
      "ufc-fight-night-emmett-vs-x"   -> ufc-fight-night-emmett-vs-x
      "https://www.ufc.com/event/..." -> slug after /event/
    """
    if isinstance(event_ref, int):
        return f"ufc-{event_ref}"

    ref = str(event_ref).strip().lower()
    ref = ref.replace("https://www.ufc.com/event/", "").strip("/")

    if ref.isdigit():
        return f"ufc-{ref}"

    ref = ref.replace("_", "-")
    ref = re.sub(r"[^a-z0-9-]+", "-", ref)
    ref = re.sub(r"-{2,}", "-", ref).strip("-")

    if not ref.startswith("ufc"):
        raise ValueError(f"Unsupported event reference: {event_ref}")

    return ref


def event_number_from_id(event_id: str) -> int | None:
    """Return the numbered UFC event if the id is of the form `ufc-324`."""
    match = re.fullmatch(r"ufc-(\d+)", event_id)
    return int(match.group(1)) if match else None


def format_event_title(event_id: str) -> str:
    """Generate a presentable title from a canonical event id."""
    event_number = event_number_from_id(event_id)
    if event_number is not None:
        return f"UFC {event_number}"

    if event_id.startswith("ufc-fight-night-"):
        main = event_id.removeprefix("ufc-fight-night-")
        main = main.replace("-vs-", " vs ")
        return f"UFC Fight Night: {main.replace('-', ' ').title()}"

    if event_id.startswith("ufc-on-espn-"):
        main = event_id.removeprefix("ufc-on-espn-")
        main = main.replace("-vs-", " vs ")
        return f"UFC on ESPN: {main.replace('-', ' ').title()}"

    if event_id.startswith("ufc-on-abc-"):
        main = event_id.removeprefix("ufc-on-abc-")
        main = main.replace("-vs-", " vs ")
        return f"UFC on ABC: {main.replace('-', ' ').title()}"

    pretty = event_id.replace("-vs-", " vs ").replace("-", " ").title()
    return pretty.replace("Ufc", "UFC")


def _candidate_event_paths(base_dir: Path, event_ref: int | str) -> list[Path]:
    """
    Return the candidate file paths for an event.

    Numbered UFC events keep backward compatibility with `ufc_324.json`.
    Fight Nights and newer slug-based events use `<slug>.json`.
    """
    event_id = normalize_event_id(event_ref)
    candidates = [base_dir / f"{event_id}.json"]

    event_number = event_number_from_id(event_id)
    if event_number is not None:
        candidates.insert(0, base_dir / f"ufc_{event_number}.json")

    return candidates


def _resolve_event_path(base_dir: Path, event_ref: int | str) -> Path | None:
    for path in _candidate_event_paths(base_dir, event_ref):
        if path.exists():
            return path
    return None


def _event_id_from_path(path: Path) -> str:
    stem = path.stem
    if re.fullmatch(r"ufc_\d+", stem):
        return stem.replace("_", "-")
    return normalize_event_id(stem)


# ── Fight Card ─────────────────────────────────────────────────────────────────

def load_fight_card(event_ref: int | str) -> list[list[str]]:
    """
    Load the fight card JSON for a given event id or numbered UFC event.

    Returns list of [fighter_a, fighter_b] pairs (uppercase names as scraped).

    Raises:
        FileNotFoundError with a helpful message if the card file is absent.
        Run `python scrape_cards.py <event_ref>` to populate it.
    """
    event_id = normalize_event_id(event_ref)
    path = _resolve_event_path(CARDS_DIR, event_id)
    if path is None:
        raise FileNotFoundError(
            f"No fight card for event '{event_id}' in {CARDS_DIR}.\n"
            f"Run: python scrape_cards.py {event_id}"
        )
    with open(path) as f:
        return json.load(f)


def list_available_events() -> list[int]:
    """
    Return sorted numbered UFC events for which a local fight card exists.

    This is preserved for legacy consumers. For the product surface, use
    `list_available_event_items()` so Fight Nights are included too.
    """
    events: list[int] = []
    for item in list_available_event_items():
        if item["event_number"] is not None:
            events.append(item["event_number"])
    return sorted(events)


def list_available_event_ids() -> list[str]:
    """Return all locally available event ids, including Fight Nights."""
    return [item["event_id"] for item in list_available_event_items()]


def list_available_event_items() -> list[dict]:
    """
    Return locally available event metadata derived from card and odds filenames.

    Each item includes:
      event_id, event_number, title, card_path, odds_path, has_card, has_odds
    """
    items_by_id: dict[str, dict] = {}

    for base_dir, kind in [(CARDS_DIR, "card"), (ODDS_DIR, "odds")]:
        for path in sorted(base_dir.glob("*.json")):
            try:
                event_id = _event_id_from_path(path)
            except ValueError:
                continue

            event_number = event_number_from_id(event_id)
            item = items_by_id.setdefault(
                event_id,
                {
                    "event_id": event_id,
                    "event_number": event_number,
                    "title": format_event_title(event_id),
                    "card_path": None,
                    "odds_path": None,
                    "has_card": False,
                    "has_odds": False,
                },
            )

            if kind == "card":
                item["card_path"] = str(path)
                item["has_card"] = True
            else:
                item["odds_path"] = str(path)
                item["has_odds"] = True

    def sort_key(item: dict) -> tuple[int, int, str]:
        event_number = item["event_number"]
        return (
            1 if event_number is not None else 0,
            event_number or 0,
            item["event_id"],
        )

    return sorted(items_by_id.values(), key=sort_key, reverse=True)


# ── Odds ───────────────────────────────────────────────────────────────────────

def load_odds_payload(event_ref: int | str) -> dict:
    """Load the raw odds payload for an event, or an empty dict when missing."""
    path = _resolve_event_path(ODDS_DIR, event_ref)
    if path is None:
        return {}
    with open(path) as f:
        return json.load(f)


def load_odds_map(event_ref: int | str) -> dict[str, list[int]]:
    """
    Load the odds JSON for a given event id or numbered UFC event.

    Returns the `odds` dict (fighter alias → list of American odds values).
    Returns an empty dict if the file doesn't exist — odds are optional;
    predictions can still be made without them.
    """
    data = load_odds_payload(event_ref)
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
