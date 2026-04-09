"""
scrape_odds.py — fetch UFC/MMA moneyline odds from The Odds API.

Usage:
    python scrape_odds.py                        # refresh all locally tracked card events
    python scrape_odds.py <EVENT_REF> [...]      # refresh one or more specific events

Required env:
    ODDS_API_KEY=...                             # The Odds API key

Output:
    odds/ufc_<event_number>.json for numbered cards
    odds/<event_slug>.json for Fight Nights and other slug events
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

from core.config import ODDS_DIR
from core.event import (
    event_number_from_id,
    list_available_event_items,
    load_fight_card,
    normalize_event_id,
)
from core.names import normalize_name

ODDS_DIR.mkdir(exist_ok=True)

ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4/sports/mma_mixed_martial_arts/odds"
ODDS_API_SOURCE = "https://the-odds-api.com"


def _output_path_for_event(event_id: str) -> Path:
    event_number = event_number_from_id(event_id)
    if event_number is not None:
        return ODDS_DIR / f"ufc_{event_number}.json"
    return ODDS_DIR / f"{event_id}.json"


def _fight_key(name_a: str, name_b: str) -> tuple[str, str]:
    return tuple(sorted((normalize_name(name_a), normalize_name(name_b))))


def _get_api_key() -> str:
    api_key = os.getenv("ODDS_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "Missing ODDS_API_KEY. Add it locally or as a GitHub Actions secret."
        )
    return api_key


def fetch_upcoming_odds() -> list[dict[str, Any]]:
    """
    Fetch upcoming MMA h2h odds from The Odds API.

    Official docs:
    https://the-odds-api.com/sports-odds-data/mma-odds.html
    """
    params = {
        "apiKey": _get_api_key(),
        "regions": os.getenv("ODDS_API_REGIONS", "us,us2"),
        "markets": "h2h",
        "oddsFormat": "american",
    }

    response = requests.get(ODDS_API_BASE_URL, params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()

    if not isinstance(payload, list):
        raise RuntimeError("Unexpected odds API response format: expected a list of fight events.")

    return payload


def _matchup_price_map(
    odds_payload: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    """
    Build matchup-keyed price maps from The Odds API response.

    Returns:
      {
        ("jiro prochazka", "carlos ulberg"): {
          "jiri prochazka": {-113, -112, ...},
          "carlos ulberg": {-107, +100, ...},
        },
        ...
      }
    """
    by_matchup: dict[tuple[str, str], dict[str, Any]] = {}

    for event in odds_payload:
        fighter_a = event.get("home_team")
        fighter_b = event.get("away_team")
        if not fighter_a or not fighter_b:
            continue

        matchup_key = _fight_key(fighter_a, fighter_b)
        matchup_prices = by_matchup.setdefault(
            matchup_key,
            {
                "fighter_names": {
                    normalize_name(fighter_a): fighter_a,
                    normalize_name(fighter_b): fighter_b,
                },
                "prices": {},
                "quotes": [],
            },
        )

        for bookmaker in event.get("bookmakers", []):
            quote_prices: dict[str, int] = {}
            for market in bookmaker.get("markets", []):
                if market.get("key") != "h2h":
                    continue

                for outcome in market.get("outcomes", []):
                    fighter_name = outcome.get("name")
                    price = outcome.get("price")
                    if not fighter_name or not isinstance(price, int):
                        continue
                    if price == 0 or abs(price) >= 5000:
                        continue

                    fighter_key = normalize_name(fighter_name)
                    matchup_prices["prices"].setdefault(fighter_key, set()).add(int(price))
                    quote_prices[fighter_key] = int(price)

            if quote_prices:
                matchup_prices["quotes"].append(
                    {
                        "sportsbook": bookmaker.get("title") or bookmaker.get("key") or "Unknown Book",
                        "last_update": bookmaker.get("last_update"),
                        "prices": quote_prices,
                    }
                )

    return by_matchup


def build_event_odds_bundle(
    event_id: str,
    fights: list[list[str]],
    odds_payload: list[dict[str, Any]],
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    """
    Map The Odds API fight-level odds into the repo's event odds format.
    """
    matchup_prices = _matchup_price_map(odds_payload)
    event_odds: dict[str, set[int]] = {}
    event_books: dict[str, Any] = {}

    for fighter_a, fighter_b in fights:
        matchup_key = _fight_key(fighter_a, fighter_b)
        data = matchup_prices.get(matchup_key)
        if not data:
            continue

        prices = data["prices"]
        for fighter in (fighter_a, fighter_b):
            fighter_key = normalize_name(fighter)
            if fighter_key in prices:
                event_odds.setdefault(fighter_key, set()).update(prices[fighter_key])

        quotes = []
        for quote in data.get("quotes", []):
            fighter_a_price = quote["prices"].get(normalize_name(fighter_a))
            fighter_b_price = quote["prices"].get(normalize_name(fighter_b))
            if fighter_a_price is None and fighter_b_price is None:
                continue

            quotes.append(
                {
                    "sportsbook": quote["sportsbook"],
                    "last_update": quote.get("last_update"),
                    "fighter_a_price": fighter_a_price,
                    "fighter_b_price": fighter_b_price,
                }
            )

        if quotes:
            event_books[f"{normalize_name(fighter_a)}-vs-{normalize_name(fighter_b)}"] = {
                "fighter_a": fighter_a,
                "fighter_b": fighter_b,
                "quotes": quotes,
            }

    return (
        {fighter: sorted(values) for fighter, values in event_odds.items()},
        event_books,
    )


def build_event_odds_map(
    event_id: str,
    fights: list[list[str]],
    odds_payload: list[dict[str, Any]],
) -> dict[str, list[int]]:
    odds_map, _ = build_event_odds_bundle(event_id, fights, odds_payload)
    return odds_map


def _default_event_ids() -> list[str]:
    """
    Refresh all locally tracked event cards.

    We use the local card inventory as the source of truth for which UFC events
    the product currently cares about, then enrich those matchups with odds if
    the external provider has them available.
    """
    return [
        item["event_id"]
        for item in list_available_event_items()
        if item.get("has_card")
    ]


def refresh_events(event_refs: list[str]) -> list[Path]:
    """
    Fetch odds once, then write one file per event when matching bouts exist.
    """
    odds_payload = fetch_upcoming_odds()
    written_paths: list[Path] = []

    for ref in event_refs:
        event_id = normalize_event_id(ref)
        fights = load_fight_card(event_id)
        odds_map, books_map = build_event_odds_bundle(event_id, fights, odds_payload)

        if not odds_map:
            print(f"No matching upcoming odds found for {event_id} — skipping write.")
            continue

        event_number = event_number_from_id(event_id)
        output_path = _output_path_for_event(event_id)
        data = {
            "event": event_number or event_id,
            "event_id": event_id,
            "event_number": event_number,
            "scraped_at": datetime.utcnow().isoformat(),
            "source": ODDS_API_SOURCE,
            "odds": odds_map,
            "books": books_map,
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        written_paths.append(output_path)
        print(f"Saved odds → {output_path}")

    return written_paths


if __name__ == "__main__":
    event_refs = sys.argv[1:] or _default_event_ids()
    if not event_refs:
        print("No local card events available to refresh odds for.")
        sys.exit(0)

    try:
        written = refresh_events(event_refs)
    except Exception as exc:
        print(f"Odds refresh failed: {exc}", file=sys.stderr)
        raise

    if not written:
        print("No odds files were updated.")
