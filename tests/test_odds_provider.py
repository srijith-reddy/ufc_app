"""
Tests for The Odds API-based odds ingestion helpers.
"""
from __future__ import annotations

from scrape_odds import build_event_odds_map


def test_build_event_odds_map_matches_fight_pairs_by_name():
    fights = [["JIRI PROCHAZKA", "CARLOS ULBERG"]]
    payload = [
        {
            "home_team": "Jiri Prochazka",
            "away_team": "Carlos Ulberg",
            "bookmakers": [
                {
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Jiri Prochazka", "price": -113},
                                {"name": "Carlos Ulberg", "price": -107},
                            ],
                        }
                    ]
                },
                {
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Jiri Prochazka", "price": -112},
                                {"name": "Carlos Ulberg", "price": -108},
                            ],
                        }
                    ]
                },
            ],
        }
    ]

    odds_map = build_event_odds_map("ufc-327", fights, payload)

    assert odds_map["jiri prochazka"] == [-113, -112]
    assert odds_map["carlos ulberg"] == [-108, -107]


def test_build_event_odds_map_skips_unmatched_bouts():
    fights = [["JOSH HOKIT", "CURTIS BLAYDES"]]
    payload = [
        {
            "home_team": "Different Fighter",
            "away_team": "Someone Else",
            "bookmakers": [],
        }
    ]

    odds_map = build_event_odds_map("ufc-327", fights, payload)

    assert odds_map == {}
