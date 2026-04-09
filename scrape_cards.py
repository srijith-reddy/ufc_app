"""
scrape_cards.py — scrape a UFC event fight card from ufc.com.

Uses Playwright (headless Chromium). Must be run after
`playwright install chromium && playwright install-deps chromium`.

Usage:
    python scrape_cards.py                                # scrape next 3 upcoming UFC events
    python scrape_cards.py <EVENT_REF> [<EVENT_REF> ...] # numbered UFC or full slug

Output:
    cards/ufc_<event_number>.json for numbered cards
    cards/<event_slug>.json for Fight Nights and other slug events
"""
import json
import re
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

# normalize_name is the single canonical implementation in core/names.py.
# The scraper must use it so all stored names match the lookup in app/API.
from core.names import normalize_name
from core.config import CARDS_DIR
from core.event import normalize_event_id, event_number_from_id

CARDS_DIR.mkdir(exist_ok=True)
BASE_URL = "https://www.ufc.com/event"
EVENTS_URL = "https://www.ufc.com/events"

def _output_path_for_event(event_id: str) -> Path:
    event_number = event_number_from_id(event_id)
    if event_number is not None:
        return CARDS_DIR / f"ufc_{event_number}.json"
    return CARDS_DIR / f"{event_id}.json"


def _discover_upcoming_event_ids(limit: int = 3) -> list[str]:
    """
    Discover the next upcoming UFC event slugs from ufc.com/events.

    This is intentionally slug-based so Fight Nights and numbered cards can
    both flow through the same automation path.
    """
    event_ids: list[str] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(EVENTS_URL, timeout=60_000)
        page.wait_for_timeout(4000)

        for anchor in page.query_selector_all("a[href*='/event/']"):
            href = anchor.get_attribute("href")
            if not href or "/event/" not in href:
                continue

            raw = href.split("/event/", 1)[1].strip("/")
            try:
                event_id = normalize_event_id(raw)
            except ValueError:
                continue

            if event_id in event_ids:
                continue

            event_ids.append(event_id)
            if len(event_ids) >= limit:
                break

        browser.close()

    if not event_ids:
        raise RuntimeError(f"Could not discover upcoming UFC event links from {EVENTS_URL}")

    return event_ids


def scrape_event_card(event_ref: int | str) -> list[list[str]]:
    """
    Scrape and return all fight matchups for a given UFC event ref.

    Returns:
        [ ["Justin Gaethje", "Paddy Pimblett"], ... ]

    Raises:
        RuntimeError if the event card is not published or no fights found.
    """
    event_id = normalize_event_id(event_ref)
    url = f"{BASE_URL}/{event_id}"
    fights: list[list[str]] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        page.goto(url, timeout=60_000)

        try:
            page.wait_for_selector("div.c-listing-fight", timeout=60_000)
        except Exception:
            browser.close()
            raise RuntimeError(f"Event '{event_id}' card not published yet at {url}")

        for fight in page.query_selector_all("div.c-listing-fight"):
            names = fight.query_selector_all("div.c-listing-fight__corner-name")
            if len(names) != 2:
                continue

            fighter_a = names[0].inner_text().strip()
            fighter_b = names[1].inner_text().strip()

            # Skip rows that don't resolve to valid names after normalization
            if not normalize_name(fighter_a) or not normalize_name(fighter_b):
                continue

            fights.append([fighter_a, fighter_b])

        browser.close()

    if not fights:
        raise RuntimeError(f"No fights scraped for event '{event_id}'")

    return fights


if __name__ == "__main__":
    event_refs = sys.argv[1:]
    if not event_refs:
        print("No event reference provided — discovering next upcoming UFC events...")
        event_refs = _discover_upcoming_event_ids(limit=3)
        print(f"Discovered: {', '.join(event_refs)}")

    for ref in event_refs:
        event_id = normalize_event_id(ref)
        output_path = _output_path_for_event(event_id)
        fights = scrape_event_card(event_id)

        with open(output_path, "w") as f:
            json.dump(fights, f, indent=2)

        print(f"Saved fight card → {output_path}  ({len(fights)} fights)")
