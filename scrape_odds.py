"""
scrape_odds.py — scrape UFC moneyline odds from fightodds.io.

Uses Playwright (headless Chromium). Must be run after
`playwright install chromium && playwright install-deps chromium`.

Usage:
    python scrape_odds.py                     # auto-detect next upcoming UFC event slug
    python scrape_odds.py <EVENT_REF>         # specific numbered card or fight-night slug

Output:
    odds/ufc_<event_number>.json for numbered cards
    odds/<event_slug>.json for Fight Nights and other slug events
"""
import json
import re
import sys
from datetime import datetime
from pathlib import Path

from playwright.sync_api import sync_playwright

# normalize_name and name_aliases are the single canonical implementations
# in core/names.py. All three entry points (scraper, app, API) must share them.
from core.names import name_aliases, normalize_name
from core.config import ODDS_DIR
from core.event import event_number_from_id, normalize_event_id

ODDS_DIR.mkdir(exist_ok=True)
BASE_URL = "https://fightodds.io"


# ── Event discovery ────────────────────────────────────────────────────────────

def _output_path_for_event(event_id: str) -> Path:
    event_number = event_number_from_id(event_id)
    if event_number is not None:
        return ODDS_DIR / f"ufc_{event_number}.json"
    return ODDS_DIR / f"{event_id}.json"


def _slug_from_href(href: str) -> str | None:
    match = re.search(r"/odds/\d+/([^/?#]+)", href.lower())
    return match.group(1) if match else None


def get_fightodds_event_url(event_ref: int | str, page) -> str | None:
    """Scroll fightodds.io homepage to find the link for a specific event slug."""
    target = normalize_event_id(event_ref)

    for _ in range(6):
        page.mouse.wheel(0, 3000)
        page.wait_for_timeout(1000)

    for a in page.query_selector_all("a[href^='/odds/']"):
        href = a.get_attribute("href")
        slug = _slug_from_href(href or "")
        if slug and (slug == target or target in slug or slug in target):
            return BASE_URL + href

    return None


def get_next_ufc_event_id(page) -> str:
    """Auto-detect the next upcoming UFC event slug from fightodds.io."""
    page.goto(BASE_URL, timeout=30_000)
    page.wait_for_timeout(3000)

    for _ in range(6):
        page.mouse.wheel(0, 3000)
        page.wait_for_timeout(800)

    for a in page.query_selector_all("a[href^='/odds/']"):
        href = a.get_attribute("href")
        if not href:
            continue
        slug = _slug_from_href(href)
        if slug and slug.startswith("ufc"):
            return slug

    raise RuntimeError("Could not auto-detect upcoming UFC event from fightodds.io")


# ── Odds scraper ───────────────────────────────────────────────────────────────

def scrape_event(event_ref: int | str) -> dict[str, list[int]]:
    """
    Scrape all available moneyline odds for the given event.

    Returns dict mapping fighter aliases → sorted list of American odds.
    """
    odds_map: dict[str, set[int]] = {}
    event_id = normalize_event_id(event_ref)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        page.goto(BASE_URL, timeout=30_000)
        page.wait_for_timeout(3000)

        event_url = get_fightodds_event_url(event_id, page)
        if not event_url:
            browser.close()
            raise RuntimeError(f"Event '{event_id}' not found on fightodds.io")

        page.goto(event_url, timeout=30_000)
        page.wait_for_selector("table tbody tr", timeout=15_000)

        for tr in page.query_selector_all("table tbody tr"):
            if not tr.is_visible():
                continue

            tds = tr.query_selector_all("td")
            if len(tds) < 2:
                continue

            raw_name = tds[0].inner_text().strip()
            aliases = name_aliases(raw_name)

            row_odds: set[int] = set()
            for td in tds[1:]:
                for s in td.query_selector_all("span"):
                    txt = s.inner_text().replace("−", "-").strip()
                    if re.fullmatch(r"[+-]\d+", txt):
                        val = int(txt)
                        if -5000 < val < 5000 and val != 0:
                            row_odds.add(val)

            if not row_odds:
                continue

            canonical = normalize_name(raw_name)
            odds_map.setdefault(canonical, set()).update(row_odds)

            # All aliases point to the same set (no duplication on lookup)
            for a in aliases:
                odds_map[a] = odds_map[canonical]

        browser.close()

    return {k: sorted(v) for k, v in odds_map.items()}


# ── CLI entrypoint ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Determine event id (manual override or auto-detect)
    if len(sys.argv) == 2:
        event_id = normalize_event_id(sys.argv[1])
    else:
        print("No event reference provided — auto-detecting next UFC event...")
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            event_id = get_next_ufc_event_id(page)
            browser.close()
        print(f"Auto-detected: {event_id}")

    output_path = _output_path_for_event(event_id)
    event_number = event_number_from_id(event_id)

    data = {
        "event": event_number or event_id,
        "event_id": event_id,
        "event_number": event_number,
        "scraped_at": datetime.utcnow().isoformat(),
        "source": BASE_URL,
        "odds": scrape_event(event_id),
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved odds → {output_path}")
