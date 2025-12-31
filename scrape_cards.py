# scrape_cards.py
import json
import re
import sys
import unicodedata
from datetime import datetime
from pathlib import Path
from playwright.sync_api import sync_playwright

# ===============================================================
# CONFIG
# ===============================================================
BASE_URL = "https://www.ufc.com/event"
CARDS_DIR = Path("cards")
CARDS_DIR.mkdir(exist_ok=True)

# ===============================================================
# NAME NORMALIZATION (MUST MATCH STREAMLIT EXACTLY)
# ===============================================================
def normalize_name(name: str) -> str:
    name = unicodedata.normalize("NFKD", name)
    name = name.encode("ascii", "ignore").decode("ascii")
    name = name.lower()
    name = re.sub(r"[^a-z\s]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name

# ===============================================================
# SCRAPE UFC EVENT CARD
# ===============================================================
def scrape_event_card(event_number: int) -> list[list[str]]:
    """
    Returns:
      [
        ["Justin Gaethje", "Paddy Pimblett"],
        ["Sean O'Malley", "Song Yadong"],
        ...
      ]
    """

    url = f"{BASE_URL}/ufc-{event_number}"
    fights: list[list[str]] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        page.goto(url, timeout=60000)

        try:
            page.wait_for_selector("div.c-listing-fight", timeout=60000)
        except Exception:
            browser.close()
            raise RuntimeError(f"UFC {event_number} card not published yet")

        for fight in page.query_selector_all("div.c-listing-fight"):
            names = fight.query_selector_all("div.c-listing-fight__corner-name")
            if len(names) != 2:
                continue

            fighter_a = names[0].inner_text().strip()
            fighter_b = names[1].inner_text().strip()

            # strict safety: ignore garbage rows
            if not normalize_name(fighter_a) or not normalize_name(fighter_b):
                continue

            fights.append([fighter_a, fighter_b])

        browser.close()

    if not fights:
        raise RuntimeError(f"No fights scraped for UFC {event_number}")

    return fights

# ===============================================================
# CLI ENTRYPOINT (FOR GITHUB ACTIONS)
# ===============================================================
if __name__ == "__main__":
    if len(sys.argv) != 2 or not sys.argv[1].isdigit():
        print("Usage: python scrape_cards.py <UFC_EVENT_NUMBER>")
        sys.exit(1)

    event_number = int(sys.argv[1])
    output_path = CARDS_DIR / f"ufc_{event_number}.json"

    fights = scrape_event_card(event_number)

    with open(output_path, "w") as f:
        json.dump(fights, f, indent=2)

    print(f"Saved fight card → {output_path}")
