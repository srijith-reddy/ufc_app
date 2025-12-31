# scrape_odds.py
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
BASE_URL = "https://fightodds.io"
ODDS_DIR = Path("odds")
ODDS_DIR.mkdir(exist_ok=True)

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


def name_aliases(name: str) -> set[str]:
    base = normalize_name(name)
    parts = base.split()

    aliases = set()
    aliases.add(base)

    if len(parts) >= 2:
        first = parts[0]
        last = parts[-1]
        middle = parts[1:-1]

        # Basic swaps
        aliases.add(f"{last} {first}")
        aliases.add(f"{first} {last}")

        # Handle compound surnames (Cortes Acosta)
        if middle:
            compound_last = " ".join(middle + [last])
            aliases.add(f"{first} {compound_last}")
            aliases.add(compound_last)
            aliases.add(f"{last} {compound_last}")
            aliases.add(f"{compound_last} {first}")

        # Full reverse
        aliases.add(" ".join(reversed(parts)))

    return {a.strip() for a in aliases if a.strip()}



# ===============================================================
# EVENT DISCOVERY
# ===============================================================
def get_fightodds_event_url(event_number: int, page) -> str | None:
    target = f"ufc-{event_number}".lower()

    for _ in range(6):
        page.mouse.wheel(0, 3000)
        page.wait_for_timeout(1000)

    for a in page.query_selector_all("a[href^='/odds/']"):
        href = a.get_attribute("href")
        if href and target in href.lower():
            return BASE_URL + href

    return None


def get_next_ufc_event_number(page) -> int:
    """
    Auto-detect the next upcoming UFC event from fightodds.io
    Used for daily scheduled scraping
    """
    page.goto(BASE_URL, timeout=30000)
    page.wait_for_timeout(3000)

    for _ in range(6):
        page.mouse.wheel(0, 3000)
        page.wait_for_timeout(800)

    for a in page.query_selector_all("a[href^='/odds/']"):
        href = a.get_attribute("href")
        if not href:
            continue

        m = re.search(r"/odds/\d+/ufc-(\d+)", href.lower())
        if m:
            return int(m.group(1))

    raise RuntimeError("Could not auto-detect upcoming UFC event")


# ===============================================================
# SCRAPER (IDENTICAL LOGIC TO STREAMLIT)
# ===============================================================
def scrape_event(event_number: int) -> dict[str, list[int]]:
    odds_map: dict[str, set[int]] = {}

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        page.goto(BASE_URL, timeout=30000)
        page.wait_for_timeout(3000)

        event_url = get_fightodds_event_url(event_number, page)
        if not event_url:
            browser.close()
            raise RuntimeError(f"UFC {event_number} not found on fightodds.io")

        page.goto(event_url, timeout=30000)
        page.wait_for_selector("table tbody tr", timeout=15000)

        for tr in page.query_selector_all("table tbody tr"):
            if not tr.is_visible():
                continue

            tds = tr.query_selector_all("td")
            if len(tds) < 2:
                continue

            raw_name = tds[0].inner_text().strip()
            aliases = name_aliases(raw_name)

            row_odds = set()
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

            # aliases point to SAME set (no duplication)
            for a in aliases:
                odds_map[a] = odds_map[canonical]

        browser.close()

    return {k: sorted(v) for k, v in odds_map.items()}


# ===============================================================
# CLI ENTRYPOINT
# ===============================================================
if __name__ == "__main__":
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        if len(sys.argv) == 2 and sys.argv[1].isdigit():
            event_number = int(sys.argv[1])
        else:
            print("No event number provided — auto-detecting next UFC event")
            event_number = get_next_ufc_event_number(page)

        browser.close()

    output_path = ODDS_DIR / f"ufc_{event_number}.json"

    data = {
        "event": event_number,
        "scraped_at": datetime.utcnow().isoformat(),
        "source": "https://fightodds.io",
        "odds": scrape_event(event_number),
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved odds → {output_path}")
