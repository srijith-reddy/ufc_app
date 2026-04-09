"""
Incremental UFCStats raw-data scraper.

This script replaces the notebook-only scraping chunks with a reproducible CLI.
It stores the raw CSV bundle inside the repo-managed data directory so monthly
or post-event refreshes can run without any desktop-only paths.
"""
from __future__ import annotations

import argparse
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover - exercised only when scrape deps are absent
    BeautifulSoup = Any

from core.config import UFCSTATS_DATA_DIR

UFCSTATS_COMPLETED_EVENTS_URL = "http://www.ufcstats.com/statistics/events/completed?page=all"
USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Mozilla/5.0 (X11; Linux x86_64)",
]

EVENTS_FILENAME = "events.csv"
FIGHTS_FILENAME = "fights.csv"
TOTALS_FILENAME = "fight_totals.csv"
PROFILES_FILENAME = "fighters_advanced.csv"
HISTORY_FILENAME = "fighters_fight_history.csv"


def extract_event_id(url: str) -> str | None:
    """Extract the UFCStats event id from an event URL."""
    match = re.search(r"/event-details/([a-zA-Z0-9\-]+)", url)
    return match.group(1) if match else None


def parse_scheduled_rounds(round_end: str) -> int:
    """Infer scheduled rounds from the ending round value, matching notebook behavior."""
    return 5 if "5" in str(round_end) else 3


def read_existing_csv(path: Path) -> pd.DataFrame:
    """Load an existing CSV if present, else return an empty frame."""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def merge_records(
    existing: pd.DataFrame,
    new: pd.DataFrame,
    key_cols: list[str],
) -> pd.DataFrame:
    """
    Merge incremental scrape output onto an existing CSV, preferring new rows.
    """
    if existing.empty:
        return new.copy()
    if new.empty:
        return existing.copy()
    merged = pd.concat([existing, new], ignore_index=True)
    merged = merged.drop_duplicates(subset=key_cols, keep="last")
    return merged.reset_index(drop=True)


def txt(node) -> str | None:
    """Extract a clean text value from a BeautifulSoup node."""
    return node.get_text(" ", strip=True) if node else None


class UFCStatsClient:
    """Thin requests wrapper with retry/backoff for UFCStats instability."""

    def __init__(self, min_sleep: float = 0.15, max_sleep: float = 0.35):
        self.min_sleep = min_sleep
        self.max_sleep = max_sleep
        self.session = requests.Session()

    def get_soup(self, url: str, retries: int = 5) -> BeautifulSoup:
        if BeautifulSoup is Any:
            raise ImportError(
                "beautifulsoup4 is required for UFCStats scraping. "
                "Install requirements-scrape.txt or the scrape extra first."
            )
        last_error = None
        for attempt in range(retries):
            time.sleep(random.uniform(self.min_sleep, self.max_sleep))
            headers = {"User-Agent": random.choice(USER_AGENTS)}
            try:
                response = self.session.get(url, headers=headers, timeout=30)
            except requests.RequestException as exc:
                last_error = exc
                time.sleep(1.5 * (attempt + 1))
                continue

            if response.status_code != 200:
                last_error = RuntimeError(f"HTTP {response.status_code} for {url}")
                time.sleep(1.5 * (attempt + 1))
                continue

            if "<h1>Internal Server Error" in response.text:
                last_error = RuntimeError(f"UFCStats temporary block for {url}")
                time.sleep(random.uniform(3, 7))
                continue

            return BeautifulSoup(response.text, "html.parser")

        raise RuntimeError(f"Failed to fetch {url}: {last_error}")


def scrape_events(data_dir: Path, client: UFCStatsClient, full: bool = False) -> pd.DataFrame:
    """Scrape the completed UFC event list into events.csv."""
    output_path = data_dir / EVENTS_FILENAME
    existing = pd.DataFrame() if full else read_existing_csv(output_path)

    soup = client.get_soup(UFCSTATS_COMPLETED_EVENTS_URL)
    event_links = [a["href"] for a in soup.select("a.b-link.b-link_style_black")]
    event_names = [a.get_text(strip=True) for a in soup.select("a.b-link.b-link_style_black")]

    records = [
        {
            "event_id": extract_event_id(url),
            "event_name": name,
            "event_url": url,
        }
        for name, url in zip(event_names, event_links)
        if extract_event_id(url)
    ]

    df_new = pd.DataFrame(records)
    merged = merge_records(existing, df_new, ["event_id"])
    merged.to_csv(output_path, index=False)
    print(f"Saved {EVENTS_FILENAME} → {output_path} ({len(merged):,} rows)")
    return merged


def scrape_fights(data_dir: Path, client: UFCStatsClient, full: bool = False) -> pd.DataFrame:
    """Scrape fight-level rows for each event into fights.csv."""
    events_path = data_dir / EVENTS_FILENAME
    if not events_path.exists():
        raise FileNotFoundError(f"Missing {events_path}. Run the events scrape first.")

    df_events = pd.read_csv(events_path)
    output_path = data_dir / FIGHTS_FILENAME
    existing = pd.DataFrame() if full else read_existing_csv(output_path)
    existing_event_ids = set() if existing.empty else set(existing["event_id"].astype(str))

    rows: list[dict] = []
    for idx, event in df_events.iterrows():
        event_id = str(event["event_id"])
        if not full and event_id in existing_event_ids:
            continue

        soup = client.get_soup(event["event_url"])
        table_rows = soup.select("tbody tr")
        fight_order = 1

        for tr in table_rows:
            cols = tr.find_all("td")
            if len(cols) < 10:
                continue

            fighter_links = cols[1].select("a")
            if len(fighter_links) < 2:
                continue

            fight_link = cols[0].select_one("a")
            if fight_link is None or not fight_link.get("href"):
                continue

            round_end = txt(cols[8])
            rows.append({
                "event_id": event_id,
                "event_name": event["event_name"],
                "event_url": event["event_url"],
                "fight_order": fight_order,
                "fight_url": fight_link["href"],
                "fighter_A": fighter_links[0].get_text(strip=True),
                "fighter_B": fighter_links[1].get_text(strip=True),
                "WL_label": txt(cols[0]),
                "KD": txt(cols[2]),
                "STR": txt(cols[3]),
                "TD": txt(cols[4]),
                "SUB": txt(cols[5]),
                "weight_class": txt(cols[6]),
                "method": txt(cols[7]),
                "round_end": round_end,
                "time_end": txt(cols[9]),
                "scheduled_rounds": parse_scheduled_rounds(round_end),
            })
            fight_order += 1

        if idx % 50 == 0:
            print(f"Processed {idx}/{len(df_events)} events...")

    df_new = pd.DataFrame(rows)
    merged = merge_records(existing, df_new, ["fight_url"])
    merged.to_csv(output_path, index=False)
    print(f"Saved {FIGHTS_FILENAME} → {output_path} ({len(merged):,} rows)")
    return merged


def scrape_fight_totals(data_dir: Path, client: UFCStatsClient, full: bool = False) -> pd.DataFrame:
    """Scrape totals/significant-strike stats into fight_totals.csv."""
    fights_path = data_dir / FIGHTS_FILENAME
    if not fights_path.exists():
        raise FileNotFoundError(f"Missing {fights_path}. Run the fights scrape first.")

    df_fights = pd.read_csv(fights_path)
    output_path = data_dir / TOTALS_FILENAME
    existing = pd.DataFrame() if full else read_existing_csv(output_path)
    scraped_fight_urls = set() if existing.empty else set(existing["fight_url"])

    totals_rows: list[dict] = []
    for idx, row in df_fights.iterrows():
        fight_url = row["fight_url"]
        if not full and fight_url in scraped_fight_urls:
            continue

        soup = client.get_soup(fight_url)
        fighters = soup.select("h3.b-fight-details__person-name a")
        if len(fighters) < 2:
            continue

        totals_header = soup.find("p", string=re.compile(r"Totals", re.I))
        if not totals_header:
            totals_header = soup.find("p", class_=re.compile("collapse-link_tot", re.I))
        totals_table = totals_header.find_next("table") if totals_header else soup.select_one("table.b-fight-details__table")
        if totals_table is None:
            continue

        rows = totals_table.select("tbody tr")
        if not rows:
            continue

        stats: dict[str, str | list[dict]] = {}
        labels = ["KD", "Sig Str", "Sig Str %", "Total Str", "Td", "Td %", "Sub Att", "Rev", "Ctrl"]
        first_total_row = rows[0].find_all("td")
        for i, label in enumerate(labels):
            col = first_total_row[i + 1]
            p = col.find_all("p")
            if len(p) < 2:
                continue
            stats[f"{label}_A"] = txt(p[0])
            stats[f"{label}_B"] = txt(p[1])

        per_round_header = soup.select_one("a.b-fight-details__collapse-link_rnd")
        if not per_round_header:
            i_tag = soup.find("i", string=re.compile("Per round", re.I))
            per_round_header = i_tag.find_parent("a") if i_tag else None

        round_tables_list: list[dict] = []
        if per_round_header:
            table = per_round_header.find_next("table")
            tbody = table.find("tbody") if table else None
            current_round = None

            if tbody:
                for section in tbody.find_all(["thead", "tr"]):
                    if section.name == "thead":
                        current_round = txt(section.find("th"))
                        continue

                    cols = section.find_all("td")
                    if len(cols) < 10:
                        continue

                    p = [c.find_all("p") for c in cols]
                    round_tables_list.append({
                        "round": current_round,
                        "KD_A": txt(p[1][0]), "KD_B": txt(p[1][1]),
                        "SigStr_A": txt(p[2][0]), "SigStr_B": txt(p[2][1]),
                        "SigStrPct_A": txt(p[3][0]), "SigStrPct_B": txt(p[3][1]),
                        "TotalStr_A": txt(p[4][0]), "TotalStr_B": txt(p[4][1]),
                        "Td_A": txt(p[5][0]), "Td_B": txt(p[5][1]),
                        "TdPct_A": txt(p[6][0]), "TdPct_B": txt(p[6][1]),
                        "SubAtt_A": txt(p[7][0]), "SubAtt_B": txt(p[7][1]),
                        "Rev_A": txt(p[8][0]), "Rev_B": txt(p[8][1]),
                        "Ctrl_A": txt(p[9][0]), "Ctrl_B": txt(p[9][1]),
                    })

        sig_header = soup.find("p", string=re.compile("Significant Strikes", re.I))
        if sig_header is None:
            continue

        sig_table = sig_header.find_next("table")
        sig_rows = sig_table.select("tbody tr") if sig_table else []
        if not sig_rows:
            continue

        row0 = sig_rows[0].find_all("td")
        sig_stats: dict[str, str] = {}
        sig_labels = ["Sig Str", "Sig Str %", "Head", "Body", "Leg", "Distance", "Clinch", "Ground"]

        for i, label in enumerate(sig_labels):
            p = row0[i + 1].find_all("p")
            if len(p) < 2:
                continue
            sig_stats[f"{label}_A"] = txt(p[0])
            sig_stats[f"{label}_B"] = txt(p[1])

        totals_rows.append({
            "fight_url": fight_url,
            "fighter_A": txt(fighters[0]),
            "fighter_B": txt(fighters[1]),
            **stats,
            **sig_stats,
            "per_round_totals": round_tables_list,
        })

        if idx % 200 == 0:
            print(f"Processed {idx}/{len(df_fights)} fights...")

    df_new = pd.DataFrame(totals_rows)
    merged = merge_records(existing, df_new, ["fight_url"])
    merged.to_csv(output_path, index=False)
    print(f"Saved {TOTALS_FILENAME} → {output_path} ({len(merged):,} rows)")
    return merged


def gather_missing_fighter_urls(
    df_fights: pd.DataFrame,
    existing_profiles: pd.DataFrame,
    client: UFCStatsClient,
    full: bool = False,
) -> set[str]:
    """Discover fighter profile URLs by visiting fight pages when needed."""
    known_names = set() if existing_profiles.empty else set(existing_profiles["name"].dropna())
    fighter_urls: set[str] = set()

    for idx, row in df_fights.iterrows():
        if not full and row["fighter_A"] in known_names and row["fighter_B"] in known_names:
            continue

        soup = client.get_soup(row["fight_url"])
        for a in soup.select("h3.b-fight-details__person-name a"):
            href = a.get("href")
            if href:
                fighter_urls.add(href)

        if idx % 300 == 0:
            print(f"Scanned {idx}/{len(df_fights)} fights for fighter profile URLs...")

    return fighter_urls


def scrape_fighter_profiles(data_dir: Path, client: UFCStatsClient, full: bool = False) -> pd.DataFrame:
    """Scrape fighter bio/advanced profile pages into fighters_advanced.csv."""
    fights_path = data_dir / FIGHTS_FILENAME
    if not fights_path.exists():
        raise FileNotFoundError(f"Missing {fights_path}. Run the fights scrape first.")

    df_fights = pd.read_csv(fights_path)
    output_path = data_dir / PROFILES_FILENAME
    existing = pd.DataFrame() if full else read_existing_csv(output_path)
    existing_urls = set() if existing.empty else set(existing["fighter_url"].dropna())

    fighter_urls = gather_missing_fighter_urls(df_fights, existing, client, full=full)
    fighter_urls = [url for url in fighter_urls if full or url not in existing_urls]

    profiles: list[dict] = []
    for idx, url in enumerate(fighter_urls):
        soup = client.get_soup(url)

        bio = {
            "height": None,
            "weight": None,
            "reach": None,
            "stance": None,
            "dob": None,
            "wins": None,
            "losses": None,
            "draws": None,
            "nc": None,
        }
        for li in soup.select("li.b-list__box-list-item"):
            line = txt(li)
            if not line:
                continue
            if line.startswith("Height:"):
                bio["height"] = line.replace("Height:", "").strip()
            elif line.startswith("Weight:"):
                bio["weight"] = line.replace("Weight:", "").strip()
            elif line.startswith("Reach:"):
                bio["reach"] = line.replace("Reach:", "").strip()
            elif "STANCE:" in line.upper():
                bio["stance"] = line.split(":")[-1].strip()
            elif line.startswith("DOB:"):
                bio["dob"] = line.replace("DOB:", "").strip()
            elif "Wins:" in line:
                bio["wins"] = line.replace("Wins:", "").strip()
            elif "Losses:" in line:
                bio["losses"] = line.replace("Losses:", "").strip()
            elif "Draws:" in line:
                bio["draws"] = line.replace("Draws:", "").strip()
            elif "No Contest:" in line:
                bio["nc"] = line.replace("No Contest:", "").strip()

        adv_stats = {
            "SLpM": None,
            "Str_Acc": None,
            "SApM": None,
            "Str_Def": None,
            "TD_Avg": None,
            "TD_Acc": None,
            "TD_Def": None,
            "Sub_Avg": None,
        }
        for li in soup.select("ul.b-list__box-list.b-list__box-list_margin-top li"):
            line = txt(li)
            if not line:
                continue
            if "SLpM:" in line:
                adv_stats["SLpM"] = line.split(":")[-1].strip()
            elif "Str. Acc.:" in line:
                adv_stats["Str_Acc"] = line.split(":")[-1].strip()
            elif "SApM:" in line:
                adv_stats["SApM"] = line.split(":")[-1].strip()
            elif "Str. Def.:" in line:
                adv_stats["Str_Def"] = line.split(":")[-1].strip()
            elif "TD Avg.:" in line:
                adv_stats["TD_Avg"] = line.split(":")[-1].strip()
            elif "TD Acc.:" in line:
                adv_stats["TD_Acc"] = line.split(":")[-1].strip()
            elif "TD Def.:" in line:
                adv_stats["TD_Def"] = line.split(":")[-1].strip()
            elif "Sub. Avg.:" in line:
                adv_stats["Sub_Avg"] = line.split(":")[-1].strip()

        profiles.append({
            "fighter_url": url,
            "name": txt(soup.select_one("span.b-content__title-highlight")),
            "nickname": txt(soup.select_one("p.b-content__Nickname")),
            **bio,
            **adv_stats,
        })

        if idx % 100 == 0:
            print(f"Scraped {idx}/{len(fighter_urls)} fighter profiles...")

    df_new = pd.DataFrame(profiles)
    merged = merge_records(existing, df_new, ["fighter_url"])
    merged.to_csv(output_path, index=False)
    print(f"Saved {PROFILES_FILENAME} → {output_path} ({len(merged):,} rows)")
    return merged


def scrape_fighter_history(data_dir: Path, client: UFCStatsClient, full: bool = False) -> pd.DataFrame:
    """Scrape each fighter's history table into fighters_fight_history.csv."""
    profiles_path = data_dir / PROFILES_FILENAME
    if not profiles_path.exists():
        raise FileNotFoundError(f"Missing {profiles_path}. Run the fighter profiles scrape first.")

    df_profiles = pd.read_csv(profiles_path)
    output_path = data_dir / HISTORY_FILENAME
    existing = pd.DataFrame() if full else read_existing_csv(output_path)
    scraped_urls = set() if existing.empty else set(existing["fighter_url"].dropna())

    rows: list[dict] = []
    for idx, profile in df_profiles.iterrows():
        fighter_url = profile["fighter_url"]
        if not full and fighter_url in scraped_urls:
            continue

        soup = client.get_soup(fighter_url)
        for tr in soup.select("tr.b-fight-details__table-row"):
            cols = tr.find_all("td")
            if len(cols) < 10:
                continue

            rows.append({
                "fighter_url": fighter_url,
                "fighter_name": profile["name"],
                "wl": txt(cols[0]),
                "opponent": txt(cols[1]),
                "kd": txt(cols[2]),
                "str": txt(cols[3]),
                "td": txt(cols[4]),
                "sub": txt(cols[5]),
                "event": txt(cols[6]),
                "event_url": cols[6].find("a")["href"] if cols[6].find("a") else None,
                "method": txt(cols[7]),
                "round": txt(cols[8]),
                "time": txt(cols[9]),
            })

        if idx % 100 == 0:
            print(f"Scraped {idx}/{len(df_profiles)} fighter histories...")

    df_new = pd.DataFrame(rows)
    merged = merge_records(existing, df_new, ["fighter_url", "event_url"])
    merged.to_csv(output_path, index=False)
    print(f"Saved {HISTORY_FILENAME} → {output_path} ({len(merged):,} rows)")
    return merged


def run_all(data_dir: Path, full: bool = False) -> dict[str, int]:
    """Run the full raw UFCStats scrape/update pipeline."""
    data_dir.mkdir(parents=True, exist_ok=True)
    client = UFCStatsClient()

    events = scrape_events(data_dir, client, full=full)
    fights = scrape_fights(data_dir, client, full=full)
    totals = scrape_fight_totals(data_dir, client, full=full)
    profiles = scrape_fighter_profiles(data_dir, client, full=full)
    history = scrape_fighter_history(data_dir, client, full=full)

    return {
        "events": len(events),
        "fights": len(fights),
        "totals": len(totals),
        "profiles": len(profiles),
        "history": len(history),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Incrementally scrape the raw UFCStats CSV bundle used by the training pipeline."
    )
    parser.add_argument(
        "step",
        choices=["events", "fights", "totals", "profiles", "history", "all"],
        help="Which scrape step to run.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=UFCSTATS_DATA_DIR,
        help=f"Directory where raw UFCStats CSVs should be stored. Defaults to {UFCSTATS_DATA_DIR}",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Ignore incremental caches and rebuild the selected step from scratch.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    client = UFCStatsClient()

    try:
        if args.step == "events":
            scrape_events(args.data_dir, client, full=args.full)
        elif args.step == "fights":
            scrape_fights(args.data_dir, client, full=args.full)
        elif args.step == "totals":
            scrape_fight_totals(args.data_dir, client, full=args.full)
        elif args.step == "profiles":
            scrape_fighter_profiles(args.data_dir, client, full=args.full)
        elif args.step == "history":
            scrape_fighter_history(args.data_dir, client, full=args.full)
        else:
            summary = run_all(args.data_dir, full=args.full)
            print(f"Completed raw UFCStats refresh: {summary}")
    except Exception as exc:
        print(f"\n❌ UFCStats scrape failed: {exc}", file=sys.stderr)
        raise
