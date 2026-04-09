"""
End-to-end automation entrypoint for the UFC app data/model refresh cycle.

This orchestrates:
  1. Incremental UFCStats raw-data updates
  2. Fighter snapshot refresh or full model retraining
  3. Optional event-card scraping
  4. Optional odds scraping
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from core.config import CARDS_DIR, ODDS_DIR, UFCSTATS_DATA_DIR
from pipelines import refresh_fighters, train_model
from pipelines.scrape_ufcstats import run_all as scrape_ufcstats_all
from scrape_cards import scrape_event_card
from scrape_odds import scrape_event


def refresh_cards(events: list[int]) -> None:
    """Scrape and persist one or more UFC event cards."""
    CARDS_DIR.mkdir(exist_ok=True)
    for event in events:
        fights = scrape_event_card(event)
        output_path = CARDS_DIR / f"ufc_{event}.json"
        with open(output_path, "w") as f:
            json.dump(fights, f, indent=2)
        print(f"Saved fight card → {output_path} ({len(fights)} fights)")


def refresh_odds(events: list[int]) -> None:
    """Scrape and persist odds files for one or more UFC events."""
    ODDS_DIR.mkdir(exist_ok=True)
    for event in events:
        output_path = ODDS_DIR / f"ufc_{event}.json"
        payload = {
            "event": event,
            "scraped_at": datetime.now(timezone.utc).isoformat(),
            "source": "https://fightodds.io",
            "odds": scrape_event(event),
        }
        with open(output_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved odds → {output_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full UFC data/model refresh workflow."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=UFCSTATS_DATA_DIR,
        help=f"Directory containing the raw UFCStats CSV bundle. Defaults to {UFCSTATS_DATA_DIR}",
    )
    parser.add_argument(
        "--full-raw",
        action="store_true",
        help="Rebuild the raw UFCStats CSV bundle from scratch instead of incrementally.",
    )
    parser.add_argument(
        "--skip-raw",
        action="store_true",
        help="Skip the raw UFCStats scrape/update step.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip full model retraining and only refresh fighters_latest.csv.",
    )
    parser.add_argument(
        "--skip-snapshot",
        action="store_true",
        help="When --skip-train is set, also skip fighters_latest.csv refresh.",
    )
    parser.add_argument(
        "--tune-trials",
        type=int,
        default=25,
        help="Optuna trial count passed to pipelines.train_model. Use 0 to skip tuning.",
    )
    parser.add_argument(
        "--card-events",
        nargs="*",
        type=int,
        default=[],
        help="Optional UFC event numbers whose cards should be scraped.",
    )
    parser.add_argument(
        "--odds-events",
        nargs="*",
        type=int,
        default=[],
        help="Optional UFC event numbers whose odds should be scraped.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run snapshot/model steps without writing model artifacts.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    try:
        if not args.skip_raw:
            summary = scrape_ufcstats_all(args.data_dir, full=args.full_raw)
            print(f"Raw UFCStats refresh complete: {summary}")

        if args.skip_train:
            if not args.skip_snapshot:
                refresh_fighters.run(args.data_dir, dry_run=args.dry_run)
        else:
            train_model.run(
                data_dir=args.data_dir,
                tune_trials=args.tune_trials,
                dry_run=args.dry_run,
            )

        if args.card_events:
            refresh_cards(args.card_events)

        if args.odds_events:
            refresh_odds(args.odds_events)

    except Exception as exc:
        print(f"\n❌ Refresh-all failed: {exc}", file=sys.stderr)
        raise
