"""
Product-facing service layer for the UFC intelligence platform.

This module translates the low-level prediction and eligibility logic in core/
into richer event and fight payloads that a polished frontend can render
without bolting on product semantics in the client.
"""
from __future__ import annotations

import math
import re
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from core import (
    american_to_decimal,
    check_event_coverage,
    event_number_from_id,
    format_event_title,
    get_odds_for_fighter,
    implied_prob_from_decimal,
    kelly_fraction,
    list_available_event_items,
    load_fight_card,
    load_odds_payload,
)
from core.names import normalize_name


STATUS_META = {
    "fully_predictable": {
        "label": "Fully Predictable",
        "tone": "success",
        "description": "Every bout on this card has a valid prefight prediction path.",
    },
    "partially_predictable": {
        "label": "Partially Predictable",
        "tone": "warning",
        "description": "Supported bouts are shown normally and unavailable fights are called out explicitly.",
    },
    "not_predictable": {
        "label": "Not Predictable",
        "tone": "muted",
        "description": "This card does not currently have enough prefight support to produce fight-level predictions.",
    },
}

ARCHIVE_STATUS_META = {
    "label": "Archived Data",
    "tone": "muted",
    "description": "Historical event data is saved locally, but a full fight-card artifact is not available for product-grade rendering.",
}


UNSUPPORTED_REASON_META = {
    "unsupported_missing_fighter_a": {
        "label": "Missing Fighter Snapshot",
        "description": "Fighter A is not available in the current prefight snapshot.",
    },
    "unsupported_missing_fighter_b": {
        "label": "Missing Fighter Snapshot",
        "description": "Fighter B is not available in the current prefight snapshot.",
    },
    "unsupported_feature_build_failed": {
        "label": "Feature Build Failed",
        "description": "A complete prefight feature vector could not be assembled for this matchup.",
    },
}


def _safe_round(value: float | None, digits: int = 3) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return round(float(value), digits)


def _median_market_prob(odds_list: list[int]) -> float | None:
    decimals = [american_to_decimal(o) for o in odds_list]
    if not decimals:
        return None
    raw = implied_prob_from_decimal(float(np.median(decimals)))
    if math.isnan(raw):
        return None
    return float(raw)


def _best_decimal(odds_list: list[int]) -> float | None:
    decimals = [american_to_decimal(o) for o in odds_list]
    if not decimals:
        return None
    return float(np.max(decimals))


def _confidence_meta(p_pick: float) -> dict[str, Any]:
    margin = abs(p_pick - 0.5)
    score = int(round(min(99, 50 + margin * 100)))
    if margin >= 0.18:
        return {"label": "High Conviction", "tone": "strong", "score": score}
    if margin >= 0.10:
        return {"label": "Measured Lean", "tone": "medium", "score": score}
    return {"label": "Tight Fight", "tone": "cautious", "score": score}


def _value_state(edge_pick: float | None) -> dict[str, str]:
    if edge_pick is None:
        return {"label": "Pass", "tone": "muted"}
    if edge_pick >= 0.05:
        return {"label": "Positive Edge", "tone": "positive"}
    if edge_pick <= -0.03:
        return {"label": "Market Favored", "tone": "negative"}
    return {"label": "Neutral", "tone": "neutral"}


def _coalesce_float(value: Any) -> float | None:
    try:
        val = float(value)
    except Exception:
        return None
    if math.isnan(val):
        return None
    return val


def _metric_advantage(a_value: float | None, b_value: float | None, prefer_lower: bool = False) -> str | None:
    if a_value is None or b_value is None:
        return None
    if abs(a_value - b_value) < 1e-9:
        return None
    if prefer_lower:
        return "a" if a_value < b_value else "b"
    return "a" if a_value > b_value else "b"


def _format_percentage(value: float | None) -> str | None:
    if value is None:
        return None
    return f"{value * 100:.0f}%"


def _parse_event_date_from_id(event_id: str) -> str | None:
    match = re.match(
        r"ufc-fight-night-([a-z]+)-(\d{1,2})-(\d{4})$",
        event_id,
    )
    if not match:
        return None

    month_name, day, year = match.groups()
    month_lookup = {
        "january": 1,
        "february": 2,
        "march": 3,
        "april": 4,
        "may": 5,
        "june": 6,
        "july": 7,
        "august": 8,
        "september": 9,
        "october": 10,
        "november": 11,
        "december": 12,
    }
    month = month_lookup.get(month_name)
    if month is None:
        return None

    return date(int(year), month, int(day)).isoformat()


def _timeline_bucket(event_meta: dict[str, Any], all_items: list[dict[str, Any]]) -> str:
    event_id = event_meta["event_id"]
    event_number = event_meta["event_number"]
    has_card = event_meta["has_card"]
    has_odds = event_meta["has_odds"]

    slug_date = _parse_event_date_from_id(event_id)
    if slug_date is not None:
        return "future" if slug_date >= date.today().isoformat() else "past"

    numbered_card_events = [
        item["event_number"]
        for item in all_items
        if item["has_card"] and item["event_number"] is not None
    ]
    max_numbered_card = max(numbered_card_events) if numbered_card_events else None

    if event_number is not None and max_numbered_card is not None:
        if event_number < max_numbered_card and has_odds:
            return "past"
        if event_number == max_numbered_card and has_card:
            return "future"

    if has_card and not has_odds:
        return "future"
    if has_odds and not has_card:
        return "past"
    return "future"


def build_feature_comparison(A: pd.Series, B: pd.Series) -> list[dict[str, Any]]:
    """Build a render-friendly prefight feature comparison list."""
    rows = [
        ("rating", "Glicko Rating", _coalesce_float(A.get("g_rating_before")), _coalesce_float(B.get("g_rating_before")), False, 1),
        ("rating_rd", "Rating Uncertainty", _coalesce_float(A.get("g_RD_before")), _coalesce_float(B.get("g_RD_before")), True, 1),
        ("age", "Age", _coalesce_float(A.get("fighter_age")), _coalesce_float(B.get("fighter_age")), True, 1),
        ("reach", "Reach", _coalesce_float(A.get("fighter_reach_inches")), _coalesce_float(B.get("fighter_reach_inches")), False, 1),
        ("height", "Height", _coalesce_float(A.get("fighter_height_inches")), _coalesce_float(B.get("fighter_height_inches")), False, 1),
        ("slpm", "Strikes Landed / Min", _coalesce_float(A.get("SLpM")), _coalesce_float(B.get("SLpM")), False, 2),
        ("sapm", "Strikes Absorbed / Min", _coalesce_float(A.get("SApM")), _coalesce_float(B.get("SApM")), True, 2),
        ("str_acc", "Strike Accuracy", _coalesce_float(A.get("Str_Acc")), _coalesce_float(B.get("Str_Acc")), False, None),
        ("str_def", "Strike Defense", _coalesce_float(A.get("Str_Def")), _coalesce_float(B.get("Str_Def")), False, None),
        ("td_avg", "Takedowns / 15", _coalesce_float(A.get("TD_Avg")), _coalesce_float(B.get("TD_Avg")), False, 2),
        ("td_def", "Takedown Defense", _coalesce_float(A.get("TD_Def")), _coalesce_float(B.get("TD_Def")), False, None),
        ("sub_avg", "Submissions / 15", _coalesce_float(A.get("Sub_Avg")), _coalesce_float(B.get("Sub_Avg")), False, 2),
        ("fights_before", "Prior UFC Bouts", _coalesce_float(A.get("fights_before")), _coalesce_float(B.get("fights_before")), False, 0),
        ("layoff", "Layoff Days", _coalesce_float(A.get("days_since_last_fight")), _coalesce_float(B.get("days_since_last_fight")), True, 0),
        ("recent_form_3", "Recent Form (3)", _coalesce_float(A.get("recent_win_rate_3")), _coalesce_float(B.get("recent_win_rate_3")), False, None),
        ("recent_form_5", "Recent Form (5)", _coalesce_float(A.get("recent_win_rate_5")), _coalesce_float(B.get("recent_win_rate_5")), False, None),
    ]

    comparisons: list[dict[str, Any]] = []
    for key, label, a_value, b_value, prefer_lower, digits in rows:
        advantage = _metric_advantage(a_value, b_value, prefer_lower=prefer_lower)
        a_display = _format_percentage(a_value) if key in {"str_acc", "str_def", "td_def", "recent_form_3", "recent_form_5"} else None
        b_display = _format_percentage(b_value) if key in {"str_acc", "str_def", "td_def", "recent_form_3", "recent_form_5"} else None
        if a_display is None and a_value is not None:
            a_display = f"{a_value:.{digits}f}" if digits is not None else f"{a_value:.2f}"
        if b_display is None and b_value is not None:
            b_display = f"{b_value:.{digits}f}" if digits is not None else f"{b_value:.2f}"

        comparisons.append({
            "key": key,
            "label": label,
            "fighter_a_value": a_value,
            "fighter_b_value": b_value,
            "fighter_a_display": a_display,
            "fighter_b_display": b_display,
            "advantage": advantage,
        })
    return comparisons


def _top_edge_side(diff: float, threshold: float) -> str | None:
    if abs(diff) < threshold:
        return None
    return "a" if diff > 0 else "b"


def build_insight_chips(
    fighter_a: str,
    fighter_b: str,
    A: pd.Series,
    B: pd.Series,
    prob_a: float,
    market_prob_a: float | None,
) -> list[dict[str, Any]]:
    """Generate grounded, feature-based insight chips for a supported fight."""
    chips: list[dict[str, Any]] = []
    favored = "a" if prob_a >= 0.5 else "b"

    def side_name(side: str) -> str:
        return fighter_a if side == "a" else fighter_b

    def diff(a_value: Any, b_value: Any) -> float | None:
        a_num = _coalesce_float(a_value)
        b_num = _coalesce_float(b_value)
        if a_num is None or b_num is None:
            return None
        return a_num - b_num

    rating_diff = diff(A.get("g_rating_before"), B.get("g_rating_before"))
    side = _top_edge_side(rating_diff, 45) if rating_diff is not None else None
    if side:
        chips.append({
            "label": "Rating Edge",
            "fighter": side_name(side),
            "tone": "strong" if side == favored else "neutral",
            "detail": f"{abs(rating_diff):.0f} point Glicko gap",
        })

    reach_diff = diff(A.get("fighter_reach_inches"), B.get("fighter_reach_inches"))
    side = _top_edge_side(reach_diff, 2) if reach_diff is not None else None
    if side:
        chips.append({
            "label": "Reach Edge",
            "fighter": side_name(side),
            "tone": "neutral",
            "detail": f"{abs(reach_diff):.0f} inch reach advantage",
        })

    form_diff = diff(A.get("recent_win_rate_5"), B.get("recent_win_rate_5"))
    side = _top_edge_side(form_diff, 0.18) if form_diff is not None else None
    if side:
        chips.append({
            "label": "Form Edge",
            "fighter": side_name(side),
            "tone": "strong" if side == favored else "neutral",
            "detail": f"{abs(form_diff) * 100:.0f} point recent form gap",
        })

    layoff_diff = diff(B.get("days_since_last_fight"), A.get("days_since_last_fight"))
    side = _top_edge_side(layoff_diff, 120) if layoff_diff is not None else None
    if side:
        chips.append({
            "label": "Activity Edge",
            "fighter": side_name(side),
            "tone": "neutral",
            "detail": "Shorter layoff entering the bout",
        })

    striking_a = diff(A.get("SLpM"), A.get("SApM"))
    striking_b = diff(B.get("SLpM"), B.get("SApM"))
    striking_diff = None if striking_a is None or striking_b is None else striking_a - striking_b
    side = _top_edge_side(striking_diff, 1.0) if striking_diff is not None else None
    if side:
        chips.append({
            "label": "Striking Profile",
            "fighter": side_name(side),
            "tone": "neutral",
            "detail": "Better landed-to-absorbed striking profile",
        })

    if market_prob_a is not None and abs(prob_a - market_prob_a) >= 0.05:
        side = "a" if prob_a > market_prob_a else "b"
        chips.append({
            "label": "Market Disagreement",
            "fighter": side_name(side),
            "tone": "positive" if side == favored else "negative",
            "detail": f"{abs(prob_a - market_prob_a) * 100:.1f} point model-market gap",
        })

    return chips[:5]


def build_model_lean(
    favored_name: str,
    chips: list[dict[str, Any]],
    value_state: dict[str, str],
) -> str:
    """Generate a concise, grounded summary of the model lean."""
    if chips:
        top_labels = ", ".join(chip["label"].lower() for chip in chips[:2])
        if value_state["label"] == "Positive Edge":
            return f"Model leans {favored_name} on {top_labels}, with a favorable gap versus market pricing."
        return f"Model leans {favored_name} on {top_labels} in the current prefight snapshot."
    if value_state["label"] == "Positive Edge":
        return f"Model favors {favored_name} and still sees value versus the current market."
    return f"Model favors {favored_name} based on the current prefight feature set."


def _humanize_unavailable(fight) -> dict[str, Any]:
    meta = UNSUPPORTED_REASON_META.get(
        fight.status,
        {"label": "Unavailable", "description": "This fight is not currently supported."},
    )
    return {
        "id": f"{normalize_name(fight.fighter_a)}-vs-{normalize_name(fight.fighter_b)}",
        "fighter_a": fight.fighter_a,
        "fighter_b": fight.fighter_b,
        "status": fight.status,
        "reason_label": meta["label"],
        "reason": fight.reason or meta["description"],
        "description": meta["description"],
    }


def _fight_books_from_payload(
    fighter_a: str,
    fighter_b: str,
    odds_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    books = odds_payload.get("books", {})
    direct_key = f"{normalize_name(fighter_a)}-vs-{normalize_name(fighter_b)}"
    reverse_key = f"{normalize_name(fighter_b)}-vs-{normalize_name(fighter_a)}"
    entry = books.get(direct_key) or books.get(reverse_key)
    if not entry:
        return []

    stored_a = normalize_name(entry.get("fighter_a", fighter_a))
    is_direct = stored_a == normalize_name(fighter_a)

    quotes: list[dict[str, Any]] = []
    for quote in entry.get("quotes", []):
        a_price = quote.get("fighter_a_price")
        b_price = quote.get("fighter_b_price")
        if not is_direct:
            a_price, b_price = b_price, a_price

        a_decimal = (
            round(float(american_to_decimal(int(a_price))), 4)
            if isinstance(a_price, int)
            else None
        )
        b_decimal = (
            round(float(american_to_decimal(int(b_price))), 4)
            if isinstance(b_price, int)
            else None
        )
        quotes.append(
            {
                "sportsbook": quote.get("sportsbook", "Unknown Book"),
                "updated_at": quote.get("last_update"),
                "fighter_a_price": a_price,
                "fighter_b_price": b_price,
                "fighter_a_decimal": a_decimal,
                "fighter_b_decimal": b_decimal,
            }
        )

    return quotes


def _fighter_rows(
    fighter_a: str,
    fighter_b: str,
    fighters_df: pd.DataFrame,
    fighter_lookup: dict[str, str],
) -> tuple[pd.Series, pd.Series]:
    a_name = fighter_lookup[normalize_name(fighter_a)]
    b_name = fighter_lookup[normalize_name(fighter_b)]
    return (
        fighters_df[fighters_df["fighter"] == a_name].iloc[0],
        fighters_df[fighters_df["fighter"] == b_name].iloc[0],
    )


def build_supported_fight_payload(
    fight,
    fighters_df: pd.DataFrame,
    fighter_lookup: dict[str, str],
    odds_map: dict[str, list[int]],
    odds_payload: dict[str, Any],
) -> dict[str, Any]:
    """Build the rich product payload for a supported fight."""
    fighter_a = fight.fighter_a
    fighter_b = fight.fighter_b
    prob_a = float(fight.prob_a)
    prob_b = float(1.0 - prob_a)
    favored = fighter_a if prob_a >= 0.5 else fighter_b
    pick_side = "a" if favored == fighter_a else "b"
    p_pick = prob_a if pick_side == "a" else prob_b

    odds_list_a = get_odds_for_fighter(fighter_a, odds_map)
    odds_list_b = get_odds_for_fighter(fighter_b, odds_map)
    market_prob_a = _median_market_prob(odds_list_a)
    market_prob_b = _median_market_prob(odds_list_b)
    market_prob_pick = market_prob_a if pick_side == "a" else market_prob_b
    best_odds_pick = _best_decimal(odds_list_a if pick_side == "a" else odds_list_b)
    edge_pick = None if market_prob_pick is None else float(p_pick - market_prob_pick)
    value_state = _value_state(edge_pick)
    confidence = _confidence_meta(p_pick)
    kelly = kelly_fraction(p_pick, best_odds_pick or float("nan"))

    A, B = _fighter_rows(fighter_a, fighter_b, fighters_df, fighter_lookup)
    chips = build_insight_chips(fighter_a, fighter_b, A, B, prob_a, market_prob_a)
    comparisons = build_feature_comparison(A, B)

    return {
        "id": f"{normalize_name(fighter_a)}-vs-{normalize_name(fighter_b)}",
        "fighter_a": fighter_a,
        "fighter_b": fighter_b,
        "probability_a": _safe_round(prob_a, 4),
        "probability_b": _safe_round(prob_b, 4),
        "favored_fighter": favored,
        "pick_side": pick_side,
        "pick_probability": _safe_round(p_pick, 4),
        "confidence": confidence,
        "odds_available": bool(best_odds_pick is not None),
        "best_odds_decimal": _safe_round(best_odds_pick, 4),
        "market_probability_a": _safe_round(market_prob_a, 4),
        "market_probability_pick": _safe_round(market_prob_pick, 4),
        "edge_pick": _safe_round(edge_pick, 4),
        "value_state": value_state,
        "kelly_fraction": _safe_round(kelly, 4),
        "insight_chips": chips,
        "model_lean": build_model_lean(favored, chips, value_state),
        "feature_comparison": comparisons,
        "odds_list_a": odds_list_a,
        "odds_list_b": odds_list_b,
        "sportsbook_quotes": _fight_books_from_payload(fighter_a, fighter_b, odds_payload),
    }


def build_event_summary(
    event_meta: dict[str, Any],
    coverage,
    odds_payload: dict[str, Any],
    timeline: str,
) -> dict[str, Any]:
    """Build a compact event summary for the event index/landing views."""
    if coverage is None:
        status_meta = ARCHIVE_STATUS_META
        fights = []
        supported = []
        unsupported = []
        featured = None
        subtitle = "Historical event artifact detected, but no local fight card is available."
    else:
        status_meta = STATUS_META[coverage.status]
        fights = coverage.fights
        supported = coverage.predictable_fights
        unsupported = coverage.unpredictable_fights
        featured = supported[0] if supported else fights[0] if fights else None
        subtitle = "Calibrated prefight intelligence for supported bouts."

    featured_matchup = None
    if featured:
        featured_matchup = f"{featured.fighter_a} vs {featured.fighter_b}"

    return {
        "event_id": event_meta["event_id"],
        "event_number": event_meta["event_number"],
        "title": event_meta["title"],
        "subtitle": subtitle,
        "date": _parse_event_date_from_id(event_meta["event_id"]),
        "venue": None,
        "location": None,
        "timeline": timeline,
        "has_card": event_meta["has_card"],
        "has_odds": event_meta["has_odds"],
        "coverage_status": coverage.status if coverage is not None else "archived_data",
        "coverage": status_meta,
        "supported_count": len(supported),
        "unsupported_count": len(unsupported),
        "total_count": len(fights),
        "coverage_ratio": _safe_round(len(supported) / len(fights), 3) if fights else 0,
        "featured_matchup": featured_matchup,
        "odds_last_updated": odds_payload.get("scraped_at"),
    }


def list_event_summaries(
    fighters_df: pd.DataFrame,
    fighter_lookup: dict[str, str],
    feature_cols: list[str],
    clip_bounds: dict,
    booster,
    calibrator,
) -> list[dict[str, Any]]:
    """Return product-friendly event summaries for every local event card."""
    events: list[dict[str, Any]] = []
    all_items = list_available_event_items()
    for event_meta in all_items:
        event_id = event_meta["event_id"]
        timeline = _timeline_bucket(event_meta, all_items)
        coverage = None

        if event_meta["has_card"]:
            fights = load_fight_card(event_id)
            coverage = check_event_coverage(
                event_meta["event_number"] or 0,
                fights,
                fighters_df,
                fighter_lookup,
                feature_cols,
                clip_bounds,
                booster,
                calibrator,
            )

        events.append(
            build_event_summary(
                event_meta,
                coverage,
                load_odds_payload(event_id),
                timeline,
            )
        )

    return sorted(
        events,
        key=lambda event: (
            1 if event["timeline"] == "future" else 0,
            1 if event["event_number"] is not None else 0,
            event["event_number"] or 0,
            event["event_id"],
        ),
        reverse=True,
    )


def build_event_detail(
    event_id: str,
    fighters_df: pd.DataFrame,
    fighter_lookup: dict[str, str],
    feature_cols: list[str],
    clip_bounds: dict,
    booster,
    calibrator,
) -> dict[str, Any]:
    """Return the full product payload for one event detail page."""
    all_items = list_available_event_items()
    event_meta = next((item for item in all_items if item["event_id"] == event_id), None)
    if event_meta is None:
        raise FileNotFoundError(f"No local event artifacts found for '{event_id}'")

    event_number = event_number_from_id(event_id)
    timeline = _timeline_bucket(event_meta, all_items)
    odds_payload = load_odds_payload(event_id)
    odds_map = odds_payload.get("odds", {})
    coverage = None
    supported: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []

    if event_meta["has_card"]:
        fights = load_fight_card(event_id)
        coverage = check_event_coverage(
            event_number or 0,
            fights,
            fighters_df,
            fighter_lookup,
            feature_cols,
            clip_bounds,
            booster,
            calibrator,
        )
        supported = [
            build_supported_fight_payload(
                fight,
                fighters_df,
                fighter_lookup,
                odds_map,
                odds_payload,
            )
            for fight in coverage.predictable_fights
        ]
        unsupported = [_humanize_unavailable(fight) for fight in coverage.unpredictable_fights]

    summary = build_event_summary(
        event_meta,
        coverage,
        odds_payload,
        timeline,
    )

    return {
        "event": summary,
        "hero": {
            "title": summary["title"],
            "subtitle": summary["coverage"]["description"],
            "date": summary["date"],
            "venue": summary["venue"],
            "location": summary["location"],
            "coverage": summary["coverage"],
            "summary": f"{summary['supported_count']} supported fights · {summary['unsupported_count']} unavailable",
            "odds_last_updated": summary["odds_last_updated"],
            "odds_status": (
                f"Odds synced {summary['odds_last_updated']}"
                if summary["odds_last_updated"]
                else (
                    "Odds not synced yet for this event."
                    if summary["timeline"] == "future"
                    else "Historical odds snapshot unavailable."
                )
            ),
        },
        "supported_fights": supported,
        "unsupported_fights": unsupported,
    }
