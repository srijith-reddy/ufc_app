"""
UFC Fight Predictor — FastAPI backend.

This is the Vercel/API entrypoint.

Exposes both low-level and product-facing read-only endpoints:
  GET /health                               liveness check
  GET /events                               list events with available cards
  GET /v1/health                            product API health
  GET /v1/events                            product event summaries
  GET /v1/events/{event_number}             premium event detail payload
  GET /events/{event_number}/coverage       eligibility per fight
  GET /events/{event_number}/predictions    full predictions + odds + betting metrics

All model artifacts are loaded once at startup via lifespan().
Inference is deterministic: given the same fighter snapshot + event card,
the same probabilities are always returned.

Deployment:
  Local:   uvicorn apps.vercel.api.main:app --reload
  Prod:    uvicorn apps.vercel.api.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import sys
from pathlib import Path
import math
from contextlib import asynccontextmanager
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.service import build_event_detail, list_event_summaries
from core import (
    american_to_decimal,
    check_event_coverage,
    ev_per_dollar,
    get_odds_for_fighter,
    implied_prob_from_decimal,
    kelly_fraction,
    kelly_note,
    list_available_event_ids,
    list_available_events,
    load_artifacts,
    load_fight_card,
    load_odds_map,
    normalize_event_id,
)

# ── Globals populated at startup ───────────────────────────────────────────────
_booster = None
_calibrator = None
_feature_cols: list[str] = []
_clip_bounds: dict = {}
_fighters_df = None
_fighter_lookup: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts once before the first request."""
    global _booster, _calibrator, _feature_cols, _clip_bounds, _fighters_df, _fighter_lookup
    _booster, _calibrator, _feature_cols, _clip_bounds, _fighters_df, _fighter_lookup = (
        load_artifacts()
    )
    yield
    # Nothing to clean up for in-process artifacts


app = FastAPI(
    title="UFC Fight Predictor API",
    version="1.0.0",
    description="Prefight-only calibrated XGBoost predictions with betting metrics.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten to your frontend origin in production
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _nan_to_none(x: Any) -> Any:
    """Convert float NaN → None for JSON-safe serialization."""
    if isinstance(x, float) and math.isnan(x):
        return None
    return x


def _build_fight_prediction(fight_eligibility, odds_map: dict) -> dict:
    """
    Given a FightEligibility result and an odds map, return the full
    prediction payload for one fight (odds, EV, Kelly, etc.).

    Called only when fight_eligibility.is_predictable is True.
    """
    f = fight_eligibility
    prob_a = f.prob_a

    winner = f.fighter_a if prob_a >= 0.5 else f.fighter_b
    pick_side = "A" if winner == f.fighter_a else "B"
    p_pick = float(prob_a if pick_side == "A" else 1.0 - prob_a)

    odds_list_a = get_odds_for_fighter(f.fighter_a, odds_map)
    odds_list_b = get_odds_for_fighter(f.fighter_b, odds_map)

    dec_a = [american_to_decimal(o) for o in odds_list_a]
    dec_b = [american_to_decimal(o) for o in odds_list_b]

    # Market probability — uses median to filter outlier books
    market_prob_a: float | None = None
    if dec_a:
        median_dec_a = float(np.median(dec_a))
        raw = implied_prob_from_decimal(median_dec_a)
        market_prob_a = None if math.isnan(raw) else float(raw)

    # Best available odds for the pick side (used for EV + Kelly)
    best_odds_dec: float | None = None
    odds_count_pick = 0
    if pick_side == "A" and dec_a:
        best_odds_dec = float(np.max(dec_a))
        odds_count_pick = len(dec_a)
    elif pick_side == "B" and dec_b:
        best_odds_dec = float(np.max(dec_b))
        odds_count_pick = len(dec_b)

    ev = _nan_to_none(ev_per_dollar(p_pick, best_odds_dec or float("nan")))
    fk = kelly_fraction(p_pick, best_odds_dec or float("nan"))

    edge_a: float | None = None
    if market_prob_a is not None:
        edge_a = float(prob_a - market_prob_a)

    return {
        "fighter_a": f.fighter_a,
        "fighter_b": f.fighter_b,
        "status": f.status,
        "reason": f.reason,
        "prob_a": round(prob_a, 4),
        "winner": winner,
        "pick_side": pick_side,
        "p_pick": round(p_pick, 4),
        "market_prob_a": round(market_prob_a, 4) if market_prob_a is not None else None,
        "edge_a": round(edge_a, 4) if edge_a is not None else None,
        "best_odds_dec": round(best_odds_dec, 4) if best_odds_dec is not None else None,
        "ev_per_dollar": round(ev, 4) if ev is not None else None,
        "kelly_fraction": round(fk, 4),
        "kelly_note": kelly_note(fk),
        "odds_count_pick": odds_count_pick,
        "odds_list_a": odds_list_a,
        "odds_list_b": odds_list_b,
    }


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Liveness check. Returns 200 if the API is running and artifacts loaded."""
    return {"status": "ok", "artifacts_loaded": _booster is not None}


@app.get("/events")
def get_events():
    """
    Return all event numbers for which a local fight card exists.

    Only events with a cards/ufc_NNN.json file are listed.
    Odds availability is separate — check /events/{n}/predictions.
    """
    return {"events": list_available_events()}


@app.get("/v1/health")
def product_health():
    """Product API liveness check for the premium frontend."""
    return {
        "status": "ok",
        "product_api": True,
        "artifacts_loaded": _booster is not None,
        "available_events": list_available_event_ids(),
        "available_numbered_events": list_available_events(),
    }


@app.get("/v1/events")
def get_product_events():
    """Return product-ready event summaries for the premium frontend."""
    events = list_event_summaries(
        _fighters_df,
        _fighter_lookup,
        _feature_cols,
        _clip_bounds,
        _booster,
        _calibrator,
    )
    future_events = [event for event in events if event.get("timeline") == "future"]
    past_events = [event for event in events if event.get("timeline") == "past"]
    return {
        "events": events,
        "future_events": future_events,
        "past_events": past_events,
        "featured_event": future_events[0] if future_events else events[0] if events else None,
    }


@app.get("/v1/events/{event_id:path}")
def get_product_event_detail(event_id: str):
    """Return the full premium event payload with supported and unsupported fights."""
    try:
        return build_event_detail(
            normalize_event_id(event_id),
            _fighters_df,
            _fighter_lookup,
            _feature_cols,
            _clip_bounds,
            _booster,
            _calibrator,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.get("/events/{event_number}/coverage")
def get_event_coverage(event_number: int):
    """
    Return fight-level eligibility for every bout on the card.

    For each fight, status is one of:
      predictable                       — prediction available
      unsupported_missing_fighter_a     — Fighter A absent from snapshot
      unsupported_missing_fighter_b     — Fighter B absent from snapshot
      unsupported_feature_build_failed  — inference error

    Use this endpoint to understand why some fights cannot be predicted.
    """
    try:
        fights = load_fight_card(event_number)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    coverage = check_event_coverage(
        event_number, fights,
        _fighters_df, _fighter_lookup,
        _feature_cols, _clip_bounds,
        _booster, _calibrator,
    )

    return {
        "event_number": event_number,
        "status": coverage.status,
        "predictable_count": len(coverage.predictable_fights),
        "total_count": len(coverage.fights),
        "fights": [
            {
                "fighter_a": f.fighter_a,
                "fighter_b": f.fighter_b,
                "status": f.status,
                "reason": f.reason,
            }
            for f in coverage.fights
        ],
    }


@app.get("/events/{event_number}/predictions")
def get_event_predictions(event_number: int):
    """
    Return full predictions for every fight on the card.

    Predictable fights include:
      prob_a, winner, pick_side, p_pick
      market_prob_a, edge_a
      best_odds_dec, ev_per_dollar, kelly_fraction, kelly_note
      odds_list_a, odds_list_b

    Unpredictable fights include only status + reason.
    """
    try:
        fights = load_fight_card(event_number)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    odds_map = load_odds_map(event_number)

    coverage = check_event_coverage(
        event_number, fights,
        _fighters_df, _fighter_lookup,
        _feature_cols, _clip_bounds,
        _booster, _calibrator,
    )

    results = []
    for f in coverage.fights:
        if not f.is_predictable:
            results.append({
                "fighter_a": f.fighter_a,
                "fighter_b": f.fighter_b,
                "status": f.status,
                "reason": f.reason,
            })
        else:
            results.append(_build_fight_prediction(f, odds_map))

    return {
        "event_number": event_number,
        "event_status": coverage.status,
        "predictable_count": len(coverage.predictable_fights),
        "total_count": len(coverage.fights),
        "fights": results,
    }
