"""
Fight and event eligibility checking.

Returns structured status codes so the UI/API can explain exactly why a
fight cannot be predicted, rather than crashing or silently returning garbage.

Status taxonomy
───────────────
Fight-level statuses:
  predictable                       — all checks passed, prob_a is set
  unsupported_missing_fighter_a     — Fighter A not found in snapshot
  unsupported_missing_fighter_b     — Fighter B not found in snapshot
  unsupported_feature_build_failed  — feature construction or inference raised

Event-level statuses:
  fully_predictable                 — every fight on the card is predictable
  partially_predictable             — at least one but not all fights predictable
  not_predictable                   — no fights on the card are predictable
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from .inference import build_feature_row, predict_win_prob, preprocess_row
from .names import normalize_name

# ── Type aliases ───────────────────────────────────────────────────────────────
FightStatus = Literal[
    "predictable",
    "unsupported_missing_fighter_a",
    "unsupported_missing_fighter_b",
    "unsupported_feature_build_failed",
]

EventStatus = Literal[
    "fully_predictable",
    "partially_predictable",
    "not_predictable",
]


# ── Result dataclasses ─────────────────────────────────────────────────────────

@dataclass
class FightEligibility:
    fighter_a: str
    fighter_b: str
    status: FightStatus
    reason: str = ""
    prob_a: float | None = None  # populated only when status == "predictable"

    @property
    def is_predictable(self) -> bool:
        return self.status == "predictable"


@dataclass
class EventCoverage:
    event_number: int
    status: EventStatus
    fights: list[FightEligibility] = field(default_factory=list)

    @property
    def predictable_fights(self) -> list[FightEligibility]:
        return [f for f in self.fights if f.is_predictable]

    @property
    def unpredictable_fights(self) -> list[FightEligibility]:
        return [f for f in self.fights if not f.is_predictable]


# ── Checkers ───────────────────────────────────────────────────────────────────

def check_fight_eligibility(
    fighter_a: str,
    fighter_b: str,
    fighters_df: pd.DataFrame,
    fighter_lookup: dict[str, str],
    feature_cols: list[str],
    clip_bounds: dict,
    booster,
    calibrator,
) -> FightEligibility:
    """
    Run all preflight eligibility checks for one matchup.

    Checks (in order):
      1. Fighter A present in snapshot
      2. Fighter B present in snapshot
      3. Feature construction + inference succeeds

    Each check returns immediately on failure with a descriptive reason.
    """
    a_norm = normalize_name(fighter_a)
    b_norm = normalize_name(fighter_b)

    if a_norm not in fighter_lookup:
        return FightEligibility(
            fighter_a=fighter_a,
            fighter_b=fighter_b,
            status="unsupported_missing_fighter_a",
            reason=(
                f"'{fighter_a}' (normalized: '{a_norm}') not found in fighter snapshot. "
                "Likely a recent debut or a name spelling mismatch."
            ),
        )

    if b_norm not in fighter_lookup:
        return FightEligibility(
            fighter_a=fighter_a,
            fighter_b=fighter_b,
            status="unsupported_missing_fighter_b",
            reason=(
                f"'{fighter_b}' (normalized: '{b_norm}') not found in fighter snapshot. "
                "Likely a recent debut or a name spelling mismatch."
            ),
        )

    try:
        A = fighters_df[fighters_df["fighter"] == fighter_lookup[a_norm]].iloc[0]
        B = fighters_df[fighters_df["fighter"] == fighter_lookup[b_norm]].iloc[0]

        row = build_feature_row(A, B)
        X = preprocess_row(pd.DataFrame([row]), feature_cols, clip_bounds)
        prob_a = predict_win_prob(booster, calibrator, X)
    except Exception as exc:
        return FightEligibility(
            fighter_a=fighter_a,
            fighter_b=fighter_b,
            status="unsupported_feature_build_failed",
            reason=f"Feature construction or inference raised: {exc}",
        )

    return FightEligibility(
        fighter_a=fighter_a,
        fighter_b=fighter_b,
        status="predictable",
        prob_a=prob_a,
    )


def check_event_coverage(
    event_number: int,
    fights: list[list[str]],
    fighters_df: pd.DataFrame,
    fighter_lookup: dict[str, str],
    feature_cols: list[str],
    clip_bounds: dict,
    booster,
    calibrator,
) -> EventCoverage:
    """
    Run eligibility checks on every fight on a card.

    Never raises — each individual fight failure is captured as a structured
    FightEligibility with the appropriate unsupported_* status.
    """
    results: list[FightEligibility] = []

    for fighter_a, fighter_b in fights:
        eligibility = check_fight_eligibility(
            fighter_a, fighter_b,
            fighters_df, fighter_lookup,
            feature_cols, clip_bounds,
            booster, calibrator,
        )
        results.append(eligibility)

    n_predictable = sum(1 for r in results if r.is_predictable)
    n_total = len(results)

    if n_total == 0 or n_predictable == 0:
        event_status: EventStatus = "not_predictable"
    elif n_predictable == n_total:
        event_status = "fully_predictable"
    else:
        event_status = "partially_predictable"

    return EventCoverage(
        event_number=event_number,
        status=event_status,
        fights=results,
    )
