"""
Betting math utilities (pure functions, no I/O, no UI dependencies).

All formulas match those documented in the Streamlit Betting Guide page.
"""
from __future__ import annotations
import numpy as np


# ── Odds Conversion ────────────────────────────────────────────────────────────

def american_to_decimal(odds: int) -> float:
    """Convert American moneyline odds to decimal odds."""
    if odds == 0:
        return np.nan
    if odds > 0:
        return 1.0 + odds / 100.0
    return 1.0 + 100.0 / abs(odds)


def implied_prob_from_decimal(decimal_odds: float) -> float:
    """Return implied win probability from decimal odds."""
    if decimal_odds is None or np.isnan(decimal_odds) or decimal_odds <= 1.0:
        return np.nan
    return 1.0 / decimal_odds


# ── Bet Sizing ─────────────────────────────────────────────────────────────────

def ev_per_dollar(p: float, d: float) -> float:
    """Expected profit per $1 bet: p*d - 1."""
    if d is None or np.isnan(d) or d <= 1.0:
        return np.nan
    return float(p * d - 1.0)


def kelly_fraction(p: float, d: float) -> float:
    """
    Full Kelly criterion fraction.

    Returns max(0, (b*p - (1-p)) / b) where b = d - 1.
    Returns 0.0 when odds are invalid or Kelly is negative.
    """
    if d is None or np.isnan(d) or d <= 1.0:
        return 0.0
    b = d - 1.0
    return float(max(0.0, (b * p - (1 - p)) / b))


def kelly_note(fk: float) -> str:
    """Human-readable sizing signal label for a Kelly fraction."""
    if fk >= 0.05:
        return "Large sizing signal"
    if fk >= 0.02:
        return "Moderate sizing signal"
    if fk > 0:
        return "Small sizing signal"
    return "No sizing signal"


def safe_float(x, default: float = np.nan) -> float:
    """Coerce x to float, returning default on any failure."""
    try:
        return float(x)
    except Exception:
        return default


# ── Simulation ─────────────────────────────────────────────────────────────────

def simulate_bankroll_paths(
    bets: list[dict],
    initial_bankroll: float,
    n_sims: int,
    kelly_mult: float,
) -> np.ndarray:
    """
    Monte Carlo bankroll simulation over a set of qualifying bets.

    Each bet dict must contain:
      p                    float  model win probability
      decimal_odds         float  decimal payout odds
      stake_frac_full_kelly float full Kelly fraction (pre-multiplier)

    Actual stake per bet = bankroll * stake_frac_full_kelly * kelly_mult

    Returns array of shape (n_sims,) with final bankroll values.
    Returns empty array if no bets provided.
    """
    if not bets:
        return np.array([])

    final_bankrolls = np.zeros(n_sims, dtype=float)

    for s in range(n_sims):
        bankroll = float(initial_bankroll)
        for b in bets:
            p = float(b["p"])
            d = float(b["decimal_odds"])
            fk = float(b["stake_frac_full_kelly"])
            stake = bankroll * fk * float(kelly_mult)

            if stake <= 1e-9:
                continue

            if np.random.rand() < p:
                bankroll += stake * (d - 1.0)
            else:
                bankroll -= stake

            if bankroll <= 0:
                bankroll = 0.0
                break

        final_bankrolls[s] = bankroll

    return final_bankrolls
