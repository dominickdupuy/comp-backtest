"""Leverage / sizing posture for a rank-based, pure-PnL tournament.

Standard growth theory says full-Kelly maximizes long-run wealth and fractional
Kelly is "production safe". None of that caution applies when only raw terminal
PnL and *rank* matter over a fixed 2-month horizon: tournament theory
(Browne 1999/2000; Brown-Harlow-Starks 1996; Dubins-Savage bold play) says to
deliberately OVER-bet -- push gross to the cap, concentrate, and escalate when
behind. These helpers express that posture; the engine still hard-clips to the
competition's leverage ceiling, so over-betting can never exceed the rules.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def continuous_kelly(returns: pd.Series, risk_free: float = 0.0) -> float:
    """Continuous-form Kelly fraction f* = mu / sigma^2 (annualized excess)."""
    mu = returns.mean() * 252 - risk_free
    var = returns.var() * 252
    if var <= 0:
        return 0.0
    return float(mu / var)


def target_gross(
    base_gross: float,
    max_gross: float,
    kelly_multiple: float = 2.0,
) -> float:
    """Super-Kelly target gross, clipped to the competition ceiling."""
    return float(min(base_gross * kelly_multiple, max_gross))


def escalation_factor(
    days_elapsed: int,
    contest_length_days: int,
    rank_pct: float | None,
    behind_threshold: float = 0.5,
) -> float:
    """Multiplier in [1, 2] that ramps risk in the final third *if behind*.

    rank_pct is the competitor's current standing in [0, 1] (1 = leading). If
    leading, returns <1 to de-risk and lock rank (Browne's cut-once-ahead).
    """
    if rank_pct is None:
        return 1.0
    progress = days_elapsed / max(1, contest_length_days)
    if progress < 0.66:
        return 1.0
    if rank_pct >= 0.9:               # clearly leading -> lock the rank
        return 0.5
    if rank_pct < behind_threshold:   # trailing -> swing for the fences
        return 1.0 + (1.0 - rank_pct)
    return 1.0
