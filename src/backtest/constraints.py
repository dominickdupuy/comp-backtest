"""Enforce the competition's position/leverage constraints on a weight matrix.

Weights are a wide DataFrame: index = rebalance dates, columns = tickers,
values = target portfolio weight (fraction of equity; negative = short).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def apply_position_cap(weights: pd.DataFrame, cap: float) -> pd.DataFrame:
    """Clip each name to +/- cap (the 10% per-name entry cap)."""
    if cap >= 1.0:
        return weights
    return weights.clip(lower=-cap, upper=cap)


def enforce_no_shorting(weights: pd.DataFrame, allow_shorting: bool) -> pd.DataFrame:
    if allow_shorting:
        return weights
    return weights.clip(lower=0.0)


def scale_to_gross(weights: pd.DataFrame, target_gross: float) -> pd.DataFrame:
    """Scale each row so sum(|w|) == target_gross (e.g. 2.0 for 2x).

    Rows that are entirely zero are left untouched.
    """
    gross = weights.abs().sum(axis=1)
    factor = pd.Series(0.0, index=weights.index)
    nonzero = gross > 0
    factor[nonzero] = target_gross / gross[nonzero]
    return weights.mul(factor, axis=0)


def cap_net_exposure(weights: pd.DataFrame, max_net: float) -> pd.DataFrame:
    """If sum(w) on a row exceeds max_net, shrink toward the cap.

    Uniformly de-levers a row whose net exceeds the cap while preserving the
    relative weights. Net below the cap is left alone.
    """
    net = weights.sum(axis=1)
    over = net.abs() > max_net
    if not over.any():
        return weights
    out = weights.copy()
    factor = (max_net / net.abs()).where(over, 1.0)
    out = out.mul(factor, axis=0)
    return out


def apply_constraints(
    weights: pd.DataFrame,
    *,
    max_position_weight: float,
    target_gross_leverage: float,
    max_gross_leverage: float,
    max_net_leverage: float,
    allow_shorting: bool,
) -> pd.DataFrame:
    """Full constraint pipeline.

    Order matters: cap names first, then scale gross to target, then re-cap
    (scaling can push a name back over the cap), then enforce hard leverage
    ceilings. A short iteration converges the cap<->gross interaction.
    """
    w = enforce_no_shorting(weights.fillna(0.0), allow_shorting)
    gross_target = min(target_gross_leverage, max_gross_leverage)

    for _ in range(3):  # cap and gross-scale interact; a few passes converge
        w = apply_position_cap(w, max_position_weight)
        w = scale_to_gross(w, gross_target)

    # Final hard ceilings (defensive; scaling above already targets gross).
    w = apply_position_cap(w, max_position_weight)
    gross = w.abs().sum(axis=1)
    over = gross > max_gross_leverage
    if over.any():
        w = w.mul((max_gross_leverage / gross).where(over, 1.0), axis=0)
    w = cap_net_exposure(w, max_net_leverage)
    return w.replace([np.inf, -np.inf], 0.0).fillna(0.0)
