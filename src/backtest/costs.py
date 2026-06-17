"""Transaction-cost model.

The competition specifies ZERO commissions and frictionless fills, so the
defaults here are 0. The model is kept fully parameterized (commission per
turnover dollar + slippage in bps) so the *same* code can stress-test the
strategy under realistic frictions -- a caveat flagged in the research: the
microcap edge depends entirely on costs being zero.
"""
from __future__ import annotations

import pandas as pd


def turnover(weights: pd.DataFrame) -> pd.Series:
    """One-way turnover per period: 0.5 * sum |w_t - w_{t-1}|.

    The first period's turnover equals 0.5 * sum|w_0| (building the book).
    """
    prev = weights.shift(1).fillna(0.0)
    return 0.5 * (weights - prev).abs().sum(axis=1)


def cost_drag(
    weights: pd.DataFrame,
    *,
    commission_per_trade: float,
    slippage_bps: float,
) -> pd.Series:
    """Per-period return drag from trading, as a positive fraction of equity.

    commission_per_trade is treated as a cost per dollar of two-way turnover
    (a flat fee model collapses to ~0 for a large book; for the zero-cost
    competition this is 0 regardless). slippage_bps applies to two-way turnover.
    """
    tno = turnover(weights)
    two_way = 2.0 * tno
    slip = (slippage_bps / 1e4) * two_way
    comm = commission_per_trade * two_way
    return slip + comm
