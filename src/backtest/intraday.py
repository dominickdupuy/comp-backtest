"""Intraday cross-sectional reversal backtest -- the turnover-sensitive sleeve.

Tests the hypothesis that, with ZERO transaction costs, rebalancing more often
than once/day adds PnL. The 15-20 min delay only kills sub-20-min alpha, so we
rebalance on intraday bars (e.g. 30/60 min) and apply an execution lag equal to
the delay. Overnight returns are excluded (separate signal). Output is a
bar-frequency return series we can compound to a total return and compare
against the once-daily sleeve on the same names/window.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class IntradayResult:
    bar_returns: pd.Series
    total_return: float
    n_bars: int
    bars_per_day: float
    rebalance_freq: str
    delay_bars: int


def _intraday_bar_returns(close: pd.DataFrame) -> pd.DataFrame:
    """Bar-over-bar returns, with the first bar of each day set to NaN so
    overnight (close->open) moves are excluded."""
    ret = close.pct_change(fill_method=None)
    day = close.index.normalize()
    first_of_day = pd.Series(day, index=close.index).groupby(day).cumcount() == 0
    ret[first_of_day.values] = np.nan
    return ret


def run_intraday_reversal(
    close: pd.DataFrame,
    *,
    rebalance_freq: str = "30min",
    lookback_bars: int = 1,
    delay_minutes: int = 20,
    delay_bars: int | None = None,
    exclude_overnight: bool = True,
    skip_bars: int = 0,
    spread_bps: float = 0.0,
    long_quantile: float = 0.2,
    short_quantile: float = 0.2,
    target_gross_leverage: float = 2.0,
    max_position_weight: float = 0.10,
) -> IntradayResult:
    """Cross-sectional reversal on intraday bars under zero costs.

    `close` is a wide (ts x symbol) panel already resampled to `rebalance_freq`.
    Score = -(trailing `lookback_bars` return). Long the biggest losers, short
    the biggest winners, dollar-neutral, scaled to `target_gross_leverage` with
    a per-name cap. Weights are lagged by ceil(delay/bar) bars (the 20-min lag).

    Set ``exclude_overnight=False`` and ``delay_bars=1`` to run the once-daily
    baseline on a daily close panel with identical construction.
    """
    if exclude_overnight:
        bar_ret = _intraday_bar_returns(close)
    else:
        bar_ret = close.pct_change(fill_method=None)

    # Trailing return as the reversal signal. `skip_bars` leaves a gap between
    # the signal window and the fill so the position is NOT formed from the very
    # bar whose last-trade bounce reverts next bar (the dominant fake-alpha source).
    signal = -(close.shift(skip_bars).pct_change(lookback_bars, fill_method=None))

    # Cross-sectional long/short weights per bar.
    ranks = signal.rank(axis=1, pct=True)
    longs = (ranks >= 1 - long_quantile).astype(float)
    shorts = (ranks <= short_quantile).astype(float)
    w = longs.div(longs.sum(axis=1).replace(0, np.nan), axis=0) \
        - shorts.div(shorts.sum(axis=1).replace(0, np.nan), axis=0)
    w = w.fillna(0.0)

    # Per-name cap, then scale gross to target.
    w = w.clip(-max_position_weight, max_position_weight)
    gross = w.abs().sum(axis=1).replace(0, np.nan)
    w = w.mul((target_gross_leverage / gross).fillna(0.0), axis=0)

    # Execution delay: act on bar t, fill `delay_bars` bars later.
    if delay_bars is None:
        offset = pd.tseries.frequencies.to_offset(rebalance_freq)
        bar_minutes = pd.Timedelta(offset).total_seconds() / 60
        delay_bars = max(1, int(np.ceil(delay_minutes / max(bar_minutes, 1))))
    held = w.shift(delay_bars).fillna(0.0)

    port_ret = (held * bar_ret.reindex_like(held).fillna(0.0)).sum(axis=1)

    # Effective half-spread on two-way turnover. "Zero commissions" does NOT
    # mean fills at the last-trade price: you buy at the ask, sell at the bid.
    # Even a 1-2 bps spread on the huge intraday turnover removes the bounce
    # mirage. Set spread_bps=0 to reproduce the (unrealistic) last-trade fantasy.
    if spread_bps:
        prev = held.shift(1).fillna(0.0)
        turn = 0.5 * (held - prev).abs().sum(axis=1)
        port_ret = port_ret - (spread_bps / 1e4) * 2.0 * turn

    port_ret = port_ret.astype("float64").fillna(0.0)

    n_days = close.index.normalize().nunique()
    return IntradayResult(
        bar_returns=port_ret,
        total_return=float((1.0 + port_ret).prod() - 1.0),
        n_bars=len(port_ret),
        bars_per_day=len(port_ret) / max(1, n_days),
        rebalance_freq=rebalance_freq,
        delay_bars=delay_bars,
    )
