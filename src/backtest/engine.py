"""Cross-sectional long/short backtest engine.

Pipeline:
    target weights (per rebalance date)  ->  apply competition constraints
      ->  lag to prevent lookahead  ->  hold between rebalances
      ->  daily portfolio return = sum_i w_i * r_i  ->  subtract cost drag
      ->  equity curve.

Everything is vectorized over a wide returns panel (index=date, cols=ticker).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .constraints import apply_constraints
from .costs import cost_drag, turnover


@dataclass
class BacktestResult:
    returns: pd.Series          # daily portfolio return (after costs)
    equity: pd.Series           # equity curve, starting_capital -> ...
    weights: pd.Series | pd.DataFrame  # held weights per day (after lag/ffill)
    gross_leverage: pd.Series
    net_exposure: pd.Series
    turnover: pd.Series
    starting_capital: float

    @property
    def total_return(self) -> float:
        return float(self.equity.iloc[-1] / self.starting_capital - 1.0)


def run_backtest(
    target_weights: pd.DataFrame,
    daily_returns: pd.DataFrame,
    *,
    starting_capital: float = 1_000_000.0,
    signal_lag_days: int = 1,
    max_position_weight: float = 0.10,
    target_gross_leverage: float = 2.0,
    max_gross_leverage: float = 2.0,
    max_net_leverage: float = 2.0,
    allow_shorting: bool = True,
    commission_per_trade: float = 0.0,
    slippage_bps: float = 0.0,
) -> BacktestResult:
    """Run the backtest.

    Parameters
    ----------
    target_weights : wide DataFrame indexed by *rebalance* dates.
    daily_returns  : wide DataFrame of simple daily returns indexed by trading
                     day. Columns are the tradable universe; the intersection of
                     columns with ``target_weights`` is used.
    """
    # Align universe.
    cols = daily_returns.columns.intersection(target_weights.columns)
    if len(cols) == 0:
        raise ValueError("No overlapping tickers between weights and returns.")
    tw = target_weights[cols].copy()
    rets = daily_returns[cols].copy()

    # 1) Enforce competition constraints on the *target* book.
    tw = apply_constraints(
        tw,
        max_position_weight=max_position_weight,
        target_gross_leverage=target_gross_leverage,
        max_gross_leverage=max_gross_leverage,
        max_net_leverage=max_net_leverage,
        allow_shorting=allow_shorting,
    )

    # 2) Expand rebalance-date weights onto every trading day, then lag so that
    #    a book formed from info through close(T-1) is held on day T.
    held = tw.reindex(rets.index).ffill()
    held = held.shift(signal_lag_days).fillna(0.0)

    # 3) Daily portfolio return = sum_i w_i * r_i  (weights are start-of-day).
    aligned_rets = rets.reindex(columns=held.columns).fillna(0.0)
    gross_ret = (held * aligned_rets).sum(axis=1)

    # 4) Cost drag (zero under competition rules).
    drag = cost_drag(
        held,
        commission_per_trade=commission_per_trade,
        slippage_bps=slippage_bps,
    )
    # Coerce to plain float (CRSP returns can arrive as pandas nullable dtype,
    # whose NAType breaks downstream float math / quantstats).
    net_ret = (gross_ret - drag).astype("float64").fillna(0.0)

    # 5) Equity curve.
    equity = starting_capital * (1.0 + net_ret).cumprod()

    return BacktestResult(
        returns=net_ret,
        equity=equity,
        weights=held,
        gross_leverage=held.abs().sum(axis=1),
        net_exposure=held.sum(axis=1),
        turnover=turnover(held),
        starting_capital=starting_capital,
    )
