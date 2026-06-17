"""End-to-end orchestration: config -> data -> signals -> weights -> backtest."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .backtest.engine import BacktestResult, run_backtest
from .config import Config, load_config
from .portfolio.construction import blend_signals, build_weights
from .signals.base import DataBundle


def _rebalance_dates(index: pd.DatetimeIndex, rule) -> pd.DatetimeIndex:
    if rule == "daily":
        return index
    if rule == "weekly":
        return index[index.to_series().groupby(index.isocalendar().week).cumcount() == 0]
    if rule == "monthly":
        return index[index.to_series().groupby([index.year, index.month]).cumcount() == 0]
    if isinstance(rule, int):
        return index[::rule]
    return index


def run(config: Config, data: DataBundle) -> BacktestResult:
    """Run the full strategy defined by ``config`` on ``data``."""
    # Clip data to the configured window.
    mask = (data.close.index.date >= config.start_date) & (
        data.close.index.date <= config.end_date
    )
    if mask.any():
        idx = data.close.index[mask]
        data = DataBundle(
            close=data.close.loc[idx],
            ret=data.ret.loc[idx],
            open=None if data.open is None else data.open.loc[idx],
            volume=None if data.volume is None else data.volume.loc[idx],
            market_cap=None if data.market_cap is None else data.market_cap.loc[idx],
            sector=data.sector,
            earnings=data.earnings,
            meta=data.meta,
        )

    combined = blend_signals(data, config.enabled_signals)

    con = config.strategies["construction"]
    rebal = _rebalance_dates(combined.index, config.strategies["backtest"]["rebalance"])
    weights = build_weights(
        combined,
        long_short=con["long_short"],
        long_quantile=con["long_quantile"],
        short_quantile=con["short_quantile"],
        weighting=con["weighting"],
        max_names_per_leg=con["sizing"].get("max_names_per_leg"),
        dollar_neutral=con["dollar_neutral"],
        rebalance_dates=rebal,
    )

    return run_backtest(
        weights,
        data.ret,
        starting_capital=config.competition["starting_capital"],
        signal_lag_days=config.signal_lag_days,
        max_position_weight=config.max_position_weight,
        target_gross_leverage=con["target_gross_leverage"],
        max_gross_leverage=config.max_gross_leverage,
        max_net_leverage=float(config.competition["max_net_leverage"]),
        allow_shorting=config.allow_shorting,
        commission_per_trade=config.commission_per_trade,
        slippage_bps=config.slippage_bps,
    )
