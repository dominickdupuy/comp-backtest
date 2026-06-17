"""Determine the 'relevant' subset of names for minute-data pulls.

Relevance is defined operationally: the names the strategies *actually trade*
over the backtest (the union of every name that enters a long or short leg),
optionally narrowed by per-strategy liquidity rules. This is exactly the set
for which intraday data could matter, and nothing more.
"""
from __future__ import annotations

import pandas as pd

from ..signals.base import DataBundle


def traded_assets(weights: pd.DataFrame, threshold: float = 0.0) -> list:
    """Every column that ever holds a non-trivial weight (long or short)."""
    held = weights.abs().max(axis=0)
    return held[held > threshold].index.tolist()


def assets_to_symbols(assets: list, data: DataBundle) -> dict:
    """Map panel asset ids (permno) -> TAQ ticker symbol via the names map."""
    names = data.meta.get("names", {})
    out = {}
    for a in assets:
        sym = names.get(a)
        if sym and str(sym).strip() and not str(sym).isdigit():
            out[a] = str(sym).strip().upper()
    return out


def liquid_universe_symbols(data: DataBundle, top_n: int = 1000) -> dict:
    """Top-N assets by median daily dollar volume -> {permno: TAQ symbol}.

    This is the cross-section a high-frequency stat-arb/reversal book trades, so
    it is the right minute-data universe for the intraday-vs-daily experiment.
    """
    if data.volume is None:
        assets = list(data.close.columns)[:top_n]
    else:
        dvol = (data.close * data.volume).median(axis=0).dropna()
        assets = dvol.sort_values(ascending=False).head(top_n).index.tolist()
    return assets_to_symbols(assets, data)


def liquidity_filter(
    assets: list,
    data: DataBundle,
    *,
    min_median_dollar_vol: float | None = None,
    max_median_dollar_vol: float | None = None,
) -> list:
    """Keep assets whose median daily dollar volume falls in a band.

    Lets a strategy target, e.g., only low-liquidity microcaps (PEAD/reversal)
    or only liquid large-caps (stat-arb), per the strategy description.
    """
    if data.volume is None:
        return assets
    dvol = (data.close * data.volume).median(axis=0)
    sel = dvol.reindex(assets).dropna()
    if min_median_dollar_vol is not None:
        sel = sel[sel >= min_median_dollar_vol]
    if max_median_dollar_vol is not None:
        sel = sel[sel <= max_median_dollar_vol]
    return sel.index.tolist()
