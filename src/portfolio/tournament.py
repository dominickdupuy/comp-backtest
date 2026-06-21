"""Concentrated long-only book construction for the winner-take-all tournament.

Per rebalance date: blend the lottery signals into a composite score, then
greedily select the highest-scoring names subject to a low mutual-correlation
constraint (GTW Prop. 8: decorrelate from the field and from each other so at
least one name can reach the right tail independently). Long-only (truncated
downside, convex payoff), equal/conviction weighted, capped at the per-name
limit and scaled to the gross-leverage cap by the engine.

Note on the cap vs leverage trade-off: with a 10% per-name cap, full 2x gross
needs >=20 names; fewer names => higher idiosyncratic concentration but lower
realized gross. Both raise outcome variance; this is the bold-play knob.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..signals.base import DataBundle, build_signal, cross_sectional_zscore


def lottery_score(data: DataBundle, signal_specs: dict[str, dict]) -> pd.DataFrame:
    """Blend enabled lottery signals into one cross-sectional z-score panel."""
    combined: pd.DataFrame | None = None
    total = sum(float(s.get("blend_weight", 0.0)) for s in signal_specs.values())
    for name, spec in signal_specs.items():
        sig = build_signal(name, spec.get("params"))
        z = cross_sectional_zscore(sig.compute(data))
        contrib = z.fillna(0.0) * (float(spec.get("blend_weight", 0.0)) / total)
        combined = contrib if combined is None else combined.add(contrib, fill_value=0.0)
    if combined is None:
        raise ValueError("No lottery signals enabled.")
    return combined


def _greedy_decorrelated(
    scores: pd.Series,
    corr: pd.DataFrame,
    n_names: int,
    corr_threshold: float,
) -> list:
    """Pick up to n_names highest-score assets whose pairwise correlation with
    everything already chosen stays below corr_threshold."""
    ranked = scores.sort_values(ascending=False).index
    chosen: list = []
    for cand in ranked:
        if len(chosen) >= n_names:
            break
        if cand not in corr.index:
            chosen.append(cand)
            continue
        if chosen:
            c = corr.loc[cand, [x for x in chosen if x in corr.columns]]
            if (c.abs() > corr_threshold).any():
                continue
        chosen.append(cand)
    return chosen


def build_tournament_weights(
    data: DataBundle,
    signal_specs: dict[str, dict],
    *,
    n_names: int = 20,
    corr_threshold: float = 0.5,
    corr_lookback: int = 63,
    weighting: str = "equal",
    min_price: float = 0.0,
    rebalance_dates: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Long-only concentrated target weights toward the lottery names."""
    scores = lottery_score(data, signal_specs)
    dates = rebalance_dates if rebalance_dates is not None else scores.index
    weights = pd.DataFrame(0.0, index=dates, columns=scores.columns)

    for dt in dates:
        if dt not in scores.index:
            continue
        row = scores.loc[dt].dropna()
        if min_price > 0:
            valid = data.close.loc[dt].reindex(row.index) >= min_price
            row = row[valid]
        # Tradable today (has a price and recent returns).
        tradable = data.close.loc[dt].reindex(row.index).notna()
        row = row[tradable]
        if len(row) < n_names:
            continue
        # Correlation among the top score candidates only (keep it cheap).
        top = row.sort_values(ascending=False).head(max(n_names * 5, 50)).index
        window = data.ret.loc[:dt, top].tail(corr_lookback)
        corr = window.corr()
        picks = _greedy_decorrelated(row[top], corr, n_names, corr_threshold)
        if not picks:
            continue
        if weighting == "score":
            s = row[picks].clip(lower=0.0)
            w = s / s.sum() if s.sum() > 0 else pd.Series(1.0 / len(picks), index=picks)
        else:
            w = pd.Series(1.0 / len(picks), index=picks)
        weights.loc[dt, picks] = w.values

    return weights.fillna(0.0)
