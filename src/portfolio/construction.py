"""Turn blended cross-sectional scores into long/short target weights.

Steps per day:
  1. z-score each signal cross-sectionally, blend by configured weights.
  2. rank the combined score across the universe.
  3. go long the top quantile, short the bottom quantile.
  4. (optional) keep only the highest-conviction ``max_names_per_leg`` per side.
  5. weight within each leg (by score or equal), scale to dollar-neutral.

Output gross is ~2.0 (1.0 per side) by default; the engine then rescales to the
competition's target gross leverage and applies the per-name cap.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..signals.base import DataBundle, build_signal, cross_sectional_zscore


def blend_signals(
    data: DataBundle, signal_specs: dict[str, dict]
) -> pd.DataFrame:
    """Combine enabled signals into one cross-sectional z-score panel."""
    combined: pd.DataFrame | None = None
    total_w = sum(float(s.get("blend_weight", 0.0)) for s in signal_specs.values())
    for name, spec in signal_specs.items():
        sig = build_signal(name, spec.get("params"))
        raw = sig.compute(data)
        z = cross_sectional_zscore(raw)
        w = float(spec.get("blend_weight", 0.0)) / total_w
        contrib = z.fillna(0.0) * w
        combined = contrib if combined is None else combined.add(contrib, fill_value=0.0)
    if combined is None:
        raise ValueError("No signals to blend.")
    return combined


def _leg_weights(
    row_rank: pd.Series,
    score: pd.Series,
    quantile: float,
    side: str,
    max_names: int | None,
    weighting: str,
) -> pd.Series:
    """Weights for one side (long/short) on a single date."""
    n = row_rank.notna().sum()
    if n == 0:
        return pd.Series(0.0, index=row_rank.index)

    if side == "long":
        mask = row_rank >= (1.0 - quantile)
    else:
        mask = row_rank <= quantile
    sel = score[mask].dropna()
    if sel.empty:
        return pd.Series(0.0, index=row_rank.index)

    # Concentrate into the most extreme names.
    if max_names is not None and len(sel) > max_names:
        sel = sel.sort_values(ascending=(side == "short")).iloc[
            -max_names:
        ] if side == "long" else sel.sort_values().iloc[:max_names]

    if weighting == "score":
        mag = sel.abs()
        w = mag / mag.sum() if mag.sum() > 0 else pd.Series(
            1.0 / len(sel), index=sel.index
        )
    else:  # equal
        w = pd.Series(1.0 / len(sel), index=sel.index)

    if side == "short":
        w = -w
    return w.reindex(row_rank.index).fillna(0.0)


def build_weights(
    combined: pd.DataFrame,
    *,
    long_short: bool = True,
    long_quantile: float = 0.10,
    short_quantile: float = 0.10,
    weighting: str = "score",
    max_names_per_leg: int | None = 30,
    dollar_neutral: bool = True,
    rebalance_dates: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Construct target weights from the blended score panel."""
    if rebalance_dates is not None:
        combined = combined.reindex(rebalance_dates)

    ranks = combined.rank(axis=1, pct=True)
    weights = pd.DataFrame(0.0, index=combined.index, columns=combined.columns)

    for dt in combined.index:
        score = combined.loc[dt]
        rank = ranks.loc[dt]
        if rank.notna().sum() < 5:
            continue
        longs = _leg_weights(rank, score, long_quantile, "long",
                             max_names_per_leg, weighting)
        if long_short:
            shorts = _leg_weights(rank, score, short_quantile, "short",
                                 max_names_per_leg, weighting)
        else:
            shorts = pd.Series(0.0, index=score.index)

        if dollar_neutral and long_short:
            # Normalize each leg to 1.0 gross so net ~ 0.
            lg = longs[longs > 0].sum()
            sg = -shorts[shorts < 0].sum()
            if lg > 0:
                longs = longs / lg
            if sg > 0:
                shorts = shorts / sg
        weights.loc[dt] = longs.add(shorts, fill_value=0.0)

    return weights.replace([np.inf, -np.inf], 0.0).fillna(0.0)
