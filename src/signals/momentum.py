"""Cross-sectional and time-series momentum."""
from __future__ import annotations

import pandas as pd

from .base import DataBundle, Signal, register


@register("xs_momentum")
class CrossSectionalMomentum(Signal):
    """Buy past winners / sell past losers (Jegadeesh-Titman).

    Formation = trailing ~6 months, skipping the most recent month to avoid
    short-term reversal contamination. ``long_bias`` is consumed downstream by
    the constructor (trims the crash-prone short leg).
    """

    def compute(self, data: DataBundle) -> pd.DataFrame:
        formation_days = int(round(self.params.get("formation_months", 6) * 21))
        skip = int(self.params.get("skip_days", 21))
        close = data.close
        # Return from (t - formation - skip) to (t - skip).
        recent = close.shift(skip)
        past = close.shift(skip + formation_days)
        mom = recent / past - 1.0
        return mom.fillna(0.0)


@register("ts_momentum")
class TimeSeriesMomentum(Signal):
    """A security's own trailing return predicts its next-period return."""

    def compute(self, data: DataBundle) -> pd.DataFrame:
        lookback = int(round(self.params.get("lookback_months", 12) * 21))
        close = data.close
        trend = close / close.shift(lookback) - 1.0
        return trend.fillna(0.0)
