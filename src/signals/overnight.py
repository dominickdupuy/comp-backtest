"""Overnight-return harvesting overlay.

The equity premium accrues largely overnight (Lou-Polk-Skouras). Rank names by
trailing overnight return (open_t / close_{t-1} - 1) and tilt long the strong
overnight performers. Structurally immune to the intraday execution delay since
the position is decided before the close and held to the open.
"""
from __future__ import annotations

import pandas as pd

from .base import DataBundle, Signal, register


@register("overnight")
class OvernightOverlay(Signal):
    def compute(self, data: DataBundle) -> pd.DataFrame:
        lookback = int(self.params.get("lookback_days", 21))
        if data.open is None:
            # No open prices -> cannot compute overnight return.
            return pd.DataFrame(0.0, index=data.close.index, columns=data.close.columns)

        prev_close = data.close.shift(1)
        overnight_ret = data.open / prev_close - 1.0
        # Trailing average overnight return as the persistence score.
        score = overnight_ret.rolling(lookback, min_periods=max(2, lookback // 2)).mean()
        return score.fillna(0.0)
