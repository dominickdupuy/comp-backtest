"""Short-term residual mean-reversion / statistical arbitrage.

Buy recent relative losers, short recent relative winners over a multi-day
window (NOT intraday -- that is what the 15-20 min delay would kill). Scores
can be sector- or beta-neutralized so the bet is on idiosyncratic reversion.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import DataBundle, Signal, neutralize_by_group, register


@register("reversal")
class ShortTermReversal(Signal):
    def compute(self, data: DataBundle) -> pd.DataFrame:
        lookback = int(self.params.get("lookback_days", 5))
        neutralize = str(self.params.get("neutralize", "sector"))

        close = data.close
        past_ret = close / close.shift(lookback) - 1.0
        score = -past_ret  # losers (negative return) get positive score

        if neutralize == "sector" and data.sector is not None:
            score = neutralize_by_group(score, data.sector)
        elif neutralize == "beta":
            # Remove the market component (equal-weight cross-section proxy).
            mkt = data.ret.mean(axis=1)
            cum_mkt = (1.0 + mkt).rolling(lookback).apply(np.prod, raw=True) - 1.0
            score = score.sub(-cum_mkt, axis=0)

        return score.fillna(0.0)
