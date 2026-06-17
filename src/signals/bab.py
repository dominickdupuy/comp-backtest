"""Betting-Against-Beta base layer.

Long low-beta names, short high-beta names (Frazzini-Pedersen). A slow-moving,
fully delay-robust diversifier. Beta is estimated against an equal-weight
cross-sectional market proxy over a rolling window.
"""
from __future__ import annotations

import pandas as pd

from .base import DataBundle, Signal, register


@register("bab")
class BettingAgainstBeta(Signal):
    def compute(self, data: DataBundle) -> pd.DataFrame:
        window = int(self.params.get("beta_window_days", 252))
        ret = data.ret
        mkt = ret.mean(axis=1)                       # equal-weight market proxy

        var_mkt = mkt.rolling(window, min_periods=window // 2).var()
        # cov(r_i, r_m) = E[r_i r_m] - E[r_i]E[r_m], computed via rolling means.
        mean_i = ret.rolling(window, min_periods=window // 2).mean()
        mean_m = mkt.rolling(window, min_periods=window // 2).mean()
        prod = ret.mul(mkt, axis=0)
        mean_prod = prod.rolling(window, min_periods=window // 2).mean()
        cov = mean_prod.sub(mean_i.mul(mean_m, axis=0))
        beta = cov.div(var_mkt, axis=0)

        # Low beta attractive -> long; high beta -> short.
        return (-beta).fillna(0.0)
