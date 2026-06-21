"""Maximum-variance / lottery signals for the winner-take-all tournament.

Objective is P(finishing #1 of N), not expected return. Contest theory
(Gaba-Tsetlin-Winkler 2004; Hvide 2002) says: when the winning proportion is
small, maximize the variance and right-skew of your outcome and minimize
correlation to the field. These signals rank the universe by exactly those
lottery characteristics so the constructor can concentrate into them.

All are computable from CRSP daily data (Bali-Cakici-Whitelaw 2011 'MAX',
idiosyncratic vol & skew, low price, recent IPO, biotech sector). Short interest
(squeeze fuel) comes from Compustat if loaded.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import DataBundle, Signal, register

# SIC ranges with the fattest binary-catalyst tails (pharma / biotech / research).
_BIOTECH_SIC = [(2833, 2836), (8731, 8731), (3826, 3826), (3841, 3851)]


@register("max_lottery")
class MaxLottery(Signal):
    """Bali-Cakici-Whitelaw MAX: max single-day return over a trailing window.

    High-MAX names are the low-priced, high-idiosyncratic-vol, high-skew
    'lottery' stocks. Score = average of the top-k daily returns in the window.
    """

    def compute(self, data: DataBundle) -> pd.DataFrame:
        window = int(self.params.get("window", 21))
        k = int(self.params.get("top_k", 1))
        ret = data.ret
        if k <= 1:
            score = ret.rolling(window, min_periods=window // 2).max()
        else:
            score = ret.rolling(window, min_periods=window // 2).apply(
                lambda x: np.sort(x)[-k:].mean(), raw=True
            )
        return score.fillna(0.0)


@register("ivol")
class IdiosyncraticVol(Signal):
    """High idiosyncratic volatility = high manufactured dispersion.

    Approximated as the volatility of market-residual returns (return minus the
    equal-weight cross-section) over a trailing window.
    """

    def compute(self, data: DataBundle) -> pd.DataFrame:
        window = int(self.params.get("window", 21))
        mkt = data.ret.mean(axis=1)
        resid = data.ret.sub(mkt, axis=0)
        return resid.rolling(window, min_periods=window // 2).std().fillna(0.0)


@register("idio_skew")
class IdiosyncraticSkew(Signal):
    """Right-skew preference: positively-skewed payoffs concentrate dispersion
    in the upper tail (where the single prize lives)."""

    def compute(self, data: DataBundle) -> pd.DataFrame:
        window = int(self.params.get("window", 42))
        mkt = data.ret.mean(axis=1)
        resid = data.ret.sub(mkt, axis=0)
        return resid.rolling(window, min_periods=window // 2).skew().fillna(0.0)


@register("low_price")
class LowPrice(Signal):
    """Lottery names are low-priced. Score = -log(price), strongest for the
    cheapest names; an optional hard ceiling drops everything above it."""

    def compute(self, data: DataBundle) -> pd.DataFrame:
        ceiling = float(self.params.get("price_ceiling", 0.0))  # 0 => no hard cut
        score = -np.log(data.close.clip(lower=0.1))
        if ceiling > 0:
            score = score.where(data.close <= ceiling, -np.inf)
        return score.replace([-np.inf, np.inf], np.nan).fillna(score.min().min())


@register("recent_ipo")
class RecentIPO(Signal):
    """Recently-listed names have fat early-life tails. Score decays from 1 at
    listing to 0 after `max_age_days` of trading history."""

    def compute(self, data: DataBundle) -> pd.DataFrame:
        max_age = int(self.params.get("max_age_days", 252))
        # Trading-day age = count of valid observations to date per name.
        age = data.close.notna().cumsum()
        score = (1.0 - age / max_age).clip(lower=0.0)
        return score.fillna(0.0)


@register("biotech")
class BiotechTilt(Signal):
    """Static tilt toward pharma/biotech/medical SICs (binary-catalyst names)."""

    def compute(self, data: DataBundle) -> pd.DataFrame:
        flag = pd.Series(0.0, index=data.close.columns)
        if data.siccd is not None:
            sic = data.siccd.reindex(data.close.columns)
            for lo, hi in _BIOTECH_SIC:
                flag[(sic >= lo) & (sic <= hi)] = 1.0
        # Broadcast the static flag across all dates.
        return pd.DataFrame(
            np.tile(flag.values, (len(data.close.index), 1)),
            index=data.close.index, columns=data.close.columns,
        )


@register("short_squeeze")
class ShortSqueeze(Signal):
    """High short interest as a fraction of shares outstanding = squeeze fuel.
    Contributes 0 if short-interest data was not loaded."""

    def compute(self, data: DataBundle) -> pd.DataFrame:
        if data.short_ratio is None:
            return pd.DataFrame(0.0, index=data.close.index, columns=data.close.columns)
        sr = data.short_ratio.reindex_like(data.close).ffill(limit=21)
        return sr.fillna(0.0)
