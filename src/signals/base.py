"""Signal framework.

A *signal* maps a :class:`DataBundle` of wide panels to a wide DataFrame of
cross-sectional scores (index=date, columns=ticker). Higher score = more
attractive to go long; lower = more attractive to short. The portfolio
constructor ranks these scores each day, so only the cross-sectional ordering
matters -- raw scale is normalized away.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class DataBundle:
    """All market data needed by the signals, as wide (date x ticker) frames.

    Only ``close`` and ``ret`` are strictly required; signals that need more
    (open, volume, market_cap, sector, earnings) check for availability.
    """

    close: pd.DataFrame                       # adjusted close
    ret: pd.DataFrame                         # daily simple returns
    open: pd.DataFrame | None = None          # daily open (for overnight)
    volume: pd.DataFrame | None = None
    market_cap: pd.DataFrame | None = None
    sector: pd.Series | None = None           # asset -> coarse sector bucket
    siccd: pd.Series | None = None            # asset -> raw 4-digit SIC code
    # PEAD inputs: long event table with columns [date, asset, sue]
    earnings: pd.DataFrame | None = None
    # Short interest: wide (date x asset) shares-short / shrout (squeeze signal)
    short_ratio: pd.DataFrame | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def tickers(self) -> pd.Index:
        return self.close.columns

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.close.index


def cross_sectional_zscore(scores: pd.DataFrame) -> pd.DataFrame:
    """Standardize each row (date) to mean 0, std 1, ignoring NaNs."""
    mu = scores.mean(axis=1)
    sd = scores.std(axis=1).replace(0.0, np.nan)
    z = scores.sub(mu, axis=0).div(sd, axis=0)
    return z


def neutralize_by_group(scores: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    """Demean each row within ``groups`` (e.g. sector) to remove group tilts."""
    if groups is None:
        return scores
    aligned = groups.reindex(scores.columns)
    out = scores.copy()
    for _, members in aligned.groupby(aligned):
        cols = members.index
        cols = cols.intersection(scores.columns)
        if len(cols) > 1:
            out[cols] = scores[cols].sub(scores[cols].mean(axis=1), axis=0)
    return out


class Signal(ABC):
    """Base class. Subclasses implement :meth:`compute`."""

    name: str = "signal"

    def __init__(self, **params: Any) -> None:
        self.params = params

    @abstractmethod
    def compute(self, data: DataBundle) -> pd.DataFrame:
        """Return a wide DataFrame of cross-sectional scores."""

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.__class__.__name__}({self.params})"


# --- registry ----------------------------------------------------------------
_REGISTRY: dict[str, type[Signal]] = {}


def register(name: str):
    def _wrap(cls: type[Signal]) -> type[Signal]:
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return _wrap


def build_signal(name: str, params: dict[str, Any] | None = None) -> Signal:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown signal '{name}'. Known: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**(params or {}))


def available_signals() -> list[str]:
    return sorted(_REGISTRY)
