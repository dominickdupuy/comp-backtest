"""Post-Earnings Announcement Drift (PEAD) -- the primary engine.

Stocks with large positive earnings surprises (SUE) drift up for ~5-60 trading
days; large negative surprises drift down. The drift is several times stronger
in small/microcaps, which zero transaction costs make freely tradeable.

Score = the standardized earnings surprise, carried forward over the holding
window, optionally tilted toward small caps.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import DataBundle, Signal, register


@register("pead")
class PEAD(Signal):
    def compute(self, data: DataBundle) -> pd.DataFrame:
        hold_days = int(self.params.get("hold_days", 30))
        smallcap_tilt = bool(self.params.get("smallcap_tilt", True))

        scores = pd.DataFrame(
            np.nan, index=data.close.index, columns=data.close.columns
        )

        if data.earnings is None or data.earnings.empty:
            # No earnings data available -> signal contributes nothing.
            return scores.fillna(0.0)

        ev = data.earnings.copy()
        ev["date"] = pd.to_datetime(ev["date"])
        # Place each surprise on the nearest trading day >= announcement.
        idx = data.close.index
        col_pos = {c: i for i, c in enumerate(scores.columns)}
        for _, row in ev.iterrows():
            asset = row["asset"]
            if asset not in col_pos:
                continue
            pos = idx.searchsorted(row["date"])
            if pos >= len(idx):
                continue
            scores.iat[pos, col_pos[asset]] = row["sue"]

        # Carry the surprise forward across the drift window.
        scores = scores.ffill(limit=hold_days)

        if smallcap_tilt and data.market_cap is not None:
            # Smaller cap -> larger drift. Multiply standardized SUE by an
            # inverse-size weight in (0.5, 1.5] based on daily cap rank.
            cap = data.market_cap.reindex_like(scores)
            rank = cap.rank(axis=1, pct=True)          # 0 (small) .. 1 (large)
            tilt = 1.5 - rank                          # small caps weighted up
            scores = scores * tilt

        return scores.fillna(0.0)
