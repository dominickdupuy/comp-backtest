"""Generate a realistic synthetic DataBundle so the full pipeline can be
exercised with no WRDS access. Returns embed a small momentum + post-event
drift so the signals have something to find."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.signals.base import DataBundle


def make_bundle(cfg=None, n_names: int = 200, seed: int = 7) -> DataBundle:
    rng = np.random.default_rng(seed)
    if cfg is not None:
        dates = pd.bdate_range(cfg.start_date, cfg.end_date)
    else:
        dates = pd.bdate_range("2023-01-01", "2026-01-01")
    tickers = [f"S{i:04d}" for i in range(n_names)]
    T, N = len(dates), n_names

    # Latent per-name drift + idiosyncratic noise + a touch of momentum.
    base_drift = rng.normal(0.0003, 0.0008, N)
    shocks = rng.normal(0, 0.02, (T, N))
    ret = shocks + base_drift
    for t in range(1, T):  # mild autocorrelation -> exploitable momentum
        ret[t] += 0.05 * ret[t - 1]

    ret_df = pd.DataFrame(ret, index=dates, columns=tickers)
    close = 50.0 * (1.0 + ret_df).cumprod()
    open_ = close.shift(1) * (1.0 + rng.normal(0, 0.005, (T, N)))  # overnight gap
    volume = pd.DataFrame(rng.integers(1e5, 1e7, (T, N)), index=dates, columns=tickers)
    shrout = rng.integers(1e7, 5e8, N)
    mktcap = close.mul(shrout, axis=1)
    sector = pd.Series(rng.integers(0, 8, N).astype(str), index=tickers)

    # Synthetic earnings events with a SUE that predicts subsequent drift.
    ev_rows = []
    for tkr in tickers:
        for d in dates[rng.integers(20, 40):len(dates):63]:  # ~quarterly
            sue = rng.normal(0, 1)
            ev_rows.append({"date": d, "asset": tkr, "sue": sue})
    earnings = pd.DataFrame(ev_rows)

    return DataBundle(
        close=close, ret=ret_df, open=open_, volume=volume,
        market_cap=mktcap, sector=sector, earnings=earnings,
        meta={"synthetic": True},
    )
