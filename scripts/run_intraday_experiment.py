"""Does higher-frequency rebalancing add PnL under ZERO transaction costs?

Runs the SAME cross-sectional reversal book at increasing rebalance frequencies
(once-daily -> 2x/day -> hourly -> 30min) on the liquid cross-section, holding
the 15-20 min execution delay fixed, and compares total PnL. Uses whatever
minute months are already cached (so it can run while the bulk pull continues).

    python scripts/run_intraday_experiment.py --top 1000
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from dotenv import load_dotenv

from src.backtest.intraday import run_intraday_reversal
from src.config import REPO_ROOT, load_config
from src.data.relevance import liquid_universe_symbols
from src.data.taq_loader import cached_months, load_cached_minute
from src.data.wrds_loader import load_data_bundle

load_dotenv(REPO_ROOT / ".env")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=1000)
    args = ap.parse_args()

    cfg = load_config()
    comp = cfg.competition
    data = load_data_bundle(cfg.start_date, cfg.end_date,
                            universe_size=int(comp["universe_size"]),
                            min_price=float(comp["min_price"]), use_cache=True)
    sym_map = liquid_universe_symbols(data, top_n=args.top)
    symbols = sorted(set(sym_map.values()))

    have = cached_months(symbols, cfg.start_date, cfg.end_date)
    if not have:
        print("No minute months cached yet -- let scripts/pull_minute.py run first.")
        return
    win_start = pd.Period(have[0], "M").start_time.date()
    win_end = pd.Period(have[-1], "M").end_time.date()
    print(f"Cached minute months: {have[0]} .. {have[-1]} ({len(have)} months), "
          f"{len(symbols)} symbols")

    gl = comp["max_gross_leverage"]
    cap = comp["max_position_weight"]

    # Three fill-realism regimes, to separate real alpha from the bounce mirage:
    #   naive    : last-trade fills, no skip, no spread (the fantasy)
    #   skip     : 1-bar gap so the position isn't formed from the bounce bar
    #   realistic: 1-bar gap + a small effective half-spread (you cross the book)
    regimes = {
        "naive (last-trade)": dict(skip_bars=0, spread_bps=0.0),
        "skip 1 bar":         dict(skip_bars=1, spread_bps=0.0),
        "skip + 2bps spread": dict(skip_bars=1, spread_bps=2.0),
    }

    daily_close = data.close.rename(columns=data.meta["names"])
    daily_close = daily_close.loc[
        (daily_close.index.date >= win_start) & (daily_close.index.date <= win_end),
        [s for s in symbols if s in daily_close.columns],
    ]
    panels = {"1x/day (daily)": daily_close}
    for freq in ["130min", "65min", "30min"]:
        p = load_cached_minute(symbols, win_start, win_end, field="close", freq=freq)
        if not p.empty:
            label = {"130min": "~2x/day", "65min": "hourly", "30min": "30-min"}[freq]
            panels[f"{label} ({freq})"] = p

    print("\nReversal sleeve total PnL: rebalance frequency x fill realism")
    print(f"({win_start} .. {win_end}, zero commissions, 20-min delay)")
    print("-" * 78)
    header = f"{'frequency':<20}" + "".join(f"{r:>19}" for r in regimes)
    print(header)
    for label, panel in panels.items():
        is_daily = label.startswith("1x")
        freq = "1D" if is_daily else label.split("(")[1].rstrip(")")
        cells = []
        for reg in regimes.values():
            res = run_intraday_reversal(
                panel, rebalance_freq=freq, lookback_bars=1, delay_minutes=20,
                delay_bars=1 if is_daily else None,
                exclude_overnight=not is_daily,
                target_gross_leverage=gl, max_position_weight=cap, **reg)
            tr = res.total_return
            cells.append(f"{tr:>18.1%}" if abs(tr) < 100 else f"{tr:>18.2e}")
        print(f"{label:<20}" + "".join(f"{c:>19}" for c in cells))


if __name__ == "__main__":
    main()
