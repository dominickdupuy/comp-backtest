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
    rows = []

    # Once-daily baseline (close-to-close, 1-day lag) on the SAME names/window.
    daily_close = data.close.rename(columns=data.meta["names"])
    daily_close = daily_close.loc[
        (daily_close.index.date >= win_start) & (daily_close.index.date <= win_end),
        [s for s in symbols if s in daily_close.columns],
    ]
    base = run_intraday_reversal(
        daily_close, exclude_overnight=False, delay_bars=1, lookback_bars=1,
        target_gross_leverage=gl, max_position_weight=cap)
    rows.append(("1x/day (daily close)", base.total_return, base.bars_per_day,
                 base.delay_bars))

    # Intraday frequencies.
    for freq in ["130min", "65min", "30min"]:
        close = load_cached_minute(symbols, win_start, win_end, field="close", freq=freq)
        if close.empty:
            continue
        res = run_intraday_reversal(
            close, rebalance_freq=freq, lookback_bars=1, delay_minutes=20,
            target_gross_leverage=gl, max_position_weight=cap)
        label = {"130min": "~2x/day", "65min": "hourly", "30min": "30-min"}[freq]
        rows.append((f"{label} ({freq})", res.total_return, res.bars_per_day,
                     res.delay_bars))

    print("\nReversal sleeve: total PnL vs rebalance frequency "
          f"({win_start} .. {win_end}, zero costs, 20-min delay)")
    print("-" * 72)
    print(f"{'frequency':<24}{'total_return':>16}{'bars/day':>12}{'delay(bars)':>14}")
    for label, tr, bpd, db in rows:
        print(f"{label:<24}{tr:>15.2%}{bpd:>12.1f}{db:>14d}")


if __name__ == "__main__":
    main()
