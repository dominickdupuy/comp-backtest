"""Bulk-pull 1-minute OHLCV bars from WRDS TAQ for the liquid cross-section.

Minute bars only (aggregated server-side), NOT tick -- a few GB for ~1000
names over 3 years. Per-month parquet cache => resumable. Use for the
intraday-vs-daily rebalancing experiment under zero transaction costs.

    python scripts/pull_minute.py --top 1000
    python scripts/pull_minute.py --top 500 --start 2025-06-01 --end 2026-06-17
"""
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
from src.config import REPO_ROOT, load_config
from src.data.relevance import liquid_universe_symbols
from src.data.taq_loader import pull_minute_bars
from src.data.wrds_loader import load_data_bundle

load_dotenv(REPO_ROOT / ".env")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=1000, help="liquid names to pull")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    args = ap.parse_args()

    cfg = load_config()
    comp = cfg.competition
    start = datetime.strptime(args.start, "%Y-%m-%d").date() if args.start else cfg.start_date
    end = datetime.strptime(args.end, "%Y-%m-%d").date() if args.end else cfg.end_date

    # Daily bundle (cached) -> pick the liquid cross-section.
    data = load_data_bundle(cfg.start_date, cfg.end_date,
                            universe_size=int(comp["universe_size"]),
                            min_price=float(comp["min_price"]), use_cache=True)
    sym_map = liquid_universe_symbols(data, top_n=args.top)
    symbols = sorted(set(sym_map.values()))
    print(f"Pulling minute bars: {len(symbols)} symbols, {start} -> {end}", flush=True)

    # Loop month-by-month with a fresh connection each, so a dropped connection
    # never loses more than one month and reruns resume from the cache.
    import pandas as pd
    months = pd.period_range(start, end, freq="M")
    t0 = time.time()
    for i, m in enumerate(months, 1):
        m_start = max(m.start_time.date(), start)
        m_end = min(m.end_time.date(), end)
        ti = time.time()
        try:
            pull_minute_bars(symbols, m_start, m_end, use_cache=True,
                             accumulate=False, db=None)
            print(f"[{i}/{len(months)}] {m} done in {(time.time()-ti)/60:.1f} min "
                  f"(elapsed {(time.time()-t0)/60:.1f} min)", flush=True)
        except Exception as exc:
            print(f"[{i}/{len(months)}] {m} FAILED: {str(exc)[:120]} -- will resume on rerun",
                  flush=True)
    print(f"ALL DONE in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
