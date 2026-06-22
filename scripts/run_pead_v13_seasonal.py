"""QUANT NOTE v13 item 1 — SEASONALLY HONEST forward estimate.

The all-windows backtest average blends every calendar position. The LIVE window starts
2026-06-21: Q1 reporting is done (drift mostly harvested), the Q2 flood arrives ~mid-July,
and names entered late in the window have TRUNCATED drift (the 40-day drift can't complete
before the 2-month window closes). So re-run the 2-month (42 trading-day) window
distribution for windows STARTING in mid-to-late June across 2017-2024 and compare to the
all-windows average. That gap is the honest expectation for THIS run.

Core (locked): illiquid bucket-5 PEAD, N=25, buy-hold-40, midpoint, full fills.
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_pead_walkforward import P
from scripts.run_pead_v7_fillrobust import prep, run_book_volcap

WIN = 42                                    # 2-month window in trading days
JUNE_LO, JUNE_HI = 12, 30                   # "mid-to-late June" start days


def windows_with_start(s, win=WIN):
    v = s.values; idx = s.index
    out = [(idx[i], float(np.prod(1 + v[i:i + win]) - 1)) for i in range(len(v) - win + 1)]
    return pd.DataFrame(out, columns=["start", "ret"])


def dist(r):
    return dict(n=int(len(r)), ev=float(r.mean()), med=float(np.median(r)),
                p40=float((r > .40).mean()), p60=float((r > .60).mean()),
                p95=float(np.percentile(r, 95)), mx=float(r.max()), mn=float(r.min()))


def main():
    D = prep()
    zb5 = np.where(D["bmat"] == 5, D["z"], np.nan)
    s = run_book_volcap(zb5, D["mp"], D["dvol"], D["rowpos"], D["days"],
                        "2017-01-01", "2024-12-31", N=25, f=None, K_exit=30, band=0.03)
    wd = windows_with_start(s)
    wd["yr"] = wd["start"].dt.year; wd["mo"] = wd["start"].dt.month; wd["dy"] = wd["start"].dt.day

    allw = dist(wd["ret"].values)
    june = wd[(wd["mo"] == 6) & (wd["dy"] >= JUNE_LO) & (wd["dy"] <= JUNE_HI)]
    junew = dist(june["ret"].values)

    print("=" * 92)
    print("v13.1 SEASONALLY HONEST FORWARD ESTIMATE — bucket-5 N=25 buy-hold-40 (2017-2024)")
    print(f"\n{'distribution':<28}{'n':>5}{'EV':>8}{'median':>8}{'P>40%':>8}{'P>60%':>8}"
          f"{'95th':>8}{'max':>8}")
    print(f"{'ALL-windows (every position)':<28}{allw['n']:>5}{allw['ev']:>8.1%}"
          f"{allw['med']:>8.1%}{allw['p40']:>8.1%}{allw['p60']:>8.1%}{allw['p95']:>8.0%}{allw['mx']:>8.0%}")
    print(f"{'JUNE-start (live position)':<28}{junew['n']:>5}{junew['ev']:>8.1%}"
          f"{junew['med']:>8.1%}{junew['p40']:>8.1%}{junew['p60']:>8.1%}{junew['p95']:>8.0%}{junew['mx']:>8.0%}")

    print("\n  per-year window starting nearest June-21 (the live entry date):")
    for y in range(2017, 2025):
        cand = wd[(wd["yr"] == y) & (wd["mo"] == 6)]
        if cand.empty:
            continue
        row = cand.iloc[(cand["dy"] - 21).abs().argsort().iloc[0]]
        print(f"    {y}  start {row['start'].date()}  2mo return {row['ret']:>+7.1%}", flush=True)

    june_ex20 = dist(june[june["yr"] != 2020]["ret"].values)
    print(f"\n{'JUNE-start EX-2020 (robust)':<28}{june_ex20['n']:>5}{june_ex20['ev']:>8.1%}"
          f"{june_ex20['med']:>8.1%}{june_ex20['p40']:>8.1%}{june_ex20['p60']:>8.1%}"
          f"{june_ex20['p95']:>8.0%}{june_ex20['mx']:>8.0%}")
    print("    (2020 = COVID melt-up outlier; ex-2020 isolates the non-regime seasonal effect)")

    gap_ev = junew["ev"] - allw["ev"]
    print(f"\n  SEASONAL GAP (June-start - all-windows):  EV {gap_ev:+.1%}   "
          f"P>60% {junew['p60']-allw['p60']:+.1%}   max {junew['mx']-allw['mx']:+.0%}")
    print(f"  -> honest expectation for THIS run = the JUNE-start row, not the all-windows blend.")

    out = dict(all_windows=allw, june_start=junew, june_ex2020=june_ex20,
               per_year_june21={int(y): float(wd[(wd['yr'] == y) & (wd['mo'] == 6)]
                                .iloc[(wd[(wd['yr'] == y) & (wd['mo'] == 6)]['dy'] - 21)
                                .abs().argsort().iloc[0]]['ret'])
                                for y in range(2017, 2025)
                                if not wd[(wd['yr'] == y) & (wd['mo'] == 6)].empty})
    Path("results").mkdir(exist_ok=True)
    json.dump(out, open("results/pead_v13_seasonal.json", "w"), indent=2, default=float)
    print("\nwrote results/pead_v13_seasonal.json", flush=True)


if __name__ == "__main__":
    main()
