"""QuantStats tearsheets for the backtest.

Generates HTML tearsheets + console metrics for:
  1. CORE (deployed): illiquid bucket-5 PEAD, N=25, buy-hold-40, midpoint, full fills.
  2. HEADLINE walkforward (broad PEAD, the +4092% book) if its daily CSV exists.
Benchmark = IWM (live small-cap index) pulled from the Yahoo chart API.
"""
from __future__ import annotations
import json, sys, urllib.request, warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

# ── numpy 2.x compatibility shims for quantstats 0.0.81 ──
for _name, _val in [("product", np.prod), ("Inf", np.inf), ("NaN", np.nan),
                    ("NAN", np.nan), ("float", float), ("int", int), ("bool", bool),
                    ("object", object)]:
    if not hasattr(np, _name):
        setattr(np, _name, _val)
import matplotlib
matplotlib.use("Agg")
import quantstats as qs

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_pead_v7_fillrobust import prep, run_book_volcap

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
OUT = Path("results"); OUT.mkdir(exist_ok=True)


def iwm_benchmark(p1=1483228800, p2=1735689600):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/IWM?period1={p1}&period2={p2}&interval=1d"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    j = json.loads(urllib.request.urlopen(req, timeout=40).read())
    r = j["chart"]["result"][0]
    ts = pd.to_datetime(r["timestamp"], unit="s").normalize()
    c = pd.Series(r["indicators"]["quote"][0]["close"], index=ts, dtype="float64").dropna()
    s = c.pct_change().dropna(); s.index = s.index.tz_localize(None)
    return s.rename("IWM")


def tearsheet(returns, bench, title, fname):
    returns = returns.copy(); returns.index = pd.to_datetime(returns.index).tz_localize(None)
    returns = returns[~returns.index.duplicated()].sort_index()
    print(f"\n{'='*70}\n{title}\n{'='*70}")
    try:
        qs.reports.metrics(returns, benchmark=bench, mode="full", rf=0.0,
                           display=True, prepare_returns=False)
    except Exception as e:
        print(f"  [metrics fallback: {e}]")
        qs.reports.metrics(returns, mode="basic", rf=0.0, display=True, prepare_returns=False)
    out = OUT / fname
    try:
        qs.reports.html(returns, benchmark=bench, rf=0.0, title=title,
                        output=str(out), download_filename=str(out))
    except Exception:                       # benchmark alignment can trip older qs; retry solo
        qs.reports.html(returns, rf=0.0, title=title, output=str(out),
                        download_filename=str(out))
    print(f"  -> wrote {out}")
    return out


def main():
    bench = iwm_benchmark()

    # 1) CORE deployed book
    D = prep()
    zb5 = np.where(D["bmat"] == 5, D["z"], np.nan)
    core = run_book_volcap(zb5, D["mp"], D["dvol"], D["rowpos"], D["days"],
                           "2017-01-01", "2024-12-31", N=25, f=None, K_exit=30, band=0.03)
    core.index = pd.to_datetime(core.index).tz_localize(None)
    core.to_csv(OUT / "core_daily_returns.csv")
    f1 = tearsheet(core, bench, "Illiquid Bucket-5 PEAD (core, N=25, buy-hold-40)",
                   "quantstats_core.html")

    files = [str(f1)]
    # 2) headline walkforward book, if present
    wf_csv = OUT / "pead_daily_returns.csv"
    if wf_csv.exists():
        wf = pd.read_csv(wf_csv, index_col=0).squeeze("columns")
        wf.index = pd.to_datetime(wf.index)
        f2 = tearsheet(wf, bench, "Broad PEAD walk-forward (headline +4092%)",
                       "quantstats_walkforward.html")
        files.append(str(f2))

    print("\nDONE:", files)


if __name__ == "__main__":
    main()
