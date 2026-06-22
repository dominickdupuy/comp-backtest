"""QUANT NOTE v13b — CONTEST-MATCHED seasonal proxy (6-week window, not 2-month).

The live contest is ~6 weeks (~30 trading days), June 22 -> ~Aug 3.  That is SHORTER than
the 40-day PEAD hold, so nothing exits on age inside the window and the final PnL is OPEN
positions marked to market at the 6-week close.  The 2-month June figures overstate what a
6-week run can harvest.

For each year 2017-2024: start fresh (cash) at the first trading day on/after June 22, run
the core (bucket-5, N=25, buy-hold-40, midpoint, signed delay slippage, rdq+1 entry) for 30
trading days, mark the book at the close with positions still open.  Score exactly as the
contest: long-only, ~98% invested, 10%-at-entry cap, appreciation-lock.

Diagnostic per window: the gap between PnL harvested in the 6 weeks and what the SAME entries
would have returned held to their full age-40 exit (entries frozen at the 6-week close, then
ridden out).  That quantifies the drift the short window truncates.

Caveat carried: this is a PROXY for the seasonal position (n=8), NOT a significance test.
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.run_pead_v3 as _v3
from scripts.run_pead_v3 import build_target
from scripts.run_pead_walkforward import P
from scripts.run_pead_v4_fillmodel import DFRAC
from scripts.run_pead_v7_fillrobust import prep

CAPITAL = P["capital"]
WIN_TD = 30            # ~6 weeks of trading days
AGE_TAIL = 45          # extra days so frozen entries can reach age-40 exit


def run_window(z, mp, s_idx, close_idx, hard_idx, N=25, CAP=0.10, TGT=0.98,
               K_exit=30, band=0.03, charge_signed=True):
    """Fresh-capital book from s_idx; no NEW entries after close_idx (frozen),
    existing positions ride to age-40 exit by hard_idx. Returns (window_pnl,
    full_pnl) where window_pnl marks open positions at close_idx."""
    _v3.CAP, _v3.TGT = CAP, TGT
    ncol = z.shape[1]
    pos = np.zeros(ncol); V = CAPITAL
    rec_v, rec_day = [], []
    di = list(range(s_idx - 1, hard_idx))
    for k in range(len(di) - 1):
        day = di[k]
        zr = z[day].copy()
        if day >= close_idx:                       # past the 6-week close: freeze entries
            held = pos > 0
            zr = np.where(held, zr, np.nan)
        cur_w = pos / V if V > 0 else pos * 0.0
        held_idx = np.where(pos > 0)[0]
        idx, w = build_target(zr, held_idx, cur_w, N, K_exit, band)
        new_w = np.zeros(ncol); new_w[idx] = w
        target = new_w * V
        filled = target - pos
        r = np.nan_to_num(mp[di[k + 1]])
        signed = float((filled / V * r).sum()) * DFRAC if charge_signed else 0.0
        new_pos = np.maximum(target, 0.0)
        cash = V - new_pos.sum()
        pos = new_pos * (1.0 + r)
        Vn = pos.sum() + cash
        rec_v.append(Vn / V - 1.0 - signed); rec_day.append(di[k + 1]); V = Vn
    rec_v = np.array(rec_v); rec_day = np.array(rec_day)
    win_mask = rec_day <= close_idx
    window_pnl = float(np.prod(1 + rec_v[win_mask]) - 1)
    full_pnl = float(np.prod(1 + rec_v) - 1)
    return window_pnl, full_pnl


def anchor_idx(days, y, month=6, day=22):
    return int(days.searchsorted(pd.Timestamp(f"{y}-{month:02d}-{day:02d}")))


def main():
    D = prep()
    z, mp, bmat = D["z"], D["mp"], D["bmat"]
    days = D["days"]
    zb5 = np.where(bmat == 5, z, np.nan)
    n = len(days)

    print("=" * 90)
    print("v13b CONTEST-MATCHED SEASONAL PROXY — 6-week window (~30 td), June 22 anchor")
    print("  core: bucket-5, N=25, buy-hold-40, midpoint, signed slippage, rdq+1; MTM at close")
    print(f"\n{'year':<6}{'start':<12}{'close':<12}{'6wk PnL':>10}{'full-40 PnL':>13}{'truncated':>11}")
    per_year = {}
    for y in range(2017, 2025):
        s = anchor_idx(days, y)
        if s < 1 or s + WIN_TD >= n:
            continue
        close = s + WIN_TD - 1
        hard = min(close + AGE_TAIL, n - 1)
        wpnl, fpnl = run_window(zb5, mp, s, close, hard)
        per_year[y] = dict(window=wpnl, full=fpnl, trunc=fpnl - wpnl,
                           start=str(days[s].date()), close=str(days[close].date()))
        print(f"{y:<6}{str(days[s].date()):<12}{str(days[close].date()):<12}"
              f"{wpnl:>+10.1%}{fpnl:>+13.1%}{fpnl - wpnl:>+11.1%}", flush=True)

    w = np.array([per_year[y]["window"] for y in per_year])
    yrs = list(per_year)
    print(f"\n=== ACROSS 8 EXACT-ANCHOR WINDOWS (primary read, n={len(w)}) ===")
    print(f"  median {np.median(w):+.1%}   min {w.min():+.1%}   max {w.max():+.1%}")
    for thr in (0.25, 0.40, 0.60):
        print(f"  cleared >+{int(thr*100)}%: {int((w > thr).sum())}/{len(w)}  "
              f"({[y for y in yrs if per_year[y]['window'] > thr]})")
    w_ex20 = np.array([per_year[y]["window"] for y in per_year if y != 2020])
    print(f"  2020 FLAGGED as regime outlier ({per_year.get(2020, {}).get('window', float('nan')):+.1%}).")
    print(f"  EX-2020 (n={len(w_ex20)}): median {np.median(w_ex20):+.1%}  min {w_ex20.min():+.1%}  "
          f"max {w_ex20.max():+.1%}  cleared>+25% {int((w_ex20>0.25).sum())}/{len(w_ex20)}  "
          f">+40% {int((w_ex20>0.40).sum())}/{len(w_ex20)}")
    tr = np.array([per_year[y]["trunc"] for y in per_year])
    print(f"\n  TRUNCATION (full-40 minus 6-week): median {np.median(tr):+.1%}  "
          f"mean {tr.mean():+.1%}  -> drift left on the table by the short window")

    # ── sample-size band: ±5 td around June 22 (overlapping => correlated) ──
    print("\n=== STABILITY BAND: start +/-5 td around anchor (overlapping, CORRELATED) ===")
    band = []
    for y in range(2017, 2025):
        a = anchor_idx(days, y)
        for off in range(-5, 6):
            s = a + off
            if s < 1 or s + WIN_TD >= n:
                continue
            close = s + WIN_TD - 1; hard = min(close + AGE_TAIL, n - 1)
            band.append(run_window(zb5, mp, s, close, hard)[0])
    band = np.array(band)
    print(f"  band n={len(band)} (8 yrs x ~11 offsets; treat as correlated, NOT independent)")
    print(f"  band median {np.median(band):+.1%}  vs exact-anchor median {np.median(w):+.1%}  "
          f"-> central estimate {'STABLE' if abs(np.median(band)-np.median(w))<0.05 else 'shifts'}")
    print(f"  band cleared >+25% {(band>0.25).mean():.0%}  >+40% {(band>0.40).mean():.0%}  "
          f">+60% {(band>0.60).mean():.0%}")

    print("\n  CAVEAT: seasonal POSITION proxy, n=8 exact windows; not a significance test.")

    out = dict(per_year=per_year, exact=dict(median=float(np.median(w)), min=float(w.min()),
               max=float(w.max()), c25=int((w > .25).sum()), c40=int((w > .40).sum()),
               c60=int((w > .60).sum())),
               ex2020=dict(median=float(np.median(w_ex20)), max=float(w_ex20.max())),
               truncation_median=float(np.median(tr)),
               band=dict(n=len(band), median=float(np.median(band)),
                         c25=float((band > .25).mean()), c40=float((band > .40).mean())))
    Path("results").mkdir(exist_ok=True)
    json.dump(out, open("results/pead_v13b_contest.json", "w"), indent=2, default=float)
    print("\nwrote results/pead_v13b_contest.json", flush=True)


if __name__ == "__main__":
    main()
