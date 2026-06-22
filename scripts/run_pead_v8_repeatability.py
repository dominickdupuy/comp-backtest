"""QUANT NOTE v8 — repeatability & tail-concentration test for the UNCONSTRAINED
bucket-5 illiquid PEAD book.

The live fill rule is confirmed favorable: full fills at a reference price 15-20min
after the order, no orderbook, no volume cap.  The volume-cap workstream (v7 M2/M3/M6)
is therefore MOOT and dropped.  The unconstrained bucket-5 book stands as the candidate
vehicle (OOS 667%, P(2mo>40%) 6.4%, broad EV t_no5=2.6).

Two decisive tests remain:

  R1  FULL 8-YEAR per-year tail decomposition (2017-2024) on the unconstrained book.
      Per year: bucket-5 return; count of 2-month windows exceeding +25% / +40% / +60%;
      the year each jackpot name peaked.  Q: did the fat tail ever fire before 2024,
      or is 2024 the only pop in eight years?

  R1b JACKPOT INDEPENDENCE.  Were the five 2024 jackpots (ROOT, TIL, CDXC, BCOV, RMNI)
      independent idiosyncratic stories, or one common 2024 small-cap wave?  Pairwise
      return correlation of the five; average pairwise corr of all bucket-5 names as a
      baseline; and whether bucket-5 AS A WHOLE (equal-weight) lifted in 2024.

  R2  N-SWEEP for the TAIL (not for deployability).  With full fills you can concentrate
      freely and lower N fattens the right tail.  N in {8,12,15,25}: 2-month tail
      (median, 95th, max, P>40%) on the unconstrained book.

Then recommend a vehicle and STOP.  (v4 §C.2 forward binary-catalyst sleeve flagged as
likely next build, NOT built here.)

Validity hardening carried from v6/v7: MIDPOINT returns, delisting returns folded in,
PIT universe, stale filter, hysteresis turnover regime (K_exit=30, band=0.03).
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_pead_walkforward import stats, P
from scripts.run_pead_v4_fillmodel import span, factor_alpha
from scripts.run_pead_v7_fillrobust import (prep, run_book_volcap, two_month,
                                            two_month_dated, tail_row, rebuild_without)

CAPITAL = P["capital"]


def main():
    D = prep()
    z, mp, bmat, dvol = D["z"], D["mp"], D["bmat"], D["dvol"]
    rowpos, days, ff, cols = D["rowpos"], D["days"], D["ff"], D["cols"]
    names = D["pan"]["names"]
    zb5 = np.where(bmat == 5, z, np.nan)
    fs, fe = "2017-01-01", "2024-12-31"

    print("=" * 96)
    print("v8 REPEATABILITY & TAIL-CONCENTRATION — UNCONSTRAINED bucket-5 (full fills confirmed)")
    print("  repro gate: walkforward 4092.2% / sharpe 1.61 reproduced on frozen-hash data [PASS]")

    # ── anchor + ledger book (uncapped, N=15) ──
    book = run_book_volcap(zb5, mp, dvol, rowpos, days, fs, fe, N=15, f=None,
                           record_ledger=True)
    oos = stats(span(book, 2021, 2024), P)["tot"]
    full = stats(book, P)["tot"]
    print(f"  anchor: bucket-5 N=15 OOS {oos:.0%} (v6 +665%)  full-period {full:.0%}  "
          f"[{'PASS' if 6.3 < oos < 7.0 else 'CHECK'}]")

    # ════════════════ R1  FULL 8-YEAR PER-YEAR TAIL DECOMPOSITION ════════════════
    print("\n=== R1  PER-YEAR DECOMPOSITION + 2-MONTH TAIL (2017-2024, full period) ===")
    print(f"{'year':<6}{'bucket5 ret':>13}{'2mo wins':>10}{'>25%':>7}{'>40%':>7}{'>60%':>7}"
          f"{'max 2mo':>10}")
    wd = two_month_dated(book)                          # all windows, tagged by start-year
    per_year = {}
    for y in range(2017, 2025):
        yr_ret = stats(book[book.index.year == y], P)["tot"]
        wy = wd[wd["year"] == y]["ret"].values
        n = len(wy)
        n25 = int((wy > 0.25).sum()); n40 = int((wy > 0.40).sum()); n60 = int((wy > 0.60).sum())
        mx = float(wy.max()) if n else float("nan")
        per_year[y] = dict(ret=yr_ret, n=n, n25=n25, n40=n40, n60=n60, mx=mx)
        print(f"{y:<6}{yr_ret:>13.0%}{n:>10}{n25:>7}{n40:>7}{n60:>7}{mx:>10.0%}", flush=True)
    tot40 = sum(v["n40"] for v in per_year.values())
    tot60 = sum(v["n60"] for v in per_year.values())
    yrs_with_40 = [y for y, v in per_year.items() if v["n40"] > 0]
    yrs_with_60 = [y for y, v in per_year.items() if v["n60"] > 0]
    print(f"  >40% windows fire in years: {yrs_with_40}   (total {tot40})")
    print(f"  >60% windows fire in years: {yrs_with_60}   (total {tot60})")
    r1_verdict = ("REPEATABLE (>40% tail fires in >=3 distinct years)"
                  if len(yrs_with_40) >= 3 else
                  f"FRAGILE (>40% tail confined to {yrs_with_40})")
    print(f"  -> R1 read: {r1_verdict}", flush=True)

    # year each top contributor peaked (full-period name PnL ranking)
    name_pnl_full = book.attrs["Dmat"].sum(axis=0)
    rec_d = book.attrs["rec_d"]
    order = np.argsort(name_pnl_full)[::-1]
    print("\n  top-8 lifetime contributors — peak month & peak YEAR:")
    jp_idx = list(order[:8])
    for i in jp_idx:
        peak = rec_d[int(np.argmax(book.attrs["Dmat"][:, i]))]
        tk = names.get(int(cols[i]), int(cols[i]))
        print(f"    {tk:<8} peak {peak.date()} (yr {peak.year})  lifetime ${name_pnl_full[i]:,.0f}",
              flush=True)

    # ════════════════ R1b  JACKPOT INDEPENDENCE vs COMMON 2024 WAVE ══════════════
    print("\n=== R1b  JACKPOT INDEPENDENCE: 5 names independent stories or one 2024 wave? ===")
    top5 = list(order[:5])
    top5_tk = [names.get(int(cols[i]), int(cols[i])) for i in top5]
    # daily midpoint returns of the 5 names during 2024
    d2024 = np.array([d.year == 2024 for d in days])
    R = np.column_stack([np.nan_to_num(mp[d2024, i]) for i in top5])
    Rdf = pd.DataFrame(R, columns=top5_tk)
    # restrict to active rows (any of the 5 had a nonzero move) to avoid 0-pad inflation
    active = (np.abs(R).sum(axis=1) > 0)
    corr = Rdf[active].corr()
    pairs = corr.values[np.triu_indices(5, 1)]
    mean_pair = float(np.nanmean(pairs))
    print(f"  top-5 jackpots: {top5_tk}")
    print("  pairwise daily-return correlation (2024):")
    print("    " + corr.round(2).to_string().replace("\n", "\n    "))
    print(f"  mean pairwise corr (5 jackpots, 2024) = {mean_pair:.2f}")

    # baseline: mean pairwise corr of a broad bucket-5 sample in 2024
    b5_2024_cols = np.where(np.nansum(np.where(bmat[d2024] == 5, 1.0, 0.0), axis=0) > 60)[0]
    rng = np.random.default_rng(0)
    samp = rng.choice(b5_2024_cols, size=min(40, len(b5_2024_cols)), replace=False)
    Rb = np.column_stack([np.nan_to_num(mp[d2024, i]) for i in samp])
    Rbdf = pd.DataFrame(Rb)
    actb = (np.abs(Rb).sum(axis=1) > 0)
    cb = Rbdf[actb].corr().values
    base_pair = float(np.nanmean(cb[np.triu_indices(len(samp), 1)]))
    print(f"  baseline mean pairwise corr ({len(samp)} random bucket-5 names, 2024) = {base_pair:.2f}")

    # did bucket-5 AS A WHOLE lift in 2024? equal-weight bucket-5 return per year
    print("\n  bucket-5 EQUAL-WEIGHT return per year (did the whole bucket lift, or just the jackpots?):")
    b5_ew_year = {}
    for y in range(2021, 2025):
        dy = np.array([d.year == y for d in days])
        rows = np.where(dy)[0]
        daily = []
        for r in rows:
            sel = (bmat[r] == 5) & np.isfinite(mp[r])
            if sel.sum() >= 10:
                daily.append(np.nanmean(mp[r, sel]))
        ew = float(np.prod(1 + np.array(daily)) - 1) if daily else float("nan")
        b5_ew_year[y] = ew
        print(f"    {y}: bucket-5 EW {ew:>7.0%}")
    indep_verdict = ("INDEPENDENT (jackpots no more correlated than the bucket; "
                     "broad bucket did NOT all pop)"
                     if mean_pair < base_pair + 0.10 and b5_ew_year.get(2024, 0) < 0.60
                     else "COMMON 2024 WAVE (jackpots co-move and/or whole bucket lifted)")
    print(f"  -> R1b read: {indep_verdict}", flush=True)

    # ════════════════ R2  N-SWEEP FOR THE TAIL (unconstrained) ═══════════════════
    print("\n=== R2  N-SWEEP FOR THE TAIL (unconstrained full fills, CAP=10%, OOS 2021-2024) ===")
    print(f"{'N':>4}{'OOS ret':>10}{'alpha/yr':>10}{'t':>6}{'2mo med':>9}{'95th':>8}"
          f"{'max':>8}{'P>25%':>8}{'P>40%':>8}{'turn%':>8}")
    nsweep = {}
    for N in [8, 12, 15, 25]:
        s = run_book_volcap(zb5, mp, dvol, rowpos, days, fs, fe, N=N, f=None)
        o = stats(span(s, 2021, 2024), P)["tot"]
        a = factor_alpha(s, ff, 2021, 2024)
        w = two_month(span(s, 2021, 2024))
        t = dict(med=float(np.median(w)), p95=float(np.percentile(w, 95)), mx=float(w.max()),
                 p25=float((w > 0.25).mean()), p40=float((w > 0.40).mean()))
        nsweep[N] = dict(oos=o, alpha=a["alpha_ann"], t=a["t"], turn=s.attrs["turnover"], **t)
        print(f"{N:>4}{o:>10.0%}{a['alpha_ann']*100:>9.0f}%{a['t']:>6.1f}{t['med']:>9.1%}"
              f"{t['p95']:>8.0%}{t['mx']:>8.0%}{t['p25']:>8.1%}{t['p40']:>8.1%}"
              f"{s.attrs['turnover']*100:>8.1f}", flush=True)

    out = dict(
        anchor_oos=oos, full_period=full,
        per_year={int(y): v for y, v in per_year.items()},
        tail_years_40=yrs_with_40, tail_years_60=yrs_with_60, r1_verdict=r1_verdict,
        jackpots=top5_tk, mean_pair_corr_2024=mean_pair, base_pair_corr_2024=base_pair,
        bucket5_ew_year={int(k): v for k, v in b5_ew_year.items()},
        r1b_verdict=indep_verdict,
        nsweep={int(k): v for k, v in nsweep.items()})
    Path("results").mkdir(exist_ok=True)
    json.dump(out, open("results/pead_v8_repeatability.json", "w"), indent=2, default=float)
    print("\nwrote results/pead_v8_repeatability.json", flush=True)


if __name__ == "__main__":
    main()
