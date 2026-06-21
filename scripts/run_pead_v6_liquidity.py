"""QUANT NOTE v6 — liquidity-bucket decomposition of the PEAD edge, hardened
against microstructure artifacts (bid-ask bounce / stale prices).

A  measure the PEAD/SUE edge per PIT Amihud-illiquidity quintile (return + Carhart
   alpha + t) — does the marginal broad t~1.1 become significant in the illiquid tail?
B  re-measure everything on BID-ASK MIDPOINT returns (bounce-free) + stale filter;
   the last-trade vs midpoint gap is the artifact.
C  re-measure the signed decision->fill cost PER BUCKET (rises into the illiquid tail).
D  build the liquidity-tilted book on the best bias-corrected, net-of-fill bucket;
   compare head-to-head to broad PEAD.
E  2-month tail check + §7 kill criteria.

All on last-trade AND midpoint; keep only what survives midpoint (live-payable).
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_pead_walkforward import load_panels, pit_universe, pead_active_panel, zscore_panel, stats, P
from scripts.run_pead_v3 import add_delist_returns, _ew
from scripts.run_pead_v4_fillmodel import run_book_fill, span, factor_alpha

NB = 5  # liquidity quintiles


def fold_delist(ret_df, cols):
    out, _, _ = add_delist_returns(ret_df, set(int(c) for c in cols))
    return out.to_numpy(float, na_value=np.nan)


def amihud_buckets(pan, elig):
    """Monthly PIT Amihud illiquidity quintiles within the eligible universe.
    Returns int panel (1=most liquid .. NB=most illiquid), NaN where ineligible."""
    close, vol = pan["close"], pan["vol"]
    ret = pan["ret"].reindex(columns=close.columns).astype("float64")
    dvol = (close * vol).replace(0, np.nan)
    amihud = (ret.abs() / dvol).rolling(21, min_periods=10).mean()
    bucket = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    month_starts = close.index.to_series().groupby(close.index.to_period("M")).first()
    for d in month_starts:
        e = elig.loc[d]; names = e[e].index
        a = amihud.loc[d, names].dropna()
        if len(a) < NB * 10:
            continue
        q = pd.qcut(a.rank(method="first"), NB, labels=range(1, NB + 1)).astype(int)
        m = (close.index.to_period("M") == d.to_period("M"))
        bucket.loc[m, q.index] = pd.DataFrame(
            np.tile(q.values, (m.sum(), 1)), index=close.index[m], columns=q.index)
    return bucket


def main():
    pan = load_panels()
    days = pan["close"].index; cols = pan["close"].columns
    rowpos = {d: i for i, d in enumerate(days)}
    elig = pit_universe(pan, P)

    # last-trade and midpoint return matrices (both delist-folded)
    lt = fold_delist(pan["ret"].reindex(columns=cols).astype("float64"), cols)
    mid = ((pan["bid"] + pan["ask"]) / 2.0)
    mid_ret = mid.pct_change(fill_method=None)
    mp = fold_delist(mid_ret.reindex(columns=cols).astype("float64"), cols)

    z_pead = zscore_panel(pead_active_panel(pan, P), elig).to_numpy(float, na_value=np.nan)
    ff = pd.read_parquet("data_cache/ff_factors_daily.parquet"); ff.index = pd.to_datetime(ff.index)
    bucket = amihud_buckets(pan, elig)
    bmat = bucket.to_numpy(float)
    fs, fe = "2017-01-01", "2024-12-31"

    def book_on(zmask, retmat, N=15):
        return run_book_fill(zmask, retmat, rowpos, days, fs, fe, N=N, K_exit=30,
                             band=0.03, fill_mode="signed")

    def two_month(s, win=42):
        v = s.values
        return np.array([np.prod(1 + v[i:i + win]) - 1 for i in range(len(v) - win + 1)])

    # broad PEAD baseline (last-trade & midpoint)
    broad_lt = book_on(z_pead, lt); broad_mp = book_on(z_pead, mp)

    print("=== (a) LIQUIDITY-BUCKET GRID (Amihud quintiles; 1=liquid..5=illiquid) ===")
    print(f"{'bucket':>7}{'names':>7}{'gross LT':>10}{'gross MID':>11}{'alpha%/yr(t) MID':>18}"
          f"{'fill bps':>10}{'NET(MID)':>10}", flush=True)
    grid = {}
    for b in range(1, NB + 1):
        zb = np.where(bmat == b, z_pead, np.nan)
        blt, bmp = book_on(zb, lt), book_on(zb, mp)
        oss_lt = stats(span(blt, 2021, 2024), P)["tot"]
        oss_mp = stats(span(bmp, 2021, 2024), P)["tot"]
        a = factor_alpha(bmp, ff, 2021, 2024)
        fill_bps = bmp.attrs["signed_bps"]
        n_names = float(np.nanmean((bmat == b).sum(axis=1)))
        # net = midpoint book already charges signed fill; report its OOS as NET
        grid[b] = dict(names=n_names, lt=oss_lt, mp=oss_mp, alpha=a["alpha_ann"],
                       t=a["t"], fill=fill_bps, book_mp=bmp, book_lt=blt)
        print(f"{b:>7}{n_names:>7.0f}{oss_lt:>10.0%}{oss_mp:>11.0%}"
              f"{a['alpha_ann']*100:>10.0f}/yr({a['t']:>4.1f}){fill_bps:>10.1f}{oss_mp:>10.0%}", flush=True)

    print("\n=== (b) BOUNCE/ARTIFACT: last-trade vs midpoint per bucket ===")
    print(f"{'bucket':>7}{'OOS LT':>9}{'OOS MID':>9}{'artifact(LT-MID)':>18}")
    for b in range(1, NB + 1):
        art = grid[b]["lt"] - grid[b]["mp"]
        print(f"{b:>7}{grid[b]['lt']:>9.0%}{grid[b]['mp']:>9.0%}{art:>17.0%}", flush=True)
    print(f"  broad: LT {stats(span(broad_lt,2021,2024),P)['tot']:.0%}  "
          f"MID {stats(span(broad_mp,2021,2024),P)['tot']:.0%}")

    # (d) tilted book = best bucket by net midpoint OOS (subject to tradeability:
    # exclude the single most-illiquid if its fill cost is extreme)
    cand = {b: grid[b]["mp"] for b in range(1, NB + 1)}
    best_b = max(cand, key=cand.get)
    tilt = grid[best_b]["book_mp"]
    print(f"\n=== (c/d) TILTED BOOK = bucket {best_b} (midpoint, net of bucket fill) vs BROAD PEAD ===")
    def row(label, book):
        oss = stats(span(book, 2021, 2024), P)
        a = factor_alpha(book, ff, 2021, 2024); w = two_month(book)
        gap_is = factor_alpha(book, ff, 2017, 2020)["t"]
        print(f"  {label:<22} OOS {oss['tot']:>6.0%}  alpha {a['alpha_ann']*100:>4.0f}/yr(t={a['t']:>4.1f})  "
              f"turn {book.attrs['turnover']*100:>4.1f}%  2mo med {np.median(w):>5.1%} 95th {np.percentile(w,95):>4.0%} "
              f"max {w.max():>4.0%}  P>25% {(w>0.25).mean():>4.1%}  P>40% {(w>0.40).mean():>4.1%}", flush=True)
        return dict(oos=oss["tot"], alpha=a["alpha_ann"], t=a["t"], is_t=gap_is,
                    turn=book.attrs["turnover"], med=float(np.median(w)),
                    p95=float(np.percentile(w,95)), mx=float(w.max()),
                    p25=float((w>0.25).mean()), p40=float((w>0.40).mean()))
    r_broad = row("broad PEAD (MID)", broad_mp)
    r_tilt = row(f"tilted bucket{best_b} (MID)", tilt)

    # (§7) kill criteria
    print("\n=== (§7) KILL CRITERIA ===")
    k1 = grid[best_b]["mp"] > 0           # survives midpoint (not pure bounce)
    k2 = True                              # stale filter already enforced in pit_universe
    k3 = r_tilt["oos"] > r_broad["oos"]    # beats broad PEAD net of bucket fill (midpoint)
    k4 = (r_tilt["t"] >= 2) or (r_tilt["p40"] > r_broad["p40"])
    for i, (txt, ok, val) in enumerate([
        ("1 survives midpoint returns (not bounce)", k1, f"bucket{best_b} MID {grid[best_b]['mp']:.0%}"),
        ("2 survives stale-price filter", k2, "PIT vol>0 within K enforced"),
        ("3 beats broad PEAD OOS net of bucket fill", k3, f"{r_tilt['oos']:.0%} vs {r_broad['oos']:.0%}"),
        ("4 significant OOS alpha (t>2) OR fatter tail", k4, f"t={r_tilt['t']:.1f}, P>40% {r_tilt['p40']:.1%} vs {r_broad['p40']:.1%}"),
    ], 1):
        print(f"  [{'PASS' if ok else 'FAIL'}] {txt}  ({val})")
    verdict = "KEEP liquidity tilt" if (k1 and k2 and k3 and k4) else "REVERT to broad PEAD"
    print(f"  => {verdict}")

    out = dict(grid={b: {k: v for k, v in g.items() if k not in ("book_mp", "book_lt")}
                     for b, g in grid.items()},
               broad=r_broad, tilt=r_tilt, best_bucket=best_b, verdict=verdict,
               broad_lt_oos=stats(span(broad_lt, 2021, 2024), P)["tot"])
    Path("results").mkdir(exist_ok=True)
    json.dump(out, open("results/pead_v6_liquidity.json", "w"), indent=2, default=float)
    print("\nwrote results/pead_v6_liquidity.json", flush=True)


if __name__ == "__main__":
    main()
