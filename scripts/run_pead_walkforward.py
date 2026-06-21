"""PEAD-core long-only drift backtest with walk-forward OOS validation (v2).

Engine = post-earnings-announcement drift on a point-in-time Russell-2000-proxy
universe. Real IBES standardized earnings surprise (SUE); enter the first close
AFTER the announcement, hold through the ~40-day drift window; long positive
surprises only; top-12 capped signal-proportional book; no leverage, no costs.

Success metric is OUT-OF-SAMPLE robustness (QUANT NOTE v2 0.1), so the report
fixes the design on an early window and evaluates on a later untouched one, and
shows the per-year + parameter-sweep plateau rather than one tuned number.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_russell_drift import target_weights  # capped sizer + cap/lock
from src.report import metrics

CRSP = "data_cache/pead_crsp_daily.parquet"
SUE = "data_cache/pead_ibes_sue.parquet"

P = dict(N=12, TGT=0.98, CAP=0.10, H=40, stale_K=5,
         band_lo=1001, band_hi=3000, sue_winsor=(-15.0, 15.0),
         rf=0.04, capital=1_000_000.0)


def load_panels():
    df = pd.read_parquet(CRSP)
    piv = lambda c: df.pivot_table(index="date", columns="permno", values=c).sort_index()
    close, open_, vol = piv("prc"), piv("openprc"), piv("vol")
    ret = piv("ret").where(lambda x: x.notna(), close.pct_change(fill_method=None))
    bid, ask, mcap = piv("bid"), piv("ask"), piv("mktcap")
    shrcd = df.dropna(subset=["shrcd"]).groupby("permno")["shrcd"].last()
    names = df.groupby("permno")["ticker"].last()
    return dict(close=close, open=open_, vol=vol, ret=ret, bid=bid, ask=ask,
                mcap=mcap, shrcd=shrcd, names=names)


def pit_universe(pan, p):
    """Monthly point-in-time membership: rank common stocks by mktcap at each
    month start, take the band_lo..band_hi band (Russell-2000 proxy). Returns a
    boolean (date x name) eligibility frame, fixed within each month."""
    close, mcap, shrcd = pan["close"], pan["mcap"], pan["shrcd"]
    common = shrcd.reindex(mcap.columns).isin([10, 11])
    elig = pd.DataFrame(False, index=close.index, columns=close.columns)
    month_starts = close.index.to_series().groupby(close.index.to_period("M")).first()
    for d in month_starts:
        mc = mcap.loc[d].where(common).dropna()
        mc = mc[mc > 0]
        rank = mc.rank(ascending=False, method="first")
        members = rank[(rank >= p["band_lo"]) & (rank <= p["band_hi"])].index
        m = (close.index.to_period("M") == d.to_period("M"))
        elig.loc[m, members] = True
    # staleness + valid quote
    traded = (pan["vol"].fillna(0) > 0).rolling(p["stale_K"], min_periods=1).max().astype(bool)
    return elig & traded & close.notna() & (close >= 1.0) & pan["bid"].notna() & pan["ask"].notna()


def pead_active_panel(pan, p):
    """For each (date, name) the winsorized SUE of the most recent announcement
    that is in the trailing [1, H] trading-day window (NaN outside). Available
    from the announcement date forward -> held the NEXT day = enter at rdq+1."""
    sue = pd.read_parquet(SUE)
    lo, hi = p["sue_winsor"]
    sue["sue"] = sue["sue"].astype(float).clip(lo, hi)
    close = pan["close"]
    # event -> nearest trading day on/after announcement
    idx = close.index
    out = pd.DataFrame(np.nan, index=idx, columns=close.columns)
    pos = pd.Series(range(len(idx)), index=idx)
    for permno, grp in sue.groupby("asset"):
        if permno not in out.columns:
            continue
        col = np.full(len(idx), np.nan)
        for _, row in grp.iterrows():
            ann = idx.searchsorted(row["date"])      # first trading day >= anndats
            if ann >= len(idx):
                continue
            end = min(ann + p["H"], len(idx))
            # value active from announcement bar; held next day (=rdq+1)
            col[ann:end] = row["sue"]
        out[permno] = col
    return out


def zscore_panel(score_panel, elig):
    """Vectorized cross-sectional z-score of `score_panel` over eligible names,
    one row (date) at a time. NaN where ineligible / no live surprise."""
    s = score_panel.where(elig)
    mu = s.mean(axis=1)
    sd = s.std(axis=1).replace(0.0, np.nan)
    return s.sub(mu, axis=0).div(sd, axis=0)


def _size_np(zr, cur_w, N, cap, TGT):
    """numpy port of target_weights: capped signal-proportional top-N + 10%
    entry cap + appreciation lock. Returns (indices, weights)."""
    valid = np.where(np.isfinite(zr) & (zr > 0))[0]
    if valid.size == 0:
        return np.empty(0, int), np.empty(0)
    vals = zr[valid]
    if valid.size > N:
        sel = valid[np.argpartition(vals, valid.size - N)[valid.size - N:]]
    else:
        sel = valid
    sz = zr[sel]
    cw = cur_w[sel]
    locked = cw >= cap
    w = np.zeros(sel.size)
    w[locked] = cw[locked]                       # locked winners hold their weight
    budget = max(0.0, TGT - w[locked].sum())
    oth = ~locked
    if budget > 0 and oth.any() and sz[oth].sum() > 0:
        ow = budget * sz[oth] / sz[oth].sum()
        for _ in range(50):                      # clip-to-cap + redistribute excess
            over = ow > cap + 1e-12
            if not over.any():
                break
            excess = (ow[over] - cap).sum()
            ow[over] = cap
            free = ~over
            if not free.any() or ow[free].sum() == 0:
                break
            ow[free] += excess * ow[free] / ow[free].sum()
        w[oth] = np.minimum(ow, cap)
    keep = w > 0
    return sel[keep], w[keep]


def backtest_np(z_mat, ret_mat, rowpos, days, p, start, end, sign=1.0):
    """Stateful long-only top-N capped book over [start,end] on numpy matrices.
    `sign=-1` runs the inverse-signal sanity check. Returns daily return Series."""
    win = days[(days >= start) & (days <= end)]
    if len(win) < 5:
        return pd.Series(dtype=float)
    first = days[days < win[0]]
    d0 = first[-1] if len(first) else win[0]
    dec = days[(days >= d0) & (days <= win[-1])]
    di = [rowpos[d] for d in dec]
    ncol = z_mat.shape[1]
    cap, TGT, N, V = p["CAP"], p["TGT"], p["N"], p["capital"]
    pos = np.zeros(ncol); cash = V
    rec_d, rec_v = [], []
    max_w = 0.0
    for k in range(len(di) - 1):
        zr = sign * z_mat[di[k]]
        cur_w = pos / V if V > 0 else pos * 0.0
        idx, w = _size_np(zr, cur_w, N, cap, TGT)
        # SANITY: long-only, no leverage, entry cap (locked names exempt)
        assert (w >= -1e-12).all() and w.sum() <= 1.0 + 1e-6
        if idx.size:
            entry = w[cur_w[idx] < cap]
            if entry.size:
                max_w = max(max_w, entry.max())
        new_pos = np.zeros(ncol); new_pos[idx] = w * V
        pos, cash = new_pos, V - new_pos.sum()
        r = np.nan_to_num(ret_mat[di[k + 1]])
        pos = pos * (1.0 + r)
        Vn = pos.sum() + cash
        rec_d.append(dec[k + 1]); rec_v.append(Vn / V - 1.0); V = Vn
    s = pd.Series(rec_v, index=pd.to_datetime(rec_d)).sort_index()
    s.attrs["max_entry_w"] = max_w
    return s


def ew_benchmark_np(elig_mat, ret_mat, rowpos, days, start, end):
    win = days[(days >= start) & (days <= end)]
    d, v = [], []
    for k in range(len(win) - 1):
        e = elig_mat[rowpos[win[k]]]
        r = ret_mat[rowpos[win[k + 1]]]
        m = e & np.isfinite(r)
        v.append(float(r[m].mean()) if m.any() else 0.0)
        d.append(win[k + 1])
    return pd.Series(v, index=pd.to_datetime(d)).sort_index()


def stats(s, p):
    if s.empty:
        return dict(tot=0, sharpe=0, vol=0, mdd=0, n=0)
    return dict(tot=metrics.total_return(s), sharpe=metrics.sharpe(s, p["rf"]),
                vol=metrics.ann_vol(s), mdd=metrics.max_drawdown(s), n=len(s))


def yr_slice(s, y):
    return s[s.index.year == y]


def span_stats(s, p, y0, y1):
    return stats(s[(s.index.year >= y0) & (s.index.year <= y1)], p)


def main():
    p = dict(P)
    pan = load_panels()
    print(f"data: {pan['close'].index[0].date()}..{pan['close'].index[-1].date()}  "
          f"{pan['close'].shape[1]} names", flush=True)
    elig = pit_universe(pan, p)
    print(f"PIT universe avg members/day: {elig.sum(axis=1).mean():.0f}", flush=True)
    pead = pead_active_panel(pan, p)
    full_start = f"{pan['close'].index[0].year}-01-01"
    full_end = f"{pan['close'].index[-1].year}-12-31"

    # precompute matrices once (numpy fast path)
    days = pan["close"].index
    rowpos = {d: i for i, d in enumerate(days)}
    ret_mat = pan["ret"].reindex(columns=pan["close"].columns).astype("float64").to_numpy(
        dtype=float, na_value=np.nan)
    elig_mat = elig.to_numpy(dtype=bool)
    z_mat = zscore_panel(pead, elig).to_numpy(dtype=float, na_value=np.nan)

    # ONE continuous book over the full period; slice by year for per-year stats.
    allr = backtest_np(z_mat, ret_mat, rowpos, days, p, full_start, full_end)
    invr = backtest_np(z_mat, ret_mat, rowpos, days, p, full_start, full_end, sign=-1.0)
    benr = ew_benchmark_np(elig_mat, ret_mat, rowpos, days, full_start, full_end)
    print(f"  [done full-period]  max entry weight {allr.attrs.get('max_entry_w', 0):.4f} "
          f"(cap {p['CAP']})", flush=True)

    yrs = sorted(set(pan["close"].index.year))
    print("\n=== PER-YEAR (params fixed at literature prior H=40,N=12; NOT tuned) ===")
    print(f"{'year':<6}{'PEAD ret':>10}{'PEAD shrp':>10}{'bench ret':>11}{'excess':>9}{'INV ret':>9}")
    rows = {}
    for y in yrs:
        st, bt, it = stats(yr_slice(allr, y), p), stats(yr_slice(benr, y), p), stats(yr_slice(invr, y), p)
        rows[y] = dict(pead=st, bench=bt, inv=it)
        print(f"{y:<6}{st['tot']:>10.1%}{st['sharpe']:>10.2f}{bt['tot']:>11.1%}"
              f"{st['tot']-bt['tot']:>9.1%}{it['tot']:>9.1%}", flush=True)

    fst = stats(allr, p)
    monthly = (1 + allr).resample("ME").prod() - 1
    wins = sum(rows[y]["pead"]["tot"] > rows[y]["bench"]["tot"] for y in yrs)
    print("\nFULL-PERIOD PEAD (one continuous book):")
    print(f"  total {fst['tot']:.1%}  ann.sharpe {fst['sharpe']:.2f}  vol {fst['vol']:.1%}  "
          f"maxDD {fst['mdd']:.1%}")
    print(f"  avg monthly return {monthly.mean():.2%}  (median {monthly.median():.2%})")
    print(f"  beat EW benchmark in {wins}/{len(yrs)} years", flush=True)

    # ---- formal IS/OOS split (design fixed on early years) -----------------
    iss, oss = span_stats(allr, p, 2017, 2020), span_stats(allr, p, 2021, 2024)
    isb, osb = span_stats(benr, p, 2017, 2020), span_stats(benr, p, 2021, 2024)
    print("\n=== IS/OOS SPLIT (design choices fixed on 2017-2020) ===")
    print(f"  IN-SAMPLE  2017-2020:  PEAD {iss['tot']:>7.1%} (shrp {iss['sharpe']:.2f})  "
          f"bench {isb['tot']:>7.1%}  excess {iss['tot']-isb['tot']:>6.1%}")
    print(f"  OUT-SAMPLE 2021-2024:  PEAD {oss['tot']:>7.1%} (shrp {oss['sharpe']:.2f})  "
          f"bench {osb['tot']:>7.1%}  excess {oss['tot']-osb['tot']:>6.1%}", flush=True)

    # ---- parameter robustness (plateau, not a peak) ------------------------
    print("\n=== PARAMETER ROBUSTNESS: full-period total return (want a plateau) ===")
    print("H \\\\ N", "".join(f"{n:>10}" for n in [10, 12, 15, 20]), flush=True)
    grid = {}
    for H in [20, 30, 40, 50, 60]:
        z_H = zscore_panel(pead_active_panel(pan, dict(p, H=H)), elig).to_numpy(
            dtype=float, na_value=np.nan)
        line = []
        for N in [10, 12, 15, 20]:
            rr = backtest_np(z_H, ret_mat, rowpos, days, dict(p, N=N), full_start, full_end)
            t = stats(rr, p)["tot"]; line.append(t); grid[(H, N)] = t
        print(f"H={H:<3}", "".join(f"{v:>10.1%}" for v in line), flush=True)

    out = dict(per_year={int(y): {k: rows[y][k]["tot"] for k in rows[y]} for y in yrs},
               full=fst, avg_monthly=float(monthly.mean()),
               is_oos=dict(is_tot=iss["tot"], is_excess=iss["tot"]-isb["tot"],
                           oos_tot=oss["tot"], oos_excess=oss["tot"]-osb["tot"],
                           oos_sharpe=oss["sharpe"]),
               robustness={f"H{h}_N{n}": v for (h, n), v in grid.items()})
    Path("results").mkdir(exist_ok=True)
    json.dump(out, open("results/pead_walkforward_summary.json", "w"), indent=2, default=float)
    allr.to_csv("results/pead_daily_returns.csv")
    monthly.to_csv("results/pead_monthly_returns.csv")
    print("\nwrote results/pead_walkforward_summary.json + daily/monthly CSVs")


if __name__ == "__main__":
    main()
