"""Illiquid Bucket-5 PEAD — the deployed strategy (self-contained).

Post-earnings-announcement drift, tilted to the most-illiquid Amihud quintile, on a
point-in-time small-cap universe.  This single module loads the data, builds the signal
and the illiquidity buckets, runs the book, and reports per-year returns + a QuantStats
tearsheet.

Locked config (validated; see DEPLOYMENT_RUNBOOK.md):
  universe  US common stock, mktcap rank 1001-3000 (Russell-2000 definition), monthly PIT
  signal    IBES-style SUE, enter rdq+1, hold to age 40 (buy-hold-40)
  liquidity  bucket-5 = most-illiquid Amihud quintile (monthly recompute)
  sizing    top-N=25, signal-proportional, 10% entry cap, appreciation-lock, ~98% invested
  returns   bid-ask MIDPOINT (bounce-free), delisting returns folded in, full fills at the
            reference price with the decision->fill signed slippage charged
  direction  long-only, no leverage

Usage:
  python pead_strategy.py                 # per-year returns + summary
  python pead_strategy.py --tearsheet     # also write results/pead_tearsheet.html
"""
from __future__ import annotations
import argparse, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

CRSP = "data_cache/pead_crsp_daily.parquet"
SUE = "data_cache/pead_ibes_sue.parquet"
DELIST = "data_cache/pead_delist.parquet"
FF = "data_cache/ff_factors_daily.parquet"

P = dict(N=25, TGT=0.98, CAP=0.10, H=40, stale_K=5, K_exit=30, band=0.03,
         band_lo=1001, band_hi=3000, sue_winsor=(-15.0, 15.0), rf=0.04,
         capital=1_000_000.0, NB=5)
DFRAC = 17.5 / 390.0                       # decision->fill window as a fraction of the day


# ─────────────────────────────── metrics ────────────────────────────────────
def total_return(s):
    return float((1 + s).prod() - 1) if len(s) else 0.0


def ann_sharpe(s, rf):
    if not len(s) or s.std() == 0:
        return 0.0
    return float((s.mean() * 252 - rf) / (s.std() * np.sqrt(252)))


def ann_vol(s):
    return float(s.std() * np.sqrt(252)) if len(s) else 0.0


def max_drawdown(s):
    if not len(s):
        return 0.0
    eq = (1 + s).cumprod()
    return float((eq / eq.cummax() - 1).min())


def stats(s):
    return dict(tot=total_return(s), sharpe=ann_sharpe(s, P["rf"]),
                vol=ann_vol(s), mdd=max_drawdown(s), n=len(s))


# ─────────────────────────────── data ───────────────────────────────────────
def load_panels():
    df = pd.read_parquet(CRSP)
    piv = lambda c: df.pivot_table(index="date", columns="permno", values=c).sort_index()
    close, vol = piv("prc"), piv("vol")
    ret = piv("ret").where(lambda x: x.notna(), close.pct_change(fill_method=None))
    bid, ask, mcap = piv("bid"), piv("ask"), piv("mktcap")
    shrcd = df.dropna(subset=["shrcd"]).groupby("permno")["shrcd"].last()
    names = df.groupby("permno")["ticker"].last()
    return dict(close=close, vol=vol, ret=ret, bid=bid, ask=ask, mcap=mcap,
                shrcd=shrcd, names=names)


def pit_universe(pan):
    """Monthly point-in-time membership: common stock, mktcap rank in [1001,3000]
    (Russell-2000 proxy), + staleness/valid-quote filter."""
    close, mcap, shrcd = pan["close"], pan["mcap"], pan["shrcd"]
    common = shrcd.reindex(mcap.columns).isin([10, 11])
    elig = pd.DataFrame(False, index=close.index, columns=close.columns)
    month_starts = close.index.to_series().groupby(close.index.to_period("M")).first()
    for d in month_starts:
        mc = mcap.loc[d].where(common).dropna(); mc = mc[mc > 0]
        rank = mc.rank(ascending=False, method="first")
        members = rank[(rank >= P["band_lo"]) & (rank <= P["band_hi"])].index
        m = (close.index.to_period("M") == d.to_period("M"))
        elig.loc[m, members] = True
    traded = (pan["vol"].fillna(0) > 0).rolling(P["stale_K"], min_periods=1).max().astype(bool)
    return elig & traded & close.notna() & (close >= 1.0) & pan["bid"].notna() & pan["ask"].notna()


def pead_active_panel(pan):
    """Winsorized SUE of the most recent announcement within the trailing [1,H] window
    (NaN outside). Available from the announcement bar -> held next day = enter rdq+1."""
    sue = pd.read_parquet(SUE); lo, hi = P["sue_winsor"]
    sue["sue"] = sue["sue"].astype(float).clip(lo, hi)
    idx = pan["close"].index
    out = pd.DataFrame(np.nan, index=idx, columns=pan["close"].columns)
    for permno, grp in sue.groupby("asset"):
        if permno not in out.columns:
            continue
        col = np.full(len(idx), np.nan)
        for _, row in grp.iterrows():
            ann = idx.searchsorted(row["date"])
            if ann >= len(idx):
                continue
            col[ann:min(ann + P["H"], len(idx))] = row["sue"]
        out[permno] = col
    return out


def zscore_panel(score_panel, elig):
    s = score_panel.where(elig)
    return s.sub(s.mean(axis=1), axis=0).div(s.std(axis=1).replace(0.0, np.nan), axis=0)


def add_delist_returns(ret_df, cols):
    d = pd.read_parquet(DELIST).dropna(subset=["dlret"])
    d = d[d["permno"].isin(cols)]; idx = ret_df.index; ret = ret_df.copy()
    for _, r in d.iterrows():
        permno, dt, dlr = int(r["permno"]), r["dlstdt"], float(r["dlret"])
        if permno not in ret.columns:
            continue
        loc = min(idx.searchsorted(dt), len(idx) - 1); day = idx[loc]
        base = ret.at[day, permno]; base = 0.0 if pd.isna(base) else float(base)
        ret.at[day, permno] = (1 + base) * (1 + dlr) - 1
    return ret


def fold_delist(ret_df, cols):
    return add_delist_returns(ret_df, set(int(c) for c in cols)).to_numpy(float, na_value=np.nan)


def amihud_buckets(pan, elig):
    """Monthly PIT Amihud-illiquidity quintiles within the eligible universe
    (1=most liquid .. NB=most illiquid)."""
    close, vol = pan["close"], pan["vol"]
    ret = pan["ret"].reindex(columns=close.columns).astype("float64")
    dvol = (close * vol).replace(0, np.nan)
    amihud = (ret.abs() / dvol).rolling(21, min_periods=10).mean()
    bucket = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    month_starts = close.index.to_series().groupby(close.index.to_period("M")).first()
    for d in month_starts:
        e = elig.loc[d]; names = e[e].index
        a = amihud.loc[d, names].dropna()
        if len(a) < P["NB"] * 10:
            continue
        q = pd.qcut(a.rank(method="first"), P["NB"], labels=range(1, P["NB"] + 1)).astype(int)
        m = (close.index.to_period("M") == d.to_period("M"))
        bucket.loc[m, q.index] = np.tile(q.values, (int(m.sum()), 1))
    return bucket.to_numpy(float)


# ─────────────────────────────── book engine ────────────────────────────────
def build_target(zr, held, cur_w, N, K_exit, band, cap, tgt):
    """Top-N capped signal-proportional target with K_exit retention band, no-trade
    weight band, 10% entry cap and appreciation-lock (locked names keep their weight)."""
    finite = np.isfinite(zr) & (zr > 0)
    if not finite.any():
        return np.empty(0, int), np.empty(0)
    order = np.argsort(np.where(finite, -zr, np.inf))[:int(finite.sum())]
    rank = np.full(zr.shape[0], 10**9); rank[order] = np.arange(len(order))
    retained = [i for i in held if rank[i] < K_exit]; rset = set(retained)
    adds = [i for i in order if i not in rset][:max(0, N - len(retained))]
    sel = np.array(retained + adds, int)
    if sel.size > N:
        sel = sel[np.argsort(-zr[sel])][:N]
    if sel.size == 0:
        return sel, np.empty(0)
    sz = np.clip(zr[sel], 0, None); cw = cur_w[sel]
    locked = cw >= cap; w = np.zeros(sel.size); w[locked] = cw[locked]
    budget = max(0.0, tgt - w[locked].sum()); oth = ~locked
    if budget > 0 and oth.any() and sz[oth].sum() > 0:
        ow = budget * sz[oth] / sz[oth].sum()
        for _ in range(50):
            over = ow > cap + 1e-12
            if not over.any():
                break
            ex = (ow[over] - cap).sum(); ow[over] = cap; free = ~over
            if not free.any() or ow[free].sum() == 0:
                break
            ow[free] += ex * ow[free] / ow[free].sum()
        w[oth] = np.minimum(ow, cap)
    if band > 0:
        for j, i in enumerate(sel):
            if cur_w[i] > 0 and abs(w[j] - cur_w[i]) < band:
                w[j] = cur_w[i]
    tot = w.sum()
    if tot > 1.0:
        w *= tgt / tot
    return sel[w > 0], w[w > 0]


def run_book(z_mat, ret_mat, rowpos, days, start, end, *, N, charge_signed=True):
    """Long-only top-N capped book, full fills at the reference price, decision->fill
    signed slippage charged on traded notional. Returns daily net-return Series."""
    cap, tgt, K_exit, band, V = P["CAP"], P["TGT"], P["K_exit"], P["band"], P["capital"]
    win = days[(days >= start) & (days <= end)]
    if len(win) < 5:
        return pd.Series(dtype=float)
    first = days[days < win[0]]; d0 = first[-1] if len(first) else win[0]
    dec = days[(days >= d0) & (days <= win[-1])]; di = [rowpos[d] for d in dec]
    ncol = z_mat.shape[1]; pos = np.zeros(ncol)
    rec_d, rec_v, turns = [], [], []
    for k in range(len(di) - 1):
        zr = z_mat[di[k]]; cur_w = pos / V if V > 0 else pos * 0.0
        idx, w = build_target(zr, np.where(pos > 0)[0], cur_w, N, K_exit, band, cap, tgt)
        new_w = np.zeros(ncol); new_w[idx] = w
        target = new_w * V; filled = target - pos
        turns.append(np.abs(filled).sum() / 2 / V)
        r = np.nan_to_num(ret_mat[di[k + 1]])
        signed = float((filled / V * r).sum()) * DFRAC if charge_signed else 0.0
        new_pos = np.maximum(target, 0.0); cash = V - new_pos.sum()
        pos = new_pos * (1.0 + r); Vn = pos.sum() + cash
        rec_d.append(dec[k + 1]); rec_v.append(Vn / V - 1.0 - signed); V = Vn
    s = pd.Series(rec_v, index=pd.to_datetime(rec_d)).sort_index()
    s.attrs["turnover"] = float(np.mean(turns))
    return s


def carhart_alpha(book, y0, y1):
    try:
        import statsmodels.api as sm
    except ImportError:
        return None
    ff = pd.read_parquet(FF); ff.index = pd.to_datetime(ff.index)
    sub = book[(book.index.year >= y0) & (book.index.year <= y1)]
    df = pd.concat([sub.rename("r"), ff], axis=1).dropna()
    if len(df) < 30:
        return None
    m = sm.OLS(df["r"] - df["RF"], sm.add_constant(df[["MktRF", "SMB", "HML", "Mom"]])).fit()
    return dict(alpha_ann=float(m.params["const"] * 252), t=float(m.tvalues["const"]))


# ─────────────────────────────── tearsheet ──────────────────────────────────
def write_tearsheet(book, out="results/pead_tearsheet.html"):
    import json, urllib.request
    for n, v in [("product", np.prod), ("Inf", np.inf), ("NaN", np.nan), ("float", float),
                 ("int", int), ("bool", bool)]:                  # numpy-2.x shims for quantstats
        if not hasattr(np, n):
            setattr(np, n, v)
    import matplotlib; matplotlib.use("Agg")
    import quantstats as qs
    try:
        url = ("https://query1.finance.yahoo.com/v8/finance/chart/IWM"
               "?period1=1483228800&period2=1735689600&interval=1d")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        j = json.loads(urllib.request.urlopen(req, timeout=40).read())
        r = j["chart"]["result"][0]; ts = pd.to_datetime(r["timestamp"], unit="s").normalize()
        bench = pd.Series(r["indicators"]["quote"][0]["close"], index=ts).pct_change().dropna()
        bench.index = bench.index.tz_localize(None)
    except Exception as e:
        print(f"  [benchmark unavailable, tearsheet without it: {e}]"); bench = None
    Path("results").mkdir(exist_ok=True)
    rets = book.copy(); rets.index = pd.to_datetime(rets.index).tz_localize(None)
    try:
        qs.reports.html(rets, benchmark=bench, rf=0.0, output=out, download_filename=out,
                        title="Illiquid Bucket-5 PEAD")
    except Exception:
        qs.reports.html(rets, rf=0.0, output=out, download_filename=out,
                        title="Illiquid Bucket-5 PEAD")
    print(f"  wrote {out}")


# ─────────────────────────────── main ───────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tearsheet", action="store_true", help="write results/pead_tearsheet.html")
    args = ap.parse_args()

    pan = load_panels()
    days = pan["close"].index; cols = pan["close"].columns
    rowpos = {d: i for i, d in enumerate(days)}
    elig = pit_universe(pan)
    mid = (pan["bid"] + pan["ask"]) / 2.0
    mp = fold_delist(mid.pct_change(fill_method=None).reindex(columns=cols).astype("float64"), cols)
    z = zscore_panel(pead_active_panel(pan), elig).to_numpy(float, na_value=np.nan)
    bmat = amihud_buckets(pan, elig)
    zb5 = np.where(bmat == 5, z, np.nan)                          # bucket-5 illiquid signal
    fs, fe = f"{days[0].year}-01-01", f"{days[-1].year}-12-31"

    book = run_book(zb5, mp, rowpos, days, fs, fe, N=P["N"])

    print("=" * 64)
    print("ILLIQUID BUCKET-5 PEAD  —  N=25, buy-hold-40, midpoint, full fills")
    print("=" * 64)
    print(f"{'year':<8}{'return':>12}{'sharpe':>10}")
    yrs = sorted(set(days.year))
    for y in yrs:
        st = stats(book[book.index.year == y])
        print(f"{y:<8}{st['tot']:>12.1%}{st['sharpe']:>10.2f}")
    full = stats(book)
    oos = stats(book[book.index.year >= 2021])
    a_oos = carhart_alpha(book, 2021, 2024)
    print("-" * 64)
    print(f"FULL  {days[0].year}-{days[-1].year}:  total {full['tot']:.0%}   sharpe {full['sharpe']:.2f}"
          f"   vol {full['vol']:.0%}   maxDD {full['mdd']:.0%}")
    print(f"OOS   2021-{days[-1].year}:  total {oos['tot']:.0%}   sharpe {oos['sharpe']:.2f}"
          + (f"   Carhart alpha {a_oos['alpha_ann']:.0%}/yr (t={a_oos['t']:.1f})" if a_oos else ""))
    print(f"turnover {book.attrs['turnover']*100:.1f}%/day")

    Path("results").mkdir(exist_ok=True)
    book.to_csv("results/pead_daily_returns.csv")
    if args.tearsheet:
        write_tearsheet(book)


if __name__ == "__main__":
    main()
