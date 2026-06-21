"""QUANT NOTE v4 Workstreams B + C.1 + D — quality core, skew sleeve, M&A base
rate, and 2-month tournament shaping, all under the corrected signed-fill model.

B  Novy-Marx gross profitability (GP/A) long core (slow -> delay-immune).
C.1 high-MAX/IVOL skew sleeve (negative-EV; size-limited for right tail) and the
    historical M&A-target base rate from CRSP merger delistings.
D  2-month (42d) window distribution + N-sweep; pick N for the right tail s.t.
    aggregate EV stays positive after the signed fill cost (F EV-floor).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_pead_walkforward import load_panels, pit_universe, zscore_panel, stats, P
from scripts.run_pead_v3 import add_delist_returns, _ew
from scripts.run_pead_v4_fillmodel import run_book_fill, span, factor_alpha
import pandas as pd


def quality_panel(pan):
    q = pd.read_parquet("data_cache/compustat_quality.parquet")
    q["avail"] = q["datadate"] + pd.Timedelta(days=150)  # reporting lag
    close = pan["close"]
    gpa = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    for permno, g in q.sort_values("avail").groupby("permno"):
        if permno not in gpa.columns:
            continue
        col = pd.Series(np.nan, index=close.index)
        for _, r in g.iterrows():
            loc = close.index.searchsorted(r["avail"])
            if loc < len(close.index):
                col.iloc[loc] = r["gpa"]
        gpa[permno] = col.ffill()
    return gpa


def max_panel(pan, k=5, window=21):
    """Bali-Cakici-Whitelaw MAX: mean of the k largest daily returns over window."""
    r = pan["ret"].reindex(columns=pan["close"].columns).astype("float64")
    return r.rolling(window, min_periods=10).apply(
        lambda x: np.mean(np.sort(x)[-k:]), raw=True)


def ma_base_rate(cols):
    d = pd.read_parquet("data_cache/pead_delist.parquet")
    d = d[d["permno"].isin(cols)]
    merg = d[(d["dlstcd"] >= 200) & (d["dlstcd"] < 300)]
    yrs = 8.0
    n_names = len(cols)
    rate = len(merg) / n_names / yrs
    mret = merg["dlret"].dropna().astype(float)
    return dict(n_merg=len(merg), base_rate=rate, mean_merg_dlret=float(mret.mean()),
                pct_pos=float((mret > 0).mean()))


def two_month(s, win=42):
    v = s.values
    return np.array([np.prod(1 + v[i:i + win]) - 1 for i in range(len(v) - win + 1)])


def main():
    pan = load_panels()
    days = pan["close"].index; cols = pan["close"].columns
    rowpos = {d: i for i, d in enumerate(days)}
    elig = pit_universe(pan, P)
    ret_df = pan["ret"].reindex(columns=cols).astype("float64")
    ret_df, n_dl, _ = add_delist_returns(ret_df, set(int(c) for c in cols))
    ret_mat = ret_df.to_numpy(float, na_value=np.nan)
    ff = pd.read_parquet("data_cache/ff_factors_daily.parquet"); ff.index = pd.to_datetime(ff.index)
    fs, fe = "2017-01-01", "2024-12-31"
    ob_oos = stats(_ew(elig, ret_mat, rowpos, days, "2021-01-01", fe), P)["tot"]
    ob_full = stats(_ew(elig, ret_mat, rowpos, days, fs, fe), P)["tot"]

    print("=== B. GROSS-PROFITABILITY QUALITY CORE (signed fill) ===")
    zq = zscore_panel(quality_panel(pan), elig).to_numpy(float, na_value=np.nan)
    zmax = zscore_panel(max_panel(pan), elig).to_numpy(float, na_value=np.nan)
    zblend = np.where(np.isnan(zq) | np.isnan(zmax), np.nan, 0.5 * zq + 0.5 * zmax)

    def show(label, z, N, K_exit, band):
        s = run_book_fill(z, ret_mat, rowpos, days, fs, fe, N=N, K_exit=K_exit,
                          band=band, fill_mode="signed")
        full, oos = stats(s, P), stats(span(s, 2021, 2024), P)
        a = factor_alpha(s, ff, 2021, 2024)
        print(f"  {label:<26} turn/d {s.attrs['turnover']*100:4.1f}%  full {full['tot']:>7.0%}  "
              f"OOS {oos['tot']:>6.0%}  shrp {oos['sharpe']:4.2f}  "
              f"alpha {a['alpha_ann']:>5.0%}/yr t={a['t']:>4.1f}", flush=True)
        return s

    print(f"  (EW benchmark: full {ob_full:.0%}, OOS {ob_oos:.0%})")
    q20 = show("quality top-20", zq, 20, 30, 0.03)
    q12 = show("quality top-12", zq, 12, 20, 0.02)

    print("\n=== C.1 SKEW SLEEVE + M&A BASE RATE ===")
    sk = show("high-MAX skew top-12", zmax, 12, 20, 0.0)
    sk_ev = float(two_month(sk).mean())
    print(f"  high-MAX sleeve 2mo EV {sk_ev:+.1%} (Bali: negative-EV on avg -> skew only)")
    ma = ma_base_rate(set(int(c) for c in cols))
    print(f"  M&A: {ma['n_merg']} merger-delistings, base rate {ma['base_rate']:.1%}/yr, "
          f"mean merger dlret {ma['mean_merg_dlret']:+.1%} ({ma['pct_pos']:.0%} positive)")

    print("\n=== F. INTEGRATED BOOK (quality core + MAX skew, 50/50 blend) + D shaping ===")
    print(f"{'N':>4}{'turn/d':>8}{'2mo med':>9}{'5th':>8}{'95th':>9}{'max':>9}{'P(>25%)':>9}{'EV':>8}")
    e_rows = {}
    for N in [5, 8, 10, 15]:
        s = run_book_fill(zblend, ret_mat, rowpos, days, fs, fe, N=N,
                          K_exit=max(N + 8, 20), band=0.02, fill_mode="signed")
        w = two_month(s); q = np.percentile(w, [5, 50, 95])
        ev = float(w.mean()); p_hi = float((w > 0.25).mean())
        e_rows[N] = dict(ev=ev, p5=float(q[0]), p50=float(q[1]), p95=float(q[2]),
                         mx=float(w.max()), p_hi=p_hi, turn=s.attrs["turnover"])
        print(f"{N:>4}{s.attrs['turnover']*100:>7.1f}%{q[1]:>9.1%}{q[0]:>8.1%}"
              f"{q[2]:>9.1%}{w.max():>9.1%}{p_hi:>9.1%}{ev:>8.1%}", flush=True)
    feas = {n: r for n, r in e_rows.items() if r["ev"] > 0}
    pick = max(feas, key=lambda n: feas[n]["p_hi"]) if feas else None
    print(f"  chosen N (max P(2mo>25%) s.t. EV>0): N={pick}" +
          (f"  [EV {e_rows[pick]['ev']:+.1%}, 95th {e_rows[pick]['p95']:.0%}, "
           f"5th {e_rows[pick]['p5']:.1%}, max {e_rows[pick]['mx']:.0%}]" if pick else ""))

    out = dict(bench_full=ob_full, bench_oos=ob_oos,
               quality_top20_oos=stats(span(q20, 2021, 2024), P)["tot"],
               quality_top12_oos=stats(span(q12, 2021, 2024), P)["tot"],
               quality_alpha_oos=factor_alpha(q20, ff, 2021, 2024),
               skew_2mo_ev=sk_ev, ma=ma,
               integrated={int(n): r for n, r in e_rows.items()}, chosen_N=pick)
    Path("results").mkdir(exist_ok=True)
    json.dump(out, open("results/pead_v4_quality.json", "w"), indent=2, default=float)
    print("\nwrote results/pead_v4_quality.json", flush=True)


if __name__ == "__main__":
    main()
