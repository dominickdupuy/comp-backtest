"""QUANT NOTE v9a — EXTENDED N-SWEEP to lock the core breadth.

The v8 tail sweep improved monotonically to N=25.  Extend to N in {30,35,50} to find
where breadth stops helping.  Watch for SUE-SIGNAL DILUTION: as N grows we hold weaker-
surprise names and per-name weight shrinks (no leverage), so there is a point where the
marginal name adds beta, not edge.  Pick the N that maximizes max / P(2mo>40%) while the
alpha-t stays strong.

Dilution diagnostic per N: mean |SUE z| of held names and mean per-name weight.

Unconstrained full fills (live rule confirmed), midpoint returns, delist-folded, PIT,
stale filter, hysteresis (K_exit=30, band=0.03).

METHODOLOGICAL NOTE: tail/per-year repeatability uses the full 2017-2024 window incl.
the 2017-2020 in-sample span.  Legitimate for a DESCRIPTIVE tail-distribution question
(the SUE/40-day signal is a literature prior, not fit to 2020), but the OOS columns
(2021-2024) are the out-of-sample read.  Both reported.
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
from scripts.run_pead_walkforward import stats, P
from scripts.run_pead_v4_fillmodel import span, factor_alpha
from scripts.run_pead_v7_fillrobust import prep, run_book_volcap, two_month

CAPITAL = P["capital"]


def dilution(zb5, mp, rowpos, days, start, end, N, CAP=0.10, TGT=0.98, K_exit=30, band=0.03):
    """Mean |SUE z| of held names and mean per-name weight across decision bars."""
    _v3.CAP, _v3.TGT = CAP, TGT
    win = days[(days >= start) & (days <= end)]
    first = days[days < win[0]]; d0 = first[-1] if len(first) else win[0]
    dec = days[(days >= d0) & (days <= win[-1])]; di = [rowpos[d] for d in dec]
    ncol = zb5.shape[1]; pos = np.zeros(ncol); V = CAPITAL
    sig, wts, nheld = [], [], []
    for k in range(len(di) - 1):
        zr = zb5[di[k]]; cur_w = pos / V if V > 0 else pos * 0.0
        held = np.where(pos > 0)[0]
        idx, w = build_target(zr, held, cur_w, N, K_exit, band)
        if len(idx):
            sig.append(float(np.nanmean(np.abs(zr[idx]))))
            wts.append(float(np.mean(w))); nheld.append(len(idx))
        r = np.nan_to_num(mp[di[k + 1]]); npz = np.zeros(ncol); npz[idx] = w * V
        pos = npz * (1 + r); V = pos.sum() + (V - npz.sum())
    return dict(sue_z=float(np.mean(sig)), wbar=float(np.mean(wts)),
                nheld=float(np.mean(nheld)))


def main():
    D = prep()
    z, mp, bmat, dvol = D["z"], D["mp"], D["bmat"], D["dvol"]
    rowpos, days, ff = D["rowpos"], D["days"], D["ff"]
    zb5 = np.where(bmat == 5, z, np.nan)
    fs, fe = "2017-01-01", "2024-12-31"

    print("=" * 104)
    print("v9a EXTENDED N-SWEEP — unconstrained illiquid bucket-5 (full fills); lock the core")
    print("  NOTE: OOS = 2021-2024 (out-of-sample). Tail repeatability uses full 8y (descriptive).")
    print(f"\n{'N':>4}{'nheld':>7}{'|SUEz|':>8}{'wbar':>7}{'OOS ret':>9}{'a/yr':>7}{'t':>5}"
          f"{'med':>7}{'95th':>7}{'max':>7}{'P>25%':>8}{'P>40%':>8}{'turn%':>7}")
    rows = {}
    for N in [12, 15, 25, 30, 35, 50]:
        s = run_book_volcap(zb5, mp, dvol, rowpos, days, fs, fe, N=N, f=None)
        sOOS = span(s, 2021, 2024)
        o = stats(sOOS, P)["tot"]; a = factor_alpha(s, ff, 2021, 2024)
        w = two_month(sOOS)
        med, p95, mx = np.median(w), np.percentile(w, 95), w.max()
        p25, p40 = (w > 0.25).mean(), (w > 0.40).mean()
        dl = dilution(zb5, mp, rowpos, days, fs, fe, N)
        rows[N] = dict(oos=o, alpha=a["alpha_ann"], t=a["t"], med=float(med), p95=float(p95),
                       mx=float(mx), p25=float(p25), p40=float(p40),
                       turn=s.attrs["turnover"], **dl)
        print(f"{N:>4}{dl['nheld']:>7.0f}{dl['sue_z']:>8.2f}{dl['wbar']:>7.1%}{o:>9.0%}"
              f"{a['alpha_ann']*100:>6.0f}%{a['t']:>5.1f}{med:>7.1%}{p95:>7.0%}{mx:>7.0%}"
              f"{p25:>8.1%}{p40:>8.1%}{s.attrs['turnover']*100:>7.1f}", flush=True)

    # pick N: maximize (max * P>40%) subject to t >= 3 (alpha stays strong)
    elig = {N: r for N, r in rows.items() if r["t"] >= 3.0}
    pick = max(elig, key=lambda N: elig[N]["mx"] * elig[N]["p40"]) if elig else \
        max(rows, key=lambda N: rows[N]["mx"] * rows[N]["p40"])
    # detect where breadth stops helping (max stops rising)
    Ns = sorted(rows)
    peak_max_N = max(rows, key=lambda N: rows[N]["mx"])
    print(f"\n  SUE dilution: |SUE z| falls {rows[Ns[0]]['sue_z']:.2f}@N{Ns[0]} -> "
          f"{rows[Ns[-1]]['sue_z']:.2f}@N{Ns[-1]};  per-name weight {rows[Ns[0]]['wbar']:.1%} -> "
          f"{rows[Ns[-1]]['wbar']:.1%}")
    print(f"  max 2mo peaks at N={peak_max_N} ({rows[peak_max_N]['mx']:.0%}); "
          f"alpha-t at N={peak_max_N} = {rows[peak_max_N]['t']:.1f}")
    print(f"  -> RECOMMENDED CORE N = {pick}  (max {rows[pick]['mx']:.0%}, "
          f"P>40% {rows[pick]['p40']:.1%}, t {rows[pick]['t']:.1f}, |SUEz| {rows[pick]['sue_z']:.2f})")

    out = dict(sweep={int(N): r for N, r in rows.items()}, recommended_N=int(pick),
               peak_max_N=int(peak_max_N))
    Path("results").mkdir(exist_ok=True)
    json.dump(out, open("results/pead_v9_extsweep.json", "w"), indent=2, default=float)
    print("\nwrote results/pead_v9_extsweep.json", flush=True)


if __name__ == "__main__":
    main()
