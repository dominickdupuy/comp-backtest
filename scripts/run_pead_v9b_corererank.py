"""v9b — re-rank the core N-sweep on the WIN-RELEVANT thresholds: P(2mo>60%) and max
single-window (not P>40%), across N in {20,25,30}.  Unconstrained illiquid bucket-5,
full fills, OOS 2021-2024.  Pick N -> user locks.
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_pead_walkforward import stats, P
from scripts.run_pead_v4_fillmodel import span, factor_alpha
from scripts.run_pead_v7_fillrobust import prep, run_book_volcap, two_month


def main():
    D = prep()
    z, mp, bmat, dvol = D["z"], D["mp"], D["bmat"], D["dvol"]
    rowpos, days, ff = D["rowpos"], D["days"], D["ff"]
    zb5 = np.where(bmat == 5, z, np.nan)
    fs, fe = "2017-01-01", "2024-12-31"
    print("=== v9b CORE RE-RANK on WIN-RELEVANT tail (P>60%, max) — OOS 2021-2024 ===")
    print(f"{'N':>4}{'OOS ret':>10}{'t':>6}{'95th':>8}{'max':>8}{'P>40%':>8}{'P>50%':>8}{'P>60%':>8}")
    res = {}
    for N in [20, 25, 30]:
        s = run_book_volcap(zb5, mp, dvol, rowpos, days, fs, fe, N=N, f=None)
        sOOS = span(s, 2021, 2024)
        o = stats(sOOS, P)["tot"]; a = factor_alpha(s, ff, 2021, 2024)
        w = two_month(sOOS)
        p40, p50, p60 = (w > .40).mean(), (w > .50).mean(), (w > .60).mean()
        res[N] = dict(oos=o, t=a["t"], p95=float(np.percentile(w, 95)), mx=float(w.max()),
                      p40=float(p40), p50=float(p50), p60=float(p60))
        print(f"{N:>4}{o:>10.0%}{a['t']:>6.1f}{np.percentile(w,95):>8.0%}{w.max():>8.0%}"
              f"{p40:>8.1%}{p50:>8.1%}{p60:>8.1%}", flush=True)
    by_p60 = max(res, key=lambda N: (res[N]["p60"], res[N]["mx"]))
    by_max = max(res, key=lambda N: (res[N]["mx"], res[N]["p60"]))
    print(f"\n  best by P(2mo>60%): N={by_p60} (P>60%={res[by_p60]['p60']:.1%}, max={res[by_p60]['mx']:.0%})")
    print(f"  best by max single window: N={by_max} (max={res[by_max]['mx']:.0%}, P>60%={res[by_max]['p60']:.1%})")


if __name__ == "__main__":
    main()
