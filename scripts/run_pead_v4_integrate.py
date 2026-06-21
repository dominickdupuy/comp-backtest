"""v4 final integration — correct the sleeve mix the literal spec got wrong.

Workstream A proved the PEAD book is the positive-EV, significant-alpha core
(not quality, which failed its OOS gate). So build: PEAD core + a SMALL high-MAX
skew dose, keeping aggregate EV > 0 (the EV floor), and pick (skew fraction, core
N) that maximizes the 2-month right tail for the 1-of-76 tournament.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_pead_walkforward import load_panels, pit_universe, pead_active_panel, zscore_panel, stats, P
from scripts.run_pead_v3 import add_delist_returns
from scripts.run_pead_v4_fillmodel import run_book_fill, span, factor_alpha
from scripts.run_pead_v4_quality import max_panel, two_month


def main():
    pan = load_panels()
    days = pan["close"].index; cols = pan["close"].columns
    rowpos = {d: i for i, d in enumerate(days)}
    elig = pit_universe(pan, P)
    ret_df = pan["ret"].reindex(columns=cols).astype("float64")
    ret_df, _, _ = add_delist_returns(ret_df, set(int(c) for c in cols))
    ret_mat = ret_df.to_numpy(float, na_value=np.nan)
    zp = zscore_panel(pead_active_panel(pan, P), elig).to_numpy(float, na_value=np.nan)
    zmax = zscore_panel(max_panel(pan), elig).to_numpy(float, na_value=np.nan)
    fs, fe = "2017-01-01", "2024-12-31"
    skew = run_book_fill(zmax, ret_mat, rowpos, days, fs, fe, N=12, K_exit=20, fill_mode="signed")

    print("=== v4 FINAL: PEAD core + small MAX-skew dose (signed fill, EV floor) ===")
    print(f"{'coreN':>6}{'skew%':>6}{'2mo EV':>8}{'med':>8}{'5th':>8}{'95th':>8}{'max':>9}{'P(>25%)':>9}{'P(>40%)':>9}")
    best = None
    rows = {}
    for N in [10, 15, 20]:
        core = run_book_fill(zp, ret_mat, rowpos, days, fs, fe, N=N, K_exit=30, band=0.03, fill_mode="signed")
        for f in [0.0, 0.10, 0.15, 0.20, 0.30]:
            r = (1 - f) * core + f * skew.reindex(core.index).fillna(0.0)
            w = two_month(r); q = np.percentile(w, [5, 50, 95])
            ev = float(w.mean()); p25 = float((w > 0.25).mean()); p40 = float((w > 0.40).mean())
            rows[(N, f)] = dict(ev=ev, p5=float(q[0]), p50=float(q[1]), p95=float(q[2]),
                                mx=float(w.max()), p25=p25, p40=p40)
            print(f"{N:>6}{f*100:>5.0f}%{ev:>8.1%}{q[1]:>8.1%}{q[0]:>8.1%}{q[2]:>8.1%}"
                  f"{w.max():>9.0%}{p25:>9.1%}{p40:>9.1%}", flush=True)
    feas = {k: v for k, v in rows.items() if v["ev"] > 0}
    pick = max(feas, key=lambda k: feas[k]["p40"])  # maximize aggressive right tail
    pv = rows[pick]
    print(f"\n  CHOSEN (max P(2mo>40%) s.t. EV>0): core N={pick[0]}, skew={pick[1]*100:.0f}%")
    print(f"    2mo: EV {pv['ev']:+.1%}, median {pv['p50']:.1%}, 5th {pv['p5']:.1%}, "
          f"95th {pv['p95']:.0%}, max {pv['mx']:.0%}, P(>25%) {pv['p25']:.1%}, P(>40%) {pv['p40']:.1%}")
    # vs pure-core (skew 0) at same N for the skew's marginal contribution
    base = rows[(pick[0], 0.0)]
    print(f"    vs pure PEAD core (N={pick[0]}): EV {base['ev']:+.1%} -> {pv['ev']:+.1%}, "
          f"P(>40%) {base['p40']:.1%} -> {pv['p40']:.1%}, max {base['mx']:.0%} -> {pv['mx']:.0%}")
    json.dump({f"N{k[0]}_skew{int(k[1]*100)}": v for k, v in rows.items()},
              open("results/pead_v4_integrate.json", "w"), indent=2, default=float)
    print("\nwrote results/pead_v4_integrate.json", flush=True)


if __name__ == "__main__":
    main()
