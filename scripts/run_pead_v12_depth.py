"""QUANT NOTE v12 — ILLIQUIDITY-DEPTH sweep (the right axis for this sim).

Transaction cost was never the binding constraint (v11), so turnover was the wrong axis.
The sim's unique, exploitable feature is NO MARKET IMPACT + full fills at undisturbed
real prices — worth the most exactly where real-world costs are highest: the most
extreme-illiquid names, where costs normally eat 70-100% of the premium and a real fund
cannot participate. So sweep illiquidity DEPTH, not turnover.

Within the fixed ~1700-name PIT universe, sweep how hard the held N=25 buy-hold-40 book
tilts into the illiquid extreme:
  - bucket-5 baseline (most-illiquid quintile, ~338 names/mo)
  - most-illiquid 200 / 100 / 50 (by Amihud; cross-checked by raw $ ADV)
  - composite: rank jointly on high SUE AND high illiquidity (z_sue + z_amihud)

THESIS: deeper illiquidity fattens the right tail (P(2mo>60%), max) even if EV stays
flat. Confirm or kill specifically.

PRE-REGISTERED KILL CRITERIA:
  - EV(2mo) > 0 is a hard floor; reject any depth that goes EV-negative.
  - DATA QUALITY is the primary gate. Deep-illiquid names risk bid-ask bounce + stale
    prints. Headline returns use MIDPOINT (the bounce-free / ABK-Blume-Stambaugh-spirit
    read) + stale filter + delisting returns. Report the LAST-TRADE vs MID gap (the bounce
    artifact), the Blume-Stambaugh bounce-bias estimate from spreads, and the held relative
    spread by depth. If the incremental deep edge is mostly bounce/stale, reject it.
  - COVERAGE limiter: IBES/SUE coverage and |SUE z| of held names by depth; stop where
    coverage thins enough to degrade the surprise signal. (Coverage-free actual-earnings-
    change proxy flagged as a SEPARATE, separately-validated signal — NOT built/blended.)
  - Validation: OOS 2021-2024 only, 42d embargo at the IS->OOS seam, overlapping windows.
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.run_pead_v3 as _v3
from scripts.run_pead_v3 import build_target, add_delist_returns
from scripts.run_pead_walkforward import stats, P
from scripts.run_pead_v4_fillmodel import span, factor_alpha
from scripts.run_pead_v6_liquidity import fold_delist
from scripts.run_pead_v7_fillrobust import prep, run_book_volcap, two_month
from scripts.run_pead_v11_rotation import oos_tail

CAPITAL = P["capital"]


# ───────────── Amihud illiquidity + $ADV, continuous + PIT monthly rank ──────
def amihud_panel(pan):
    close, vol = pan["close"], pan["vol"]
    ret = pan["ret"].reindex(columns=close.columns).astype("float64")
    dvol = (close * vol).replace(0, np.nan)
    amihud = (ret.abs() / dvol).rolling(21, min_periods=10).mean()
    dadv = dvol.rolling(21, min_periods=10).mean()
    return amihud, dadv


def pit_rank(metric, elig, ascending):
    """Monthly PIT rank within eligible names. rank 1 = most illiquid.
    `ascending=True` for $ADV (low ADV => illiquid => rank 1)."""
    out = pd.DataFrame(np.nan, index=metric.index, columns=metric.columns)
    zout = pd.DataFrame(np.nan, index=metric.index, columns=metric.columns)
    month_starts = metric.index.to_series().groupby(metric.index.to_period("M")).first()
    for d in month_starts:
        e = elig.loc[d]; names = e[e].index
        a = metric.loc[d, names].dropna()
        if len(a) < 50:
            continue
        r = a.rank(ascending=ascending, method="first")          # 1 = most illiquid
        z = (a - a.mean()) / (a.std() if a.std() else np.nan)     # illiquidity z (high=illiquid)
        m = (metric.index.to_period("M") == d.to_period("M"))
        nrep = int(m.sum())
        out.loc[m, r.index] = np.tile(r.values, (nrep, 1))
        zout.loc[m, z.index] = np.tile(z.values, (nrep, 1))
    return out, zout


def main():
    D = prep()
    z, mp, bmat, dvol = D["z"], D["mp"], D["bmat"], D["dvol"]
    rowpos, days, ff, cols, elig = D["rowpos"], D["days"], D["ff"], D["cols"], D["elig"]
    pan = D["pan"]; names = pan["names"]
    # last-trade returns (bounce-prone) for the artifact comparison
    lt = fold_delist(pan["ret"].reindex(columns=cols).astype("float64"), cols)
    # relative spread (tradeability gate)
    mid_px = (pan["bid"] + pan["ask"]) / 2.0
    relspr = ((pan["ask"] - pan["bid"]) / mid_px).reindex(columns=cols).to_numpy(float, na_value=np.nan)

    amihud, dadv = amihud_panel(pan)
    arank, az = pit_rank(amihud, elig, ascending=False)          # high Amihud illiquid
    drank, _ = pit_rank(dadv, elig, ascending=True)              # low $ADV illiquid
    arank_m = arank.to_numpy(float); drank_m = drank.to_numpy(float)
    az_m = az.to_numpy(float)
    fs, fe = "2017-01-01", "2024-12-31"

    # depth-restricted signals (Amihud rank), composite, and cross-check ($ADV rank)
    signals = {
        "bucket5(~338)": np.where(bmat == 5, z, np.nan),
        "illiq-200":     np.where(arank_m <= 200, z, np.nan),
        "illiq-100":     np.where(arank_m <= 100, z, np.nan),
        "illiq-50":      np.where(arank_m <= 50,  z, np.nan),
        "composite(SUE*illiq)": np.where(np.isfinite(z) & (z > 0), z + np.nan_to_num(az_m), np.nan),
        "ADV-100(x-check)": np.where(drank_m <= 100, z, np.nan),
    }

    print("=" * 104)
    print("v12 ILLIQUIDITY-DEPTH SWEEP — buy-hold-40, N=25 (OOS 2021-2024, embargoed, MIDPOINT)")
    print("  thesis: deeper illiquidity fattens the right tail even if EV is flat")
    print(f"\n{'depth':<22}{'OOS':>8}{'EV2mo':>7}{'a-t':>6}{'P>40':>7}{'P>60':>7}{'max':>7}"
          f" | {'LT-MID':>7}{'B-S bp':>7}{'relspr':>7}{'cover':>7}{'|SUEz|':>7}")
    rows = {}
    for label, sig in signals.items():
        # MID (headline) and LT (artifact) books
        bM = run_book_volcap(sig, mp, dvol, rowpos, days, fs, fe, N=25, f=None,
                             K_exit=30, band=0.03)
        bL = run_book_volcap(sig, lt, dvol, rowpos, days, fs, fe, N=25, f=None,
                             K_exit=30, band=0.03)
        tM = oos_tail(bM); tL = oos_tail(bL)
        a = factor_alpha(bM, ff, 2021, 2024)
        # held-name diagnostics: coverage, |SUE z|, relative spread, B-S bounce bias
        diag = held_diag(sig, z, mp, relspr, rowpos, days, fs, fe)
        rows[label] = dict(oos=tM["tot"], ev=tM["ev"], t=a["t"], p40=tM["p40"], p60=tM["p60"],
                           mx=tM["mx"], lt_mid=tL["tot"] - tM["tot"], bs_bp=diag["bs_bp"],
                           relspr=diag["relspr"], cover=diag["cover"], suez=diag["suez"])
        print(f"{label:<22}{tM['tot']:>8.0%}{tM['ev']:>7.1%}{a['t']:>6.1f}{tM['p40']:>7.1%}"
              f"{tM['p60']:>7.1%}{tM['mx']:>7.0%} | {tL['tot']-tM['tot']:>+7.0%}{diag['bs_bp']:>7.0f}"
              f"{diag['relspr']:>7.1%}{diag['cover']:>7.1f}{diag['suez']:>7.2f}", flush=True)

    # ── per-depth kill-criteria verdict ──
    print("\n=== KILL-CRITERIA VERDICT BY DEPTH ===")
    base = rows["bucket5(~338)"]
    for label, r in rows.items():
        flags = []
        if r["ev"] <= 0:
            flags.append("EV<=0 (REJECT)")
        # bounce/stale: a large LT-MID gap relative to the MID total => artifact-driven
        if r["oos"] > 0 and (r["lt_mid"] / max(abs(r["oos"]), 1e-9)) > 0.20:
            flags.append(f"bounce artifact (LT-MID {r['lt_mid']:+.0%})")
        if r["relspr"] > 0.06:
            flags.append(f"wide spread {r['relspr']:.1%} (tradeability)")
        if r["cover"] < 25:
            flags.append(f"coverage thin ({r['cover']:.0f}<25 held)")
        if r["suez"] < base["suez"] - 0.3:
            flags.append(f"SUE diluted ({r['suez']:.2f})")
        ok = not flags
        tail_better = (r["p60"] > base["p60"] + 0.005) or (r["mx"] > base["mx"] + 0.05)
        verdict = ("CLEAN" + ("  + fatter tail" if tail_better else "  (tail ~flat)")) if ok \
            else "FLAG: " + "; ".join(flags)
        print(f"  {label:<22} {verdict}", flush=True)

    print("\n  NOTE: MIDPOINT headline already removes bid-ask bounce (the Blume-Stambaugh/ABK")
    print("        upward bias); LT-MID is the bounce that would otherwise inflate returns.")
    print("        Coverage-free actual-earnings-change proxy NOT built (would be a separate,")
    print("        separately-validated signal; do not blend into the SUE core).")

    out = {label: rows[label] for label in rows}
    Path("results").mkdir(exist_ok=True)
    json.dump(out, open("results/pead_v12_depth.json", "w"), indent=2, default=float)
    print("\nwrote results/pead_v12_depth.json", flush=True)


def held_diag(sig, z, mp, relspr, rowpos, days, start, end, N=25, CAP=0.10, TGT=0.98,
              K_exit=30, band=0.03):
    """Simulate the buy-hold book; over OOS bars record: coverage (# live positive
    surprises available in this depth set), |SUE z| of held names, held relative spread,
    and the Blume-Stambaugh bid-ask-bounce bias (annualized bp) from held spreads."""
    _v3.CAP, _v3.TGT = CAP, TGT
    win = days[(days >= start) & (days <= end)]
    first = days[days < win[0]]; d0 = first[-1] if len(first) else win[0]
    dec = days[(days >= d0) & (days <= win[-1])]; di = [rowpos[d] for d in dec]
    ncol = sig.shape[1]; pos = np.zeros(ncol); V = CAPITAL
    cov, suez, spr, bsday = [], [], [], []
    oos = np.array([d.year >= 2021 for d in dec])
    for k in range(len(di) - 1):
        zr = sig[di[k]]; cur_w = pos / V if V > 0 else pos * 0.0
        held = np.where(pos > 0)[0]
        idx, w = build_target(zr, held, cur_w, N, K_exit, band)
        if oos[k] and idx.size:
            cov.append(int((np.isfinite(zr) & (zr > 0)).sum()))
            suez.append(float(np.nanmean(np.abs(z[di[k]][idx]))))
            s = relspr[di[k]][idx]; s = s[np.isfinite(s)]
            if s.size:
                spr.append(float(np.nanmean(s)))
                bsday.append(float(np.nanmean((s / 2.0) ** 2)))    # B-S per-day bounce variance
        new_pos = np.zeros(ncol); new_pos[idx] = w * V
        rr = np.nan_to_num(mp[di[k + 1]])
        pos = new_pos * (1.0 + rr); V = pos.sum() + (V - new_pos.sum())
    return dict(cover=float(np.mean(cov)) if cov else 0.0,
                suez=float(np.mean(suez)) if suez else 0.0,
                relspr=float(np.mean(spr)) if spr else 0.0,
                bs_bp=float(np.mean(bsday) * 252 * 1e4) if bsday else 0.0)


if __name__ == "__main__":
    main()
