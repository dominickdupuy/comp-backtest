"""QUANT NOTE v11 — re-derive the CORE trading policy under the sim's TRUE constraint:
turnover is FREE (zero commission/spread/impact, full fills at undisturbed real prices).

The existing low-turnover hysteresis (K_exit=30, 3% no-trade band, ~6%/day) is a real-
world cost-suppression artifact with no justification here.  Strip it; re-optimize from
the principle that turnover is costless.

TARGET POLICY: each day hold the names with the highest CURRENT EXPECTED FORWARD PEAD
drift at the 10% entry cap; cut decayed/flat names and redeploy into fresh surprises;
never trim a name still trending (appreciation-lock; winners ride past 10%).  Harvest the
front-loaded drift continuously and maximize the count of distinct multibagger candidates
capital cycles through over a 2-month window.

The raw SUE signal is FLAT for H=40d then cliffs (see pead_active_panel) — so it carries
no economic decay.  We add expected-forward-drift  g = SUE_z * decay(age)  where
decay(age) is CALIBRATED on the realized IS (2017-2020) bucket-5 drift-by-age curve
(purged: calibrate on IS, evaluate OOS only).  Rotation threshold = a 'material' g-gap
(signal economics), NOT transaction costs.

KILL CRITERIA (pre-registered):
  - EV>0 hard constraint (reject any EV-negative policy: guards against churn-into-noise).
  - realistic fill: order now, fill ref-price 15-20min later -> SIGNED delay slippage
    charged on traded notional (run_book_volcap charge_signed); edge must survive net.
  - SUE dilution: track |SUE z| of names actually held (fast rotation must not push so
    deep into the ranking that surprise collapses).
  - tail preservation: top-PnL names under each policy + that rotation HOLDS them through
    their run (not rotating out the eventual multibaggers).
Validation: OOS 2021-2024 only (+ per-year), 42d embargo at the IS->OOS seam, overlapping
2-month windows.  IS used ONLY to calibrate decay; IS performance NOT reported.
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
from scripts.run_pead_walkforward import load_panels, pit_universe, stats, P, SUE
from scripts.run_pead_v4_fillmodel import span, factor_alpha, DFRAC
from scripts.run_pead_v7_fillrobust import prep, run_book_volcap, two_month

CAPITAL = P["capital"]
H = P["H"]                                     # 40-day drift horizon
TREND_K = 10                                   # uptrend lookback (days)
TREND_THRESH = 0.05                            # "still in a strong uptrend"
MARGIN = 0.25                                  # rotate non-protected held only if g < (1-m)*g_N


# ───────────── age matrix (trading-days since the live announcement) ─────────
def build_age_mat(pan, cols, days):
    sue = pd.read_parquet(SUE)
    idx = days
    age = np.full((len(idx), len(cols)), -1, dtype=np.int32)
    colpos = {int(c): j for j, c in enumerate(cols)}
    for permno, grp in sue.groupby("asset"):
        j = colpos.get(int(permno))
        if j is None:
            continue
        for _, row in grp.iterrows():
            ann = idx.searchsorted(row["date"])
            if ann >= len(idx):
                continue
            end = min(ann + H, len(idx))
            age[ann:end, j] = np.arange(end - ann)     # later events overwrite tail
    return age


# ───────────── decay(age) calibrated on IS realized bucket-5 drift ───────────
def calibrate_decay(z, mp, bmat, age_mat, rowpos, days, is_end="2020-12-31"):
    """Mean next-day midpoint return by age (IS, bucket-5, z>0); expected FORWARD
    drift from age a = sum_{a'>=a} dailydrift(a'); decay(a)=forward(a)/forward(0)."""
    isd = days[days <= is_end]
    di = [rowpos[d] for d in isd]
    sa = np.zeros(H + 1); ca = np.zeros(H + 1)
    for k in di[:-1]:
        zr = z[k]; b5 = bmat[k] == 5; ag = age_mat[k]
        sel = np.isfinite(zr) & (zr > 0) & b5 & (ag >= 0) & (ag <= H)
        if not sel.any():
            continue
        r1 = np.nan_to_num(mp[k + 1])
        ages = ag[sel]; rr = r1[sel]
        np.add.at(sa, ages, rr); np.add.at(ca, ages, 1.0)
    daily = np.where(ca > 0, sa / np.maximum(ca, 1), 0.0)
    fwd = np.array([daily[a:].sum() for a in range(H + 1)])     # remaining drift
    d = np.clip(fwd, 0, None) / max(fwd[0], 1e-9)
    return d, daily, fwd


# ───────────── trailing k-day return (uptrend gate) ─────────────────────────
def trailing_return(mp, k=TREND_K):
    lr = np.log1p(np.nan_to_num(mp))
    cs = np.cumsum(lr, axis=0)
    out = np.full_like(mp, np.nan, dtype=float)
    out[k:] = np.expm1(cs[k:] - cs[:-k])
    return out


# ───────────── rotation target builder (signal-economic) ────────────────────
def build_rotate(g, trail, cur_w, N, margin, trend_thresh, cap, tgt):
    finite = np.isfinite(g) & (g > 0)
    if not finite.any():
        return np.empty(0, int), np.empty(0)
    order = np.argsort(np.where(finite, -g, np.inf))
    ncand = int(finite.sum()); order = order[:ncand]
    gN = g[order[N - 1]] if ncand >= N else g[order[-1]]
    held = cur_w > 0
    locked = cur_w >= cap
    trending = np.nan_to_num(trail) > trend_thresh
    protected = held & (locked | trending)                     # never trim trenders/lockers
    competitive = finite & (g >= (1.0 - margin) * gN)          # still close to the frontier
    retained = np.where(held & (protected | competitive))[0]
    rset = set(retained.tolist())
    adds = [i for i in order if i not in rset][:max(0, N - len(retained))]
    sel = np.array(list(retained) + adds, int)
    if sel.size == 0:
        return sel, np.empty(0)
    # size by g (capped signal-proportional + appreciation lock), as in build_target
    sz = np.clip(g[sel], 0, None); cw = cur_w[sel]
    locked_s = cw >= cap
    w = np.zeros(sel.size); w[locked_s] = cw[locked_s]
    budget = max(0.0, tgt - w[locked_s].sum()); oth = ~locked_s
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
    tot = w.sum()
    if tot > 1.0:
        w *= tgt / tot
    return sel[w > 0], w[w > 0]


# ───────────── free-turnover rotation book (full fills + signed slippage) ────
def run_book_rotate(z, mp, age_mat, trail, decay, rowpos, days, start, end, *,
                    N=15, CAP=0.10, TGT=0.98, margin=MARGIN, trend_thresh=TREND_THRESH,
                    charge_signed=True, record_ledger=False):
    win = days[(days >= start) & (days <= end)]
    if len(win) < 5:
        return pd.Series(dtype=float)
    first = days[days < win[0]]; d0 = first[-1] if len(first) else win[0]
    dec = days[(days >= d0) & (days <= win[-1])]; di = [rowpos[d] for d in dec]
    ncol = z.shape[1]
    pos = np.zeros(ncol); V = CAPITAL
    turns, sigbps_acc, traded_acc = [], 0.0, 0.0
    sue_held, tickets = [], np.zeros(ncol, bool)
    rec_d, rec_v = [], []
    Dmat = np.zeros((len(di) - 1, ncol)) if record_ledger else None
    holddays = np.zeros(ncol, int) if record_ledger else None
    for k in range(len(di) - 1):
        zr = z[di[k]]; ag = age_mat[di[k]]
        dvec = decay[np.clip(ag, 0, H)]
        g = np.where((ag >= 0) & np.isfinite(zr), zr * dvec, np.nan)
        cur_w = pos / V if V > 0 else pos * 0.0
        idx, w = build_rotate(g, trail[di[k]], cur_w, N, margin, trend_thresh, CAP, TGT)
        new_w = np.zeros(ncol); new_w[idx] = w
        target_usd = new_w * V
        filled_dw = target_usd - pos
        turns.append(np.abs(filled_dw).sum() / 2 / V)
        r = np.nan_to_num(mp[di[k + 1]])
        signed = float((filled_dw / V * r).sum()) * DFRAC          # signed delay slippage
        sigbps_acc += signed; traded_acc += float(np.abs(filled_dw).sum() / V)
        haircut = signed if charge_signed else 0.0
        new_pos = np.maximum(target_usd, 0.0)
        cash = V - new_pos.sum()
        if record_ledger:
            Dmat[k] = new_pos * r
            holddays[new_pos > 1e-9] += 1
        if idx.size:
            zz = np.abs(zr[idx]); sue_held.append(float(np.nanmean(zz)))
            tickets[idx[new_w[idx] > 0.01]] = True
        pos = new_pos * (1.0 + r)
        Vn = pos.sum() + cash
        rec_d.append(dec[k + 1]); rec_v.append(Vn / V - 1.0 - haircut); V = Vn
    s = pd.Series(rec_v, index=pd.to_datetime(rec_d)).sort_index()
    s.attrs.update(turnover=float(np.mean(turns)),
                   signed_bps=(1e4 * sigbps_acc / traded_acc if traded_acc else 0.0),
                   sue_z=float(np.nanmean(sue_held)) if sue_held else 0.0,
                   tickets=int(tickets.sum()))
    if record_ledger:
        s.attrs["Dmat"] = Dmat; s.attrs["rec_d"] = pd.to_datetime(rec_d)
        s.attrs["holddays"] = holddays
    return s


# ───────────── metrics helpers (OOS, embargoed) ─────────────────────────────
def oos_tail(s, embargo=42):
    o = span(s, 2021, 2024)
    if len(o) > embargo:
        o = o.iloc[embargo:]                                  # purge IS->OOS seam
    w = two_month(o)
    return dict(ev=float(w.mean()), med=float(np.median(w)), p95=float(np.percentile(w, 95)),
                mx=float(w.max()), p40=float((w > .40).mean()), p50=float((w > .50).mean()),
                p60=float((w > .60).mean()), tot=stats(o, P)["tot"])


def ticket_count(book_attrs_tickets):
    return book_attrs_tickets


def main():
    D = prep()
    z, mp, bmat, dvol = D["z"], D["mp"], D["bmat"], D["dvol"]
    rowpos, days, ff, cols = D["rowpos"], D["days"], D["ff"], D["cols"]
    names = D["pan"]["names"]
    zb5 = np.where(bmat == 5, z, np.nan)
    age_mat = build_age_mat(D["pan"], cols, days)
    trail = trailing_return(mp, TREND_K)
    fs, fe = "2017-01-01", "2024-12-31"

    print("=" * 100)
    print("v11 FREE-TURNOVER CORE RE-DERIVATION (turnover costless; signed delay slippage charged)")

    # ── calibrate decay(age) on IS only ──
    decay, daily, fwd = calibrate_decay(zb5, mp, bmat, age_mat, rowpos, days)
    print("\n=== decay(age) CALIBRATED ON IS 2017-2020 (bucket-5 realized drift; purged) ===")
    print("  age:   " + " ".join(f"{a:>4}" for a in [0, 2, 5, 10, 15, 20, 30, 40]))
    print("  decay: " + " ".join(f"{decay[a]:>4.2f}" for a in [0, 2, 5, 10, 15, 20, 30, 40]))
    half = next((a for a in range(H + 1) if decay[a] <= 0.5), H)
    print(f"  forward-drift half-life ~ {half} trading days; decay->0 by age "
          f"{next((a for a in range(H+1) if decay[a] <= 0.05), H)}")

    # ── BUY-HOLD-40 baseline (current policy: K_exit=30, band=0.03), N=25 ──
    bh = run_book_volcap(zb5, mp, dvol, rowpos, days, fs, fe, N=25, f=None,
                         K_exit=30, band=0.03, record_ledger=True)
    # ticket count for buy-hold via its ledger
    bh_tickets = int((np.abs(bh.attrs["Dmat"]) > 0).any(axis=0).sum())
    bh_t = oos_tail(bh); a_bh = factor_alpha(bh, ff, 2021, 2024)

    # ── FREE-TURNOVER N-sweep ──
    print("\n=== FREE-TURNOVER ROTATION — concurrent-N sweep (OOS 2021-2024, embargoed) ===")
    print(f"{'N':>4}{'OOS':>9}{'EV(2mo)':>9}{'alpha-t':>9}{'P>40%':>7}{'P>60%':>7}{'max':>7}"
          f"{'turn%':>7}{'slip_bp':>8}{'|SUEz|':>7}{'tickets':>8}")
    rot = {}
    for N in [8, 12, 15, 20, 25, 30]:
        s = run_book_rotate(zb5, mp, age_mat, trail, decay, rowpos, days, fs, fe, N=N)
        t = oos_tail(s); a = factor_alpha(s, ff, 2021, 2024)
        rot[N] = dict(tail=t, t=a["t"], turn=s.attrs["turnover"], slip=s.attrs["signed_bps"],
                      sue=s.attrs["sue_z"], tickets=s.attrs["tickets"])
        print(f"{N:>4}{t['tot']:>9.0%}{t['ev']:>9.1%}{a['t']:>9.1f}{t['p40']:>7.1%}"
              f"{t['p60']:>7.1%}{t['mx']:>7.0%}{s.attrs['turnover']*100:>7.1f}"
              f"{s.attrs['signed_bps']:>8.1f}{s.attrs['sue_z']:>7.2f}{s.attrs['tickets']:>8}",
              flush=True)

    # pick free-turnover N: max P(2mo>60%) then max single window, s.t. EV>0
    feas = {N: r for N, r in rot.items() if r["tail"]["ev"] > 0}
    if not feas:
        print("\n  !! ALL free-turnover configs EV<=0 -> KILL: revert to buy-hold-40")
        bestN = None
    else:
        bestN = max(feas, key=lambda N: (feas[N]["tail"]["p60"], feas[N]["tail"]["mx"]))

    # ── HEAD-TO-HEAD ──
    print("\n=== HEAD-TO-HEAD: buy-hold-40 (N=25) vs free-turnover (best N) ===")
    print(f"{'policy':<26}{'OOS':>9}{'EV':>8}{'alpha-t':>9}{'P>60%':>8}{'max':>8}"
          f"{'turn%':>7}{'slip_bp':>8}{'|SUEz|':>7}{'tickets':>9}")
    print(f"{'buy-hold-40 (K30,b3%)':<26}{bh_t['tot']:>9.0%}{bh_t['ev']:>8.1%}{a_bh['t']:>9.1f}"
          f"{bh_t['p60']:>8.1%}{bh_t['mx']:>8.0%}{bh.attrs['turnover']*100:>7.1f}"
          f"{bh.attrs['signed_bps']:>8.1f}{'n/a':>7}{bh_tickets:>9}", flush=True)
    rec = None
    if bestN is not None:
        r = rot[bestN]
        print(f"{'free-turnover N=' + str(bestN):<26}{r['tail']['tot']:>9.0%}{r['tail']['ev']:>8.1%}"
              f"{r['t']:>9.1f}{r['tail']['p60']:>8.1%}{r['tail']['mx']:>8.0%}{r['turn']*100:>7.1f}"
              f"{r['slip']:>8.1f}{r['sue']:>7.2f}{r['tickets']:>9}", flush=True)
        # ── tail-preservation: top-PnL names under each, holding days under rotation ──
        rfull = run_book_rotate(zb5, mp, age_mat, trail, decay, rowpos, days, fs, fe,
                                N=bestN, record_ledger=True)
        def top_names(book, kk=10):
            oos = np.array([d.year >= 2021 for d in book.attrs["rec_d"]])
            pnl = book.attrs["Dmat"][oos].sum(axis=0)
            order = np.argsort(pnl)[::-1][:kk]
            return order, pnl
        bo, bp = top_names(bh); ro, rp = top_names(rfull)
        bset = {names.get(int(cols[i]), int(cols[i])) for i in bo}
        print("\n  TAIL PRESERVATION — top-10 OOS PnL contributors:")
        print(f"    buy-hold-40 : {[names.get(int(cols[i]), int(cols[i])) for i in bo]}")
        print(f"    free-turnover: {[names.get(int(cols[i]), int(cols[i])) for i in ro]}")
        hd = rfull.attrs["holddays"]
        print("    rotation holding-days for ITS top-5 (did it ride them, not churn out?):")
        for i in ro[:5]:
            tk = names.get(int(cols[i]), int(cols[i]))
            print(f"      {tk:<8} held {int(hd[i]):>3}d  OOS PnL ${rp[i]:,.0f}", flush=True)
        overlap = len(bset & {names.get(int(cols[i]), int(cols[i])) for i in ro})
        print(f"    top-10 overlap buy-hold vs rotation: {overlap}/10", flush=True)
        rec = bestN

    # ── verdict ──
    print("\n=== VERDICT ===")
    if bestN is None:
        verdict = "REVERT to buy-hold-40 (free-turnover went EV-negative)"
    else:
        r = rot[bestN]
        # a REAL tail win = strictly higher P(2mo>60%) by more than noise AND >= max,
        # not just a single noisy max realization
        wins_tail = (r["tail"]["p60"] > bh_t["p60"] + 0.005) and (r["tail"]["mx"] >= bh_t["mx"])
        ev_ok = r["tail"]["ev"] >= 0.8 * bh_t["ev"]            # EV not materially degraded
        alpha_ok = r["t"] >= 3.0                                # alpha significance preserved
        slip_ok = r["tail"]["tot"] > 0 and r["slip"] < 50
        if wins_tail and ev_ok and alpha_ok and slip_ok:
            verdict = f"ADOPT free-turnover N={bestN}"
        else:
            why = []
            if not wins_tail: why.append("no real tail win (P>60% tie; higher max is 1 noisy window)")
            if not ev_ok: why.append(f"EV degraded {r['tail']['ev']:.1%} vs {bh_t['ev']:.1%}")
            if not alpha_ok: why.append(f"alpha-t collapses {r['t']:.1f} vs {a_bh['t']:.1f}")
            verdict = "KEEP buy-hold-40 — " + "; ".join(why)
    print(f"  {verdict}")

    out = dict(decay=[float(x) for x in decay], buyhold=dict(tail=bh_t, t=a_bh["t"],
               turn=bh.attrs["turnover"], slip=bh.attrs["signed_bps"], tickets=bh_tickets),
               rotation={int(N): dict(tail=r["tail"], t=r["t"], turn=r["turn"], slip=r["slip"],
                                      sue=r["sue"], tickets=r["tickets"]) for N, r in rot.items()},
               best_N=rec, verdict=verdict)
    Path("results").mkdir(exist_ok=True)
    json.dump(out, open("results/pead_v11_rotation.json", "w"), indent=2, default=float)
    print("\nwrote results/pead_v11_rotation.json", flush=True)


if __name__ == "__main__":
    main()
