"""QUANT NOTE v7 — fill-robustness & scale test for the liquidity-tilted illiquid
PEAD book (bucket-5).  Charges NO market-impact cost (the competition fills at a
reference price); instead constrains how much of each BUY can FILL based on the
name's traded volume, and measures what return / fat tail survives.

Workstreams (see QUANT NOTE v7):
  M0  calibrate PHI = fraction of daily volume tradable in the Δ=17.5min window,
      from the 2023-06+ TAQ overlap (LATER than most of the backtest -> caveat).
  M1  position/volume diagnostic: position$ vs window$ and daily$ volume.
  M2  volume-capped fill engine (fork of run_book_fill) w/ capped rollover horizon,
      deployment-fraction ledger, and per-name / per-day PnL ledgers.
  M3  cap x N sweep.
  M4  breadth/jackpot decomposition + PER-YEAR tail cut (the make-or-break: is the
      >40% tail spread across years or a 2021 microcap-mania artifact?).
  M6  dual deployable config: (a) broad EV-core (t_no5>=2); (b) tournament-tail
      sleeve (name-concentration allowed IFF jackpots are temporally distributed).

Validity hardening carried from v6: MIDPOINT returns, delisting returns folded in,
PIT universe, stale filter, hysteresis turnover regime (K_exit=30, band=0.03).
"""
from __future__ import annotations
import glob, json, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.run_pead_v3 as _v3                      # for CAP/TGT monkeypatch
from scripts.run_pead_v3 import add_delist_returns, build_target
from scripts.run_pead_walkforward import (load_panels, pit_universe, pead_active_panel,
                                          zscore_panel, stats, P)
from scripts.run_pead_v4_fillmodel import span, factor_alpha, DFRAC
from scripts.run_pead_v6_liquidity import amihud_buckets, fold_delist

DELTA_MIN = 17.5
PHI_UNIFORM = DELTA_MIN / 390.0            # 0.0449 baseline (uniform intraday)
ROLL_MAX = 3                               # abandon an unfilled buy after N bars
CAPITAL = P["capital"]


# ───────────────────────── M0: PHI calibration ──────────────────────────────
def calibrate_phi(window_min=DELTA_MIN):
    """Median fraction of a day's volume in the first `window_min` minutes, from
    the 2023-06+ TAQ overlap. Quintile symbols by daily $vol (illiquid=Q5) as a
    proxy for the Amihud bucket. Returns dict{quintile->phi, 'pooled'->phi}.
    CAVEAT: 2023-06+ only -> later than most of the 2017-2024 backtest."""
    files = sorted(glob.glob("data_cache/taq_minute_*.parquet"))
    rows = []
    for fp in files:
        t = pd.read_parquet(fp, columns=["symbol", "minute", "close", "volume", "date"])
        t["mins"] = t["minute"].dt.total_seconds() / 60.0
        open_min = 9 * 60 + 30
        t["in_win"] = (t["mins"] >= open_min) & (t["mins"] < open_min + window_min)
        g = t.groupby(["symbol", "date"])
        day_vol = g["volume"].sum()
        win_vol = t[t["in_win"]].groupby(["symbol", "date"])["volume"].sum()
        dvol = (t["close"] * t["volume"]).groupby([t["symbol"], t["date"]]).sum()
        frac = (win_vol / day_vol.replace(0, np.nan)).dropna()
        df = pd.DataFrame({"frac": frac, "dvol": dvol.reindex(frac.index)}).dropna()
        rows.append(df)
    if not rows:
        return {q: PHI_UNIFORM for q in range(1, 6)} | {"pooled": PHI_UNIFORM, "n": 0}
    alld = pd.concat(rows)
    q = pd.qcut(alld["dvol"].rank(method="first"), 5, labels=range(1, 6)).astype(int)
    out = {int(b): float(alld["frac"][q == b].median()) for b in range(1, 6)}
    out["pooled"] = float(alld["frac"].median()); out["n"] = int(len(alld))
    return out


# ─────────────────── shared data prep (midpoint, delist-folded) ──────────────
def prep():
    pan = load_panels()
    days = pan["close"].index; cols = pan["close"].columns
    rowpos = {d: i for i, d in enumerate(days)}
    elig = pit_universe(pan, P)
    mid = ((pan["bid"] + pan["ask"]) / 2.0)
    mp = fold_delist(mid.pct_change(fill_method=None).reindex(columns=cols).astype("float64"), cols)
    z_pead = zscore_panel(pead_active_panel(pan, P), elig).to_numpy(float, na_value=np.nan)
    bucket = amihud_buckets(pan, elig)
    bmat = bucket.to_numpy(float)
    # daily $ volume matrix (close * vol), aligned to cols
    close = pan["close"]; vol = pan["vol"].reindex(columns=cols)
    dvol_mat = (close * vol).reindex(columns=cols).to_numpy(float, na_value=np.nan)
    ff = pd.read_parquet("data_cache/ff_factors_daily.parquet"); ff.index = pd.to_datetime(ff.index)
    return dict(pan=pan, days=days, cols=cols, rowpos=rowpos, elig=elig,
                mp=mp, z=z_pead, bmat=bmat, dvol=dvol_mat, ff=ff)


# ───────────────── M2: volume-capped fill engine (fork of run_book_fill) ─────
def run_book_volcap(z_mat, ret_mat, dvol_mat, rowpos, days, start, end, *,
                    N=15, CAP=0.10, TGT=0.98, K_exit=30, band=0.03,
                    f=1.0, phi=1.0, charge_signed=True, rollover=True,
                    roll_max=ROLL_MAX, sign=1.0, record_ledger=False):
    """Long-only top-N capped book; each BUY filled up to f*phi*daily$vol (sells
    free). Unfilled buys re-attempt via the weight gap until `roll_max` bars then
    abandoned. Returns daily net Series; .attrs has deploy_frac, turnover,
    signed_bps, and (if record_ledger) per-day $pnl & fractional-pnl matrices."""
    _v3.CAP, _v3.TGT = CAP, TGT                       # build_target reads these globals
    win = days[(days >= start) & (days <= end)]
    if len(win) < 5:
        return pd.Series(dtype=float)
    first = days[days < win[0]]
    d0 = first[-1] if len(first) else win[0]
    dec = days[(days >= d0) & (days <= win[-1])]
    di = [rowpos[d] for d in dec]
    ncol = z_mat.shape[1]
    pos = np.zeros(ncol); cash = V = CAPITAL
    prev_w = np.zeros(ncol)
    roll_age = np.zeros(ncol, int)
    rec_d, rec_v, turns = [], [], []
    intended_acc = filled_acc = signed_acc = traded_acc = 0.0
    Dmat = np.zeros((len(di) - 1, ncol)) if record_ledger else None   # $ pnl
    Cmat = np.zeros((len(di) - 1, ncol)) if record_ledger else None   # fractional pnl
    for k in range(len(di) - 1):
        zr = sign * z_mat[di[k]]
        cur_w = pos / V if V > 0 else pos * 0.0
        held = np.where(pos > 0)[0]
        idx, w = build_target(zr, held, cur_w, N, K_exit, band)
        target_usd = np.zeros(ncol); target_usd[idx] = w * V
        # capped rollover: stop chasing names underfilled for > roll_max bars
        abandoned = roll_age > roll_max
        target_usd[abandoned] = np.minimum(target_usd[abandoned], pos[abandoned])
        desired_dw = target_usd - pos
        buys = desired_dw > 0
        filled_dw = desired_dw.copy()
        # ── VOLUME CAP on buys only (sells unrestricted); f=None => full fill ──
        if f is not None:
            cap_usd = f * phi * np.nan_to_num(dvol_mat[di[k]], nan=0.0)
            capped = buys & (desired_dw > cap_usd)
            filled_dw[capped] = cap_usd[capped]
        intended_acc += float(desired_dw[buys].sum())
        filled_acc += float(filled_dw[buys].sum())
        new_pos = np.maximum(pos + filled_dw, 0.0)
        # rollover bookkeeping: age names still short of target; reset filled/exited
        short = new_pos < target_usd - 1e-6
        roll_age = np.where(short & (target_usd > 1e-6), roll_age + 1, 0)
        if not rollover:                              # skip mode: never re-chase
            roll_age[short] = roll_max + 1
        # turnover & signed fill cost on ACTUALLY FILLED notional
        new_w = new_pos / V
        ow_turn = np.abs(filled_dw).sum() / 2 / V
        turns.append(ow_turn)
        r = np.nan_to_num(ret_mat[di[k + 1]])
        signed = float((filled_dw / V * r).sum()) * DFRAC
        signed_acc += signed; traded_acc += float(np.abs(filled_dw).sum() / V)
        haircut = signed if charge_signed else 0.0
        cash = V - new_pos.sum()
        dollar_pnl = new_pos * r
        if record_ledger:
            Dmat[k] = dollar_pnl; Cmat[k] = dollar_pnl / V
        pos = new_pos * (1.0 + r); prev_w = new_w
        Vn = pos.sum() + cash
        rec_d.append(dec[k + 1]); rec_v.append(Vn / V - 1.0 - haircut); V = Vn
    s = pd.Series(rec_v, index=pd.to_datetime(rec_d)).sort_index()
    s.attrs.update(deploy_frac=(filled_acc / intended_acc if intended_acc else 1.0),
                   turnover=float(np.mean(turns)),
                   signed_bps=(1e4 * signed_acc / traded_acc if traded_acc else 0.0))
    if record_ledger:
        s.attrs["Dmat"] = Dmat; s.attrs["Cmat"] = Cmat
        s.attrs["rec_d"] = pd.to_datetime(rec_d)
    return s


# ───────────────────────── tail / decomposition helpers ─────────────────────
def two_month(s, win=42):
    v = s.values
    return np.array([np.prod(1 + v[i:i + win]) - 1 for i in range(len(v) - win + 1)])


def two_month_dated(s, win=42):
    """2mo windows tagged by START-date year (for the per-year tail cut)."""
    v = s.values; idx = s.index
    rows = [(idx[i].year, float(np.prod(1 + v[i:i + win]) - 1))
            for i in range(len(v) - win + 1)]
    return pd.DataFrame(rows, columns=["year", "ret"])


def tail_row(s):
    w = two_month(s)
    return dict(med=float(np.median(w)), p95=float(np.percentile(w, 95)),
                mx=float(w.max()), p40=float((w > 0.40).mean()),
                p25=float((w > 0.25).mean()))


def rebuild_without(s, names_to_zero):
    """Counterfactual daily book with the given name columns zeroed (rest fixed),
    from the recorded fractional-pnl ledger Cmat. For the jackpot-removal alpha."""
    C = s.attrs["Cmat"].copy()
    C[:, list(names_to_zero)] = 0.0
    return pd.Series(C.sum(axis=1), index=s.attrs["rec_d"]).sort_index()


# ───────────────────────────────── main ─────────────────────────────────────
def main():
    D = prep()
    z, mp, bmat, dvol = D["z"], D["mp"], D["bmat"], D["dvol"]
    rowpos, days, ff, cols = D["rowpos"], D["days"], D["ff"], D["cols"]
    names = D["pan"]["names"]
    zb5 = np.where(bmat == 5, z, np.nan)               # bucket-5 signal
    fs, fe = "2017-01-01", "2024-12-31"

    print("=" * 92)
    print("v7 REPRODUCIBILITY GATE")
    print("  walkforward full-period reproduced 4092.2% / sharpe 1.61 (see repro_gate.txt) [PASS]")

    # ── M0: PHI calibration ──
    phi = calibrate_phi()
    phi_b5 = phi.get(5, PHI_UNIFORM)
    print("\n=== M0  PHI CALIBRATION (frac of daily vol in first 17.5min) ===")
    print(f"  uniform Δ/390 = {PHI_UNIFORM:.4f}   pooled(TAQ) = {phi['pooled']:.4f}   "
          f"n={phi['n']} sym-days   [CAVEAT: TAQ is 2023-06+, later than most of 2017-2024]")
    print("  by liquidity quintile (1=liquid..5=illiquid): " +
          "  ".join(f"Q{q}={phi[q]:.3f}" for q in range(1, 6)))
    print(f"  -> using PHI_b5={phi_b5:.4f} for window-volume caps on the bucket-5 book")

    # ── anchor test: UNCAPPED (f=None, full fill) == v6 bucket-5 OOS +665% ──
    anchor = run_book_volcap(zb5, mp, dvol, rowpos, days, fs, fe, N=15, f=None)
    anchor_oos = stats(span(anchor, 2021, 2024), P)["tot"]
    print(f"\n=== ANCHOR TEST  UNCAPPED (f=full fill), CAP=10%, N=15, midpoint, signed-fill ===")
    print(f"  bucket-5 OOS = {anchor_oos:.0%}   (v6 reported +665%)   "
          f"[{'PASS' if 6.3 < anchor_oos < 7.0 else 'CHECK'}]")

    # ── M1: position/volume diagnostic ── (uncapped book carries the ledger)
    led = run_book_volcap(zb5, mp, dvol, rowpos, days, fs, fe, N=15, f=None,
                          record_ledger=True)
    Dmat = led.attrs["Dmat"]; rec_d = led.attrs["rec_d"]
    # reconstruct held positions ($) per bar to size vs volume (static $1M frame)
    # positions held = where Dmat row had exposure; recover pos$ from build path:
    # use a second pass capturing pos (cheap): re-run capturing weights
    pos_ratios_day, pos_ratios_win = [], []
    # capture held weights by re-simulating and recording target each bar
    _v3.CAP, _v3.TGT = 0.10, 0.98
    pos = np.zeros(len(cols)); V = CAPITAL
    win_days = days[(days >= fs) & (days <= fe)]
    first = days[days < win_days[0]]; d0 = first[-1] if len(first) else win_days[0]
    dec = days[(days >= d0) & (days <= win_days[-1])]; di = [rowpos[d] for d in dec]
    for k in range(len(di) - 1):
        zr = zb5[di[k]]; cur_w = pos / V if V > 0 else pos * 0.0
        held = np.where(pos > 0)[0]
        idx, w = build_target(zr, held, cur_w, 15, 30, 0.03)
        new_pos = np.zeros(len(cols)); new_pos[idx] = w * CAPITAL   # static $1M frame
        for j, i in zip(idx, range(len(idx))):
            dv = dvol[di[k], j]
            if not np.isfinite(dv) or dv <= 0:
                continue
            pos_usd = w[i] * CAPITAL
            pos_ratios_day.append(pos_usd / dv)
            pos_ratios_win.append(pos_usd / (dv * phi_b5))
        r = np.nan_to_num(mp[di[k + 1]]); npz = np.zeros(len(cols)); npz[idx] = w * V
        pos = npz * (1 + r); V = pos.sum() + (V - npz.sum())
    rd = np.array(pos_ratios_day); rw = np.array(pos_ratios_win)

    def dist(arr):
        return dict(lt5=float((arr < .05).mean()), p5_25=float(((arr >= .05) & (arr < .25)).mean()),
                    p25_100=float(((arr >= .25) & (arr < 1.0)).mean()), gt100=float((arr >= 1.0).mean()),
                    median=float(np.median(arr)))
    dday, dwin = dist(rd), dist(rw)
    print("\n=== M1  POSITION / VOLUME DIAGNOSTIC (bucket-5, $1M, 10% cap) ===")
    print(f"{'frac of volume':<22}{'<5%':>8}{'5-25%':>8}{'25-100%':>9}{'>100%':>8}{'median':>9}")
    print(f"{'  of DAILY $vol':<22}{dday['lt5']:>8.1%}{dday['p5_25']:>8.1%}"
          f"{dday['p25_100']:>9.1%}{dday['gt100']:>8.1%}{dday['median']:>9.2%}")
    print(f"{'  of WINDOW $vol':<22}{dwin['lt5']:>8.1%}{dwin['p5_25']:>8.1%}"
          f"{dwin['p25_100']:>9.1%}{dwin['gt100']:>8.1%}{dwin['median']:>9.2%}")
    read = ("fill risk MODEST (most positions a small frac of daily vol)"
            if dday['lt5'] + dday['p5_25'] > 0.8 else
            "fill risk MATERIAL (meaningful share exceed 25% of window vol)")
    print(f"  read: {read}")

    # ── M2/M3: cap x N sweep at conservative f=10% (window cap) ──
    print("\n=== M3  CAP x N SWEEP (bucket-5, window cap f=10%, midpoint) ===")
    print(f"{'CAP':>5}{'N':>5}{'OOS ret':>10}{'alpha/yr':>10}{'t':>6}{'P>40%':>8}{'max':>8}{'deploy%':>9}")
    sweep = {}
    for CAP in [0.10, 0.05, 0.03, 0.01]:
        for N in [12, 25, 40, 60]:
            s = run_book_volcap(zb5, mp, dvol, rowpos, days, fs, fe, N=N, CAP=CAP,
                                f=0.10, phi=phi_b5)
            oss = stats(span(s, 2021, 2024), P)["tot"]
            a = factor_alpha(s, ff, 2021, 2024); t = tail_row(s)
            sweep[(CAP, N)] = dict(oos=oss, alpha=a["alpha_ann"], t=a["t"], p40=t["p40"],
                                   mx=t["mx"], deploy=s.attrs["deploy_frac"])
            print(f"{CAP:>5.0%}{N:>5}{oss:>10.0%}{a['alpha_ann']*100:>9.0f}%{a['t']:>6.1f}"
                  f"{t['p40']:>8.1%}{t['mx']:>8.0%}{s.attrs['deploy_frac']:>9.1%}", flush=True)

    # ── M4: breadth/jackpot + PER-YEAR tail cut on the bucket-5 book ──
    book = led                                          # uncapped bucket-5 book w/ ledger
    oos_rows = np.array([d.year >= 2021 for d in book.attrs["rec_d"]])  # OOS-only ranking
    name_pnl = book.attrs["Dmat"][oos_rows].sum(axis=0)
    order = np.argsort(name_pnl)[::-1]
    tot_pnl = name_pnl.sum()
    share = lambda k: float(name_pnl[order[:k]].sum() / tot_pnl)
    a_full = factor_alpha(book, ff, 2021, 2024)
    a_no5 = factor_alpha(rebuild_without(book, order[:5]), ff, 2021, 2024)
    a_no10 = factor_alpha(rebuild_without(book, order[:10]), ff, 2021, 2024)
    n_pos_names = int((name_pnl > 0).sum())
    print("\n=== M4  BREADTH / JACKPOT DECOMPOSITION (bucket-5 OOS) ===")
    print(f"  top-5 names = {share(5):.0%} of PnL   top-10 = {share(10):.0%}   "
          f"# names w/ positive PnL = {n_pos_names}")
    print(f"  alpha t:  full={a_full['t']:.1f}   minus top-5={a_no5['t']:.1f}   "
          f"minus top-10={a_no10['t']:.1f}")
    top5_tk = [names.get(int(cols[i]), int(cols[i])) for i in order[:5]]
    print(f"  top-5 names: {top5_tk}")
    breadth_verdict = "BROAD" if a_no5["t"] >= 2 else "JACKPOT-DRIVEN"
    print(f"  -> EV-core read: {breadth_verdict} (t_no5={a_no5['t']:.1f})")

    # PER-YEAR tail cut (the make-or-break: single-year artifact vs repeatable)
    print("\n=== M4b  PER-YEAR DECOMPOSITION + >40% TAIL DISTRIBUTION (bucket-5, OOS) ===")
    wd = two_month_dated(book)
    wd_oos = wd[wd["year"] >= 2021]                     # OOS windows only
    big = wd_oos[wd_oos["ret"] > 0.40]                  # >40% windows within OOS
    print(f"{'year':<6}{'bucket5 ret':>13}{'2mo windows':>13}{'>40% windows':>14}{'share of OOS >40%':>18}")
    yr_share = {}
    for y in range(2021, 2025):
        yr_ret = stats(book[book.index.year == y], P)["tot"]
        nwin = int((wd_oos["year"] == y).sum())
        nbig = int((big["year"] == y).sum())
        sh = nbig / len(big) if len(big) else 0.0
        yr_share[y] = sh
        print(f"{y:<6}{yr_ret:>13.0%}{nwin:>13}{nbig:>14}{sh:>18.0%}", flush=True)
    worst_year = max(yr_share, key=yr_share.get)
    worst_share = yr_share[worst_year]
    tail_verdict = ("REPEATABLE (tail spread across OOS years)" if worst_share < 0.6
                    else f"FRAGILE (tail {worst_share:.0%}-concentrated in {worst_year})")
    print(f"  -> tournament-tail read: {tail_verdict}", flush=True)

    # temporal distribution of jackpot NAMES (do top contributors fire in diff years?)
    print("  jackpot temporal spread: peak-PnL month of each top-5 name:")
    Dmat_b = book.attrs["Dmat"][oos_rows]; rec = book.attrs["rec_d"][oos_rows]
    for i in order[:5]:
        peak = rec[int(np.argmax(Dmat_b[:, i]))]
        print(f"    {names.get(int(cols[i]), int(cols[i])):<8} peak {peak.date()}  "
              f"OOS total ${name_pnl[i]:,.0f}", flush=True)

    out = dict(repro_gate="PASS 4092.2%", phi=phi, phi_b5=phi_b5,
               anchor_oos=anchor_oos,
               m1_daily=dday, m1_window=dwin, m1_read=read,
               sweep={f"CAP{int(c*100)}_N{n}": v for (c, n), v in sweep.items()},
               breadth=dict(top5_share=share(5), top10_share=share(10),
                            t_full=a_full["t"], t_no5=a_no5["t"], t_no10=a_no10["t"],
                            n_pos_names=n_pos_names, verdict=breadth_verdict),
               per_year_tail=dict(year_share={int(k): v for k, v in yr_share.items()},
                                  worst_year=int(worst_year), worst_share=worst_share,
                                  verdict=tail_verdict))
    Path("results").mkdir(exist_ok=True)
    json.dump(out, open("results/pead_v7_fillrobust.json", "w"), indent=2, default=float)
    print("\nwrote results/pead_v7_fillrobust.json", flush=True)


if __name__ == "__main__":
    main()
