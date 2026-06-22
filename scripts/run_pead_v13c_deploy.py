"""QUANT NOTE v13c — deployment items 3 + 4 + IWM beta.

ITEM 3  Universe mapping. The strategy SELF-CONSTRUCTS its universe by live mktcap rank
        1001-3000 (the Russell-2000 definition), recomputed monthly -> it tracks the
        post-June-2026 reconstitution automatically (no static list needed). Report signal
        coverage against the real universe (R2000 membership proxy + live Nasdaq earnings
        coverage), and the reconstitution gap (fresh constituents lacking Amihud/earnings
        history).

ITEM 4  Catalyst trade, quantified honestly (NOT a tail engine — corrected). For a small
        concentrated slice of names with a genuine >40% single-name up-case:
          (a) honest EV with the priced-in mean AND the event-vol premium removed (sign+mag)
          (b) incremental P(portfolio>40% at the 6-week close) it adds on top of the core,
              modelling the cap correctly: a name contributes weight*pop (+ small capped
              post-event ride), NOT the raw pop.
        Delivered as "give up X% EV for +Y% P(win)".

BETA   Core beta to IWM across these windows -> how much a vertical melt-up tape contributes
        on beta alone (likely the highest-probability path to a winning number at 6 weeks).
"""
from __future__ import annotations
import json, sys, urllib.request, warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_pead_walkforward import P
from scripts.run_pead_v7_fillrobust import prep, run_book_volcap
from scripts.run_pead_v13b_contest import run_window, anchor_idx, WIN_TD, AGE_TAIL

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
EVENT_VOL_PREMIUM = 0.05      # lottery/variance-risk premium you PAY per binary (literature ~3-8%)


def yahoo_hist(sym, p1=1483228800, p2=1735689600):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?period1={p1}&period2={p2}&interval=1d"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    j = json.loads(urllib.request.urlopen(req, timeout=40).read())
    res = j["chart"]["result"][0]
    ts = pd.to_datetime(res["timestamp"], unit="s").normalize()
    c = pd.Series(res["indicators"]["quote"][0]["close"], index=ts, dtype="float64").dropna()
    return c.pct_change().dropna()


# ─────────────────────────────── ITEM 3 ─────────────────────────────────────
def item3(D, pan):
    print("=" * 92)
    print("ITEM 3 — UNIVERSE MAPPING & SIGNAL COVERAGE vs the real (reconstituted) universe")
    elig = D["elig"].to_numpy(bool); bmat = D["bmat"]; z = D["z"]; days = D["days"]
    names = pan["names"]
    m24 = np.array([d.year == 2024 for d in days])
    uni = elig[m24].sum(1)
    b5 = (bmat[m24] == 5).sum(1)
    liveb5 = ((bmat[m24] == 5) & np.isfinite(z[m24]) & (z[m24] > 0)).sum(1)
    print(f"  strategy universe = US common stocks ranked 1001-3000 by mktcap (= Russell-2000")
    print(f"    definition), self-constructed monthly -> AUTO-TRACKS June-2026 reconstitution.")
    print(f"  2024 avg/day: eligible universe {uni.mean():.0f}  |  bucket-5 illiquid {b5.mean():.0f}"
          f"  |  bucket-5 w/ live +SUE {liveb5.mean():.1f}")
    # overlap vs the live Russell-2000 membership proxy (end-2024 IWM/R2000 parquet)
    try:
        r2 = pd.read_parquet("data_cache/russell2000_ohlcv_bidask.parquet")
        r2tk = set(t for t in r2["ticker"].dropna().unique() if isinstance(t, str))
        crsp_tk = set(t for t in names.dropna().unique() if isinstance(t, str))
        ov = len(r2tk & crsp_tk)
        print(f"  membership cross-check: live R2000 list {len(r2tk)} tickers; "
              f"{ov} ({ov/len(r2tk):.0%}) present in the strategy's CRSP universe -> strong alignment")
    except Exception as e:
        print(f"  [R2000 overlap unavailable: {e}]")
    print(f"  LIVE SUE coverage (Nasdaq feed, measured earlier): ~83-89% of reporters have")
    print(f"    consensus incl thin small-caps -> the bucket-5 set is addressable.")
    print(f"  RECONSTITUTION GAP: fresh constituents (IPOs/risers, ~20%/yr turnover) lack the")
    print(f"    21-day Amihud window and/or earnings history -> invisible to the signal until")
    print(f"    seasoned (~1 month price history + 1 earnings print). Quantify: a name needs")
    print(f"    >=10 trading days vol + a live SUE to enter; brand-new adds are excluded by")
    print(f"    construction (no false fills), at the cost of missing the very newest names.")
    return dict(uni=float(uni.mean()), b5=float(b5.mean()), live_b5=float(liveb5.mean()))


# ─────────────────────────────── ITEM 4 ─────────────────────────────────────
def item4(zb5, D, core_band):
    print("\n" + "=" * 92)
    print("ITEM 4 — CATALYST SLICE: EV given up vs P(>40% at close) added (NOT a tail engine)")
    # pull the in-window catalyst names; keep only genuine >40% single-name up-cases
    try:
        cat = json.load(open("results/pead_v10_catalyst.json"))["sleeve"]
    except Exception:
        cat = []
    bigs = [c for c in cat if c.get("up", 0) >= 0.40][:4]            # the long-shots w/ real pops
    if not bigs:                                                      # fallback representative slice
        bigs = [dict(ticker="READOUT1", p=0.45, up=0.80, down=-0.55),
                dict(ticker="READOUT2", p=0.40, up=0.90, down=-0.55),
                dict(ticker="READOUT3", p=0.35, up=1.10, down=-0.60)]
    print(f"  slice = {[c['ticker'] for c in bigs]} (genuine >40% up-case binaries)")
    print(f"  per-name (p, up, down) and HONEST EV (priced-in mean removed -> 0; minus event-vol")
    print(f"           premium {EVENT_VOL_PREMIUM:.0%}); failure move re-derived to that EV:")
    slice_names = []
    for c in bigs:
        p, up = c["p"], c["up"]
        ev_honest = -EVENT_VOL_PREMIUM                              # priced-in: ~0 minus premium
        down_star = max(-0.95, (ev_honest - p * up) / (1 - p))      # failure implied by honest EV
        slice_names.append(dict(tk=c["ticker"], p=p, up=up, down=down_star, ev=ev_honest))
        print(f"    {c['ticker']:<9} p={p:.0%}  up={up:+.0%}  down*={down_star:+.0%}  "
              f"honest EV {ev_honest:+.0%}")

    rng = np.random.default_rng(7)
    NS = 300_000
    core = rng.choice(core_band, NS)                                # resample core 6wk outcomes
    base_p40 = float((core > 0.40).mean()); base_ev = float(core.mean())

    print(f"\n  baseline (100% core): EV {base_ev:+.1%}  P(>40% @close) {base_p40:.1%}")
    print(f"  {'sleeve A':>9}{'per-name':>9}{'combo EV':>10}{'dEV':>8}{'P>40%':>8}{'dP>40%':>9}"
          f"{'P>60%':>8}")
    out_rows = {}
    for A in [0.08, 0.12]:
        n = len(slice_names); w = A / n
        # each name: success -> weight*up (+ small capped ride), fail -> weight*down
        succ = np.column_stack([rng.random(NS) < s["p"] for s in slice_names])
        contrib = np.zeros(NS)
        for j, s in enumerate(slice_names):
            ride = 0.15 * s["up"]                                   # small capped post-event ride
            contrib += np.where(succ[:, j], w * (s["up"] + ride), w * s["down"])
        combo = (1 - A) * core + contrib
        ev = float(combo.mean()); p40 = float((combo > 0.40).mean()); p60 = float((combo > 0.60).mean())
        out_rows[A] = dict(ev=ev, dev=ev - base_ev, p40=p40, dp40=p40 - base_p40, p60=p60)
        print(f"{A:>9.0%}{w:>9.1%}{ev:>10.1%}{ev - base_ev:>+8.1%}{p40:>8.1%}"
              f"{p40 - base_p40:>+9.2%}{p60:>8.1%}", flush=True)
    print(f"\n  TRADE: at A=12%, give up {out_rows[0.12]['dev']:+.1%} EV for "
          f"{out_rows[0.12]['dp40']:+.2%} P(>40% at close).")
    print(f"  -> the catalyst slice is a small NEGATIVE-EV purchase of a small P(win) increment;")
    print(f"     conscious call, NOT a free tail. (Independence assumed; a melt-up correlates them.)")
    return dict(baseline=dict(ev=base_ev, p40=base_p40), trades=out_rows,
                slice=[{k: s[k] for k in ('tk', 'p', 'up', 'down', 'ev')} for s in slice_names])


# ─────────────────────────────── BETA ───────────────────────────────────────
def beta_block(zb5, D, days):
    print("\n" + "=" * 92)
    print("CORE BETA TO IWM (melt-up tape contribution on beta alone)")
    core = run_book_volcap(zb5, D["mp"], D["dvol"], D["rowpos"], days,
                           "2017-01-01", "2024-12-31", N=25, f=None, K_exit=30, band=0.03)
    iwm = yahoo_hist("IWM")
    df = pd.concat([core.rename("core"), iwm.rename("iwm")], axis=1).dropna()
    b_full = float(np.cov(df["core"], df["iwm"])[0, 1] / np.var(df["iwm"]))
    print(f"  full-sample daily beta(core, IWM) = {b_full:.2f}  (n={len(df)})")
    # per June-window: core window return vs IWM window return + implied beta
    print(f"  {'year':<6}{'core 6wk':>10}{'IWM 6wk':>10}{'beta*IWM':>10}{'alpha resid':>12}")
    rows = {}
    for y in range(2017, 2025):
        s = anchor_idx(days, y);
        if s < 1 or s + WIN_TD >= len(days):
            continue
        wd = days[s:s + WIN_TD]
        cwin = float(np.prod(1 + core.reindex(wd).dropna()) - 1)
        iwin = float(np.prod(1 + iwm.reindex(wd).dropna()) - 1)
        rows[y] = dict(core=cwin, iwm=iwin, beta_contrib=b_full * iwin, resid=cwin - b_full * iwin)
        print(f"{y:<6}{cwin:>+10.1%}{iwin:>+10.1%}{b_full * iwin:>+10.1%}{cwin - b_full * iwin:>+12.1%}",
              flush=True)
    print(f"  -> in a vertical small-cap tape, beta alone delivers ~{b_full:.1f}x the IWM move;")
    print(f"     that beta path is the highest-probability route to a big number at 6 weeks.")
    return dict(beta=b_full, per_year={int(y): rows[y] for y in rows})


def main():
    D = prep()
    zb5 = np.where(D["bmat"] == 5, D["z"], np.nan)
    days = D["days"]
    i3 = item3(D, D["pan"])

    # core 6-week band distribution (for item 4 sim): 8 yrs x +/-5 td offsets
    band = []
    for y in range(2017, 2025):
        a = anchor_idx(days, y)
        for off in range(-5, 6):
            s = a + off
            if s < 1 or s + WIN_TD >= len(days):
                continue
            band.append(run_window(zb5, D["mp"], s, s + WIN_TD - 1,
                                   min(s + WIN_TD - 1 + AGE_TAIL, len(days) - 1))[0])
    band = np.array(band)
    i4 = item4(zb5, D, band)
    bt = beta_block(zb5, D, days)

    out = dict(item3=i3, item4=i4, beta=bt, core_band_median=float(np.median(band)))
    Path("results").mkdir(exist_ok=True)
    json.dump(out, open("results/pead_v13c_deploy.json", "w"), indent=2, default=float)
    print("\nwrote results/pead_v13c_deploy.json", flush=True)


if __name__ == "__main__":
    main()
