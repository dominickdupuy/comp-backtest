"""QUANT NOTE v4 Workstream A — corrected fill model + alpha re-test.

v3 charged a FLAT adverse bps on all traded notional. That is wrong for a slow
signal: the 15-20 min decision->fill drift is mostly SYMMETRIC variance (zero EV),
with only a small SIGNED component equal to the signal's short-horizon
continuation. Here we:

  A.1 decompose the per-trade fill drift into signed (EV) vs symmetric (variance),
  A.2 re-charge ONLY the measured signed component, with Δ/390 scaling, and
  A.4 report the corrected OOS return + Carhart factor attribution on the
      corrected book (≈ gross, since the signed cost is tiny) to test whether
      v3's "alpha ≈ 0" was itself an artifact of the over-pessimistic haircut.

The signed cost per trade is estimated as  Δw_i · r_{fill,i} · (Δ/390): this
charges only the part of the fill-window move correlated with the trade
direction (continuation); uncorrelated noise averages to zero across trades.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_pead_walkforward import (load_panels, pit_universe, pead_active_panel,
                                          zscore_panel, stats, P)
from scripts.run_pead_v3 import add_delist_returns, build_target, _ew
from src.report import metrics
import statsmodels.api as sm

CAP, TGT = P["CAP"], P["TGT"]
DELTA_MIN = 17.5
DFRAC = DELTA_MIN / 390.0          # fraction of a trading day the fill lags


def run_book_fill(z_mat, ret_mat, rowpos, days, start, end, *, N=12, K_exit=12,
                  band=0.0, fill_mode="none", flat_bps=0.0, sign=1.0):
    """fill_mode: 'none' (zero-drift), 'flat' (v3: flat_bps on all traded notional),
    'signed' (v4: only the trade-direction-correlated continuation, Δ/390 scaled).
    Returns daily net Series; .attrs has turnover and measured signed/symmetric bps."""
    win = days[(days >= start) & (days <= end)]
    if len(win) < 5:
        return pd.Series(dtype=float)
    first = days[days < win[0]]
    d0 = first[-1] if len(first) else win[0]
    dec = days[(days >= d0) & (days <= win[-1])]
    di = [rowpos[d] for d in dec]
    ncol = z_mat.shape[1]
    pos = np.zeros(ncol); cash = V = P["capital"]
    prev_w = np.zeros(ncol)
    rec_d, rec_v, turns = [], [], []
    signed_acc, sym_acc, traded_acc = 0.0, 0.0, 0.0
    for k in range(len(di) - 1):
        zr = sign * z_mat[di[k]]
        cur_w = pos / V if V > 0 else pos * 0.0
        held = np.where(pos > 0)[0]
        idx, w = build_target(zr, held, cur_w, N, K_exit, band)
        new_w = np.zeros(ncol); new_w[idx] = w
        dw = new_w - prev_w
        r = np.nan_to_num(ret_mat[di[k + 1]])
        ow_turn = np.abs(dw).sum() / 2
        turns.append(ow_turn)
        # measured signed component: only the part of the fill move correlated
        # with trade direction, scaled to the Δ-minute window.
        signed = float((dw * r).sum()) * DFRAC
        signed_acc += signed
        traded_acc += np.abs(dw).sum()
        # symmetric magnitude (variance only, NOT subtracted from EV) for logging
        sym_acc += float((np.abs(dw) * np.abs(r)).sum()) * DFRAC
        if fill_mode == "signed":
            haircut = signed
        elif fill_mode == "flat":
            haircut = np.abs(dw).sum() * (flat_bps / 1e4)
        else:
            haircut = 0.0
        new_pos = np.zeros(ncol); new_pos[idx] = w * V
        pos, cash, prev_w = new_pos, V - new_pos.sum(), new_w
        pos = pos * (1.0 + r)
        Vn = pos.sum() + cash
        rec_d.append(dec[k + 1]); rec_v.append(Vn / V - 1.0 - haircut); V = Vn
    s = pd.Series(rec_v, index=pd.to_datetime(rec_d)).sort_index()
    eff_signed_bps = 1e4 * signed_acc / traded_acc if traded_acc else 0.0
    eff_sym_bps = 1e4 * sym_acc / traded_acc if traded_acc else 0.0
    s.attrs.update(turnover=float(np.mean(turns)),
                   signed_bps=float(eff_signed_bps), sym_bps=float(eff_sym_bps))
    return s


def span(s, y0, y1):
    return s[(s.index.year >= y0) & (s.index.year <= y1)]


def factor_alpha(book, ff, y0, y1):
    sub = book[(book.index.year >= y0) & (book.index.year <= y1)]
    df = pd.concat([sub.rename("r"), ff], axis=1).dropna()
    y = df["r"] - df["RF"]
    X = sm.add_constant(df[["MktRF", "SMB", "HML", "Mom"]])
    m = sm.OLS(y, X).fit()
    return dict(alpha_ann=m.params["const"] * 252, t=m.tvalues["const"],
                bMkt=m.params["MktRF"], bSMB=m.params["SMB"],
                bHML=m.params["HML"], bMom=m.params["Mom"], r2=m.rsquared)


def main():
    pan = load_panels()
    days = pan["close"].index; cols = pan["close"].columns
    rowpos = {d: i for i, d in enumerate(days)}
    elig = pit_universe(pan, P)
    ret_df = pan["ret"].reindex(columns=cols).astype("float64")
    ret_df, n_dl, _ = add_delist_returns(ret_df, set(int(c) for c in cols))
    ret_mat = ret_df.to_numpy(float, na_value=np.nan)
    z_mat = zscore_panel(pead_active_panel(pan, P), elig).to_numpy(float, na_value=np.nan)
    ff = pd.read_parquet("data_cache/ff_factors_daily.parquet"); ff.index = pd.to_datetime(ff.index)
    fs, fe = "2017-01-01", "2024-12-31"
    ob = stats(_ew(elig, ret_mat, rowpos, days, "2021-01-01", fe), P)["tot"]

    print("=== A. CORRECTED FILL MODEL (signed EV vs symmetric variance) ===")
    print(f"delisting events folded in: {n_dl};  OOS EW benchmark = {ob:.0%}\n")

    # baseline top-12 book (v3's high-turnover config) under 3 fill models
    print("v3 config (top-12, no hysteresis):")
    print(f"{'fill model':<26}{'turn/d':>8}{'OOS tot':>10}{'OOS shrp':>10}")
    configs = [("none (zero-drift)", dict(fill_mode="none")),
               ("v3 FLAT 25bps", dict(fill_mode="flat", flat_bps=25)),
               ("v4 SIGNED (measured)", dict(fill_mode="signed"))]
    sgn = {}
    for label, kw in configs:
        s = run_book_fill(z_mat, ret_mat, rowpos, days, fs, fe, N=12, K_exit=12, **kw)
        oss = stats(span(s, 2021, 2024), P)
        sgn[label] = s
        extra = ""
        if kw.get("fill_mode") == "signed":
            extra = f"   [measured signed {s.attrs['signed_bps']:.1f} bps/side vs " \
                    f"symmetric {s.attrs['sym_bps']:.0f} bps (variance only)]"
        print(f"{label:<26}{s.attrs['turnover']*100:>7.1f}%{oss['tot']:>10.0%}"
              f"{oss['sharpe']:>10.2f}{extra}", flush=True)

    # low-turnover hysteresis config (v3 chosen) under signed model
    print("\nlow-turnover config (K_exit=30, band=0.03):")
    print(f"{'fill model':<26}{'turn/d':>8}{'OOS tot':>10}{'OOS shrp':>10}")
    for label, kw in configs:
        s = run_book_fill(z_mat, ret_mat, rowpos, days, fs, fe, N=12, K_exit=30, band=0.03, **kw)
        oss = stats(span(s, 2021, 2024), P)
        if kw.get("fill_mode") == "signed":
            sgn["lowturn_signed"] = s
        print(f"{label:<26}{s.attrs['turnover']*100:>7.1f}%{oss['tot']:>10.0%}"
              f"{oss['sharpe']:>10.2f}", flush=True)

    # ---- alpha re-test on the CORRECTED book (settles v3's "0 alpha") -------
    print("\n=== ALPHA RE-TEST: Carhart 4-factor on the corrected (signed-fill) book ===")
    book = sgn["v4 SIGNED (measured)"]
    for label, (y0, y1) in [("IS  2017-2020", (2017, 2020)), ("OOS 2021-2024", (2021, 2024))]:
        a = factor_alpha(book, ff, y0, y1)
        sig = "SIGNIFICANT" if abs(a["t"]) >= 2 else "not sig."
        print(f"  {label}: alpha {a['alpha_ann']:>6.0%}/yr  t={a['t']:>5.1f} ({sig})  "
              f"R2={a['r2']:.2f}  betas Mkt {a['bMkt']:.2f} SMB {a['bSMB']:.2f} "
              f"HML {a['bHML']:.2f} Mom {a['bMom']:.2f}", flush=True)
    full = stats(book, P)
    print(f"\n  corrected book full-period: total {full['tot']:.0%}  sharpe {full['sharpe']:.2f}  "
          f"vs v3's mislabeled 25bps OOS +4%", flush=True)

    out = dict(n_delist=n_dl, oos_bench=ob,
               signed_bps=float(book.attrs["signed_bps"]),
               sym_bps=float(book.attrs["sym_bps"]),
               oos=dict((lab, stats(span(s, 2021, 2024), P)["tot"]) for lab, s in sgn.items()),
               alpha_is=factor_alpha(book, ff, 2017, 2020),
               alpha_oos=factor_alpha(book, ff, 2021, 2024))
    Path("results").mkdir(exist_ok=True)
    json.dump(out, open("results/pead_v4_fillmodel.json", "w"), indent=2, default=float)
    print("\nwrote results/pead_v4_fillmodel.json", flush=True)


if __name__ == "__main__":
    main()
