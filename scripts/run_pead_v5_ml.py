"""QUANT NOTE v5 — gradient-boosted cross-sectional signal search vs PEAD baseline.

ML as a cross-sectional factor combiner on SLOW, delay-robust features. Two
targets: T1 forward-40d return (EV), T2 P(forward-40d > +25%) (right tail).
Models: ElasticNet, LightGBM, XGBoost, CatBoost, RandomForest, ensemble.
Headline split: train 2017-2020 (purged/embargoed), test OOS 2021-2024. Every
model runs the SAME gauntlet PEAD passed (signed 1.3bps fill, Carhart alpha,
2-month tail) and is kept ONLY if it beats the PEAD baseline OOS.
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_pead_walkforward import load_panels, pit_universe, pead_active_panel, zscore_panel, stats, P
from scripts.run_pead_v3 import add_delist_returns, _ew
from scripts.run_pead_v4_fillmodel import run_book_fill, span, factor_alpha
from scipy.stats import spearmanr

H = 40  # forward horizon (matches validated PEAD hold)


# ---------------- feature panel (wide, PIT, slow) ---------------------------
def _comp_panel(comp, col, index, columns, lag_days=150):
    out = pd.DataFrame(np.nan, index=index, columns=columns)
    for permno, g in comp.sort_values("datadate").groupby("permno"):
        if permno not in out.columns:
            continue
        s = pd.Series(np.nan, index=index)
        for _, r in g.iterrows():
            loc = index.searchsorted(r["datadate"] + pd.Timedelta(days=lag_days))
            if loc < len(index):
                s.iloc[loc] = r[col]
        out[permno] = s.ffill()
    return out


def build_features(pan):
    close, vol = pan["close"], pan["vol"]
    ret = pan["ret"].reindex(columns=close.columns).astype("float64")
    mcap = pan["mcap"]
    dvol = close * vol
    ff = pd.read_parquet("data_cache/ff_factors_daily.parquet"); ff.index = pd.to_datetime(ff.index)
    mkt = ff["MktRF"].reindex(close.index).fillna(0.0)

    F = {}
    F["mom_12_1"] = close.shift(21) / close.shift(252) - 1
    F["mom_6_1"] = close.shift(21) / close.shift(126) - 1
    F["rev_1m"] = close / close.shift(21) - 1
    F["hi52"] = close / close.rolling(252, min_periods=120).max()
    F["ivol"] = ret.rolling(63, min_periods=30).std()
    F["max21"] = ret.rolling(21, min_periods=10).max()
    cov = (ret.mul(mkt, axis=0)).rolling(126).mean().sub(
        ret.rolling(126).mean().mul(mkt.rolling(126).mean(), axis=0))
    F["beta"] = cov.div(mkt.rolling(126).var(), axis=0)
    F["logmcap"] = np.log(mcap.clip(lower=1))
    F["logdvol"] = np.log(dvol.clip(lower=1))
    F["amihud"] = (ret.abs() / dvol.replace(0, np.nan)).rolling(21).mean()
    F["logprice"] = np.log(close.clip(lower=0.1))

    # PEAD features
    sue_val = pead_active_panel(pan, dict(P, H=60))      # winsorized SUE, active 60d
    F["sue"] = sue_val
    ann_ret = ret.where(sue_val.notna() & sue_val.diff().ne(0))  # ret near fresh surprise
    F["sue_x_mom"] = sue_val * F["mom_12_1"]

    # fundamentals
    comp = pd.read_parquet("data_cache/compustat_quality.parquet")
    idx, colz = close.index, close.columns
    ceq = _comp_panel(comp, "ceq", idx, colz); revt = _comp_panel(comp, "revt", idx, colz)
    F["bm"] = (ceq * 1e6) / mcap.replace(0, np.nan)
    F["sp"] = (revt * 1e6) / mcap.replace(0, np.nan)
    F["gpa"] = _comp_panel(comp, "gpa", idx, colz)
    F["net_issue"] = _comp_panel(comp, "net_issue", idx, colz)
    at = _comp_panel(comp, "at", idx, colz)
    F["asset_growth"] = at / at.shift(252) - 1
    F["month"] = pd.DataFrame(np.tile(close.index.month.values[:, None], (1, len(colz))),
                              index=idx, columns=colz).astype(float)
    return F, list(F.keys())


def stack(F, feats, elig, fwd, dates, with_target=True):
    """Long frame [date, permno, feat..., (y)] over eligible names on `dates`."""
    rows = []
    eligd = elig.reindex(dates)
    for d in dates:
        e = eligd.loc[d]
        names = e[e].index
        if len(names) == 0:
            continue
        m = {f: F[f].loc[d, names].values for f in feats}
        df = pd.DataFrame(m, index=names)
        df["date"] = d; df["permno"] = names
        if with_target:
            df["y"] = fwd.loc[d, names].values
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    return out


def main():
    pan = load_panels()
    days = pan["close"].index; cols = pan["close"].columns
    rowpos = {d: i for i, d in enumerate(days)}
    elig = pit_universe(pan, P)
    ret_df = pan["ret"].reindex(columns=cols).astype("float64")
    ret_df, _, _ = add_delist_returns(ret_df, set(int(c) for c in cols))
    ret_mat = ret_df.to_numpy(float, na_value=np.nan)
    ff = pd.read_parquet("data_cache/ff_factors_daily.parquet"); ff.index = pd.to_datetime(ff.index)

    close = pan["close"]
    fwd = close.shift(-H) / close - 1.0                       # forward-H return (labels/IC only)
    F, feats = build_features(pan)
    print(f"features: {len(feats)} -> {feats}", flush=True)

    # sampling: weekly training rows (reduce overlap); predict daily
    wk = days[::5]
    train_dates = [d for d in wk if 2017 <= d.year <= 2020 and rowpos[d] + H < len(days)
                   and d < pd.Timestamp("2020-11-01")]   # embargo last ~40d (purge leak into 2021)
    oos_dates = [d for d in wk if 2021 <= d.year <= 2024 and rowpos[d] + H < len(days)]
    tr = stack(F, feats, elig, fwd, train_dates).dropna(subset=["y"])
    te = stack(F, feats, elig, fwd, oos_dates).dropna(subset=["y"])
    print(f"train rows {len(tr)}  oos rows {len(te)}", flush=True)

    # targets: T1 = per-date rank of fwd; T2 = fwd>0.25
    def rankpct(g):
        return g.rank(pct=True)
    tr["y1"] = tr.groupby("date")["y"].transform(rankpct)
    te["y1"] = te.groupby("date")["y"].transform(rankpct)
    tr["y2"] = (tr["y"] > 0.25).astype(int)
    Xtr, Xte = tr[feats], te[feats]

    import lightgbm as lgb, xgboost as xgb
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import ElasticNet
    try:
        from catboost import CatBoostRegressor
        HAVE_CB = True
    except Exception:
        HAVE_CB = False

    def fit_predict_T1(name):
        if name == "ElasticNet":
            Xtr_ = Xtr.fillna(Xtr.median()); Xte_ = Xte.fillna(Xtr.median())
            m = ElasticNet(alpha=1e-3, l1_ratio=0.5, max_iter=5000).fit(Xtr_, tr["y1"])
            p = m.predict(Xte_); ptr = m.predict(Xtr_); imp = dict(zip(feats, np.abs(m.coef_)))
        elif name == "LightGBM":
            m = lgb.LGBMRegressor(n_estimators=400, num_leaves=31, learning_rate=0.03,
                                  min_child_samples=200, subsample=0.8, colsample_bytree=0.7,
                                  reg_lambda=5.0, n_jobs=-1, verbose=-1).fit(Xtr, tr["y1"])
            p = m.predict(Xte); ptr = m.predict(Xtr); imp = dict(zip(feats, m.feature_importances_))
        elif name == "XGBoost":
            m = xgb.XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.03,
                                 min_child_weight=200, subsample=0.8, colsample_bytree=0.7,
                                 reg_lambda=5.0, n_jobs=-1).fit(Xtr, tr["y1"])
            p = m.predict(Xte); ptr = m.predict(Xtr); imp = dict(zip(feats, m.feature_importances_))
        elif name == "CatBoost":
            m = CatBoostRegressor(iterations=400, depth=5, learning_rate=0.03,
                                  l2_leaf_reg=5.0, verbose=0).fit(Xtr.fillna(-999), tr["y1"])
            p = m.predict(Xte.fillna(-999)); ptr = m.predict(Xtr.fillna(-999)); imp = dict(zip(feats, m.feature_importances_))
        elif name == "RandomForest":
            Xtr_ = Xtr.fillna(Xtr.median()); Xte_ = Xte.fillna(Xtr.median())
            m = RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=200,
                                      max_features=0.5, n_jobs=-1).fit(Xtr_, tr["y1"])
            p = m.predict(Xte_); ptr = m.predict(Xtr_); imp = dict(zip(feats, m.feature_importances_))
        return p, ptr, imp

    def ic_of(pred, frame):
        f = frame[["date", "y"]].copy(); f["p"] = pred
        ics = f.groupby("date").apply(lambda g: spearmanr(g["p"], g["y"])[0] if len(g) > 5 else np.nan)
        return float(ics.mean())

    models = ["ElasticNet", "LightGBM", "XGBoost", "RandomForest"] + (["CatBoost"] if HAVE_CB else [])
    preds, ptrs, imps, is_ic = {}, {}, {}, {}
    for nm in models:
        preds[nm], ptrs[nm], imps[nm] = fit_predict_T1(nm)
        is_ic[nm] = ic_of(ptrs[nm], tr)
        print(f"  fit {nm}  IS_IC={is_ic[nm]:.3f}", flush=True)
    # ensemble = rank-average of tree models
    tree = [m for m in models if m != "ElasticNet"]
    preds["Ensemble"] = np.mean([pd.Series(preds[m]).rank(pct=True).values for m in tree], axis=0)
    ptr_ens = np.mean([pd.Series(ptrs[m]).rank(pct=True).values for m in tree], axis=0)
    is_ic["Ensemble"] = ic_of(ptr_ens, tr)

    # T2 tail model (LightGBM classifier)
    clf = lgb.LGBMClassifier(n_estimators=400, num_leaves=31, learning_rate=0.03,
                             min_child_samples=200, subsample=0.8, colsample_bytree=0.7,
                             reg_lambda=5.0, n_jobs=-1, verbose=-1).fit(Xtr, tr["y2"])
    preds["LGBM-tail(T2)"] = clf.predict_proba(Xte)[:, 1]
    is_ic["LGBM-tail(T2)"] = ic_of(clf.predict_proba(Xtr)[:, 1], tr)
    imps["LGBM-tail(T2)"] = dict(zip(feats, clf.feature_importances_))

    # ---- assemble prediction panels & run the gauntlet --------------------
    def pred_panel(p):
        te2 = te[["date", "permno"]].copy(); te2["p"] = p
        return te2.pivot_table(index="date", columns="permno", values="p").reindex(
            index=days, columns=cols)

    def oos_ic(p):
        te2 = te[["date", "y"]].copy(); te2["p"] = p
        ics = te2.groupby("date").apply(lambda g: spearmanr(g["p"], g["y"])[0] if len(g) > 5 else np.nan)
        return float(ics.mean()), float(ics.mean() / ics.std()) if ics.std() > 0 else 0.0

    def two_month(s, win=42):
        v = s.values
        return np.array([np.prod(1 + v[i:i + win]) - 1 for i in range(len(v) - win + 1)])

    # baselines
    z_pead = zscore_panel(pead_active_panel(pan, P), elig).to_numpy(float, na_value=np.nan)
    pead_book = run_book_fill(z_pead, ret_mat, rowpos, days, "2021-01-01", "2024-12-31",
                              N=15, K_exit=30, band=0.03, fill_mode="signed")
    ob = stats(_ew(elig, ret_mat, rowpos, days, "2021-01-01", "2024-12-31"), P)["tot"]

    def gauntlet(label, target, p, N=15):
        zp = pred_panel(p)
        z = zscore_panel(zp, elig).to_numpy(float, na_value=np.nan)
        book = run_book_fill(z, ret_mat, rowpos, days, "2021-01-01", "2024-12-31",
                             N=N, K_exit=30, band=0.03, fill_mode="signed")
        ic, ir = oos_ic(p)
        oss = stats(book, P)
        a = factor_alpha(book, ff, 2021, 2024)
        w = two_month(book)
        peadt = stats(pead_book, P)["tot"]
        return dict(label=label, target=target, ic=ic, ir=ir, is_ic=is_ic.get(label, np.nan),
                    gap=is_ic.get(label, np.nan) - ic, tot=oss["tot"], sharpe=oss["sharpe"],
                    vs_pead=oss["tot"] - peadt, alpha=a["alpha_ann"], at=a["t"],
                    turn=book.attrs["turnover"], med=float(np.median(w)),
                    p95=float(np.percentile(w, 95)), mx=float(w.max()),
                    p25=float((w > 0.25).mean()), p40=float((w > 0.40).mean()))

    results = []
    for nm in models + ["Ensemble"]:
        results.append(gauntlet(nm, "T1", preds[nm]))
        print(f"  gauntlet {nm}", flush=True)
    results.append(gauntlet("LGBM-tail(T2)", "T2", preds["LGBM-tail(T2)"]))

    # PEAD + EW baseline rows
    pead_w = two_month(pead_book)
    pa = factor_alpha(pead_book, ff, 2021, 2024)
    pead_row = dict(label="PEAD baseline", target="SUE", ic=np.nan, ir=np.nan,
                    tot=stats(pead_book, P)["tot"], sharpe=stats(pead_book, P)["sharpe"], vs_pead=0.0,
                    alpha=pa["alpha_ann"], at=pa["t"], turn=pead_book.attrs["turnover"],
                    med=float(np.median(pead_w)), p95=float(np.percentile(pead_w, 95)),
                    mx=float(pead_w.max()), p25=float((pead_w > 0.25).mean()),
                    p40=float((pead_w > 0.40).mean()))

    # ---- report in the §7 format ------------------------------------------
    print("\n" + "=" * 140)
    hdr = (f"{'Model':<15}{'Tgt':>5}{'OOS IC(IR)':>13}{'OOStop ret':>11}{'vsPEAD':>9}"
           f"{'alpha%/yr(t)':>14}{'turn/d':>8}{'2mo med':>9}{'2mo95':>8}{'2mo max':>9}"
           f"{'P>25%':>7}{'P>40%':>7}{'verdict':>9}")
    print(hdr); print("-" * 140)
    print(f"{'EW universe':<15}{'-':>5}{'-':>13}{ob:>11.0%}{ob-pead_row['tot']:>9.0%}"
          f"{'-':>14}{'-':>8}{'-':>9}{'-':>8}{'-':>9}{'-':>7}{'-':>7}{'bench':>9}")
    print(f"{pead_row['label']:<15}{pead_row['target']:>5}{'-':>13}{pead_row['tot']:>11.0%}"
          f"{0.0:>9.0%}{pead_row['alpha']*100:>8.0f}/yr({pead_row['at']:>3.1f}){pead_row['turn']*100:>7.1f}%"
          f"{pead_row['med']:>9.1%}{pead_row['p95']:>8.0%}{pead_row['mx']:>9.0%}"
          f"{pead_row['p25']:>7.1%}{pead_row['p40']:>7.1%}{'BASELINE':>9}")
    for r in results:
        verdict = "BEAT" if r["vs_pead"] > 0.05 and r["at"] >= 2 else ("~tie" if abs(r["vs_pead"]) <= 0.05 else "worse")
        ics = f"{r['ic']:.3f}({r['ir']:.2f})" if not np.isnan(r["ic"]) else "-"
        print(f"{r['label']:<15}{r['target']:>5}{ics:>13}{r['tot']:>11.0%}{r['vs_pead']:>9.0%}"
              f"{r['alpha']*100:>8.0f}/yr({r['at']:>3.1f}){r['turn']*100:>7.1f}%{r['med']:>9.1%}"
              f"{r['p95']:>8.0%}{r['mx']:>9.0%}{r['p25']:>7.1%}{r['p40']:>7.1%}{verdict:>9}")
    print("\nIS vs OOS rank IC (overfit gap):")
    for r in results:
        print(f"  {r['label']:<15} IS_IC {r['is_ic']:>6.3f}  OOS_IC {r['ic']:>6.3f}  gap {r['gap']:>6.3f}")

    print("\nTop-5 features by importance:")
    for nm in models + ["LGBM-tail(T2)"]:
        top = sorted(imps[nm].items(), key=lambda kv: -kv[1])[:5]
        print(f"  {nm:<15} " + ", ".join(f"{k}" for k, _ in top))

    out = dict(features=feats, models=results, pead=pead_row, ew_oos=ob,
               importances={k: sorted(v.items(), key=lambda kv: -kv[1])[:10] for k, v in imps.items()})
    Path("results").mkdir(exist_ok=True)
    json.dump(out, open("results/pead_v5_ml.json", "w"), indent=2, default=float)
    print("\nwrote results/pead_v5_ml.json", flush=True)


if __name__ == "__main__":
    main()
