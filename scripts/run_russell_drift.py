"""Long-only, delay-robust drift-harvesting backtest on the Russell 2000.

Implements V1 of the QUANT NOTE: a single daily-rotated book that blends three
delay-robust, long-only return sources -- trailing momentum, overnight-drift
propensity, and post-jump (PEAD-proxy) drift -- into a cross-sectional composite
z-score, then holds the top-N names under a hard 10% entry cap with an
appreciation lock, no leverage (gross <= TGT), and zero transaction costs.

Data: data_cache/russell2000_ohlcv_bidask.parquet  (CRSP daily OHLCV + bid/ask
for the ~2000-name Russell-2000 proxy, pulled from WRDS).

No transaction costs, single-reference-price fills (we are NOT capturing the
spread; we harvest directional drift that survives a 15-20 min fill delay).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.report import metrics  # noqa: E402

# ----------------------------------------------------------------------------- params
PARAMS = dict(
    L=63, s=5,                 # momentum: 63d formation, skip last 5d
    onight_lb=20,              # overnight: mean overnight ret, trailing 20d
    j_thresh=0.07, v_mult=2.0, # jump: +7% day on >=2x 20d avg volume
    jump_D=10,                 # jump stays "active" 10 trading days
    w_mom=0.40, w_on=0.40, w_jump=0.20,
    N=12, TGT=0.98, CAP=0.10,  # 12 names, 98% invested, 10% per-name entry cap
    stale_K=5,                 # require a real trade within last 5 bars
    rf=0.04,                   # annual risk-free for Sharpe (competition.yaml)
    capital=1_000_000.0,
    bt_start="2024-01-01", bt_end="2024-12-31",
)


def zscore_row(s: pd.Series) -> pd.Series:
    mu, sd = s.mean(), s.std()
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sd


def build_panels(df: pd.DataFrame):
    piv = lambda c: df.pivot_table(index="date", columns="permno", values=c).sort_index()
    close = piv("prc"); open_ = piv("openprc"); vol = piv("vol")
    ret = piv("ret").where(lambda x: x.notna(), close.pct_change(fill_method=None))
    bid = piv("bid"); ask = piv("ask")
    return close, open_, vol, ret, bid, ask


def compute_signals(close, open_, vol, p):
    # 4.1 momentum: close.shift(s)/close.shift(s+L) - 1
    mom = close.shift(p["s"]) / close.shift(p["s"] + p["L"]) - 1.0
    # 4.2 overnight: trailing mean of open_t/close_{t-1} - 1
    onight_d = open_ / close.shift(1) - 1.0
    onight = onight_d.rolling(p["onight_lb"], min_periods=p["onight_lb"] // 2).mean()
    # 4.3 jump: day_ret>=j_thresh AND vol>=v_mult*avg20; active D days; size-scaled
    day_ret = close.pct_change(fill_method=None)
    avg20 = vol.rolling(20, min_periods=10).mean()
    is_jump = (day_ret >= p["j_thresh"]) & (vol >= p["v_mult"] * avg20)
    jump_sz = (day_ret.where(is_jump)).fillna(0.0)               # jump-day return
    # most recent jump size within the trailing D-day window (0 if none)
    jump_active = jump_sz.rolling(p["jump_D"], min_periods=1).max()
    return mom, onight, jump_active


def _capped_proportional(top: pd.Series, budget: float, cap: float) -> pd.Series:
    """Signal-proportional weights within `budget`, each clipped to `cap`,
    excess redistributed iteratively (remainder falls to cash)."""
    if top.empty or budget <= 0 or top.sum() <= 0:
        return pd.Series(0.0, index=top.index)
    w = budget * top / top.sum()
    for _ in range(50):
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        free = w.index[~over]
        if len(free) == 0 or w[free].sum() == 0:
            break
        w[free] += excess * w[free] / w[free].sum()
    return w.clip(upper=cap)


def target_weights(scores_row: pd.Series, cur_w: pd.Series, p):
    """Scheme B (capped signal-proportional) + 10% entry cap + appreciation lock.

    Locked names (held weight >= cap, reached via appreciation) keep their
    drifted weight: no further buys, no forced trim. The rest of the top-N is
    funded from the *remaining* budget so gross never exceeds 100% (no leverage).
    """
    cand = scores_row[scores_row > 0].dropna()
    if cand.empty:
        return pd.Series(dtype=float)
    top = cand.nlargest(p["N"])
    cap = p["CAP"]
    locked = {nm: cur_w.get(nm, 0.0) for nm in top.index if cur_w.get(nm, 0.0) >= cap}
    others = top.drop(list(locked))
    budget = max(0.0, p["TGT"] - sum(locked.values()))
    w = _capped_proportional(others, budget, cap)
    for nm, wv in locked.items():  # locked winners hold their appreciated weight
        w[nm] = wv
    return w[w > 0]


def run():
    p = PARAMS
    df = pd.read_parquet("data_cache/russell2000_ohlcv_bidask.parquet")
    meta = json.load(open("results/russell2000_permnos.json"))
    names = {int(k): v for k, v in meta["ticker_map"].items()}
    universe = set(meta["permnos"])
    assert set(df["permno"]).issubset(universe), "UNIVERSE GUARD: off-list ticker present"

    close, open_, vol, ret, bid, ask = build_panels(df)
    mom, onight, jump_active = compute_signals(close, open_, vol, p)

    # staleness/eligibility mask: traded within last K bars, valid price + quote.
    traded = (vol.fillna(0) > 0).rolling(p["stale_K"], min_periods=1).max().astype(bool)
    eligible = traded & close.notna() & (close >= 1.0) & bid.notna() & ask.notna()

    days = close.index
    bt = days[(days >= p["bt_start"]) & (days <= p["bt_end"])]
    # decision starts the trading day before the first backtest day (close of t-1)
    first_dec = days[days < bt[0]][-1]
    dec_days = days[(days >= first_dec) & (days <= bt[-1])]

    cap_total = p["capital"]
    pos = pd.Series(0.0, index=close.columns)  # dollar holdings
    cash = cap_total
    V = cap_total
    rec, w_hist, ncand = [], [], []
    max_entry_w = 0.0

    for i in range(len(dec_days) - 1):
        d, nxt = dec_days[i], dec_days[i + 1]
        elig = eligible.loc[d]
        # composite score from data through close(d) -- no lookahead
        z = (p["w_mom"] * zscore_row(mom.loc[d].where(elig))
             + p["w_on"] * zscore_row(onight.loc[d].where(elig))
             + p["w_jump"] * zscore_row(jump_active.loc[d].replace(0.0, np.nan).where(elig)).reindex(close.columns).fillna(0.0))
        z = z.where(elig)
        cur_w = (pos / V) if V > 0 else pos * 0.0
        tw = target_weights(z, cur_w, p)
        ncand.append(int((z > 0).sum()))

        # SANITY: entry weights respect cap (locked/appreciated names exempt)
        for nm, wv in tw.items():
            if cur_w.get(nm, 0.0) < p["CAP"]:
                max_entry_w = max(max_entry_w, wv)
        assert (tw >= -1e-12).all(), "LONG-ONLY GUARD violated"
        assert tw.sum() <= 1.0 + 1e-6, "GROSS GUARD: leverage (sum(w) > 1.0)"

        # rebalance to target dollar positions at close(d)
        new_pos = pd.Series(0.0, index=close.columns)
        new_pos[tw.index] = tw.values * V
        pos, cash = new_pos, V - new_pos.sum()
        w_hist.append((d, tw.copy()))

        # realize next-day return (held book earns day nxt's close-to-close ret)
        r = ret.loc[nxt].reindex(pos.index).fillna(0.0)
        pos = pos * (1.0 + r)
        V_new = pos.sum() + cash
        rec.append((nxt, V_new / V - 1.0))
        V = V_new

    rser = pd.Series({d: r for d, r in rec}).sort_index()
    rser.index = pd.to_datetime(rser.index)
    equity = cap_total * (1.0 + rser).cumprod()

    # ---- diagnostics -------------------------------------------------------
    # turnover & avg holding period from the held-weight history
    wmat = pd.DataFrame({d: w for d, w in w_hist}).T.fillna(0.0)
    turn = (wmat.diff().abs().sum(axis=1) / 2).mean()
    held_mask = wmat > 0
    spans = held_mask.sum(axis=0)  # days each name was held
    avg_hold = spans[spans > 0].mean()
    # monthly returns (calendar) + average
    monthly = (1.0 + rser).resample("ME").prod() - 1.0
    roll21 = rser.rolling(21).apply(lambda x: (1 + x).prod() - 1.0).dropna()

    st = metrics.summary(rser, rf=p["rf"])
    out = {
        "window": f"{rser.index[0].date()} .. {rser.index[-1].date()}",
        "trading_days": len(rser),
        "universe": len(universe),
        "avg_candidates_per_day": float(np.mean(ncand)),
        "total_return": st["total_return"],
        "cagr": st["cagr"],
        "ann_vol": st["ann_vol"],
        "sharpe_rf0": metrics.sharpe(rser, 0.0),
        "sharpe_rf4": st["sharpe"],
        "sortino": st["sortino"],
        "max_drawdown": st["max_drawdown"],
        "hit_rate_daily": float((rser > 0).mean()),
        "avg_monthly_return": float(monthly.mean()),
        "median_monthly_return": float(monthly.median()),
        "avg_rolling21_return": float(roll21.mean()),
        "daily_turnover": float(turn),
        "avg_holding_days": float(avg_hold),
        "max_entry_weight": float(max_entry_w),
        "final_equity": float(equity.iloc[-1]),
    }

    print("=" * 70)
    print("RUSSELL 2000 -- LONG-ONLY DRIFT-HARVESTING BACKTEST (V1)")
    print("=" * 70)
    print(f"Window            {out['window']}  ({out['trading_days']} trading days)")
    print(f"Universe          {out['universe']} names (Russell-2000 proxy)")
    print(f"Avg buy cands/day {out['avg_candidates_per_day']:.0f}")
    print(f"Starting capital  ${p['capital']:,.0f}   -> Final  ${out['final_equity']:,.0f}")
    print("-" * 70)
    print(f"TOTAL RETURN              {out['total_return']:>10.2%}")
    print(f"Avg monthly return        {out['avg_monthly_return']:>10.2%}  (median {out['median_monthly_return']:.2%})")
    print(f"Avg 21d rolling return    {out['avg_rolling21_return']:>10.2%}")
    print(f"Sharpe (rf=0)             {out['sharpe_rf0']:>10.2f}")
    print(f"Sharpe (rf=4%)            {out['sharpe_rf4']:>10.2f}")
    print(f"Sortino (rf=4%)           {out['sortino']:>10.2f}")
    print(f"CAGR                      {out['cagr']:>10.2%}")
    print(f"Annualized vol            {out['ann_vol']:>10.2%}")
    print(f"Max drawdown              {out['max_drawdown']:>10.2%}")
    print(f"Daily hit rate            {out['hit_rate_daily']:>10.2%}")
    print(f"Daily turnover            {out['daily_turnover']:>10.2%}")
    print(f"Avg holding (days)        {out['avg_holding_days']:>10.1f}")
    print("-" * 70)
    print("SANITY:  max entry weight  %.4f  (cap 0.10)  -- %s" %
          (out['max_entry_weight'], "OK" if out['max_entry_weight'] <= 0.10 + 1e-9 else "FAIL"))
    print("\nMonthly returns:")
    for dt, mr in monthly.items():
        print(f"   {dt.strftime('%Y-%m')}   {mr:>8.2%}")

    Path("results").mkdir(exist_ok=True)
    json.dump(out, open("results/russell_drift_summary.json", "w"), indent=2)
    rser.to_csv("results/russell_drift_daily_returns.csv")
    monthly.to_csv("results/russell_drift_monthly_returns.csv")
    print("\nwrote results/russell_drift_summary.json + daily/monthly CSVs")
    return out


if __name__ == "__main__":
    run()
