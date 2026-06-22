"""Live small-cap REGIME read for the core-vs-catalyst tilt (staged; allocation set later).
IWM trend (vs 50/200 DMA, momentum) + small-vs-large breadth proxy (IWM/SPY ratio trend).
Data: Yahoo Finance chart API (no key)."""
from __future__ import annotations
import json, urllib.request
from pathlib import Path
import numpy as np
import pandas as pd

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"


def yahoo(sym):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?range=1y&interval=1d"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=40) as r:
        j = json.loads(r.read())
    res = j["chart"]["result"][0]
    ts = pd.to_datetime(res["timestamp"], unit="s")
    close = pd.Series(res["indicators"]["quote"][0]["close"], index=ts, dtype="float64")
    return close.dropna()


def trend(px):
    last = float(px.iloc[-1])
    ma50 = float(px.tail(50).mean()); ma200 = float(px.tail(200).mean())
    r1 = last / float(px.iloc[-21]) - 1 if len(px) > 21 else np.nan
    r3 = last / float(px.iloc[-63]) - 1 if len(px) > 63 else np.nan
    r6 = last / float(px.iloc[-126]) - 1 if len(px) > 126 else np.nan
    return dict(last=last, ma50=ma50, ma200=ma200, above50=last > ma50,
                above200=last > ma200, r1m=r1, r3m=r3, r6m=r6)


def main():
    iwm = yahoo("IWM"); spy = yahoo("SPY")
    print(f"=== LIVE SMALL-CAP REGIME (as of {iwm.index[-1].date()}) ===")
    ti = trend(iwm)
    print(f"  IWM {ti['last']:.2f}  vs 50DMA {ti['ma50']:.2f} ({'ABOVE' if ti['above50'] else 'below'})  "
          f"200DMA {ti['ma200']:.2f} ({'ABOVE' if ti['above200'] else 'below'})")
    print(f"  IWM momentum: 1m {ti['r1m']:+.1%}  3m {ti['r3m']:+.1%}  6m {ti['r6m']:+.1%}")
    # small-vs-large breadth proxy: IWM/SPY ratio trend
    idx = iwm.index.intersection(spy.index)
    ratio = (iwm.reindex(idx) / spy.reindex(idx)).dropna()
    rr3 = float(ratio.iloc[-1] / ratio.iloc[-63] - 1) if len(ratio) > 63 else np.nan
    rr6 = float(ratio.iloc[-1] / ratio.iloc[-126] - 1) if len(ratio) > 126 else np.nan
    ratio_above50 = float(ratio.iloc[-1]) > float(ratio.tail(50).mean())
    print(f"  IWM/SPY (small vs large): 3m {rr3:+.1%}  6m {rr6:+.1%}  "
          f"{'small LEADING' if ratio_above50 else 'small LAGGING'} (vs 50DMA of ratio)")
    # regime classification
    melt = ti["above200"] and ti["above50"] and (ti["r3m"] or 0) > 0.05 and rr3 > 0
    down = (not ti["above200"]) and (ti["r3m"] or 0) < 0
    regime = "MELT-UP (lean CORE)" if melt else "DOWN/WEAK (lean CATALYSTS)" if down \
        else "NEUTRAL/MIXED (balanced)"
    print(f"  -> REGIME: {regime}")
    out = dict(asof=str(iwm.index[-1].date()), iwm=ti, iwm_spy_3m=rr3, iwm_spy_6m=rr6,
               small_leading=ratio_above50, regime=regime)
    Path("results").mkdir(exist_ok=True)
    json.dump(out, open("results/regime.json", "w"), indent=2, default=float)
    print("wrote results/regime.json")


if __name__ == "__main__":
    main()
