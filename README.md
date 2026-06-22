# comp-backtest — Illiquid Bucket-5 PEAD

A single, self-contained equity strategy for a **2-month, pure-PnL, rank-based trading
competition** (76-player winner-take-all; long-only; no leverage; 10% per-name cap; zero
transaction cost; full fills at a reference price ~15-20 min after the order).

## The strategy

**Post-earnings-announcement drift (PEAD), tilted to the most-illiquid names.** The
competition's no-impact / full-fill rule is worth the most exactly where real-world costs
are highest — the illiquid tail — so the book concentrates there.

| Component | Choice |
|---|---|
| Universe | US common stock, mktcap rank 1001-3000 (Russell-2000 definition), monthly point-in-time |
| Signal | IBES-style standardized earnings surprise (SUE), enter the close after the announcement (rdq+1), hold ~40 days |
| Liquidity tilt | bucket-5 = most-illiquid Amihud quintile (recomputed monthly) |
| Sizing | top **N=25**, signal-proportional, **10% entry cap**, appreciation-lock, ~98% invested |
| Returns | bid-ask **midpoint** (bounce-free), delisting returns folded in, decision→fill signed slippage charged |
| Direction | long-only, no leverage |

## Backtest results (2017-2024)

| | value |
|---|---|
| Cumulative (full period) | **+11,116%** |
| OOS 2021-2024 | +955%, Carhart alpha 57%/yr (t=5.1) |
| Sharpe / Vol / Max DD | 1.43 / 44% / −46% |
| Best / worst year | +288% (2020) / −1.5% (2022) |

Per-year returns: 2017 +50%, 2018 +13%, 2019 +62%, 2020 +288%, 2021 +72%, 2022 −1.5%,
2023 +121%, 2024 +182%.

> **Honest live expectation:** the contest is ~6 weeks, shorter than the 40-day drift, so
> the deployable expectation is the seasonal proxy — **~+10% median over the window, ~+28%
> historical max, never >+40% in 8 years.** The win path is a small-cap melt-up (~1:1 IWM
> beta) amplifying a strong Q2 earnings-season alpha. See `DEPLOYMENT_RUNBOOK.md`.

## Usage

```bash
python -m venv .venv && .venv/Scripts/python -m pip install -r requirements.txt

python pead_strategy.py                 # per-year returns + summary
python pead_strategy.py --tearsheet     # also write results/pead_tearsheet.html
python scripts/run_regime.py            # live IWM trend / small-cap breadth read
```

## Layout

```
pead_strategy.py        the entire strategy: data -> signal -> buckets -> book -> report
scripts/run_regime.py   live small-cap regime read (IWM trend + breadth), runbook input
DEPLOYMENT_RUNBOOK.md   operational daily loop, desert/day-1 ramp, undeployed-capital toggle
data_cache/*.parquet    price/SUE/delisting/factor panels (tracked via Git LFS)
results/                derived artifacts (per-day returns CSV, QuantStats tearsheet)
memory/                 strategy record
```

Data panels are cached as parquet and tracked via Git LFS.
