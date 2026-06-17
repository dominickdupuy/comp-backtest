# comp-backtest

A config-driven backtesting environment for a **2-month, pure-PnL, rank-based
equity trading competition**, scored on **total return**. Built to test a
blended multi-anomaly long/short book under the competition's exact rules and
to run a 3-year historical backtest on UPenn **WRDS** data.

## The competition rules (baked in)

All rules live in [`config/competition.yaml`](config/competition.yaml) — nothing
is hard-coded in the engine:

| Rule | Value | Why it matters |
|------|-------|----------------|
| Scoring | **total return** (rank-based) | Variance is *not* penalized → over-bet (super-Kelly) |
| Horizon | ~2 months (`contest_length_days`) | Short sprint; tournament theory applies |
| Transaction costs | **zero** | Free turnover; resurrects illiquid/microcap alpha |
| Data + execution delay | 15–20 min | Kills sub-20-min alpha; multi-day signals intact |
| Max gross leverage | **2.0×** | Push to the cap |
| Max per-name weight | **10%** | Concentrate near the cap |
| Shorting | allowed | Long/short, dollar-neutral by default |
| Universe | ~8,000 US equities | Liquidity-ranked from CRSP common shares |

## Strategy book

[`config/strategies.yaml`](config/strategies.yaml) blends delay-robust,
cost-sensitive anomalies into one cross-sectional score, then constructs a
long/short book:

- **PEAD** (post-earnings drift) — primary; SUE from IBES, small-cap tilt
- **Cross-sectional momentum** — 6-mo formation, skip 1 mo, long-biased
- **Short-term reversal / stat-arb** — multi-day, sector-neutralized
- **Overnight overlay** — trailing overnight-return tilt (delay-immune)
- **Time-series momentum** and **betting-against-beta** (optional sleeves)

Sizing follows tournament theory (Browne; Brown-Harlow-Starks; Dubins-Savage):
super-Kelly gross to the 2× cap, concentrate into the highest-conviction names,
escalate when behind in the final weeks.

## Layout

```
config/            competition.yaml, strategies.yaml   (all rules & params)
src/
  config.py        load + validate config, resolve backtest window
  data/
    wrds_loader.py CRSP daily + IBES SUE -> wide panels, parquet-cached
    taq_loader.py  TAQ millisecond -> 1-min OHLCV bars (relevant subset)
    relevance.py   pick the traded / liquid cross-section for minute pulls
    cache.py       parquet cache
  signals/         pead, momentum, reversal, overnight, bab  (+ base/registry)
  portfolio/       construction (blend->L/S weights), sizing (super-Kelly)
  backtest/
    engine.py      signals->constraints->lag->daily PnL->equity
    constraints.py 2x gross, 10% cap, shorting toggle
    costs.py       parameterized (zero by default)
    intraday.py    intraday reversal engine (frequency experiment)
    execution.py   20-min delayed-fill pricing from minute bars
  report/          metrics + matplotlib/quantstats tearsheet
tests/             synthetic data + engine/constraint unit tests
scripts/           pull_daily, pull_minute, run_intraday_experiment
run_backtest.py    CLI
```

## Setup

```bash
python -m venv .venv && .venv/Scripts/python -m pip install -r requirements.txt
cp .env.example .env        # add WRDS_USERNAME / WRDS_PASSWORD
```

## Usage

```bash
# Validate the engine with no WRDS access (synthetic data):
python run_backtest.py --synthetic

# Real 3-year backtest from WRDS (parquet-cached after first pull):
python run_backtest.py

# Pull daily universe / minute bars explicitly:
python scripts/pull_daily.py
python scripts/pull_minute.py --top 1000

# Frequency experiment: does more rebalancing add PnL at zero cost?
python scripts/run_intraday_experiment.py --top 1000
```

Results (equity curve, drawdown, metrics, HTML tearsheet) land in `results/`.

## Data frequency: why daily is primary, minute is targeted

The winning strategies are **multi-day horizon**, so daily CRSP (plus open/close
for the overnight overlay) is the correct signal granularity — the 15–20 min
delay makes sub-daily *signal* worthless. Minute bars (from TAQ, aggregated
server-side — **not** tick) are pulled only for the **liquid cross-section** to
(a) precisely price the delayed fill and (b) test whether higher-frequency
rebalancing adds PnL now that **zero costs make turnover free**. See
`scripts/run_intraday_experiment.py`. Tick data is deliberately avoided:
infeasible at universe scale and counterproductive given the delay.
