---
name: pead-strategy-finding
description: Key result of the comp-backtest drift strategy — PEAD core works OOS but is cost-fragile
metadata:
  type: project
---

The competition (2-month pure-PnL, zero-cost, long-only, no leverage, 10% name cap)
strategy work converged on a **PEAD core** after the v1 multi-signal book failed.

**v1 (momentum+overnight+jump, top-12, 2024):** lost −47%. Attribution proved the
**overnight selector** (trailing close→open return) was the −46% loss driver — it
buys recent microcap pumps that mean-revert. Removed it. Jump/PEAD was the only
positive leg (+24%).

**v2 (PEAD core, IBES SUE, PIT Russell-2000 proxy = mktcap rank 1001–3000, enter
rdq+1, hold ~40d, top-12 capped):** zero-drift gross 2017–2024 = +4092%, Sharpe
1.61, beat EW benchmark 8/8. **This was the UPPER BOUND, not the live result.**

**v3 reckoning (the conclusion that supersedes v2's optimism):**
- **Fill-drift is decisive.** Charging k_adverse bps on traded notional (the
  decision→fill price drift over the 15–20 min delay; book turns ~22%/day):
  OOS(2021–24) total = 0bps +311%, **25bps +4% (≈benchmark)**, 50bps −74%.
  Breakeven ≈25bps. Expected-live (small-cap 17.5-min drift = tens of bps) sits at
  roughly benchmark-level.
- **Turnover hysteresis is the only real win.** K_exit=30 + 3% no-trade band cuts
  turnover 22%→6%/day and lifts OOS@25bps from +4% to **+29%** (vs bench +16%).
- **Factor attribution kills the alpha story.** Carhart 4-factor on the realistic
  (25bps, hysteresis) book: **alpha ≈0, t=0.6 IS / −0.2 OOS (insignificant)**;
  betas Mkt≈1.0, **SMB≈1.0**. The apparent edge is small-cap market beta, NOT PEAD
  alpha. Held median price $29 (not penny-stock junk), so it's beta not data glitch.
- **Tournament tail is thin.** 2-month windows (42d, @25bps): median +3–4%, 95th
  ~20%, max ~48%, **P(2mo>25%) only ~2%**, 5th pct −15%. N=10 best for tail.
- B.2 delisting returns folded in (1162 events, mean −7.3%): negligible (+3997% vs
  +4092%). All no-lookahead/cap/lock/PIT audits PASS (caveats: IBES restatement,
  full-sample liquidity support pool).

**v4 REVERSAL (corrects v3's two negative conclusions):** v3's flat 25bps haircut
on ALL traded notional was wrong — for a slow signal the 15-20min decision→fill
drift is mostly SYMMETRIC variance (zero EV), not signed cost. Measured signed cost
= **1.3 bps/side** (symmetric 12bps = variance only). Under the corrected signed
fill: OOS(2021-24) = **+294%** (not +4%). And re-running Carhart on the corrected
book: **IS alpha +46%/yr (t=5.5), OOS alpha +28%/yr (t=3.4) — SIGNIFICANT.** So the
book IS small-cap beta (Mkt~1.0, SMB~0.95) BUT with large genuine alpha on top. v3's
"~0 alpha, fill-fragile" verdict was an artifact of the over-pessimistic haircut.
`scripts/run_pead_v4_fillmodel.py`.

**v4 component tests (what was tried for tournament shaping & REJECTED):**
- Novy-Marx GP/A quality core (Compustat `compustat_quality.parquet`): turnover
  tiny (1.1%/d, delay-immune ✓) but OOS −9% vs bench +16%, alpha t=−0.5. No OOS
  edge in small caps. REJECTED as a core.
- High-MAX/IVOL skew sleeve: confirmed negative-EV (Bali), full −100%, 2mo EV
  −11.6%. Adding ANY dose to the PEAD core only lowers EV and worsens the 5th pct
  WITHOUT fattening the upper tail (P(2mo>40%) stays ~0.3%). REJECTED.
- M&A tilt: base rate 2.9%/yr (928 merger-delistings), but delisting-return method
  understates premium (misses announcement jump) — inconclusive, needs
  announcement-date CARs.

**v5 (ML signal search) — NOTHING beat PEAD.** LightGBM/XGBoost/CatBoost/RF/
ElasticNet/ensemble on 19 slow features, T1 (fwd-40d return) + T2 (P(fwd>+25%)),
purged train 2017-20 / OOS 2021-24. Every model OOS top-K +6..+25% vs PEAD +82%
(all "worse"), none significant alpha. Trees badly overfit: LightGBM IS_IC 0.373 →
OOS 0.043 (gap 0.33); T2 tail model OOS_IC went NEGATIVE (−0.15) and did NOT fatten
the tail (P(2mo>40%)=0). Models leaned on gpa/month/sp/ivol, IGNORED sue → diluted
the one working signal. `scripts/run_pead_v5_ml.py`. KEEP PEAD; ML rejected.

**v6 (liquidity tilt) — THE ALPHA LIVES IN THE ILLIQUID TAIL.** Sorting the PEAD
book into Amihud-illiquidity quintiles (PIT, monthly): the marginal broad alpha
(t~0.9-1.1) is concentrated in the MOST ILLIQUID quintile — bucket 5 OOS +665%,
**Carhart alpha +50%/yr t=3.9 (significant)**, vs buckets 1-4 ~+82-113% t=0.9-1.8.
Crucially it SURVIVES bid-ask MIDPOINT returns (665% mid vs 698% last-trade → bounce
is only ~5%, NOT an artifact) and the stale filter; measured signed fill stays low
(2.6 bps). Also fattens the tail: P(2mo>40%) 6.4% vs broad 3.7%, max +179% vs +104%.
All 4 kill criteria PASS. `scripts/run_pead_v6_liquidity.py`.
**BUT the one unremovable bias = NO MARKET IMPACT:** the edge is large precisely
because illiquid anomalies stay un-arbitraged (costs/impact eat them in reality);
the competition's zero-cost+no-impact rules make it capturable HERE but it would NOT
survive real implementation at scale. Right tilt FOR THIS COMPETITION; not real-world.

**Bottom line (current best): liquidity-tilted PEAD (illiquid quintile).** OOS +665%,
alpha t=3.9, P(2mo>40%) 6.4% — addresses BOTH the alpha and the tournament tail,
within the sim's rules. Caveat: impact-fragile (untradeable at scale live). Pure
broad PEAD remains the conservative fallback. Forward catalysts (v4 §C.2) still the
only OTHER tail source, not yet built. Scripts: `run_pead_v4_*.py`, `run_pead_v6_*.py`.

Data/code: `data_cache/pead_crsp_daily.parquet`, `pead_ibes_sue.parquet`,
`pead_delist.parquet`, `ff_factors_daily.parquet`; `scripts/run_pead_walkforward.py`,
`scripts/run_pead_v3.py`, `scripts/pull_pead_history.py`. Related: [[wrds-data-access]].
