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

**v7 (fill-robustness) — VOLUME-CAP WORKSTREAM MOOT.** The live competition fill rule
is confirmed: full fills at a reference price 15-20min after the order, no orderbook,
no volume cap (sim just follows prices). So the illiquid book's fills are NOT volume-
constrained; there is no execution reason to pull back from the most illiquid names.
`scripts/run_pead_v7_fillrobust.py`.

**v8 (repeatability) — THE TAIL IS REPEATABLE, NOT a 2024 artifact.** Full 8-year per-
year cut: >40% 2-month windows fire in 2018, 2020, 2023, 2024 (4 distinct years); >60%
in 2020 + 2024. The biggest tail year is 2020 (79 windows >40%), hidden when you look
OOS-only (2021-24). The 5 jackpots (ROOT/TIL/CDXC/BCOV/RMNI) are idiosyncratically
uncorrelated (mean pairwise corr 0.07 ≈ random bucket-5 baseline 0.09) = real breadth,
not one trade — BUT bucket-5 equal-weight lifted broadly in 2024 (+99%) and 2023 (+76%),
so the extreme (>60%) tail is melt-up-regime-amplified (2020 COVID rebound, 2024 small-
cap). `scripts/run_pead_v8_repeatability.py`.

**v9 (N-sweep) — CORE LOCKED at N≈25.** Breadth (not concentration) catches the
idiosyncratic jackpots; N=8 collapses the tail (P>40%=0). Extended sweep: OOS return
peaks N=30 (1140%) then dilutes (SUE z falls 1.50@N12->0.79@N50, per-name weight
8.2%->3.5%). Re-ranked on WIN-RELEVANT thresholds (P(2mo>60%), max), **N=25 wins both**
(P>60%=2.2%, max single 2mo window +81%, OOS 955%, alpha t=5.1). `run_pead_v9*.py`.

**v10 (§C.2 forward catalyst sleeve) — BUILT; but the 10% cap kills it as a tail engine.**
Live pipeline: CT.gov v2 Phase 2/3 trials (primary-completion Mar-Jun 2026 -> readout in
the live window), EDGAR 8-Ks with PDUFA action dates PARSED from filing text and hard-
filtered to the live 2-month window, sponsor->ticker via SEC company_tickers.json. KEY
FINDING: under the competition's 10% name cap + 2-month buy-hold, a basket of independent
binary catalysts CANNOT manufacture a >60% window (cap ceilings any single hit at ~+7%;
>=10 names needed for full deployment average the tail away). With EV assumed ~0 (dated
catalysts are priced in; clinical base rates aren't stock-reaction-conditional) the sleeve
is roughly SYMMETRIC: P>60%≈0, P>40%~2%, P(<0)~50%, 5th pct ~-30%. So the sleeve is a
REGIME-INDEPENDENT DIVERSIFIER (lifts the floor in a flat tape), NOT a grand-slam source.
The >60% tail stays the CORE's job. `scripts/run_pead_v10_catalyst.py`, `run_regime.py`.

**Live small-cap regime (as of 2026-06-18): MELT-UP** — IWM +22% 3m, above 50/200 DMA,
small leading large. Favors the core's melt-up-amplified tail now; catalyst sleeve is
insurance, not the lead.

**v11 (free-turnover re-derivation) — REJECTED; buy-hold-40 stands.** Tested the
hypothesis that the low-turnover hysteresis (K_exit=30, 3% band) is a real-world cost
artifact with no justification under the sim's free turnover. Built a signal-economic
rotation policy: g = SUE_z * decay(age), decay CALIBRATED on IS-2017-20 realized bucket-5
drift-by-age (forward-drift half-life ~23 trading days, ->0 by age ~38), hold top-N by g,
free turnover, uptrend+appreciation-lock protection, signed delay slippage charged. OOS-
only (42d embargo). RESULT: free-turnover LOSES. Head-to-head (N=25): buy-hold-40 OOS
743% / EV 11.0% / alpha-t 5.1 / P(2mo>60%) 2.3% / max 81%; free-turnover OOS 187% / EV
6.4% / alpha-t 1.8 (insignificant) / P>60% 2.4% / max 93%. The only "win" is a single
noisy max window; EV halves and alpha-t collapses. WHY: PEAD drift ACCRUES over ~40 days
and must be HELD to collect — the binding reality was never transaction cost, it's that
rotating into fresher surprises truncates the drift accrual of aged-but-still-drifting
names. The 40-day hold coincidentally matches the drift horizon. Rotation also did NOT
increase jackpot shots (621 tickets vs buy-hold 801) and pushed |SUE z| held down to 1.17.
`scripts/run_pead_v11_rotation.py`.

**v12 (illiquidity-DEPTH sweep) — deeper-than-bucket-5 KILLED; bucket-5 is the optimal depth.**
Reframe: tx cost was never binding, so the sim edge is no-impact + full fills at undisturbed
prices, worth most in the extreme-illiquid tail. Swept how hard N=25 tilts illiquid: bucket-5
(~338/mo) vs most-illiquid 200/100/50 by Amihud (x-checked $ADV) + a SUE+illiq composite. OOS
embargoed, MIDPOINT. THESIS (deeper fattens the right tail even if EV flat) is FALSE: bucket-5
has the FATTEST tail (P(2mo>60%) 2.3%, max 81%); every deeper cut collapses to P>60%≈0 and lower
max. WHY: the bottleneck is SUE COVERAGE, not liquidity — live-surprise count collapses 57->28
->12->6 as depth deepens (can't even fill N=25 below illiq-100), and |SUE z| of held names dilutes
1.17->0.6. Data quality also degrades with depth: Blume-Stambaugh bid-ask-bounce bias 62->309 bp/yr,
held relative spread 0.7%->1.6%, LT-MID bounce gap grows (ADV-100 x-check had a +94pp LT-MID gap =
mostly artifact). So bucket-5 already sits at the optimal depth: captures the illiquidity premium
while retaining enough strong-surprise coverage to populate the tail. Coverage-free actual-earnings-
change proxy noted as the ONLY way to probe deeper rigorously, but deprioritized (deeper showed no
tail gain at valid depths). `scripts/run_pead_v12_depth.py`.

**v13 (deployment) — CONTEST-MATCHED EXPECTATION IS MODEST; live feed CONFIRMED.**
Live window = ~6 weeks (~30 td), June 22 -> ~Aug 3, SHORTER than the 40-day hold, so the
2-month figures overstate. Contest-matched proxy (fresh capital at June-22 anchor, MTM open
positions at 6wk close, signed slippage), per-year 2017-2024: +5.9/+7.9/-7.1/+22.6/+0.4/
+28.1/+20.3/+12.9%. Across 8: MEDIAN +10.4% (ex-2020 +7.9%), min -7.1%, max +28.1% (2022);
cleared >+25% 1/8, **>+40% 0/8, >+60% 0/8**. Even 2020 gives only +22.6% at 6wk (its +149%
2-month came later, Aug-Sep). TRUNCATION (full-40 minus 6wk) median only +1.1% -> the short
window leaves little drift on the table; raising N / front-loading recovers ~nothing. Band
(+/-5 td, correlated) median +12.4%, STABLE. KEY DEPLOY READ: the core is a solid positive-EV
bet (~+10% median, rarely negative) but 6 weeks is TOO SHORT for the illiquid-PEAD multibaggers
to compound into a >40% grand-slam — the winner-take-all tail is essentially absent at the
contest horizon. This strengthens the case for a small catalyst tail-shot (single-day binary
pops fit in 6 weeks where drift cannot). LIVE FEED confirmed: Nasdaq earnings calendar API
(free, no key, same-session) gives actual EPS + consensus + #ests; ~83-89% of reporters have
consensus incl thin small-caps; SUE = (actual-consensus)/rolling-std, x-sec z within bucket-5;
price/vol via Yahoo chart API. `run_pead_v13_seasonal.py`, `run_pead_v13b_contest.py`.

**v13c (deploy items 3/4/beta).** ITEM 3 universe: strategy self-constructs its universe by
live mktcap rank 1001-3000 (= Russell-2000 definition), so it AUTO-TRACKS the June-2026
reconstitution (no static list needed). 2024 coverage: ~1696 eligible/day, ~340 bucket-5,
~58 bucket-5 with live +SUE/day (enough to fill N=25 in season). Live R2000 list overlaps
the CRSP universe 1610/2051 (78%). Reconstitution gap: fresh adds lack 21d-Amihud/earnings
history, excluded until seasoned (conservative, no false fills). Desert weeks (late June)
have low coverage -> book underinvested until the mid-July flood. ITEM 4 catalyst slice =
KILLED (zeroed per pre-registered criterion). Modeling the cap correctly (a name contributes
weight*pop, not raw pop) + honest EV (priced-in mean ->0 minus ~5% event-vol premium ->
EV -5%/name): at A=12% the slice gives up -1.3% EV AND ADDS -0.6% P(>40%) (NEGATIVE), P(>60%)
contribution 0.0%. Under the 10% cap the slice is too small to create +40% portfolio outcomes
and the diverted capital dilutes the core's own tail -> strictly dominated, zero it. BETA:
full-sample daily beta(core, IWM) = 0.89 (~1:1 small-cap beta, NOT an amplifier). Per-window
decomposition: the big core years were beta + alpha TOGETHER (2022 +27% = +11% beta + +15%
alpha; 2020 +25% = +6% beta + +19% alpha), and the alpha residual is the larger, higher-
variance driver. So a melt-up helps ~1:1 but is NOT alone enough for a winning number; you
need melt-up AND a strong earnings-season alpha firing together. `run_pead_v13c_deploy.py`.

**Bottom line (current best): illiquid bucket-5 PEAD, N=25, BUY-HOLD-40 — optimal liquidity depth.** OOS +955%,
alpha t=5.1, P(2mo>40%) 4.6%, P(2mo>60%) 2.2%, max +81%. Tail repeatable across 4 years
but extreme tail melt-up-amplified. Catalyst sleeve = downside/regime hedge, not a tail
engine (10%-cap-limited). Strategy is fundamentally a small-cap-melt-up bet. Caveat:
illiquid edge is impact-fragile live but capturable under the sim's zero-impact rule.

Data/code: `data_cache/pead_crsp_daily.parquet`, `pead_ibes_sue.parquet`,
`pead_delist.parquet`, `ff_factors_daily.parquet`; the whole strategy is consolidated,
self-contained, in `pead_strategy.py` (per-year returns + QuantStats tearsheet); operational
loop in `DEPLOYMENT_RUNBOOK.md`; live regime read in `scripts/run_regime.py`.
