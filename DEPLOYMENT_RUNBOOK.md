# DEPLOYMENT RUNBOOK — Illiquid Bucket-5 PEAD

**Contest:** 76-player winner-take-all · starts **Mon 2026-06-22** · ~6 weeks (~30 trading days) to **~2026-08-03** · long-only, no leverage, 10% name cap, zero cost / full fills at reference price.

**Undeployed-capital toggle (default): `BETA-BRIDGE`.** To switch: set `MODE = STRICT` (one line, §C).

---

## HONEST EXPECTATION (read before you start)
- **Median ~+10% over the 6 weeks**; historical max in this exact window **+28%** (2022); **0 of 8 years (2017-24) cleared +40%.** Worst June window −7% (2019).
- The core is a **strong floor, low standalone win-probability** at 6 weeks — too short for illiquid-PEAD multibaggers to compound into a >40% grand-slam.
- **Win path = melt-up beta (~1:1 to IWM) PLUS a strong Q2 earnings-season alpha — BOTH required.** Beta alone is not enough (IWM would need ~+45% in 6wk). Alpha residual is the larger, higher-variance driver.
- Current tape (as of 6/18): **MELT-UP** (IWM +22% 3m, above 50/200 DMA, small leading) — supplies the beta leg; the alpha leg depends on the mid-July flood delivering strong illiquid surprises.

---

# ☐ DAILY MORNING CHECKLIST (run every trading day)

1. **Pull reporters** from prior session (Nasdaq earnings API). → §D
2. **Build qualifiers Q** = live universe (mktcap rank 1001-3000) ∩ bucket-5 (Amihud) ∩ **positive SUE** ∩ age 0-40 ∩ seasoned & eligible. → §D, §E
3. **Compute SUE & cross-sectional z**; rank desc. → §D
4. **Hold top min(|Q|, 25)**, each sized signal-proportional **capped 10%**, PEAD budget 98%. **Do NOT force-fill to 25 from a thin pool, do NOT lower the SUE bar.** → §B
5. **Fund new entries:** reduce undeployed bucket (cash in STRICT; sell an **equal-dollar bridge slice** in BETA-BRIDGE). → §C
6. **Hold existing PEAD names** — never trim, even past 10% (appreciation-lock). Only sell on **loss of eligibility**. → §B
7. **Place orders once** (single daily batch); fills land ~15-20 min later at the reference price. No intraday timing. → §B
8. **BETA-BRIDGE only:** check trim trigger — if **IWM < 50DMA OR IWM/SPY ratio < its 50DMA**, liquidate the bridge to cash (keep all PEAD). → §C
9. **Monitoring sweep:** coverage ramp, tape/breadth, drift-past-10% (hold), eligibility drops, feed health. → §F
10. **Month start (Jul 1):** rebuild universe band + recompute Amihud buckets before step 2. → §E

---

# REFERENCE DETAIL

## §A. Config (LOCKED — do not change mid-contest)
| Param | Value |
|---|---|
| Signal | PEAD, IBES-style SUE, enter rdq+1 |
| Universe | US common stock, mktcap rank **1001-3000** (Russell-2000 def), monthly rebuild |
| Liquidity filter | **Bucket-5** = most-illiquid Amihud quintile within universe, monthly recompute |
| Concurrent N | **25** |
| Sizing | signal-proportional to cross-sectional SUE z, **10% entry cap**, 98% invested target |
| Hold | **buy-hold to age 40** (→ build-and-hold-to-close; see §B) |
| Winners | **appreciation-lock** — never trim a name that drifted past 10% |
| Direction | long-only, no leverage |
| **Catalyst sleeve** | **ZERO** (negative EV, negative ΔP(win) under the cap — do not deploy) |

## §B. Entry / exit mechanics
- **Entry:** a name that reported in the **prior session** enters **this session** (rdq+1). One daily order batch; **fills ~15-20 min later at the reference price** — no intraday timing, no chasing.
- **Sizing per name:** cross-sectional SUE z → signal-proportional weight, **hard-capped at 10%** at entry; total PEAD target 98%.
- **Exit:** nominal hold is age 40, **but the 40-day hold exceeds the 30-day contest window, so NOTHING exits on age — this is build-and-hold-to-close.** The **only exits are loss of eligibility:** delisting/halt, drop out of the 1001-3000 band, price < $1, or stale/no valid quote.
- **Appreciation-lock:** a name that appreciates past 10% is **held at its grown weight, never trimmed.** Winners ride. New entries are funded from the undeployed bucket (§C), not by trimming winners.
- Mark the book to market at the 6-week close with positions still open — that is the final score.

## §C. Undeployed-capital handling — TOGGLE
Set at top: **`MODE = BETA-BRIDGE`** (default) or **`MODE = STRICT`**.

**STRICT** — undeployed capital sits in **cash**; it ramps into PEAD names as they report. Zero tape risk, zero beta on the idle sleeve. Use if you do not want to bet the early tape holds.

**BETA-BRIDGE** (default) — undeployed capital holds a **diversified small-cap beta basket**, rotating name-by-name into fresh PEAD positions:
- **Basket construction:** broad, diversified small-cap exposure to track field beta. If ETFs are permitted, hold **IWM**. If single-stocks only: **equal-weight ≥15 liquid names** spread across sectors from the eligible universe (use liquid buckets 1-2, NOT bucket-5), **10% cap still applies** (so ≥10 names; 15-20 preferred for clean beta ≈ 1).
- **Rotation rule:** when a fresh PEAD name qualifies, **sell an equal-dollar slice of the bridge** to fund the new 10%-capped PEAD position. Bridge shrinks as PEAD fills; by the mid-July flood the bridge should be ~0.
- **Conscious bet:** beta-bridge is a bet the **early tape holds** through the desert weeks. It adds field-correlated beta exactly when you want it in a melt-up — and exactly what hurts if the tape rolls.
- **TRIM TRIGGER (check daily):** if **IWM closes below its 50DMA OR below its 200DMA**, **OR** small-cap breadth rolls (**IWM/SPY ratio < its own 50DMA**) → **liquidate the entire bridge to cash immediately** (keep all PEAD). Do not re-enter the bridge for the remainder of the contest once triggered. (Regime read: `python scripts/run_regime.py`.)

## §D. Daily loop — exact procedure
1. **Reporters:** `GET api.nasdaq.com/api/calendar/earnings?date=<prior session>` (UA + Accept headers). Fields: symbol, eps (actual), epsForecast (consensus), noOfEsts, marketCap.
2. **Map to universe:** keep names in the live **mktcap rank 1001-3000** band (use marketCap + a daily rank; band rebuilt monthly, §E). Drop names with **no consensus** (epsForecast blank) — **do NOT substitute a coverage-free proxy** (decision pre-registered).
3. **Bucket-5 filter:** keep only names in the most-illiquid Amihud quintile (monthly buckets, §E).
4. **SUE:** `SUE = (actual − consensus) / σ`, σ = rolling std of the name's own past (actual − consensus) surprises (time-series std). Keep **SUE > 0** (positive surprise).
5. **Cross-sectional z** of SUE across the eligible bucket-5 set today; rank descending.
6. **Hold set:** top **min(|Q|, 25)**. Size signal-proportional, cap 10%, budget 98%.
7. Fund per §C; hold existing per §B; place one order batch (§B).

## §E. Data & infra
| Need | Source | Cadence |
|---|---|---|
| Earnings actuals + consensus + #ests | **Nasdaq earnings API** (free, no key, same-session) | daily |
| Price / volume / reference prices | **Yahoo chart API** (`/v8/finance/chart/<sym>`) | daily |
| Amihud illiquidity buckets | computed from price/vol (21-day) | **monthly recompute** |
| Universe band (1001-3000) | mktcap rank, common stock | **monthly rebuild** |
| Tape / breadth (regime) | IWM, SPY via Yahoo (`scripts/run_regime.py`) | daily |

- **Monthly recompute within the window: Jul 1** (rebuild band + Amihud buckets before the daily loop). Aug falls at/after the close.
- **Reconstitution gap:** post-June-2026 Russell reconstitution is auto-tracked by the live mktcap ranking. **New constituents (IPOs/risers) are EXCLUDED until seasoned** — a name needs ≥10 trading days of volume (for Amihud) **and** one live SUE to enter. Accept missing the very newest names (the alternative is false fills on stale data).
- **No-coverage-proxy decision:** names without analyst consensus are **dropped, not substituted.** ~83-89% of reporters carry consensus incl. thin small-caps, so the bucket-5 set remains addressable.

## §F. Daily monitoring checklist
- **Coverage ramp:** count of bucket-5 names with live +SUE held vs the **in-season target ~58/day**. Expect **near-zero on Day 1**, low through early July, ramping mid-July. Underinvestment early is EXPECTED (see §G), not an error.
- **Tape & breadth:** IWM vs 50/200 DMA, IWM/SPY ratio vs its 50DMA (`run_regime.py`). Drives the BETA-BRIDGE trim trigger (§C).
- **Drift past 10%:** confirm any name above 10% is **held, not trimmed** (appreciation-lock intact).
- **Eligibility drops:** delistings/halts, band exits, price<$1, stale quotes → sell those names; redeploy per §C.
- **Feed backstop:** if Nasdaq earnings API is down, fall back to a secondary feed (FMP/Finnhub with key, or manual check of the day's reporters); **do not enter without a verified actual+consensus.** If price/volume (Yahoo) is down, hold existing book, place no new orders that day.

## §G. Day-1 & desert protocol (CRITICAL — first ~2 weeks)
- **Day 1 (6/22):** the qualifier list comes from **6/18 (Thu) and 6/19 (Fri)** reporters — late-June is an **earnings desert**, so the list will be **short, possibly near zero.**
- **Do NOT force-fill 25 names from a thin pool. Do NOT lower the SUE bar to manufacture entries.** Hold only genuine qualifiers (positive-SUE bucket-5), each capped 10%.
- **Ramp rule (explicit):** each day, **PEAD names held = min(|Q|, 25)**; **undeployed = 98% − Σ PEAD weights**, routed by the toggle (cash in STRICT; beta-bridge basket in BETA-BRIDGE). As Q2 earnings flood from **~mid-July**, |Q| grows, PEAD fills toward 25, and the undeployed bucket shrinks toward 0.
- **Expected shape:** book is **light through early July** (mostly cash or bridge), **fills mid-to-late July**, fully invested by the back third of the window. This is by design and matches the seasonal estimate.
- **Default behavior in the desert:** with `MODE = BETA-BRIDGE`, the idle capital tracks the (currently melt-up) tape while you wait for the flood — subject to the trim trigger. With `MODE = STRICT`, it waits in cash.

---

*Strategy basis frozen in `memory/pead-strategy-finding.md`; validation in `scripts/run_pead_v6…v13c`. Config is locked for the contest — no mid-window re-optimization.*
