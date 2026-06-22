"""QUANT NOTE v10 (v4 §C.2) — FORWARD BINARY-CATALYST SLEEVE, the regime hedge.
REV2: real in-window dates + skew-only sizing (EV assumed ~0) + simulated PnL dist.

The illiquid-PEAD core (v8/v9) is a strong positive-EV base whose GRAND-SLAM tail is
melt-up-regime amplified (2020, 2024).  This sleeve supplies jackpot potential that is
INDEPENDENT of the small-cap regime: binary clinical/regulatory events whose payoff is
idiosyncratic to the drug, not the tape.  It is the tail source for a flat/down 2026.

Pulls, LIVE, hard-filtered to the forward window, across the ~1700-name universe:
  1. ClinicalTrials.gov v2 — Phase 2/3 trials whose PRIMARY-COMPLETION falls ~Mar-Jun
     2026, so the topline READOUT (which lags completion by 1-3mo) lands in the live
     2026-06-21..08-21 window.  Lead sponsor matched to a universe ticker.
  2. EDGAR 8-Ks mentioning "PDUFA" -> fetch the filing and PARSE THE ACTUAL ACTION DATE;
     hard-filter to action date in 2026-06-21..08-21 (drops already-resolved names).
  3. M&A-target heuristic screen (weakest leg; base rate 2.9%/yr, v4 §C.1).

SIZING: positive SKEW only.  EV is assumed ~0 (known dated catalysts are priced in;
clinical base rates are NOT stock-reaction-conditional, so p*up double-counts the move
the price already embeds).  We size on upside skew and report per-name (p, up, down)
plus the FULL simulated 2-month PnL distribution INCLUDING THE LEFT TAIL.

FORWARD/PROSPECTIVE by construction.  Universe ticker list is the end-2024 cache snapshot
(proxy for the live 2026 universe; refresh for deployment).
"""
from __future__ import annotations
import json, re, sys, time, urllib.parse, urllib.request
from datetime import date, timedelta
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_pead_walkforward import load_panels, pit_universe, P

UA = "comp-backtest research domdd305@gmail.com"
CACHE = Path("data_cache")
TODAY = date(2026, 6, 21)
WIN_START, WIN_END = date(2026, 6, 21), date(2026, 8, 21)     # live 2-month window
CT_COMPL_START, CT_COMPL_END = "2026-03-01", "2026-06-30"     # readout lags completion 1-3mo
READOUT_LAG = 60                                              # days, completion -> topline

ONCO_KEYS = ("cancer", "carcinoma", "tumor", "tumour", "oncolog", "lymphoma", "leukemia",
             "leukaemia", "myeloma", "melanoma", "glioma", "sarcoma", "neoplas",
             "metasta", "malignan", "solid tumor")
DERISK_KEYS = ("biomarker", "mutation", "positive", "selected", "egfr", "alk", "kras",
               "her2", "braf", "ros1", "ntrk")
MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"], 1)}


def _get(url, tries=3, sleep=1.0):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for i in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=40) as r:
                return r.read().decode("utf-8", "replace")
        except Exception as e:                       # noqa
            if i == tries - 1:
                return None
            time.sleep(sleep * (i + 1))
    return None


# ───────────────────────── universe + name map ──────────────────────────────
SUFFIX = re.compile(
    r"\b(inc|incorporated|corp|corporation|co|company|companies|ltd|limited|plc|llc|lp|"
    r"holdings?|group|the|sa|nv|ag|ab|asa|oyj|kk|pharmaceuticals?|pharma|therapeutics?|"
    r"biosciences?|bioscience|biopharma|biopharmaceuticals?|biotechnology|biotech|bio|"
    r"sciences?|medical|medicines?|health|healthcare|laboratories|labs?|technologies|"
    r"technology|tech|systems?|international|industries|enterprises?|trust|com)\b", re.I)


def normalize(name):
    n = re.sub(r"[^a-z0-9 ]+", " ", name.lower())
    return re.sub(r"\s+", " ", SUFFIX.sub(" ", n)).strip()


def universe_tickers():
    pan = load_panels()
    elig = pit_universe(pan, P)
    last = elig.iloc[-1]; live = last[last].index
    tk = {pan["names"].get(int(c)) for c in live}
    return {t for t in tk if isinstance(t, str)}, pan


def sec_namemap(univ):
    j = json.loads(_get("https://www.sec.gov/files/company_tickers.json"))
    tk2title, norm2tk = {}, {}
    for v in j.values():
        t = v["ticker"].upper()
        if t in univ:
            tk2title[t] = v["title"]
            norm2tk.setdefault(normalize(v["title"]), t)
    return tk2title, norm2tk


def match_sponsor(sponsor, norm2tk):
    ns = normalize(sponsor)
    if not ns:
        return None
    if ns in norm2tk:
        return norm2tk[ns]
    toks = set(ns.split())
    if not toks:
        return None
    best, best_j = None, 0.0
    for nm, tk in norm2tk.items():
        nt = set(nm.split())
        if not nt:
            continue
        j = len(toks & nt) / len(toks | nt)
        if j > best_j and (toks <= nt or nt <= toks or (j >= 0.6 and len(toks & nt) >= 2)):
            best, best_j = tk, j
    return best if best_j >= 0.5 else None


# ───────────────────────── leg 1: ClinicalTrials.gov ────────────────────────
def pull_ctgov(norm2tk, tk2title):
    cache_fp = CACHE / "catalyst_ctgov_2026_marjun.json"
    if cache_fp.exists():
        studies = json.loads(cache_fp.read_text())
    else:
        term = (f"AREA[Phase](PHASE2 OR PHASE3) AND "
                f"AREA[PrimaryCompletionDate]RANGE[{CT_COMPL_START},{CT_COMPL_END}]")
        base = ("https://clinicaltrials.gov/api/v2/studies?"
                "filter.overallStatus=RECRUITING%7CACTIVE_NOT_RECRUITING%7CENROLLING_BY_INVITATION"
                "&fields=NCTId,LeadSponsorName,Phase,PrimaryCompletionDate,Condition,OverallStatus"
                "&pageSize=1000&query.term=" + urllib.parse.quote(term))
        studies, token = [], None
        for _ in range(40):
            raw = _get(base + (f"&pageToken={token}" if token else ""))
            if not raw:
                break
            j = json.loads(raw)
            studies.extend(j.get("studies", []))
            token = j.get("nextPageToken")
            if not token:
                break
        cache_fp.write_text(json.dumps(studies))
    print(f"  CT.gov: {len(studies)} Phase 2/3 trials completing {CT_COMPL_START}..{CT_COMPL_END}",
          flush=True)
    rows = []
    for s in studies:
        ps = s.get("protocolSection", {})
        spons = ps.get("sponsorCollaboratorsModule", {}).get("leadSponsor", {}).get("name", "")
        tk = match_sponsor(spons, norm2tk)
        if not tk:
            continue
        pcd = ps.get("statusModule", {}).get("primaryCompletionDateStruct", {}).get("date", "")
        # estimate readout date = completion + lag; keep only if readout in live window
        try:
            cd = pd.to_datetime(pcd).date()
        except Exception:
            continue
        est = cd + timedelta(days=READOUT_LAG)
        if not (WIN_START - timedelta(days=20) <= est <= WIN_END + timedelta(days=20)):
            continue
        phases = ps.get("designModule", {}).get("phases", [])
        cond = "; ".join(ps.get("conditionsModule", {}).get("conditions", []))[:60]
        ph = "PHASE3" if "PHASE3" in phases else "PHASE2"
        rows.append(dict(ticker=tk, company=tk2title.get(tk, ""), type=f"CT-{ph[-1]}",
                         date=str(est), detail=f"{cond} (readout est, compl {pcd})",
                         onco=any(k in cond.lower() for k in ONCO_KEYS),
                         derisk=any(k in cond.lower() for k in DERISK_KEYS),
                         nct=ps.get("identificationModule", {}).get("nctId", "")))
    return rows


# ───────────────────────── leg 2: EDGAR PDUFA action dates ──────────────────
TICK_RE = re.compile(r"\(([A-Z]{1,5})\)")
KW = r"(?:PDUFA|target action date|goal date|action date|prescription drug user fee)"
DATE_TXT = (r"((?:January|February|March|April|May|June|July|August|September|October|"
            r"November|December)\s+\d{1,2},?\s+20\d{2})")
PAT_FWD = re.compile(KW + r"[^.;<>]{0,140}?" + DATE_TXT, re.I)
PAT_REV = re.compile(DATE_TXT + r"[^.;<>]{0,60}?" + KW, re.I)


def parse_action_dates(text):
    t = re.sub(r"<[^>]+>", " ", text)
    t = t.replace("&nbsp;", " ").replace("&#160;", " ")
    t = re.sub(r"\s+", " ", t)
    out = []
    for pat in (PAT_FWD, PAT_REV):
        for m in pat.finditer(t):
            ds = m.group(1)
            try:
                mm = MONTHS[re.match(r"([A-Za-z]+)", ds).group(1).lower()]
                dd = int(re.search(r"(\d{1,2}),?\s+(20\d{2})", ds).group(1))
                yy = int(re.search(r"(20\d{2})", ds).group(1))
                out.append(date(yy, mm, dd))
            except Exception:
                continue
    return out


def pull_pdufa(univ, tk2title):
    hits_fp = CACHE / "catalyst_pdufa_2026.json"
    if hits_fp.exists():
        hits = json.loads(hits_fp.read_text())
    else:
        hits, frm = [], 0
        for _ in range(20):
            url = ("https://efts.sec.gov/LATEST/search-index?q=%22PDUFA%22&forms=8-K"
                   f"&startdt=2026-01-01&enddt=2026-06-21&from={frm}")
            raw = _get(url)
            if not raw:
                break
            j = json.loads(raw)
            hh = j.get("hits", {}).get("hits", [])
            if not hh:
                break
            hits.extend(hh); frm += len(hh)
            if frm >= min(1000, j.get("hits", {}).get("total", {}).get("value", 0)):
                break
        hits_fp.write_text(json.dumps(hits))

    # group universe-matched hits by ticker, keep up to 4 most recent docs each
    by_tk = {}
    for h in hits:
        src = h.get("_source", {})
        for dn in src.get("display_names", []):
            m = TICK_RE.search(dn)
            if not m:
                continue
            tk = m.group(1).upper()
            if tk not in univ:
                continue
            cik = (src.get("ciks") or ["0"])[0]
            adsh = src.get("adsh", "")
            fname = h.get("_id", ":").split(":", 1)[1]
            by_tk.setdefault(tk, []).append(
                (src.get("file_date", ""), int(str(cik)), adsh, fname))

    doc_cache_fp = CACHE / "catalyst_pdufa_docdates.json"
    doc_dates = json.loads(doc_cache_fp.read_text()) if doc_cache_fp.exists() else {}
    rows = []
    print(f"  EDGAR: parsing action dates for {len(by_tk)} universe names w/ PDUFA 8-Ks...",
          flush=True)
    for tk, docs in by_tk.items():
        docs.sort(reverse=True)
        cand = []
        for fd, cik, adsh, fname in docs[:4]:
            key = f"{adsh}:{fname}"
            if key in doc_dates:
                cand += [date.fromisoformat(x) for x in doc_dates[key]]
                continue
            url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{adsh.replace('-','')}/{fname}"
            txt = _get(url)
            dts = parse_action_dates(txt) if txt else []
            doc_dates[key] = [d.isoformat() for d in dts]
            cand += dts
            time.sleep(0.12)                          # SEC fair-access
        # current-cycle action dates only; qualifies iff one lands in the live window
        fut = sorted(set(d for d in cand if d >= date(2026, 1, 1)))
        inwin = [d for d in fut if WIN_START <= d <= WIN_END]
        if inwin:
            rows.append(dict(ticker=tk, company=tk2title.get(tk, ""), type="PDUFA",
                             date=str(min(inwin)), detail="FDA action date (parsed from 8-K)",
                             onco=False, derisk=True))
    doc_cache_fp.write_text(json.dumps(doc_dates))
    print(f"  EDGAR: {len(rows)} names with a PARSED PDUFA action date in "
          f"{WIN_START}..{WIN_END}", flush=True)
    return rows


# ───────────────────────── leg 3: M&A-target screen ─────────────────────────
def screen_ma(univ, pan, tk2title):
    try:
        q = pd.read_parquet(CACHE / "compustat_quality.parquet")
    except Exception:
        return []
    last = q.sort_values("datadate").groupby("permno").last()
    name = pan["names"]; rows = []
    if "gpa" in last.columns:
        hi = last[last["gpa"] > last["gpa"].quantile(0.80)]
        for permno, r in hi.iterrows():
            tk = name.get(int(permno))
            if isinstance(tk, str) and tk in univ:
                rows.append(dict(ticker=tk, company=tk2title.get(tk, ""), type="M&A",
                                 date="", detail=f"high GP/A {r['gpa']:.2f} (consolidation cand.)",
                                 onco=False, derisk=False))
    return rows[:40]


# ─────── per-name conditional reaction priors (p, up, down): EV NOT banked ───
# p: literature LOA / approval base rate; up/down: typical illiquid-biotech event-day
# reaction magnitudes. We do NOT bank E[r]=p*up-(1-p)*down (priced in); sizing = skew.
PRIORS = {  # (p_succ, up_on_success, down_on_failure)
    "CT-3":  (0.55, 0.70, -0.55),
    "CT-2":  (0.30, 0.60, -0.50),
    "PDUFA": (0.88, 0.28, -0.45),
    "M&A":   (0.03, 0.45, -0.02),
}


def reaction(r):
    p, up, dn = PRIORS[r["type"]]
    if r["onco"]:
        p *= 0.45
    if r["derisk"]:
        p = min(0.96, p * 1.35)
    return p, up, dn


def simulate(sleeve, n=200_000, seed=0):
    """independent binary catalysts -> sleeve 2-month return distribution."""
    rng = np.random.default_rng(seed)
    w = np.array([s["w"] for s in sleeve])
    p = np.array([s["p"] for s in sleeve])
    up = np.array([s["up"] for s in sleeve])
    dn = np.array([s["down"] for s in sleeve])
    succ = rng.random((n, len(sleeve))) < p
    R = np.where(succ, up, dn)
    port = R @ w
    def stats_of(x):
        q = lambda v: float(np.percentile(x, v))
        return dict(mean=float(x.mean()), p5=q(5), p50=q(50), p95=q(95),
                    mx=float(x.max()), mn=float(x.min()),
                    P_gt25=float((x > .25).mean()), P_gt40=float((x > .40).mean()),
                    P_gt60=float((x > .60).mean()), P_loss=float((x < 0).mean()))
    raw = stats_of(port)
    cen = stats_of(port - port.mean())                # EV~0: priced-in, mean removed
    return dict(raw=raw, cen=cen, mean=raw["mean"])


def main():
    print("=" * 100)
    print("v10 REV2 (§C.2) FORWARD CATALYST SLEEVE — real in-window dates, skew-only sizing")
    print(f"  live window {WIN_START}..{WIN_END}")
    univ, pan = universe_tickers()
    tk2title, norm2tk = sec_namemap(univ)
    print(f"  universe {len(univ)} tickers ({len(tk2title)} name-matched; end-2024 snapshot)")

    ct = pull_ctgov(norm2tk, tk2title)
    pd_rows = pull_pdufa(univ, tk2title)
    ma = screen_ma(univ, pan, tk2title)
    allrows = ct + pd_rows + ma
    for r in allrows:
        p, up, dn = reaction(r)
        r.update(p=p, up=up, down=dn, skew=p * up)

    # dedup by ticker (highest skew), rank by skew
    allrows.sort(key=lambda r: r["skew"], reverse=True)
    seen, dedup = set(), []
    for r in allrows:
        if r["ticker"] in seen:
            continue
        seen.add(r["ticker"]); dedup.append(r)

    # ── sizing: positive skew, 10% name cap, EV assumed ~0 (not a sizing input) ──
    top = dedup[:25]
    sk = np.array([r["skew"] for r in top])
    w = np.minimum(sk / sk.sum(), 0.10) if sk.sum() else np.ones(len(top)) / max(1, len(top))
    w = w / w.sum()
    for r, wi in zip(top, w):
        r["w"] = float(wi)

    print("\n=== IN-WINDOW CATALYST CALENDAR + per-name (p, up, down) ===")
    print(f"{'tk':<6}{'type':<7}{'date':<12}{'p':>6}{'up':>7}{'down':>7}{'wt':>7}  detail")
    for r in top:
        print(f"{r['ticker']:<6}{r['type']:<7}{(r['date'] or '-'):<12}{r['p']:>6.0%}"
              f"{r['up']:>+7.0%}{r['down']:>+7.0%}{r['w']:>7.1%}  {r['detail'][:42]}", flush=True)
    cnt = {}
    for r in top:
        cnt[r["type"]] = cnt.get(r["type"], 0) + 1
    print(f"  legs in sleeve: " + ", ".join(f"{k}={v}" for k, v in cnt.items()))
    print(f"  matched pool: CT(readout-in-win)={len(ct)}  PDUFA(action-in-win)={len(pd_rows)}  "
          f"M&A-screen={len(ma)}")

    # ── COMPOSITION SWEEP: the 10% cap forces >=10 names for full deployment,
    #    which diversifies away the right tail. Show concentration-vs-tail tradeoff. ──
    print("\n=== COMPOSITION SWEEP: standalone 2-month sleeve distribution vs # names ===")
    print("  two reads per row: [raw] uses conditional priors as-is; [EV~0] removes the mean")
    print("  (priced-in: known dated catalysts embed the expected move) -> the honest skew/tails.")
    print(f"{'k':>4}{'depl':>6} | {'raw:med':>8}{'95th':>6}{'P>40':>6}{'P>60':>6}{'mx':>6}"
          f" | {'EV0:med':>8}{'5th':>6}{'95th':>6}{'P>40':>6}{'P>60':>6}{'P<0':>6}")
    pool = dedup[:25]
    sweep = {}
    for k in [6, 8, 10, 12, 15, 20, 25]:
        sub = pool[:k]
        sk = np.array([r["skew"] for r in sub])
        wk = np.minimum(sk / sk.sum(), 0.10)          # 10% competition name cap
        deploy = float(wk.sum())                       # <1 => remainder sits in cash
        ss = [dict(w=float(wi), p=r["p"], up=r["up"], down=r["down"]) for r, wi in zip(sub, wk)]
        sm = simulate(ss)
        rw, cn = sm["raw"], sm["cen"]
        sweep[k] = dict(deploy=deploy, raw=rw, cen=cn)
        print(f"{k:>4}{deploy:>6.0%} | {rw['p50']:>+8.0%}{rw['p95']:>+6.0%}{rw['P_gt40']:>6.1%}"
              f"{rw['P_gt60']:>6.1%}{rw['mx']:>+6.0%} | {cn['p50']:>+8.0%}{cn['p5']:>+6.0%}"
              f"{cn['p95']:>+6.0%}{cn['P_gt40']:>6.1%}{cn['P_gt60']:>6.1%}{cn['P_loss']:>6.1%}",
              flush=True)
    print("  NOTE: weights are NOT renormalized — with k<10 the 10% cap leaves the sleeve")
    print("        partly in CASH (deploy<100%), so concentrating to chase skew also dilutes.")
    print("  READ: under the 10% name cap + 2-month buy-hold, a basket of INDEPENDENT binaries")
    print("        cannot manufacture a >60% window — the cap ceilings any single hit at ~+7%")
    print("        and >=10 names (needed for full deployment) average the tail away. The sleeve")
    print("        is a REGIME-INDEPENDENT DIVERSIFIER (lifts the floor/median in a flat tape),")
    print("        NOT a standalone grand-slam. The >60% tail stays the CORE's job (illiquid")
    print("        multibaggers riding past the cap via appreciation-lock).", flush=True)

    out = dict(window=[str(WIN_START), str(WIN_END)],
               counts=dict(ct=len(ct), pdufa=len(pd_rows), ma=len(ma)),
               sleeve=[{k: r[k] for k in ("ticker", "company", "type", "date", "detail",
                        "p", "up", "down", "skew", "w")} for r in top],
               composition_sweep={int(k): v for k, v in sweep.items()})
    Path("results").mkdir(exist_ok=True)
    json.dump(out, open("results/pead_v10_catalyst.json", "w"), indent=2, default=float)
    print("\nwrote results/pead_v10_catalyst.json", flush=True)


if __name__ == "__main__":
    main()
