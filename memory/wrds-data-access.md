---
name: wrds-data-access
description: WRDS access for comp-backtest — credentials, tables, pull speeds, Russell proxy
metadata:
  type: reference
---

WRDS works from this machine. Creds in `.env` (WRDS_USERNAME/WRDS_PASSWORD,
user `dominickdupuy`). `wrds` pip package installed. Connect:
`wrds.Connection(wrds_username=os.environ["WRDS_USERNAME"])`.

**Tables used:** `crsp.dsf` (daily OHLCV — has `openprc`, `prc`, `vol`, `ret`,
`bid`, `ask`, `bidlo`, `askhi`, `shrout`; prc/bid/ask come negative when a
bid-ask average — take abs). `crsp.dsenames` (ticker, siccd, shrcd; common =
shrcd 10/11). `ibes.actu_epsus` + `ibes.statsumu_epsus` for SUE.

**Speeds:** daily CRSP for ~4000 names is fast — 8 years pulled in ~4 min when
**chunked by year** (one big query drops the connection). The liquidity-ranking
`percentile_cont` aggregate is the slow part otherwise. **TAQ minute** bars are
brutally slow (hours, frequent disconnects) — avoid unless required.

**No Russell 2000 constituent table** in this subscription. Proxy used: US common
stocks ranked by market cap, take **rank 1001–3000** (point-in-time, monthly).
Band ≈ $84mm–$3.7bn = correct Russell 2000 range.

Related: [[pead-strategy-finding]].
