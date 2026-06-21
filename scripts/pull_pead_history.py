"""Pull multi-year CRSP daily + IBES SUE for the PEAD walk-forward (QUANT NOTE v2).

Point-in-time universe support: pull the top-N most-liquid US common stocks over
the whole span, then rebuild the Russell-2000-proxy band (market-cap rank
1001-3000) at each rebalance date in the backtest. Chunked by year so a dropped
connection only loses one year, not the whole pull.
"""
import os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from dotenv import load_dotenv
load_dotenv(".env")
import wrds

START, END = "2017-01-01", "2024-12-31"
UNIV_N = 4000           # liquidity support set; contains the 1001-3000 mktcap band
MIN_PRICE = 1.0
OUT_CRSP = "data_cache/pead_crsp_daily.parquet"
OUT_SUE  = "data_cache/pead_ibes_sue.parquet"


def select_support(db):
    sql = f"""
        select a.permno,
               percentile_cont(0.5) within group (order by abs(a.prc)*a.vol) as med_dvol
        from crsp.dsf a
        inner join crsp.dsenames b
          on a.permno=b.permno and b.namedt<=a.date and a.date<=b.nameendt
         and b.shrcd in (10,11)
        where a.date between '{START}' and '{END}'
          and a.prc is not null and abs(a.prc) >= {MIN_PRICE}
        group by a.permno
        order by med_dvol desc nulls last
        limit {UNIV_N}
    """
    return [int(p) for p in db.raw_sql(sql)["permno"].dropna()]


def pull_crsp_year(db, permnos, yr):
    pl = ",".join(map(str, permnos))
    sql = f"""
        select a.permno, a.date, a.openprc, a.prc, a.vol, a.ret,
               a.bid, a.ask, a.shrout, b.ticker, b.siccd, b.shrcd
        from crsp.dsf a
        left join crsp.dsenames b
          on a.permno=b.permno and b.namedt<=a.date and a.date<=b.nameendt
        where a.date between '{yr}-01-01' and '{yr}-12-31'
          and a.permno in ({pl})
    """
    return db.raw_sql(sql, date_cols=["date"])


def pull_sue(db, permnos):
    pl = ",".join(map(str, permnos))
    sql = f"""
        with surp as (
            select distinct on (a.ticker, a.pends)
                   a.cusip as ncusip, a.anndats as date,
                   (a.value - s.medest) / s.stdev as sue
            from ibes.actu_epsus a
            join ibes.statsumu_epsus s
              on s.ticker=a.ticker and s.fpedats=a.pends and s.fpi='6'
             and s.measure='EPS' and s.statpers<=a.anndats and s.stdev>0
            where a.measure='EPS' and a.pdicity='QTR'
              and a.anndats between '{START}' and '{END}'
            order by a.ticker, a.pends, s.statpers desc
        )
        select n.permno as asset, surp.date, surp.sue
        from surp
        join crsp.dsenames n
          on n.ncusip=surp.ncusip and n.namedt<=surp.date and surp.date<=n.nameendt
        where n.permno in ({pl})
    """
    return db.raw_sql(sql, date_cols=["date"]).dropna()[["date", "asset", "sue"]]


def main():
    t0 = time.time()
    db = wrds.Connection(wrds_username=os.environ["WRDS_USERNAME"])
    print("selecting liquidity support set...")
    permnos = select_support(db)
    print(f"  support set: {len(permnos)} permnos  ({(time.time()-t0)/60:.1f} min)")

    frames = []
    for yr in range(int(START[:4]), int(END[:4]) + 1):
        for attempt in range(3):
            try:
                d = pull_crsp_year(db, permnos, yr)
                frames.append(d)
                print(f"  {yr}: {len(d):>7} rows  (elapsed {(time.time()-t0)/60:.1f} min)")
                break
            except Exception as e:
                print(f"  {yr}: retry {attempt} ({repr(e)[:80]})")
                try: db.close()
                except Exception: pass
                db = wrds.Connection(wrds_username=os.environ["WRDS_USERNAME"])
    crsp = pd.concat(frames, ignore_index=True)
    for c in ["prc", "openprc", "bid", "ask"]:
        crsp[c] = crsp[c].abs()
    crsp["mktcap"] = crsp["prc"] * crsp["shrout"] * 1000.0
    crsp["ticker"] = crsp["ticker"].fillna(crsp["permno"].astype(str))
    crsp.to_parquet(OUT_CRSP)
    print(f"saved {OUT_CRSP}: {crsp.shape}, {crsp['permno'].nunique()} names, "
          f"{crsp['date'].min().date()}..{crsp['date'].max().date()}")

    print("pulling IBES SUE...")
    sue = pull_sue(db, permnos)
    sue.to_parquet(OUT_SUE)
    print(f"saved {OUT_SUE}: {len(sue)} events, {sue['asset'].nunique()} names")
    db.close()
    print(f"DONE in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
