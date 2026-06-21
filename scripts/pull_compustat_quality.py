"""Pull Compustat fundamentals for the v4 quality core (Workstream B).

Novy-Marx gross profitability GP/A = (revt - cogs) / at, plus net issuance
(buyback) inputs, linked CRSP permno via the CCM link table. Annual funda, so
the resulting signal rebalances slowly -> the 15-20 min fill delay is irrelevant.
"""
import os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from dotenv import load_dotenv
load_dotenv(".env")
import wrds

OUT = "data_cache/compustat_quality.parquet"


def main():
    t0 = time.time()
    crsp = pd.read_parquet("data_cache/pead_crsp_daily.parquet", columns=["permno"])
    perms = sorted(int(x) for x in crsp["permno"].unique())
    pl = ",".join(map(str, perms))
    db = wrds.Connection(wrds_username=os.environ["WRDS_USERNAME"])
    # funda linked to permno via CCM; one row per gvkey-fiscal year.
    sql = f"""
        select l.lpermno as permno, f.gvkey, f.datadate, f.fyear,
               f.revt, f.cogs, f.at, f.prstkc, f.sstk, f.csho, f.prcc_f, f.ceq
        from comp.funda f
        join crsp.ccmxpf_lnkhist l
          on f.gvkey = l.gvkey
         and l.linktype in ('LC','LU') and l.linkprim in ('P','C')
         and f.datadate >= l.linkdt
         and (l.linkenddt is null or f.datadate <= l.linkenddt)
        where f.indfmt='INDL' and f.datafmt='STD' and f.popsrc='D' and f.consol='C'
          and f.datadate between '2015-01-01' and '2024-12-31'
          and f.at > 0 and f.revt is not null and f.cogs is not null
          and l.lpermno in ({pl})
    """
    df = db.raw_sql(sql, date_cols=["datadate"])
    db.close()
    df["gpa"] = (df["revt"] - df["cogs"]) / df["at"]
    # net issuance proxy: (sstk - prstkc) / market cap; negative => net buyback
    df["net_issue"] = (df["sstk"].fillna(0) - df["prstkc"].fillna(0)) / (
        df["csho"] * df["prcc_f"]).replace(0, pd.NA)
    df = df.dropna(subset=["gpa"]).sort_values(["permno", "datadate"])
    df.to_parquet(OUT)
    print(f"saved {OUT}: {len(df)} firm-years, {df['permno'].nunique()} names, "
          f"{df['datadate'].min().date()}..{df['datadate'].max().date()} in "
          f"{(time.time()-t0)/60:.1f} min")
    print("gpa describe:"); print(df["gpa"].describe())


if __name__ == "__main__":
    main()
