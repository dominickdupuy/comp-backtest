"""Pull market data from UPenn WRDS and assemble a :class:`DataBundle`.

Sources
-------
* CRSP daily stock file (``crsp.dsf``) -> prices, returns, volume, shares,
  open prices, market cap.
* CRSP names (``crsp.dsenames``) -> ticker + SIC (sector) per permno.
* IBES (``ibes.statsumu_epsus`` + ``ibes.actpsumu_epsus``) -> earnings surprise
  (SUE) for the PEAD signal.

The exact table/column availability depends on your WRDS subscription. Each
pull is wrapped so that a missing optional source (e.g. IBES) degrades
gracefully: the dependent signal simply contributes zero rather than crashing.
All pulls are parquet-cached keyed by date range, so WRDS is hit once.

Credentials come from the environment (.env): WRDS_USERNAME / WRDS_PASSWORD.
"""
from __future__ import annotations

import os
from datetime import date

import numpy as np
import pandas as pd

from ..signals.base import DataBundle
from . import cache

try:  # wrds is optional at import time (e.g. for offline/synthetic runs)
    import wrds  # type: ignore
except Exception:  # pragma: no cover
    wrds = None


def _sic_to_sector(sic: float | int | None) -> str:
    """Coarse SIC -> sector bucket for neutralization."""
    try:
        s = int(sic)
    except (TypeError, ValueError):
        return "UNK"
    buckets = [
        (1, 999, "AGRIC"), (1000, 1499, "MINE"), (1500, 1799, "CONSTR"),
        (2000, 3999, "MFG"), (4000, 4999, "UTIL_TRANSP"), (5000, 5199, "WHOLE"),
        (5200, 5999, "RETAIL"), (6000, 6799, "FIN"), (7000, 8999, "SERVICES"),
    ]
    for lo, hi, label in buckets:
        if lo <= s <= hi:
            return label
    return "OTHER"


def connect() -> "wrds.Connection":
    if wrds is None:
        raise RuntimeError(
            "The 'wrds' package is not installed. `pip install wrds`."
        )
    user = os.environ.get("WRDS_USERNAME")
    if not user:
        raise RuntimeError("WRDS_USERNAME not set (see .env.example).")
    # wrds reads WRDS_PASSWORD from env / .pgpass automatically.
    return wrds.Connection(wrds_username=user)


def _select_universe(db, start: date, end: date, universe_size: int,
                     min_price: float) -> list[int]:
    """Pick the most-liquid `universe_size` permnos via an aggregated SQL pass.

    Done server-side so we never transfer all of CRSP just to rank liquidity.
    Restricted to ordinary common shares (shrcd 10/11).
    """
    sql = f"""
        select a.permno,
               percentile_cont(0.5) within group (
                   order by abs(a.prc) * a.vol) as med_dvol
        from crsp.dsf a
        inner join crsp.dsenames b
          on a.permno = b.permno
         and b.namedt <= a.date and a.date <= b.nameendt
         and b.shrcd in (10, 11)
        where a.date between '{start}' and '{end}'
          and a.prc is not null
          and abs(a.prc) >= {float(min_price)}
        group by a.permno
        order by med_dvol desc nulls last
        limit {int(universe_size)}
    """
    perm = db.raw_sql(sql)
    return [int(p) for p in perm["permno"].dropna().tolist()]


def _pull_crsp(db, start: date, end: date, universe_size: int,
               min_price: float) -> pd.DataFrame:
    """Daily CRSP panel (long format) for the most-liquid `universe_size` names."""
    permnos = _select_universe(db, start, end, universe_size, min_price)
    if not permnos:
        raise RuntimeError("Universe selection returned no permnos.")
    perm_list = ",".join(str(p) for p in permnos)
    sql = f"""
        select a.permno, a.date, a.prc, a.openprc, a.ret, a.vol, a.shrout,
               b.ticker, b.siccd
        from crsp.dsf a
        left join crsp.dsenames b
          on a.permno = b.permno
         and b.namedt <= a.date
         and a.date <= b.nameendt
        where a.date between '{start}' and '{end}'
          and a.permno in ({perm_list})
    """
    df = db.raw_sql(sql, date_cols=["date"])
    df["prc"] = df["prc"].abs()                       # negative = bid/ask avg
    df["openprc"] = df["openprc"].abs()
    df["mktcap"] = df["prc"] * df["shrout"] * 1000.0  # shrout in thousands
    df["dollar_vol"] = df["prc"] * df["vol"]
    df["ticker"] = df["ticker"].fillna(df["permno"].astype(str))
    return df


def _pull_ibes_sue(db, start: date, end: date, permnos: list[int]) -> pd.DataFrame:
    """Earnings surprise (SUE) events keyed to CRSP permno.

    SUE = (actual quarterly EPS - latest median estimate) / dispersion of
    estimates, using the most recent consensus *before* the announcement.
    IBES is linked to CRSP via historical CUSIP (IBES.cusip = CRSP.ncusip),
    filtered to our universe. Returns columns [date, asset, sue] where asset is
    the permno (the panel's column key).
    """
    perm_list = ",".join(str(p) for p in permnos)
    sql = f"""
        with surp as (
            select distinct on (a.ticker, a.pends)
                   a.cusip as ncusip, a.anndats as date,
                   (a.value - s.medest) / s.stdev as sue
            from ibes.actu_epsus a
            join ibes.statsumu_epsus s
              on s.ticker = a.ticker
             and s.fpedats = a.pends
             and s.fpi = '6'
             and s.measure = 'EPS'
             and s.statpers <= a.anndats
             and s.stdev > 0
            where a.measure = 'EPS'
              and a.pdicity = 'QTR'
              and a.anndats between '{start}' and '{end}'
            order by a.ticker, a.pends, s.statpers desc
        )
        select n.permno as asset, surp.date, surp.sue
        from surp
        join crsp.dsenames n
          on n.ncusip = surp.ncusip
         and n.namedt <= surp.date and surp.date <= n.nameendt
        where n.permno in ({perm_list})
    """
    df = db.raw_sql(sql, date_cols=["date"])
    return df.dropna()[["date", "asset", "sue"]]


def load_data_bundle(
    start: date,
    end: date,
    *,
    universe_size: int = 8000,
    min_price: float = 1.0,
    use_cache: bool = True,
    want_earnings: bool = True,
) -> DataBundle:
    """Top-level loader. Returns a wide-panel DataBundle, cached to parquet."""
    span = (str(start), str(end), str(universe_size), str(min_price))
    panel = cache.load("crsp_panel", *span) if use_cache else None

    if panel is None:
        db = connect()
        try:
            panel = _pull_crsp(db, start, end, universe_size, min_price)
            if use_cache:
                cache.save(panel, "crsp_panel", *span)
            earnings = None
            if want_earnings:
                try:
                    permnos = [int(p) for p in panel["permno"].unique()]
                    earnings = _pull_ibes_sue(db, start, end, permnos)
                    cache.save(earnings, "ibes_sue", *span[:2])
                except Exception as exc:  # pragma: no cover
                    print(f"[wrds] IBES SUE unavailable ({exc}); PEAD -> 0.")
        finally:
            db.close()
    else:
        earnings = cache.load("ibes_sue", *span[:2]) if want_earnings else None

    return _panel_to_bundle(panel, earnings)


def _panel_to_bundle(panel: pd.DataFrame, earnings: pd.DataFrame | None) -> DataBundle:
    # Key the panel by permno (unique, stable) rather than ticker (reused over
    # time). A permno->ticker name map is kept in meta for readable reporting.
    def pivot(col: str) -> pd.DataFrame:
        return panel.pivot_table(index="date", columns="permno", values=col)

    close = pivot("prc").sort_index()
    ret = pivot("ret").sort_index()
    # Fall back to pct_change where CRSP ret is missing.
    ret = ret.where(ret.notna(), close.pct_change(fill_method=None))

    sector = (
        panel.dropna(subset=["siccd"])
        .groupby("permno")["siccd"].last()
        .map(_sic_to_sector)
    )
    names = panel.groupby("permno")["ticker"].last()

    return DataBundle(
        close=close,
        ret=ret,
        open=pivot("openprc").sort_index(),
        volume=pivot("vol").sort_index(),
        market_cap=pivot("mktcap").sort_index(),
        sector=sector,
        earnings=earnings,
        meta={"universe_size": close.shape[1], "names": names.to_dict()},
    )
