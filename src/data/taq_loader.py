"""Minute-bar loader built from WRDS TAQ Millisecond (taqm_YYYY).

The competition's 15-20 min delay means sub-daily data adds no *signal* value
(see the strategy research), so minute bars are pulled ONLY for the relevant
subset of names the strategies actually trade -- to (a) model the delayed
execution price precisely and (b) support any intraday-execution sleeve.

TAQ Millisecond stores one consolidated-trades table per day,
``taqm_<YYYY>.ctm_<YYYYMMDD>``, keyed by ``sym_root``. We aggregate trades to
1-minute OHLCV server-side (regular session 09:30-16:00) and cache one parquet
per (symbol-set, month) so a pull is done once.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from . import cache


def _sym_key(symbols: list[str]) -> str:
    symbols = sorted(set(symbols))
    return str(len(symbols)) + ":" + ",".join(symbols)


def load_cached_minute(
    symbols: list[str],
    start: date,
    end: date,
    *,
    field: str = "close",
    freq: str | None = "30min",
) -> pd.DataFrame:
    """Read cached monthly minute parquets -> wide (ts x symbol) panel.

    Only months already present in the cache are loaded (so this works while a
    bulk pull is still running). If ``freq`` is set, intraday bars are resampled
    to that frequency *within each day* (overnight gaps are not bridged).
    """
    symbols = sorted(set(symbols))
    key = _sym_key(symbols)
    months = pd.period_range(start, end, freq="M")
    wides = []
    for m in months:
        mdf = cache.load("taq_minute", f"{m}", key)
        if mdf is None or mdf.empty:
            continue
        wide = mdf.pivot_table(index="ts", columns="symbol", values=field)
        if freq:
            wide = (
                wide.groupby(wide.index.normalize())
                .resample(freq, level=0).last()
                .droplevel(0)
            )
        wides.append(wide)
    if not wides:
        return pd.DataFrame()
    out = pd.concat(wides).sort_index()
    return out[~out.index.duplicated(keep="last")]


def cached_months(symbols: list[str], start: date, end: date) -> list[str]:
    """Which months are already in the cache for this symbol set."""
    key = _sym_key(symbols)
    found = []
    for m in pd.period_range(start, end, freq="M"):
        if cache.load("taq_minute", f"{m}", key) is not None:
            found.append(str(m))
    return found


def _trading_days(db, start: date, end: date) -> list[date]:
    """Trading days that actually have a TAQ trades table, from CRSP calendar."""
    sql = f"""
        select distinct date from crsp.dsf
        where date between '{start}' and '{end}'
        order by date
    """
    d = db.raw_sql(sql, date_cols=["date"])
    return [x.date() for x in d["date"]]


def _minute_sql(schema_day: str, symbols: list[str]) -> str:
    syms = ",".join("'" + s.replace("'", "") + "'" for s in symbols)
    return f"""
        select sym_root as symbol,
               date_trunc('minute', time_m) as minute,
               (array_agg(price order by time_m asc))[1]  as open,
               max(price) as high,
               min(price) as low,
               (array_agg(price order by time_m desc))[1] as close,
               sum(size)  as volume
        from {schema_day}
        where sym_root in ({syms})
          and price > 0
          and time_m between time '09:30:00' and time '16:00:00'
        group by sym_root, date_trunc('minute', time_m)
    """


def pull_minute_bars(
    symbols: list[str],
    start: date,
    end: date,
    *,
    use_cache: bool = True,
    db=None,
    verbose: bool = True,
    accumulate: bool = True,
) -> pd.DataFrame | None:
    """Return long-format minute OHLCV for ``symbols`` over [start, end].

    Columns: [date, ts, minute, symbol, open, high, low, close, volume].
    Cached per calendar month so reruns are cheap and a crash is resumable.

    For bulk multi-year pulls pass ``accumulate=False``: the function only
    populates the per-month parquet cache and returns None, avoiding holding
    100M+ rows in memory at once. The intraday engine then reads months lazily.
    """
    from .wrds_loader import connect

    symbols = sorted(set(symbols))
    own_db = db is None
    if own_db:
        db = connect()
    try:
        days = _trading_days(db, start, end)
        frames: list[pd.DataFrame] = []
        # Group by month for cache granularity.
        by_month: dict[str, list[date]] = {}
        for d in days:
            by_month.setdefault(f"{d:%Y-%m}", []).append(d)

        # Full symbol list (sha1-hashed by the cache) -> collision-proof key.
        sym_key = str(len(symbols)) + ":" + ",".join(symbols)
        for month, mdays in by_month.items():
            cached = cache.load("taq_minute", month, sym_key) if use_cache else None
            if cached is not None:
                if accumulate:
                    frames.append(cached)
                if verbose:
                    print(f"[taq] {month} cached ({len(cached)} rows)", flush=True)
                continue
            month_frames = []
            for d in mdays:
                schema_day = f"taqm_{d:%Y}.ctm_{d:%Y%m%d}"
                try:
                    df = db.raw_sql(_minute_sql(schema_day, symbols))
                except Exception as exc:  # missing day table (holiday) etc.
                    if verbose:
                        print(f"[taq] skip {d} ({str(exc)[:60]})", flush=True)
                    continue
                if not df.empty:
                    df["date"] = pd.Timestamp(d)
                    # `minute` comes back as a time-of-day timedelta; combine
                    # with the date into a single intraday timestamp.
                    df["ts"] = df["date"] + pd.to_timedelta(df["minute"])
                    month_frames.append(df)
            if month_frames:
                mdf = pd.concat(month_frames, ignore_index=True)
                if use_cache:
                    cache.save(mdf, "taq_minute", month, sym_key)
                if accumulate:
                    frames.append(mdf)
                if verbose:
                    print(f"[taq] {month} pulled ({len(mdf)} rows)", flush=True)
                del month_frames, mdf
        if not accumulate:
            return None
        if not frames:
            return pd.DataFrame(
                columns=["date", "minute", "symbol", "open", "high",
                         "low", "close", "volume"]
            )
        out = pd.concat(frames, ignore_index=True)
        return out.sort_values(["symbol", "date", "minute"]).reset_index(drop=True)
    finally:
        if own_db:
            db.close()
