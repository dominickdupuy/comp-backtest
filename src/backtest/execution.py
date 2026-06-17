"""Delayed-execution model using minute bars.

The competition imposes a 15-20 minute data+execution delay: you act on a
signal computed at time T but your order fills ~20 min later. At daily horizon
this is negligible, but where minute bars exist for a traded name we can price
the fill exactly: fill at the close of the (T + delay) minute bar rather than
the daily close. This module computes that delayed fill price so a strategy's
realized entry/exit can be compared against the idealized daily-close fill.
"""
from __future__ import annotations

import pandas as pd


def minute_panel(minute_df: pd.DataFrame, field: str = "close") -> pd.DataFrame:
    """Pivot long minute bars to (timestamp x symbol) for one OHLCV field."""
    df = minute_df.copy()
    ts = pd.to_datetime(df["date"].astype(str)) + (
        pd.to_datetime(df["minute"].astype(str)).dt.time.map(_as_timedelta)
    )
    df["ts"] = ts
    return df.pivot_table(index="ts", columns="symbol", values=field)


def _as_timedelta(t) -> pd.Timedelta:
    return pd.Timedelta(hours=t.hour, minutes=t.minute, seconds=t.second)


def delayed_fill_price(
    minute_close: pd.DataFrame,
    decision_time: str = "16:00",
    delay_minutes: int = 20,
) -> pd.DataFrame:
    """Fill price per day = the minute-bar close `delay_minutes` after the
    decision time. Returns a (date x symbol) frame of fill prices.

    If the decision is at the session close (16:00), the realistic fill is the
    next session's open-plus-delay; here we approximate with the first
    available bar at/after (decision_time + delay) on the same day, falling back
    to the last bar of the day.
    """
    out = {}
    target = pd.Timedelta(decision_time + ":00") + pd.Timedelta(minutes=delay_minutes)
    by_day = minute_close.groupby(minute_close.index.normalize())
    for day, block in by_day:
        tod = block.index - day
        at = block[tod >= target]
        row = at.iloc[0] if len(at) else block.iloc[-1]
        out[day] = row
    return pd.DataFrame(out).T
