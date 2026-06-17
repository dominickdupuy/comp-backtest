"""Performance metrics. total_return is the competition's scoring metric;
the rest are reported for context."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def total_return(returns: pd.Series) -> float:
    return float((1.0 + returns).prod() - 1.0)


def cagr(returns: pd.Series) -> float:
    n = len(returns)
    if n == 0:
        return 0.0
    growth = (1.0 + returns).prod()
    if growth <= 0:
        return -1.0
    return float(growth ** (TRADING_DAYS / n) - 1.0)


def ann_vol(returns: pd.Series) -> float:
    return float(returns.std() * np.sqrt(TRADING_DAYS))


def sharpe(returns: pd.Series, rf: float = 0.0) -> float:
    excess = returns - rf / TRADING_DAYS
    sd = excess.std()
    if sd == 0:
        return 0.0
    return float(excess.mean() / sd * np.sqrt(TRADING_DAYS))


def sortino(returns: pd.Series, rf: float = 0.0) -> float:
    excess = returns - rf / TRADING_DAYS
    downside = excess[excess < 0].std()
    if downside == 0 or np.isnan(downside):
        return 0.0
    return float(excess.mean() / downside * np.sqrt(TRADING_DAYS))


def max_drawdown(returns: pd.Series) -> float:
    equity = (1.0 + returns).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def calmar(returns: pd.Series) -> float:
    mdd = abs(max_drawdown(returns))
    if mdd == 0:
        return 0.0
    return float(cagr(returns) / mdd)


def summary(returns: pd.Series, rf: float = 0.0) -> dict[str, float]:
    return {
        "total_return": total_return(returns),
        "cagr": cagr(returns),
        "ann_vol": ann_vol(returns),
        "sharpe": sharpe(returns, rf),
        "sortino": sortino(returns, rf),
        "calmar": calmar(returns),
        "max_drawdown": max_drawdown(returns),
    }


def format_summary(stats: dict[str, float], scoring_metric: str = "total_return") -> str:
    lines = ["", "Performance summary", "-" * 40]
    for k, v in stats.items():
        marker = "  <-- SCORING" if k == scoring_metric else ""
        if k in ("ann_vol", "total_return", "cagr", "max_drawdown"):
            lines.append(f"  {k:<14} {v:>10.2%}{marker}")
        else:
            lines.append(f"  {k:<14} {v:>10.2f}{marker}")
    return "\n".join(lines)
