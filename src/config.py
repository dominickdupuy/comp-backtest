"""Load and validate the competition + strategy configuration.

All competition rules and strategy parameters live in YAML under ``config/``.
Nothing about the rules is hard-coded in the engine: it reads everything from
the :class:`Config` object returned here.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "config"


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@dataclass
class Config:
    """Merged, validated view of competition.yaml + strategies.yaml."""

    competition: dict[str, Any]
    strategies: dict[str, Any]

    # Resolved window
    start_date: date = field(init=False)
    end_date: date = field(init=False)

    def __post_init__(self) -> None:
        self._validate()
        self.start_date, self.end_date = self._resolve_window()

    # -- convenience accessors -------------------------------------------------
    @property
    def signal_lag_days(self) -> int:
        return int(self.competition.get("signal_lag_days", 1))

    @property
    def max_gross_leverage(self) -> float:
        return float(self.competition["max_gross_leverage"])

    @property
    def max_position_weight(self) -> float:
        return float(self.competition["max_position_weight"])

    @property
    def allow_shorting(self) -> bool:
        return bool(self.competition["allow_shorting"])

    @property
    def commission_per_trade(self) -> float:
        return float(self.competition.get("commission_per_trade", 0.0))

    @property
    def slippage_bps(self) -> float:
        return float(self.competition.get("slippage_bps", 0.0))

    @property
    def scoring_metric(self) -> str:
        return str(self.competition.get("scoring_metric", "total_return"))

    @property
    def enabled_signals(self) -> dict[str, dict[str, Any]]:
        return {
            name: spec
            for name, spec in self.strategies["signals"].items()
            if spec.get("enabled")
        }

    # -- internals -------------------------------------------------------------
    def _resolve_window(self) -> tuple[date, date]:
        bt = self.strategies["backtest"]
        end = bt.get("end_date")
        end_d = _parse_date(end) if end else date.today()
        start = bt.get("start_date")
        if start:
            start_d = _parse_date(start)
        else:
            years = int(bt.get("lookback_years", 3))
            start_d = end_d - timedelta(days=int(years * 365.25))
        if start_d >= end_d:
            raise ValueError(f"start_date {start_d} must precede end_date {end_d}")
        return start_d, end_d

    def _validate(self) -> None:
        c = self.competition
        # Weights of enabled signals must be positive.
        total = sum(
            float(s.get("blend_weight", 0.0)) for s in self.enabled_signals.values()
        )
        if total <= 0:
            raise ValueError("No enabled signals with positive blend_weight.")
        if c["max_position_weight"] <= 0 or c["max_position_weight"] > 1:
            raise ValueError("max_position_weight must be in (0, 1].")
        if c["max_gross_leverage"] <= 0:
            raise ValueError("max_gross_leverage must be > 0.")
        if not c["allow_shorting"] and self.strategies["construction"]["long_short"]:
            raise ValueError(
                "Strategy is long/short but competition forbids shorting."
            )


def _parse_date(value: str | date) -> date:
    if isinstance(value, date):
        return value
    return datetime.strptime(str(value), "%Y-%m-%d").date()


def load_config(
    competition_path: Path | None = None,
    strategies_path: Path | None = None,
) -> Config:
    competition_path = competition_path or CONFIG_DIR / "competition.yaml"
    strategies_path = strategies_path or CONFIG_DIR / "strategies.yaml"
    return Config(
        competition=_load_yaml(competition_path),
        strategies=_load_yaml(strategies_path),
    )
