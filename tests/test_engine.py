"""End-to-end + unit tests that run with no WRDS access."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.constraints import apply_constraints
from src.backtest.engine import run_backtest
from src.config import load_config
from src.pipeline import run
from tests.synthetic import make_bundle


def test_constraints_respect_caps():
    idx = pd.bdate_range("2024-01-01", periods=3)
    cols = [f"S{i}" for i in range(10)]
    w = pd.DataFrame(np.random.randn(3, 10), index=idx, columns=cols)
    out = apply_constraints(
        w, max_position_weight=0.10, target_gross_leverage=2.0,
        max_gross_leverage=2.0, max_net_leverage=2.0, allow_shorting=True,
    )
    assert (out.abs() <= 0.10 + 1e-9).all().all(), "per-name cap violated"
    gross = out.abs().sum(axis=1)
    assert (gross <= 2.0 + 1e-6).all(), "gross leverage cap violated"


def test_no_shorting_toggle():
    idx = pd.bdate_range("2024-01-01", periods=2)
    cols = ["A", "B", "C"]
    w = pd.DataFrame([[1, -1, 0.5], [-2, 1, 1]], index=idx, columns=cols)
    out = apply_constraints(
        w, max_position_weight=1.0, target_gross_leverage=1.0,
        max_gross_leverage=1.0, max_net_leverage=1.0, allow_shorting=False,
    )
    assert (out >= -1e-12).all().all(), "shorts present when shorting disabled"


def test_engine_runs_and_is_lagged():
    bundle = make_bundle(n_names=30)
    # Equal long book on every day; check no lookahead (day-0 weight is 0).
    w = pd.DataFrame(1.0 / 30, index=bundle.close.index, columns=bundle.close.columns)
    res = run_backtest(w, bundle.ret, signal_lag_days=1, allow_shorting=False,
                       target_gross_leverage=1.0, max_gross_leverage=1.0,
                       max_net_leverage=1.0, max_position_weight=1.0)
    assert res.weights.iloc[0].abs().sum() == 0.0, "first day should be flat (lag)"
    assert len(res.returns) == len(bundle.ret)
    assert np.isfinite(res.total_return)


def test_full_pipeline_synthetic():
    cfg = load_config()
    bundle = make_bundle(cfg, n_names=120)
    res = run(cfg, bundle)
    assert len(res.returns) > 100
    # Gross leverage should be at/under the 2x cap on active days.
    active = res.gross_leverage[res.gross_leverage > 0]
    assert (active <= 2.0 + 1e-6).all()
    assert np.isfinite(res.total_return)
