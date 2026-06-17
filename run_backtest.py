#!/usr/bin/env python
"""CLI entry point.

Examples
--------
    # Validate the engine with no WRDS access (synthetic data):
    python run_backtest.py --synthetic

    # Real 3-year backtest from WRDS (uses .env credentials, parquet-cached):
    python run_backtest.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from src.config import REPO_ROOT, load_config
from src.pipeline import run
from src.report import metrics
from src.report.tearsheet import write_report


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    ap = argparse.ArgumentParser(description="Competition strategy backtester")
    ap.add_argument("--synthetic", action="store_true",
                    help="use generated data instead of WRDS (engine smoke test)")
    ap.add_argument("--no-cache", action="store_true", help="ignore parquet cache")
    ap.add_argument("--out", default=str(REPO_ROOT / "results"))
    args = ap.parse_args()

    cfg = load_config()

    if args.synthetic:
        from tests.synthetic import make_bundle
        data = make_bundle(cfg)
    else:
        from src.data.wrds_loader import load_data_bundle
        comp = cfg.competition
        data = load_data_bundle(
            cfg.start_date, cfg.end_date,
            universe_size=int(comp["universe_size"]),
            min_price=float(comp["min_price"]),
            use_cache=not args.no_cache,
        )

    result = run(cfg, data)
    stats = metrics.summary(result.returns, cfg.competition["risk_free_rate"])
    print(metrics.format_summary(stats, cfg.scoring_metric))

    report = write_report(
        result, Path(args.out),
        scoring_metric=cfg.scoring_metric,
        rf=cfg.competition["risk_free_rate"],
        title=cfg.competition["name"],
    )
    print(f"\nReport: {report}")


if __name__ == "__main__":
    main()
