"""Render an HTML tearsheet + PNG plots for a backtest result."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from . import metrics


def save_plots(returns: pd.Series, equity: pd.Series, out_dir: Path,
               benchmark: pd.Series | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    ax = axes[0]
    (equity / equity.iloc[0]).plot(ax=ax, label="Strategy", lw=1.4)
    if benchmark is not None and len(benchmark):
        bench_eq = (1.0 + benchmark.reindex(returns.index).fillna(0.0)).cumprod()
        bench_eq.plot(ax=ax, label="Benchmark", lw=1.0, alpha=0.7)
    ax.set_title("Cumulative growth of $1")
    ax.legend()
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    eq = (1.0 + returns).cumprod()
    dd = eq / eq.cummax() - 1.0
    dd.plot(ax=ax2, color="firebrick", lw=1.0)
    ax2.fill_between(dd.index, dd.values, 0, color="firebrick", alpha=0.3)
    ax2.set_title("Drawdown")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    png = out_dir / "performance.png"
    fig.savefig(png, dpi=120)
    plt.close(fig)
    return png


def write_report(
    result,
    out_dir: Path,
    *,
    scoring_metric: str = "total_return",
    rf: float = 0.0,
    benchmark: pd.Series | None = None,
    title: str = "Competition Backtest",
) -> Path:
    out_dir = Path(out_dir)
    stats = metrics.summary(result.returns, rf)
    png = save_plots(result.returns, result.equity, out_dir, benchmark)

    # quantstats tearsheet is optional (nice-to-have); fall back to our own.
    try:
        import quantstats as qs  # type: ignore

        qs.reports.html(
            result.returns,
            benchmark=benchmark,
            output=str(out_dir / "quantstats.html"),
            title=title,
        )
    except Exception as exc:  # pragma: no cover
        print(f"[report] quantstats tearsheet skipped ({exc}).")

    rows = "".join(
        f"<tr><td>{k}</td><td>{v:.4f}</td>"
        f"<td>{'SCORING' if k == scoring_metric else ''}</td></tr>"
        for k, v in stats.items()
    )
    html = f"""<html><head><meta charset="utf-8"><title>{title}</title></head>
<body style="font-family:system-ui;max-width:900px;margin:2em auto">
<h1>{title}</h1>
<p>Start equity: {result.starting_capital:,.0f} &nbsp;
   End equity: {result.equity.iloc[-1]:,.0f} &nbsp;
   <b>Total return: {stats['total_return']:.2%}</b></p>
<table border="1" cellpadding="6" cellspacing="0">
<tr><th>metric</th><th>value</th><th></th></tr>{rows}</table>
<img src="performance.png" style="max-width:100%;margin-top:1em">
</body></html>"""
    report = out_dir / "report.html"
    report.write_text(html, encoding="utf-8")
    return report
