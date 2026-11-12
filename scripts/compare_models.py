"""Compare all models on the design period only (issue #4).

The OHLCV data is cut at `holdout_start` *before* anything is fitted, so the holdout period is
never touched. Each model is run once with its fixed configuration from `cq.models.make_model`.

Usage: python scripts/compare_models.py            # writes results/model_comparison*.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cq.data import load_ohlcv
from cq.models import MODEL_NAMES
from cq.pipeline import buy_and_hold_metrics, load_config, run_experiment

COLUMNS = ["auc", "accuracy", "sharpe", "max_drawdown", "total_return", "n_trades", "fit_seconds"]


def design_period_rows(symbol: str, timeframe: str, cfg: dict) -> list[dict]:
    ohlcv = load_ohlcv(symbol, timeframe, cfg["since"], cfg["exchange"])
    ohlcv = ohlcv[ohlcv.index < pd.Timestamp(cfg["holdout_start"], tz="UTC")]
    rows = []
    for model in MODEL_NAMES:
        e = run_experiment(ohlcv, timeframe, model, cfg)
        if not rows:
            rows.append({"symbol": symbol, "timeframe": timeframe, **buy_and_hold_metrics(e.bt, timeframe)})
        fit_seconds = float(e.wf.folds["fit_seconds"].sum())
        rows.append({"symbol": symbol, "timeframe": timeframe, **e.metrics, "fit_seconds": fit_seconds})
    return rows


def main() -> None:
    cfg = load_config()
    rows = [r for s in cfg["symbols"] for tf in cfg["timeframes"] for r in design_period_rows(s, tf, cfg)]
    per_pair = pd.DataFrame(rows).set_index(["symbol", "timeframe", "model"])[COLUMNS]
    order = ["buy_and_hold", *MODEL_NAMES]
    average = per_pair.groupby("model").mean().loc[order]

    out = Path("results")
    out.mkdir(exist_ok=True)
    per_pair.to_csv(out / "model_comparison_per_pair.csv")
    average.to_csv(out / "model_comparison.csv")
    with pd.option_context("display.float_format", "{:,.3f}".format, "display.width", 200):
        print(f"Design period (< {cfg['holdout_start']}), average over {len(per_pair) // len(order)} pairs\n")
        print(average)
        print("\nPer pair\n")
        print(per_pair)


if __name__ == "__main__":
    main()
