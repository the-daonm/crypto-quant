"""Pipeline end-to-end: dữ liệu → feature → walk-forward → backtest → metrics.

Chạy: python -m cq.pipeline --symbol BTC/USDT --tf 1d
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from cq.backtest import backtest
from cq.data import load_ohlcv
from cq.features import build_features
from cq.labels import make_dataset
from cq.metrics import PERIODS_PER_YEAR, classification_metrics, trading_metrics
from cq.models import MODEL_NAMES
from cq.validation import WalkForwardResult, run_walk_forward

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"


def load_config(path: Path = ROOT / "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@dataclass
class Experiment:
    model: str
    wf: WalkForwardResult
    bt: pd.DataFrame
    metrics: dict


def run_experiment(
    ohlcv: pd.DataFrame,
    timeframe: str,
    model: str,
    cfg: dict,
    threshold: float | str | None = None,
    fee: float | None = None,
) -> Experiment:
    fee = cfg["fee"] if fee is None else fee
    threshold = cfg["threshold"] if threshold is None else threshold
    wf_cfg = cfg["walk_forward"]

    X, y, fwd = make_dataset(build_features(ohlcv), ohlcv["close"])
    wf = run_walk_forward(
        model, X, y, fwd,
        train_min=wf_cfg["train_min"][timeframe],
        test_size=wf_cfg["test_size"][timeframe],
        gap=wf_cfg["gap"],
        threshold=threshold,
        fee=fee,
    )
    p = wf.predictions
    bt = backtest(p["proba"], p["fwd_ret"], threshold=p["threshold"], fee=fee)

    ppy = PERIODS_PER_YEAR[timeframe]
    metrics = {
        "model": model,
        **classification_metrics(p["y"], p["proba"]),
        **trading_metrics(bt["strat_ret"], ppy, bt["trades"], bt["position"]),
    }
    return Experiment(model, wf, bt, metrics)


def buy_and_hold_metrics(bt: pd.DataFrame, timeframe: str) -> dict:
    return {"model": "buy_and_hold", **trading_metrics(bt["bh_ret"], PERIODS_PER_YEAR[timeframe])}


def run_all(symbol: str, timeframe: str, cfg: dict, threshold=None, fee=None,
            models=MODEL_NAMES, refresh=False) -> tuple[pd.DataFrame, dict[str, Experiment]]:
    ohlcv = load_ohlcv(symbol, timeframe, cfg["since"], cfg["exchange"], refresh=refresh)
    exps = {m: run_experiment(ohlcv, timeframe, m, cfg, threshold, fee) for m in models}
    any_bt = next(iter(exps.values())).bt
    rows = [buy_and_hold_metrics(any_bt, timeframe)] + [e.metrics for e in exps.values()]
    return pd.DataFrame(rows).set_index("model"), exps


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default=cfg["symbols"][0])
    ap.add_argument("--tf", default=cfg["timeframes"][0], choices=list(PERIODS_PER_YEAR))
    ap.add_argument("--threshold", default=None, help="số (vd 0.55) hoặc 'auto'")
    ap.add_argument("--fee", type=float, default=None)
    ap.add_argument("--refresh", action="store_true", help="tải lại dữ liệu, bỏ qua cache")
    args = ap.parse_args()

    thr = args.threshold
    if thr is not None and thr != "auto":
        thr = float(thr)

    summary, exps = run_all(args.symbol, args.tf, cfg, thr, args.fee, refresh=args.refresh)
    start, end = next(iter(exps.values())).bt.index[[0, -1]]
    print(f"\n{args.symbol} {args.tf} | out-of-sample {start.date()} → {end.date()} "
          f"| threshold={thr if thr is not None else cfg['threshold']} fee={args.fee or cfg['fee']}\n")
    with pd.option_context("display.float_format", "{:,.3f}".format, "display.width", 200, "display.max_columns", None):
        print(summary)

    RESULTS_DIR.mkdir(exist_ok=True)
    tag = f"{args.symbol.replace('/', '-')}_{args.tf}"
    summary.to_csv(RESULTS_DIR / f"{tag}_summary.csv")
    for name, e in exps.items():
        e.bt.join(e.wf.predictions[["proba", "y", "fold", "threshold"]]).to_csv(
            RESULTS_DIR / f"{tag}_{name}_backtest.csv"
        )
    print(f"\nĐã lưu kết quả vào {RESULTS_DIR}/{tag}_*.csv")


if __name__ == "__main__":
    main()
