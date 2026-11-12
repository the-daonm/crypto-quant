"""Metric cho phân loại và cho chiến lược giao dịch."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

PERIODS_PER_YEAR = {"1d": 365, "4h": 365 * 6, "1h": 365 * 24}


def classification_metrics(y: pd.Series, proba: pd.Series, threshold: float = 0.5) -> dict:
    pred = (proba > threshold).astype(int)
    # AUC vô nghĩa khi proba là hằng số (baseline) hoặc y chỉ có một lớp.
    auc = roc_auc_score(y, proba) if proba.nunique() > 1 and y.nunique() > 1 else float("nan")
    return {
        "accuracy": accuracy_score(y, pred),
        "auc": auc,
        "f1": f1_score(y, pred, zero_division=0),
        "up_rate": float(y.mean()),
    }


def max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1).min())


def trading_metrics(
    log_ret: pd.Series, periods_per_year: int, trades: pd.Series | None = None, position: pd.Series | None = None
) -> dict:
    equity = np.exp(log_ret.cumsum())
    years = len(log_ret) / periods_per_year
    simple = np.expm1(log_ret)
    sd = simple.std()
    out = {
        "total_return": float(equity.iloc[-1] - 1),
        "cagr": float(equity.iloc[-1] ** (1 / years) - 1) if years > 0 else float("nan"),
        "sharpe": float(simple.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else float("nan"),
        "max_drawdown": max_drawdown(equity),
    }
    if position is not None:
        held = simple[position > 0]
        out["exposure"] = float(position.mean())
        out["win_rate"] = float((held > 0).mean()) if len(held) else float("nan")
    if trades is not None:
        out["n_trades"] = int(trades.sum())
    return out
