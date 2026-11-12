"""Walk-forward validation (expanding window) and train-only threshold selection."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from cq.backtest import positions
from cq.models import feature_importance, make_model

THRESHOLD_GRID = np.round(np.arange(0.45, 0.61, 0.01), 2)


def walk_forward_splits(n: int, train_min: int, test_size: int, gap: int = 1):
    """Yield (train_idx, test_idx). Train expands from the start; `gap` bars separate train and test."""
    start = train_min + gap
    while start < n:
        end = min(start + test_size, n)
        yield np.arange(0, start - gap), np.arange(start, end)
        start = end


def _sharpe_long_flat(proba, fwd, thr, fee, band=0.0):
    pos = positions(pd.Series(proba), thr, band).to_numpy()
    trades = np.abs(np.diff(pos, prepend=0.0))
    pnl = pos * fwd - trades * fee
    sd = pnl.std()
    return pnl.mean() / sd if sd > 0 else -np.inf


def choose_threshold(model_name, X, y, fwd, fee, band=0.0, val_frac=0.25):
    """Hold out the tail of the training set and pick the threshold with the best Sharpe on it."""
    cut = int(len(X) * (1 - val_frac))
    m = make_model(model_name).fit(X.iloc[:cut], y.iloc[:cut])
    proba = m.predict_proba(X.iloc[cut:])[:, 1]
    f = fwd.iloc[cut:].to_numpy()
    scores = [_sharpe_long_flat(proba, f, t, fee, band) for t in THRESHOLD_GRID]
    return float(THRESHOLD_GRID[int(np.argmax(scores))])


@dataclass
class WalkForwardResult:
    predictions: pd.DataFrame  # time index: proba, y, fwd_ret, fold, threshold
    folds: pd.DataFrame  # per-fold summary
    importance: pd.Series | None = field(default=None)


def run_walk_forward(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    fwd: pd.Series,
    train_min: int,
    test_size: int,
    gap: int = 1,
    threshold: float | str = 0.5,
    fee: float = 0.001,
    band: float = 0.0,
) -> WalkForwardResult:
    """Refit the model in every fold and predict only on its test block (out-of-sample).

    `threshold="auto"` picks the threshold on the tail of each fold's training set, never on test.
    """
    preds, folds, imps = [], [], []
    for k, (tr, te) in enumerate(walk_forward_splits(len(X), train_min, test_size, gap)):
        Xtr, ytr = X.iloc[tr], y.iloc[tr]
        model = make_model(model_name).fit(Xtr, ytr)
        proba = model.predict_proba(X.iloc[te])[:, 1]

        thr = (
            choose_threshold(model_name, Xtr, ytr, fwd.iloc[tr], fee, band) if threshold == "auto" else float(threshold)
        )
        preds.append(
            pd.DataFrame(
                {"proba": proba, "y": y.iloc[te], "fwd_ret": fwd.iloc[te], "fold": k, "threshold": thr},
                index=X.index[te],
            )
        )
        acc = float(((proba > 0.5).astype(int) == y.iloc[te].to_numpy()).mean())
        folds.append(
            {
                "fold": k,
                "train_end": X.index[tr[-1]],
                "test_start": X.index[te[0]],
                "test_end": X.index[te[-1]],
                "n_test": len(te),
                "accuracy": acc,
                "threshold": thr,
            }
        )
        imp = feature_importance(model, X.columns)
        if imp is not None:
            imps.append(imp)

    if not preds:
        raise ValueError("Not enough data for walk-forward: reduce train_min or test_size.")
    importance = pd.DataFrame(imps).mean().sort_values(ascending=False) if imps else None
    return WalkForwardResult(pd.concat(preds), pd.DataFrame(folds), importance)
