"""Labels: price direction over the next `horizon` bars."""

from __future__ import annotations

import numpy as np
import pandas as pd


def forward_return(close: pd.Series, horizon: int = 1) -> pd.Series:
    """Log return from close_t to close_{t+horizon}. NaN for the last `horizon` rows."""
    logc = np.log(close)
    return logc.shift(-horizon) - logc


def direction_label(close: pd.Series, threshold: float = 0.0, horizon: int = 1) -> pd.Series:
    """1 if the `horizon`-bar forward return > threshold, else 0; NaN when unknown."""
    fwd = forward_return(close, horizon)
    return (fwd > threshold).astype(float).where(fwd.notna())


def make_dataset(features: pd.DataFrame, close: pd.Series, threshold: float = 0.0, horizon: int = 1):
    """Join X, y and the 1-bar forward return (used by the backtest), dropping incomplete rows.

    The label looks `horizon` bars ahead to reduce noise, while the backtest still re-evaluates
    the position every bar. Labels of neighbouring bars overlap, so train/test splits need an
    embargo of at least `horizon` bars.
    """
    y = direction_label(close, threshold, horizon).rename("y")
    fwd = forward_return(close, 1).rename("fwd_ret")
    data = pd.concat([features, y, fwd], axis=1).dropna()
    return data[features.columns], data["y"].astype(int), data["fwd_ret"]
