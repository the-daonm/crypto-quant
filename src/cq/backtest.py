"""Vectorized backtest for a long/flat strategy.

Timing convention: the signal is computed after bar t closes and the position is held
from t to t+1, so it earns bar t's `fwd_ret` (= log return from close_t to close_{t+1}).
This is equivalent to `position.shift(1) * return` on an ordinary return series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def positions(proba: pd.Series, threshold: float | pd.Series = 0.5, band: float = 0.0) -> pd.Series:
    """Long/flat positions with hysteresis.

    Enter when proba > threshold + band, exit when proba < threshold - band, and keep the
    previous position in between. band = 0 is the plain rule proba > threshold.
    The band stops the strategy from flipping every bar while proba hovers around the
    threshold, which saves fees.
    """
    if band == 0:
        return (proba > threshold).astype(float)
    sig = pd.Series(np.nan, index=proba.index)
    sig[proba > threshold + band] = 1.0
    sig[proba < threshold - band] = 0.0
    return sig.ffill().fillna(0.0)


def backtest(
    proba: pd.Series,
    fwd_ret: pd.Series,
    threshold: float | pd.Series = 0.5,
    fee: float = 0.001,
    band: float = 0.0,
) -> pd.DataFrame:
    """Return a DataFrame with position, trades, strat_ret, bh_ret, equity, bh_equity (log returns)."""
    position = positions(proba, threshold, band)
    trades = position.diff().abs().fillna(position.iloc[0])  # count the first entry if we start long
    cost = np.log1p(-fee * trades)  # fee charged on capital whenever the position changes
    strat = position * fwd_ret + cost
    out = pd.DataFrame(
        {
            "position": position,
            "trades": trades,
            "strat_ret": strat,
            "bh_ret": fwd_ret,
        },
        index=proba.index,
    )
    out["equity"] = np.exp(out["strat_ret"].cumsum())
    out["bh_equity"] = np.exp(out["bh_ret"].cumsum())
    return out
