"""Nhãn: hướng giá của nến kế tiếp."""

from __future__ import annotations

import numpy as np
import pandas as pd


def forward_return(close: pd.Series) -> pd.Series:
    """Log return của nến t+1 (từ close_t tới close_{t+1}). NaN ở dòng cuối."""
    return np.log(close).diff().shift(-1)


def direction_label(close: pd.Series, threshold: float = 0.0) -> pd.Series:
    """1 nếu return nến kế tiếp > threshold, 0 nếu ngược lại, NaN ở dòng cuối."""
    fwd = forward_return(close)
    return (fwd > threshold).astype(float).where(fwd.notna())


def make_dataset(features: pd.DataFrame, close: pd.Series, threshold: float = 0.0):
    """Ghép X, y, forward return và bỏ các dòng thiếu dữ liệu."""
    y = direction_label(close, threshold).rename("y")
    fwd = forward_return(close).rename("fwd_ret")
    data = pd.concat([features, y, fwd], axis=1).dropna()
    return data[features.columns], data["y"].astype(int), data["fwd_ret"]
