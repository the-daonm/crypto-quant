"""Nhãn: hướng giá trong `horizon` nến kế tiếp."""

from __future__ import annotations

import numpy as np
import pandas as pd


def forward_return(close: pd.Series, horizon: int = 1) -> pd.Series:
    """Log return từ close_t tới close_{t+horizon}. NaN ở `horizon` dòng cuối."""
    logc = np.log(close)
    return logc.shift(-horizon) - logc


def direction_label(close: pd.Series, threshold: float = 0.0, horizon: int = 1) -> pd.Series:
    """1 nếu return `horizon` nến tới > threshold, 0 nếu ngược lại, NaN khi chưa biết."""
    fwd = forward_return(close, horizon)
    return (fwd > threshold).astype(float).where(fwd.notna())


def make_dataset(features: pd.DataFrame, close: pd.Series, threshold: float = 0.0, horizon: int = 1):
    """Ghép X, y, forward return 1 nến (dùng cho backtest) và bỏ các dòng thiếu dữ liệu.

    Nhãn nhìn xa `horizon` nến cho bớt nhiễu, còn backtest vẫn đánh giá lại vị thế sau mỗi nến.
    Vì nhãn của các nến liền nhau chồng lấn nhau, khi chia train/test phải dùng embargo >= horizon.
    """
    y = direction_label(close, threshold, horizon).rename("y")
    fwd = forward_return(close, 1).rename("fwd_ret")
    data = pd.concat([features, y, fwd], axis=1).dropna()
    return data[features.columns], data["y"].astype(int), data["fwd_ret"]
