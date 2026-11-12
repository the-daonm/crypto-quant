"""Backtest vectorized cho chiến lược long/flat.

Quy ước thời gian: tín hiệu được tính sau khi nến t đóng cửa, vị thế được giữ trong
khoảng t -> t+1, nên nó hưởng `fwd_ret` của nến t (= log return từ close_t tới close_{t+1}).
Như vậy là tương đương với `position.shift(1) * return` trên chuỗi return thông thường.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def positions(proba: pd.Series, threshold: float | pd.Series = 0.5, band: float = 0.0) -> pd.Series:
    """Vị thế long/flat có hysteresis.

    Vào lệnh khi proba > threshold + band, thoát khi proba < threshold - band,
    ở giữa thì giữ nguyên vị thế cũ. band = 0 tương đương với quy tắc proba > threshold.
    Vùng đệm giúp không mua/bán liên tục khi proba dao động quanh ngưỡng, nên tiết kiệm phí.
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
    """Trả về DataFrame: position, trades, strat_ret, bh_ret, equity, bh_equity (log-return)."""
    position = positions(proba, threshold, band)
    trades = position.diff().abs().fillna(position.iloc[0])   # lệnh đầu tiên nếu vào ngay
    cost = np.log1p(-fee * trades)                             # phí tính trên vốn khi đổi vị thế
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
