"""Feature kỹ thuật. Mọi feature tại thời điểm t chỉ dùng dữ liệu <= t (causal)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def build_features(df: pd.DataFrame, n_lags: int = 5) -> pd.DataFrame:
    """Trả về DataFrame feature cùng index với `df` (các dòng đầu có NaN do warm-up)."""
    close, volume = df["close"], df["volume"]
    logret = np.log(close).diff()
    f = pd.DataFrame(index=df.index)

    for k in range(1, n_lags + 1):
        f[f"ret_lag{k}"] = logret.shift(k - 1)  # ret_lag1 = return của chính nến t
    for w in (5, 10, 20):
        f[f"mom_{w}"] = np.log(close / close.shift(w))  # momentum
        f[f"vol_{w}"] = logret.rolling(w).std()  # volatility

    f["rsi_14"] = rsi(close, 14)

    ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    f["macd_hist"] = (macd - signal) / close  # chuẩn hoá theo giá

    ma20, sd20 = close.rolling(20).mean(), close.rolling(20).std()
    f["bb_pctb"] = (close - (ma20 - 2 * sd20)) / (4 * sd20)

    f["atr_pct"] = atr(df, 14) / close
    for w in (20, 50, 200):
        f[f"dist_ma{w}"] = close / close.rolling(w).mean() - 1

    lv = np.log1p(volume)
    f["volume_z"] = (lv - lv.rolling(20).mean()) / lv.rolling(20).std()
    f["hl_range"] = (df["high"] - df["low"]) / close
    f["close_pos"] = (close - df["low"]) / (df["high"] - df["low"]).replace(0, np.nan)
    return f
