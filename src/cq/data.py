"""Tải dữ liệu OHLCV công khai qua ccxt và cache ra parquet.

Nguồn: public market-data API của sàn (mặc định Binance), không cần API key.
Chỉ dùng cho mục đích học tập; không phân phối lại dữ liệu thô.
"""

from __future__ import annotations

from pathlib import Path

import ccxt
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def _cache_path(exchange: str, symbol: str, timeframe: str) -> Path:
    return DATA_DIR / f"{exchange}_{symbol.replace('/', '-')}_{timeframe}.parquet"


def fetch_ohlcv(
    symbol: str,
    timeframe: str = "1d",
    since: str = "2019-01-01",
    exchange: str = "binance",
    limit: int = 1000,
) -> pd.DataFrame:
    """Tải toàn bộ nến từ `since` tới hiện tại, phân trang theo `limit`."""
    ex = getattr(ccxt, exchange)({"enableRateLimit": True})
    cursor = ex.parse8601(f"{since}T00:00:00Z")
    tf_ms = ex.parse_timeframe(timeframe) * 1000
    rows: list[list] = []
    while True:
        batch = ex.fetch_ohlcv(symbol, timeframe, since=cursor, limit=limit)
        if not batch:
            break
        rows.extend(batch)
        cursor = batch[-1][0] + tf_ms
        if len(batch) < limit:
            break

    df = pd.DataFrame(rows, columns=COLUMNS).drop_duplicates("timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index().astype(float)
    # Bỏ nến cuối nếu chưa đóng (đang chạy) để tránh dùng giá chưa hoàn chỉnh.
    now = pd.Timestamp.now(tz="UTC")
    if len(df) and df.index[-1] + pd.Timedelta(milliseconds=tf_ms) > now:
        df = df.iloc[:-1]
    return df


def load_ohlcv(
    symbol: str,
    timeframe: str = "1d",
    since: str = "2019-01-01",
    exchange: str = "binance",
    refresh: bool = False,
) -> pd.DataFrame:
    """Đọc từ cache nếu có, nếu không thì tải và lưu cache."""
    path = _cache_path(exchange, symbol, timeframe)
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    df = fetch_ohlcv(symbol, timeframe, since, exchange)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df
