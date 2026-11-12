"""Download public OHLCV data via ccxt and cache it as parquet.

Source: the exchange's public market-data API (Binance by default); no API key needed.
For educational use only; raw data is not redistributed.
"""

from __future__ import annotations

from pathlib import Path

import ccxt
import pandas as pd

DATA_DIR = Path("data")  # relative to the current working directory, so it also works from an installed wheel
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
    """Download every bar from `since` until now, paginating `limit` bars at a time."""
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
    # Drop the last bar if it has not closed yet, so we never use an incomplete price.
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
    """Read from the cache if present; otherwise download and cache."""
    path = _cache_path(exchange, symbol, timeframe)
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    df = fetch_ohlcv(symbol, timeframe, since, exchange)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df
