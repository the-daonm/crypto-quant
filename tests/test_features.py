import numpy as np
import pandas as pd

from cq.features import build_features
from cq.labels import direction_label, make_dataset


def _synthetic_ohlcv(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    open_ = close * np.exp(rng.normal(0, 0.005, n))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.01, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.01, n))
    volume = rng.uniform(1e3, 1e4, n)
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)


def test_features_do_not_use_future_data():
    """Thay đổi dữ liệu sau thời điểm t không được làm thay đổi feature tại <= t."""
    df = _synthetic_ohlcv()
    t = 300
    base = build_features(df)

    tampered = df.copy()
    tampered.iloc[t + 1 :] *= 3.0  # phá toàn bộ tương lai
    after = build_features(tampered)

    pd.testing.assert_frame_equal(base.iloc[: t + 1], after.iloc[: t + 1])


def test_label_is_next_bar_direction():
    close = pd.Series([1.0, 2.0, 1.5, 1.5, 3.0])
    y = direction_label(close)
    assert y.tolist()[:4] == [1.0, 0.0, 0.0, 1.0]
    assert np.isnan(y.iloc[-1])


def test_make_dataset_aligns_and_drops_nan():
    df = _synthetic_ohlcv()
    X, y, fwd = make_dataset(build_features(df), df["close"])
    assert len(X) == len(y) == len(fwd)
    assert not X.isna().any().any()
    assert X.index[-1] < df.index[-1]  # nến cuối không có nhãn
    assert ((fwd > 0).astype(int) == y).all()


def test_multi_bar_label_and_dataset():
    close = pd.Series([1.0, 2.0, 1.5, 1.2, 3.0])
    y = direction_label(close, horizon=2)
    assert y.tolist()[:3] == [1.0, 0.0, 1.0]
    assert y.iloc[-2:].isna().all()

    df = _synthetic_ohlcv()
    X, y, fwd = make_dataset(build_features(df), df["close"], horizon=3)
    assert X.index[-1] == df.index[-4]  # 3 nến cuối chưa có nhãn
    expected = np.log(df["close"]).diff().shift(-1).reindex(fwd.index)
    np.testing.assert_allclose(fwd, expected)  # backtest vẫn dùng return 1 nến
