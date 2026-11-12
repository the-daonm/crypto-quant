import numpy as np
import pandas as pd
import pytest

from cq.backtest import backtest
from cq.metrics import max_drawdown, trading_metrics


@pytest.fixture
def fwd():
    rng = np.random.default_rng(1)
    idx = pd.date_range("2021-01-01", periods=200, freq="D", tz="UTC")
    return pd.Series(rng.normal(0.001, 0.03, 200), index=idx)


def test_always_long_without_fee_equals_buy_and_hold(fwd):
    proba = pd.Series(1.0, index=fwd.index)
    bt = backtest(proba, fwd, threshold=0.5, fee=0.0)
    np.testing.assert_allclose(bt["equity"], bt["bh_equity"])


def test_single_entry_pays_fee_once(fwd):
    fee = 0.001
    proba = pd.Series(1.0, index=fwd.index)
    bt = backtest(proba, fwd, fee=fee)
    assert bt["trades"].sum() == 1
    assert bt["equity"].iloc[-1] == pytest.approx(bt["bh_equity"].iloc[-1] * (1 - fee))


def test_flat_strategy_stays_flat(fwd):
    proba = pd.Series(0.0, index=fwd.index)
    bt = backtest(proba, fwd, fee=0.001)
    assert bt["trades"].sum() == 0
    assert (bt["equity"] == 1.0).all()


def test_round_trip_pays_two_fees():
    idx = pd.date_range("2021-01-01", periods=4, freq="D", tz="UTC")
    fwd = pd.Series([0.0, 0.0, 0.0, 0.0], index=idx)
    proba = pd.Series([0.9, 0.9, 0.1, 0.1], index=idx)
    bt = backtest(proba, fwd, fee=0.01)
    assert bt["trades"].sum() == 2
    assert bt["equity"].iloc[-1] == pytest.approx(0.99 * 0.99)


def test_max_drawdown():
    assert max_drawdown(pd.Series([1.0, 2.0, 1.0, 3.0])) == pytest.approx(-0.5)


def test_trading_metrics_flat_returns():
    m = trading_metrics(pd.Series(np.zeros(365)), 365)
    assert m["total_return"] == 0 and m["max_drawdown"] == 0


def test_hysteresis_holds_position_inside_band():
    from cq.backtest import positions

    idx = pd.date_range("2021-01-01", periods=6, freq="D", tz="UTC")
    proba = pd.Series([0.52, 0.56, 0.52, 0.48, 0.44, 0.50], index=idx)
    assert positions(proba, 0.5, band=0.05).tolist() == [0, 1, 1, 1, 0, 0]
    assert positions(proba, 0.5, band=0.0).tolist() == [1, 1, 1, 0, 0, 0]


def test_band_reduces_trades(fwd):
    rng = np.random.default_rng(2)
    proba = pd.Series(0.5 + rng.normal(0, 0.04, len(fwd)), index=fwd.index)
    assert backtest(proba, fwd, band=0.05)["trades"].sum() < backtest(proba, fwd, band=0.0)["trades"].sum()
