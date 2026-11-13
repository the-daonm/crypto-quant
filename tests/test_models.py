import numpy as np
import pandas as pd
import pytest

from cq.models import MODEL_NAMES, feature_importance, make_model
from cq.validation import run_walk_forward

NO_IMPORTANCE = {"baseline", "mlp"}


def _requires(name):
    """Skip the parametrised case when the model's optional dependency is not installed."""
    if name == "xgboost":
        pytest.importorskip("xgboost")


@pytest.fixture(scope="module")
def dataset():
    """Synthetic features with a weak signal in the first column (no network access)."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2021-01-01", periods=400, freq="D", tz="UTC")
    X = pd.DataFrame(rng.normal(size=(400, 6)), index=idx, columns=[f"f{i}" for i in range(6)])
    y = pd.Series((X["f0"] + rng.normal(0, 1.0, 400) > 0).astype(int), index=idx)
    fwd = pd.Series(rng.normal(0.001, 0.03, 400), index=idx)
    return X, y, fwd


def test_new_models_are_registered():
    assert {"random_forest", "xgboost", "mlp"} <= set(MODEL_NAMES)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_fit_and_predict_proba_shapes(name, dataset):
    _requires(name)
    X, y, _ = dataset
    model = make_model(name).fit(X.iloc[:300], y.iloc[:300])
    proba = model.predict_proba(X.iloc[300:])
    assert proba.shape == (100, 2)
    assert np.isfinite(proba).all() and (proba >= 0).all() and (proba <= 1).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_same_seed_gives_same_predictions(name, dataset):
    _requires(name)
    X, y, _ = dataset
    first = make_model(name, seed=7).fit(X.iloc[:300], y.iloc[:300]).predict_proba(X.iloc[300:])
    second = make_model(name, seed=7).fit(X.iloc[:300], y.iloc[:300]).predict_proba(X.iloc[300:])
    np.testing.assert_allclose(first, second, rtol=1e-6)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_feature_importance_is_normalised_or_none(name, dataset):
    _requires(name)
    X, y, _ = dataset
    imp = feature_importance(make_model(name).fit(X, y), X.columns)
    if name in NO_IMPORTANCE:
        assert imp is None
    else:
        assert list(imp) == list(X.columns)
        assert sum(imp.values()) == pytest.approx(1.0)
        assert max(imp, key=imp.get) == "f0"  # the only informative feature


def test_unknown_model_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        make_model("transformer")


@pytest.mark.parametrize("name", ["random_forest", "xgboost", "mlp"])
def test_walk_forward_runs_out_of_sample_and_records_fit_time(name, dataset):
    _requires(name)
    X, y, fwd = dataset
    wf = run_walk_forward(name, X, y, fwd, train_min=250, test_size=50, gap=3)
    assert wf.predictions.index.equals(X.index[253:])  # only bars after train_min + gap are predicted
    assert wf.predictions["proba"].between(0, 1).all()
    assert (wf.folds["fit_seconds"] > 0).all()
    assert (wf.importance is None) == (name in NO_IMPORTANCE)


def test_xgboost_without_the_optional_dependency_explains_how_to_install(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "xgboost", None)  # makes `import xgboost` raise ImportError
    with pytest.raises(ImportError, match=r"crypto-quant\[models\]"):
        make_model("xgboost")


def test_default_models_are_a_fast_subset_of_the_registry():
    from cq.models import DEFAULT_MODELS

    assert set(DEFAULT_MODELS) <= set(MODEL_NAMES)
    assert not {"random_forest", "xgboost", "mlp"} & set(DEFAULT_MODELS)
