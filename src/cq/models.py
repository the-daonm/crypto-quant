"""Models: baseline, Logistic Regression, LightGBM, Random Forest, XGBoost and a small MLP."""

from __future__ import annotations

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODEL_NAMES = ["baseline", "logreg", "lightgbm", "random_forest", "xgboost", "mlp"]


class AlwaysUp(BaseEstimator, ClassifierMixin):
    """Baseline: always predicts up (always long) at any threshold, i.e. Buy & Hold."""

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.zeros(n), np.ones(n)])


def make_model(name: str, seed: int = 42):
    if name == "baseline":
        return AlwaysUp()
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000))
    if name == "lightgbm":
        # Small, heavily regularised trees: financial data is noisy and easy to overfit.
        return LGBMClassifier(
            n_estimators=200,
            learning_rate=0.03,
            num_leaves=8,
            max_depth=3,
            min_child_samples=50,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=seed,
            verbose=-1,
        )
    if name == "random_forest":
        # Shallow trees with large leaves: averaging many weak, decorrelated trees limits overfitting.
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=4,
            min_samples_leaf=50,
            max_features="sqrt",
            n_jobs=-1,
            random_state=seed,
        )
    if name == "xgboost":
        # Optional dependency (`pip install "crypto-quant[models]"`): on Linux the wheel pulls in a large
        # CUDA library, so it is imported only when the model is actually requested.
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                'The "xgboost" model needs the optional dependency: pip install "crypto-quant[models]"'
            ) from exc

        # Mirrors the LightGBM configuration so the two boosting libraries are comparable.
        return XGBClassifier(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=3,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            n_jobs=-1,
            random_state=seed,
            verbosity=0,
        )
    if name == "mlp":
        # One small hidden layer with strong L2 regularisation. sklearn's early stopping validates
        # on a shuffled split, which is optimistic for overlapping time-series labels, so it is not used.
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(16,), alpha=1.0, max_iter=2000, random_state=seed),
        )
    raise ValueError(f"Unknown model: {name}. Choose from {MODEL_NAMES}")


def feature_importance(model, columns) -> dict[str, float] | None:
    """Normalised importance per feature (sums to 1), or None for models without one (baseline, MLP)."""
    if isinstance(model, LGBMClassifier):
        imp = model.booster_.feature_importance(importance_type="gain")
        return dict(zip(columns, imp / imp.sum(), strict=True))
    if hasattr(model, "feature_importances_"):  # Random Forest, XGBoost (checked by attribute: lazy import)
        imp = model.feature_importances_  # impurity decrease (Random Forest) / gain (XGBoost)
        total = imp.sum()
        return dict(zip(columns, imp / total if total > 0 else imp, strict=True))
    if hasattr(model, "named_steps") and "logisticregression" in model.named_steps:
        coef = np.abs(model.named_steps["logisticregression"].coef_[0])
        return dict(zip(columns, coef / coef.sum(), strict=True))
    return None
