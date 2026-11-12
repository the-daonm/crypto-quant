"""Các mô hình: baseline, Logistic Regression, LightGBM."""

from __future__ import annotations

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODEL_NAMES = ["baseline", "logreg", "lightgbm"]


class AlwaysUp(BaseEstimator, ClassifierMixin):
    """Baseline: luôn dự đoán tăng (luôn long) với mọi ngưỡng; tương đương Buy & Hold."""

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
        # Cấu hình nhỏ, regularize mạnh: dữ liệu tài chính rất nhiễu, dễ overfit.
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
    raise ValueError(f"Unknown model: {name}. Choose from {MODEL_NAMES}")


def feature_importance(model, columns) -> dict[str, float] | None:
    if isinstance(model, LGBMClassifier):
        imp = model.booster_.feature_importance(importance_type="gain")
        return dict(zip(columns, imp / imp.sum(), strict=True))
    if hasattr(model, "named_steps") and "logisticregression" in model.named_steps:
        coef = np.abs(model.named_steps["logisticregression"].coef_[0])
        return dict(zip(columns, coef / coef.sum(), strict=True))
    return None
