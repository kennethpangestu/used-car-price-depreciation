"""Price prediction models and evaluation helpers.

All models predict log(price) and are scored in pounds after converting back,
because a £1,000 miss means very different things on a £5,000 Fiesta and a
£60,000 G-Class; the log target treats errors as percentages during training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

CATEGORICAL = ["brand", "model", "transmission", "fuelType"]
NUMERIC = ["age", "mileage", "engineSize", "mpg", "tax"]
FEATURES = CATEGORICAL + NUMERIC


class GroupMedianBaseline:
    """Predict the median price of training cars with the same brand, model
    and year, falling back to brand + model, then brand, then the overall median.

    This is what a person would do by eyeballing similar listings, so a real
    model has to beat it to be worth anything.
    """

    levels = (["brand", "model", "year"], ["brand", "model"], ["brand"])

    def fit(self, X: pd.DataFrame, y_log: np.ndarray):
        frame = X.assign(_y=np.asarray(y_log))
        self.tables_ = [frame.groupby(cols)["_y"].median() for cols in self.levels]
        self.global_ = float(np.median(y_log))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pred = pd.Series(np.nan, index=X.index)
        for cols, table in zip(self.levels, self.tables_):
            keys = pd.MultiIndex.from_frame(X[cols]) if len(cols) > 1 else X[cols[0]]
            looked_up = pd.Series(table.reindex(keys).to_numpy(), index=X.index)
            pred = pred.fillna(looked_up)
        return pred.fillna(self.global_).to_numpy()


def make_ridge(alpha: float = 1.0) -> Pipeline:
    """Linear model: one-hot categories + scaled numerics (+ log mileage)."""
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=5), CATEGORICAL),
        ("num", StandardScaler(), NUMERIC),
    ])
    return Pipeline([("pre", pre), ("model", Ridge(alpha=alpha))])


def make_gbm(learning_rate: float = 0.1, max_leaf_nodes: int = 63,
             max_iter: int = 600, random_state: int = 42) -> Pipeline:
    """Gradient-boosted trees with native categorical handling."""
    pre = ColumnTransformer(
        [("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                encoded_missing_value=-1), CATEGORICAL)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    model = HistGradientBoostingRegressor(
        learning_rate=learning_rate,
        max_leaf_nodes=max_leaf_nodes,
        max_iter=max_iter,
        categorical_features=list(range(len(CATEGORICAL))),
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
        random_state=random_state,
    )
    return Pipeline([("pre", pre), ("model", model)])


def price_metrics(y_true_price: np.ndarray, y_pred_price: np.ndarray) -> dict:
    """MAE and median absolute error in pounds, MAPE, and R² on prices."""
    y_true_price = np.asarray(y_true_price, dtype=float)
    y_pred_price = np.asarray(y_pred_price, dtype=float)
    err = y_pred_price - y_true_price
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true_price - y_true_price.mean()) ** 2))
    return {
        "MAE (£)": float(np.mean(np.abs(err))),
        "median abs error (£)": float(np.median(np.abs(err))),
        "MAPE": float(np.mean(np.abs(err) / y_true_price)),
        "R²": 1 - ss_res / ss_tot,
    }
