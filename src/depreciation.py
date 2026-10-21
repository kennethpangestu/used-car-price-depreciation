"""Estimate how fast each brand loses value.

Comparing raw average prices by age is misleading: brands sell different mixes
of models, and older cars have more miles. So the estimate compares cars *of
the same model* and holds mileage, fuel type, transmission and engine size
constant:

    log(price) = model effect + b_brand * age + controls + error

``b_brand`` is the brand's yearly change in log price, so ``exp(b_brand)`` is
the share of value a car keeps each year. The model effects are removed by
subtracting each model's average from every variable (the "within"
transformation), which gives the same coefficients as adding one dummy
variable per model, with far less memory.

Standard errors are heteroskedasticity-robust (HC1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _design(df: pd.DataFrame, brands: list[str]) -> tuple[np.ndarray, list[str]]:
    cols, names = [], []
    for brand in brands:
        cols.append(np.where(df["brand"] == brand, df["age"], 0.0))
        names.append(f"age x {brand}")
    cols.append(np.log1p(df["mileage"].to_numpy(dtype=float)))
    names.append("log(1 + mileage)")
    cols.append(df["engineSize"].to_numpy(dtype=float))
    names.append("engine size")
    for col in ["fuelType", "transmission"]:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True, dtype=float)
        cols.extend(dummies[c].to_numpy() for c in dummies.columns)
        names.extend(dummies.columns)
    return np.column_stack(cols), names


def _within(values: np.ndarray, groups: pd.Series) -> np.ndarray:
    """Subtract each group's mean from every row (column-wise)."""
    frame = pd.DataFrame(values)
    return (frame - frame.groupby(groups.to_numpy()).transform("mean")).to_numpy()


def fit_depreciation(df: pd.DataFrame, alpha: float = 0.05) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit the fixed-effects regression.

    Returns
    -------
    brand_table : one row per brand with the yearly log-price slope, the share of
        value kept per year (with a confidence interval) and the implied value
        left after 3 and 5 years.
    coef_table : every coefficient with its robust standard error.
    """
    brands = sorted(df["brand"].unique())
    groups = df["brand"] + " | " + df["model"]

    X, names = _design(df, brands)
    y = np.log(df["price"].to_numpy(dtype=float))
    Xw = _within(X, groups)
    yw = _within(y.reshape(-1, 1), groups).ravel()

    # Drop control columns that are constant within every model (no variation left).
    keep = Xw.std(axis=0) > 1e-12
    Xw, names = Xw[:, keep], [n for n, k in zip(names, keep) if k]

    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    resid = yw - Xw @ beta
    n, k = Xw.shape
    n_groups = groups.nunique()
    dof = n - k - n_groups
    xtx_inv = np.linalg.pinv(Xw.T @ Xw)
    meat = (Xw * resid[:, None] ** 2).T @ Xw
    cov = xtx_inv @ meat @ xtx_inv * n / dof
    se = np.sqrt(np.diag(cov))

    coef = pd.DataFrame({"coef": beta, "se": se}, index=names)
    z = stats.norm.ppf(1 - alpha / 2)

    rows = []
    for brand in brands:
        b, s = coef.loc[f"age x {brand}", ["coef", "se"]]
        rows.append({
            "brand": brand,
            "listings": int((df["brand"] == brand).sum()),
            "log_slope": b,
            "yearly_value_kept": np.exp(b),
            "kept_ci_low": np.exp(b - z * s),
            "kept_ci_high": np.exp(b + z * s),
            "value_after_3y": np.exp(3 * b),
            "value_after_5y": np.exp(5 * b),
        })
    brand_table = (
        pd.DataFrame(rows).sort_values("yearly_value_kept", ascending=False).reset_index(drop=True)
    )

    r2_within = 1 - (resid @ resid) / (yw @ yw)
    coef.attrs["r2_within"] = float(r2_within)
    coef.attrs["n"] = n
    return brand_table, coef


def fit_depreciation_curves(df: pd.DataFrame, max_age: int = 7) -> pd.DataFrame:
    """Same comparison as :func:`fit_depreciation`, but with a separate effect
    for every (brand, age) pair instead of one constant yearly rate.

    This checks whether depreciation really is a constant percentage per year.
    Cars aged 0 (2020 models) are the reference, so each value is the share of
    a nearly-new car's price that a car of that age keeps, same model and
    mileage held constant. Only ages 0..max_age are used.
    """
    data = df[df["age"] <= max_age].reset_index(drop=True)
    groups = data["brand"] + " | " + data["model"]
    brands = sorted(data["brand"].unique())

    cols, names = [], []
    for brand in brands:
        for age in range(1, max_age + 1):
            cols.append(((data["brand"] == brand) & (data["age"] == age)).to_numpy(dtype=float))
            names.append((brand, age))
    controls, _ = _design(data, brands)
    controls = controls[:, len(brands):]  # drop the linear age terms
    X = np.column_stack(cols + [controls])
    y = np.log(data["price"].to_numpy(dtype=float))

    Xw = _within(X, groups)
    yw = _within(y.reshape(-1, 1), groups).ravel()
    keep = Xw.std(axis=0) > 1e-12
    beta = np.zeros(X.shape[1])
    beta[keep], *_ = np.linalg.lstsq(Xw[:, keep], yw, rcond=None)

    rows = [{"brand": b, "age": 0, "value_kept": 1.0} for b in brands]
    for (brand, age), b in zip(names, beta[: len(names)]):
        rows.append({"brand": brand, "age": age, "value_kept": float(np.exp(b))})
    return pd.DataFrame(rows).sort_values(["brand", "age"]).reset_index(drop=True)
