import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data import clean, load_raw  # noqa: E402
from depreciation import fit_depreciation  # noqa: E402
from modeling import GroupMedianBaseline, price_metrics  # noqa: E402


def _row(**kw):
    base = dict(model="Fiesta", year=2017, price=10000, transmission="Manual", mileage=20000,
                fuelType="Petrol", tax=145, mpg=55.0, engineSize=1.0, brand="Ford")
    base.update(kw)
    return base


def test_load_raw_combines_brands_and_fixes_tax_column(tmp_path):
    pd.DataFrame([_row()]).drop(columns="brand").to_csv(tmp_path / "ford.csv", index=False)
    pd.DataFrame([_row(model=" I20")]).drop(columns="brand").rename(columns={"tax": "tax(£)"}).to_csv(
        tmp_path / "hyundi.csv", index=False)
    df = load_raw(str(tmp_path))
    assert set(df["brand"]) == {"Ford", "Hyundai"}
    assert "tax" in df.columns and "tax(£)" not in df.columns
    assert "I20" in set(df["model"])  # whitespace stripped


def test_clean_applies_each_rule():
    raw = pd.DataFrame([
        _row(),
        _row(),                                          # exact duplicate
        _row(year=2060),                                 # impossible year
        _row(engineSize=0.0),                            # 0-litre petrol: missing value
        _row(engineSize=0.0, fuelType="Electric", price=30000),  # electric: keep
    ])
    out, removed = clean(raw)
    assert removed == {"year after 2020": 1, "exact duplicates": 1, "0-litre engine (not electric)": 1}
    assert len(out) == 2
    assert out.loc[out["fuelType"] == "Petrol", "age"].iloc[0] == 3


def test_depreciation_recovers_known_rates():
    rng = np.random.default_rng(0)
    rows = []
    true_slope = {"Toyota": -0.10, "BMW": -0.15}
    for brand, slope in true_slope.items():
        for m in range(5):
            model_level = rng.normal(9.5, 0.4)  # each model has its own price level
            for _ in range(400):
                age = rng.integers(0, 8)
                mileage = age * 8000 + rng.integers(0, 5000)
                log_price = (model_level + slope * age - 0.03 * np.log1p(mileage)
                             + rng.normal(0, 0.05))
                rows.append(dict(brand=brand, model=f"{brand}-{m}", age=age, mileage=mileage,
                                 engineSize=1.5, fuelType="Petrol", transmission="Manual",
                                 price=float(np.exp(log_price))))
    table, _ = fit_depreciation(pd.DataFrame(rows))
    est = table.set_index("brand")["log_slope"]
    for brand, slope in true_slope.items():
        assert est[brand] == pytest.approx(slope, abs=0.005)


def test_baseline_falls_back_for_unseen_models():
    X = pd.DataFrame({"brand": ["Ford", "Ford", "BMW"], "model": ["Fiesta", "Focus", "X5"],
                      "year": [2017, 2017, 2018]})
    y = np.log([10000.0, 12000.0, 40000.0])
    base = GroupMedianBaseline().fit(X, y)
    new = pd.DataFrame({"brand": ["Ford", "Audi"], "model": ["Kuga", "A3"], "year": [2017, 2017]})
    pred = np.exp(base.predict(new))
    assert pred[0] == pytest.approx(np.exp(np.median(np.log([10000, 12000]))))  # brand median
    assert pred[1] == pytest.approx(12000.0)  # overall median


def test_price_metrics_perfect_and_known_error():
    perfect = price_metrics([100, 200, 300], [100, 200, 300])
    assert perfect["MAE (£)"] == 0 and perfect["R²"] == pytest.approx(1.0)
    off = price_metrics([100, 200], [110, 180])
    assert off["MAE (£)"] == pytest.approx(15)
    assert off["MAPE"] == pytest.approx(0.10)
