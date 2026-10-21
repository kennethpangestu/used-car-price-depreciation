"""Load and clean the UK used car listings.

The raw data is one CSV per brand. Column names are identical except that one
file calls the road tax column ``tax(£)``. Listings were collected in 2020, so a
car's age is measured as ``2020 - year``.
"""

from __future__ import annotations

import glob
import os

import pandas as pd

LISTING_YEAR = 2020

BRAND_NAMES = {
    "audi": "Audi",
    "bmw": "BMW",
    "ford": "Ford",
    "hyundi": "Hyundai",  # the source file name is misspelled
    "merc": "Mercedes-Benz",
    "skoda": "Skoda",
    "toyota": "Toyota",
    "vauxhall": "Vauxhall",
    "vw": "Volkswagen",
}


def load_raw(raw_dir: str) -> pd.DataFrame:
    """Read every brand CSV in ``raw_dir`` into one DataFrame with a ``brand`` column."""
    frames = []
    for path in sorted(glob.glob(os.path.join(raw_dir, "*.csv"))):
        key = os.path.splitext(os.path.basename(path))[0]
        if key not in BRAND_NAMES:
            continue
        frame = pd.read_csv(path).rename(columns={"tax(£)": "tax"})
        frame["brand"] = BRAND_NAMES[key]
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No brand CSV files found in {raw_dir}")
    df = pd.concat(frames, ignore_index=True)
    df["model"] = df["model"].astype(str).str.strip()
    return df


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply the cleaning rules and return (clean_df, rows_removed_per_rule).

    Rules, in order:
    1. Drop cars with a model year after the listing year (data entry errors).
    2. Drop exact duplicate rows (the same listing scraped twice).
    3. Drop non-electric cars with a 0-litre engine (missing value coded as 0).
    """
    removed = {}
    out = df.copy()

    mask = out["year"] > LISTING_YEAR
    removed["year after 2020"] = int(mask.sum())
    out = out[~mask]

    before = len(out)
    out = out.drop_duplicates()
    removed["exact duplicates"] = before - len(out)

    mask = (out["engineSize"] == 0) & (out["fuelType"] != "Electric")
    removed["0-litre engine (not electric)"] = int(mask.sum())
    out = out[~mask]

    out = out.reset_index(drop=True)
    out["age"] = LISTING_YEAR - out["year"]
    return out, removed
