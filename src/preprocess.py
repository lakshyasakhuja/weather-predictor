"""
preprocess.py
Cleans a city's raw historical weather CSV and engineers features used by
the regression / ML models: calendar features (day-of-year, month, year
trend), lag features, and rolling averages.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CITIES, DEFAULT_CITY, city_data_dir  # noqa: E402

LAGS = [1, 2, 3, 7, 14]
ROLLING_WINDOWS = [7, 14, 30]


def load_raw(city_key: str) -> pd.DataFrame:
    path = city_data_dir(city_key) / "historical_weather.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values and obvious data issues."""
    df = df.copy()

    numeric_cols = ["temp_max", "temp_min", "temp_mean", "precipitation", "wind_speed_max", "humidity_mean", "pressure_mean"]
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    # Ensure one row per calendar day (reindex to a full daily range so lag /
    # rolling features are computed correctly across any missing dates).
    full_range = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    df = df.set_index("date").reindex(full_range)
    df.index.name = "date"

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].interpolate(method="linear", limit_direction="both")

    df = df.reset_index()

    # Drop columns that are entirely empty (e.g. humidity/pressure not
    # available from an offline fallback source) rather than modeling on them.
    for col in ["humidity_mean", "pressure_mean"]:
        if col in df.columns and df[col].isna().all():
            df.drop(columns=[col], inplace=True)

    if "weather" in df.columns:
        df["weather"] = df["weather"].ffill().bfill()

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar, trend, lag, and rolling-average features."""
    df = df.copy()

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_year"] = df["date"].dt.dayofyear
    df["day_of_week"] = df["date"].dt.dayofweek

    # Continuous trend index (days since the start of the record) — this is
    # the single feature a plain linear-regression trend model uses.
    df["time_index"] = (df["date"] - df["date"].min()).dt.days

    # Cyclical encodings of day-of-year so the model understands that day
    # 365 is adjacent to day 1 (seasonality repeats every year).
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)

    target_col = "temp_mean"

    for lag in LAGS:
        df[f"temp_lag_{lag}"] = df[target_col].shift(lag)

    for window in ROLLING_WINDOWS:
        df[f"temp_roll_mean_{window}"] = (
            df[target_col].shift(1).rolling(window=window, min_periods=1).mean()
        )
        df[f"temp_roll_std_{window}"] = (
            df[target_col].shift(1).rolling(window=window, min_periods=2).std()
        )

    # Same-day humidity/pressure (when available) are legitimate predictors
    # of mean temperature — no lag needed since they're used as same-day
    # context, not to predict themselves.
    return df


def process_city(city_key: str) -> pd.DataFrame:
    df = load_raw(city_key)
    df = clean(df)
    df = engineer_features(df)

    before = len(df)
    df.dropna(subset=[c for c in df.columns if c.startswith("temp_lag_") or c.startswith("temp_roll_")], inplace=True)
    after = len(df)
    print(f"[{city_key}] Dropped {before - after} rows with incomplete lag/rolling features.")

    out_path = city_data_dir(city_key) / "processed_weather.csv"
    df.to_csv(out_path, index=False)
    print(f"[{city_key}] Saved processed dataset with {len(df)} rows and {len(df.columns)} columns -> {out_path}")
    return df


def main(city_keys=None):
    city_keys = city_keys or list(CITIES.keys())
    for city_key in city_keys:
        process_city(city_key)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--city", action="append", help="City key to process (repeatable). Default: all configured cities.")
    args = parser.parse_args()
    main(args.city)
