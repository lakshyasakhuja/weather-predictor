"""
preprocess.py
Cleans the raw historical weather CSV and engineers features used by the
regression / ML models: calendar features (day-of-year, month, year trend),
lag features, and rolling averages.
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_PATH = DATA_DIR / "historical_weather.csv"
PROCESSED_PATH = DATA_DIR / "processed_weather.csv"

LAGS = [1, 2, 3, 7, 14]
ROLLING_WINDOWS = [7, 14, 30]


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values and obvious data issues."""
    df = df.copy()

    # Fill small gaps in the daily series via interpolation, since weather
    # is smooth day-to-day; drop any columns that are entirely empty.
    numeric_cols = ["temp_max", "temp_min", "temp_mean", "precipitation", "wind_speed_max"]
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    # Ensure one row per calendar day (reindex to a full daily range so lag /
    # rolling features are computed correctly across any missing dates).
    full_range = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    df = df.set_index("date").reindex(full_range)
    df.index.name = "date"

    for col in numeric_cols:
        df[col] = df[col].interpolate(method="linear", limit_direction="both")

    df = df.reset_index()

    # humidity_mean may be fully NaN if not provided by the fallback source —
    # drop it in that case rather than modeling on an empty column.
    if "humidity_mean" in df.columns and df["humidity_mean"].isna().all():
        df.drop(columns=["humidity_mean"], inplace=True)

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

    return df


def main():
    df = load_raw()
    df = clean(df)
    df = engineer_features(df)

    # Drop the initial rows where lag features are NaN (can't be computed).
    before = len(df)
    df.dropna(inplace=True)
    after = len(df)
    print(f"Dropped {before - after} rows with incomplete lag/rolling features.")

    df.to_csv(PROCESSED_PATH, index=False)
    print(f"Saved processed dataset with {len(df)} rows and {len(df.columns)} columns to {PROCESSED_PATH}")
    print(df.dtypes)


if __name__ == "__main__":
    main()
