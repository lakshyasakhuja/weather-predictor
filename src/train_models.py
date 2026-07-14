"""
train_models.py
Trains and evaluates several regression / ML models for daily mean-
temperature prediction, and runs a classical seasonal-trend decomposition.

Models:
  1. Linear Regression on the time trend only        -> long-term climate trend
  2. Linear Regression on trend + seasonality (cyclical day-of-year) -> seasonal-aware trend
  3. Polynomial Regression (degree 2) on trend + seasonality
  4. Random Forest Regressor on the full feature set (lags, rolling
     averages, calendar features) -> short-term forecasting

A time-based (chronological) train/test split is used throughout, since
shuffling would leak future information into the past for a time series.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.tsa.seasonal import seasonal_decompose

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

PROCESSED_PATH = DATA_DIR / "processed_weather.csv"
TARGET = "temp_mean"

TREND_FEATURES = ["time_index"]
SEASONAL_FEATURES = ["time_index", "doy_sin", "doy_cos"]
RF_FEATURES = SEASONAL_FEATURES + [
    "month",
    "day_of_week",
    "temp_lag_1",
    "temp_lag_2",
    "temp_lag_3",
    "temp_lag_7",
    "temp_lag_14",
    "temp_roll_mean_7",
    "temp_roll_std_7",
    "temp_roll_mean_14",
    "temp_roll_std_14",
    "temp_roll_mean_30",
    "temp_roll_std_30",
]

TEST_FRACTION = 0.15


def time_based_split(df: pd.DataFrame, test_fraction: float = TEST_FRACTION):
    n_test = int(len(df) * test_fraction)
    train_df = df.iloc[: len(df) - n_test].copy()
    test_df = df.iloc[len(df) - n_test :].copy()
    return train_df, test_df


def evaluate(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_all():
    df = pd.read_csv(PROCESSED_PATH, parse_dates=["date"])
    train_df, test_df = time_based_split(df)

    metrics = {}

    # --- 1. Linear trend only ---------------------------------------------
    lin_trend = LinearRegression()
    lin_trend.fit(train_df[TREND_FEATURES], train_df[TARGET])
    pred = lin_trend.predict(test_df[TREND_FEATURES])
    metrics["linear_trend"] = evaluate(test_df[TARGET], pred)
    joblib.dump(lin_trend, MODELS_DIR / "linear_trend.pkl")

    # --- 2. Linear trend + seasonality --------------------------------------
    lin_seasonal = LinearRegression()
    lin_seasonal.fit(train_df[SEASONAL_FEATURES], train_df[TARGET])
    pred = lin_seasonal.predict(test_df[SEASONAL_FEATURES])
    metrics["linear_seasonal"] = evaluate(test_df[TARGET], pred)
    joblib.dump(lin_seasonal, MODELS_DIR / "linear_seasonal.pkl")

    # --- 3. Polynomial regression (degree 2) on trend + seasonality --------
    poly_model = make_pipeline(
        PolynomialFeatures(degree=2, include_bias=False),
        LinearRegression(),
    )
    poly_model.fit(train_df[SEASONAL_FEATURES], train_df[TARGET])
    pred = poly_model.predict(test_df[SEASONAL_FEATURES])
    metrics["polynomial_seasonal"] = evaluate(test_df[TARGET], pred)
    joblib.dump(poly_model, MODELS_DIR / "poly_seasonal.pkl")

    # --- 4. Random Forest on full feature set -------------------------------
    rf_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    rf_model.fit(train_df[RF_FEATURES], train_df[TARGET])
    pred = rf_model.predict(test_df[RF_FEATURES])
    metrics["random_forest"] = evaluate(test_df[TARGET], pred)
    joblib.dump(rf_model, MODELS_DIR / "random_forest.pkl")

    feature_importances = dict(
        zip(RF_FEATURES, rf_model.feature_importances_.round(4).tolist())
    )

    # --- Seasonal decomposition (trend / seasonal / residual) ---------------
    series = df.set_index("date")[TARGET].asfreq("D")
    series = series.interpolate()
    decomposition = seasonal_decompose(series, model="additive", period=365)
    decomp_df = pd.DataFrame(
        {
            "date": series.index,
            "observed": decomposition.observed.values,
            "trend": decomposition.trend.values,
            "seasonal": decomposition.seasonal.values,
            "residual": decomposition.resid.values,
        }
    )
    decomp_df.to_csv(DATA_DIR / "seasonal_decomposition.csv", index=False)

    # --- Persist everything needed by the app -------------------------------
    metadata = {
        "target": TARGET,
        "trend_features": TREND_FEATURES,
        "seasonal_features": SEASONAL_FEATURES,
        "rf_features": RF_FEATURES,
        "test_fraction": TEST_FRACTION,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "metrics": metrics,
        "feature_importances": feature_importances,
        "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
    }
    with open(MODELS_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print("\nBest model by RMSE:", min(metrics, key=lambda k: metrics[k]["rmse"]))
    print("\nTop feature importances (Random Forest):")
    for feat, imp in sorted(feature_importances.items(), key=lambda x: -x[1])[:8]:
        print(f"  {feat:20s} {imp}")

    return metrics


if __name__ == "__main__":
    train_all()
