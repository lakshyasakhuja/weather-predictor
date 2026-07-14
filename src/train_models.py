"""
train_models.py
Trains and evaluates several regression / ML / deep-learning models for
daily mean-temperature prediction, for a given city, and runs a classical
seasonal-trend decomposition.

Models:
  1. Linear Regression on the time trend only        -> long-term climate trend
  2. Linear Regression on trend + seasonality (cyclical day-of-year) -> seasonal-aware trend
  3. Polynomial Regression (degree 2) on trend + seasonality
  4. Random Forest Regressor on the full feature set (lags, rolling
     averages, calendar features, humidity/pressure if available) -> short-term forecasting
  5. LSTM (deep learning, optional — requires TensorFlow) on a sliding
     window of recent temperatures -> sequence-based short-term forecasting

A time-based (chronological) train/test split is used throughout, since
shuffling would leak future information into the past for a time series.
"""

import json
import sys
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CITIES, city_data_dir, city_models_dir  # noqa: E402

TARGET = "temp_mean"

TREND_FEATURES = ["time_index"]
SEASONAL_FEATURES = ["time_index", "doy_sin", "doy_cos"]
OPTIONAL_FEATURES = ["humidity_mean", "pressure_mean"]

TEST_FRACTION = 0.15


def rf_feature_list(df: pd.DataFrame) -> list:
    base = SEASONAL_FEATURES + [
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
    for col in OPTIONAL_FEATURES:
        if col in df.columns:
            base.append(col)
    return base


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


def train_lstm(train_df, test_df, models_dir, window=30, epochs=30):
    """
    Trains a small LSTM on a sliding window of past `window` days of
    temp_mean to predict the next day's temp_mean. Requires TensorFlow;
    if it isn't installed, this is skipped gracefully (lazy import) and
    the rest of the pipeline still completes.
    """
    try:
        import tensorflow as tf
        from sklearn.preprocessing import MinMaxScaler
    except ImportError:
        print("[lstm] TensorFlow not installed — skipping LSTM training. "
              "Install with `pip install tensorflow` to enable it.")
        return None, None

    tf.random.set_seed(42)

    series = pd.concat([train_df[TARGET], test_df[TARGET]]).reset_index(drop=True)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()

    n_train = len(train_df)

    def make_windows(values, start, end):
        X, y = [], []
        for i in range(max(start, window), end):
            X.append(values[i - window : i])
            y.append(values[i])
        return np.array(X), np.array(y)

    X_train, y_train = make_windows(scaled, 0, n_train)
    X_test, y_test = make_windows(scaled, n_train, len(scaled))

    if len(X_train) < 50 or len(X_test) < 5:
        print("[lstm] Not enough data for a sequence model — skipping.")
        return None, None

    X_train = X_train.reshape((-1, window, 1))
    X_test = X_test.reshape((-1, window, 1))

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(window, 1)),
            tf.keras.layers.LSTM(32, return_sequences=False),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    early_stop = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=4, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0, callbacks=[early_stop])

    pred_scaled = model.predict(X_test, verbose=0).flatten()
    pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    metrics = evaluate(actual, pred)

    model_path = models_dir / "lstm_model.keras"
    scaler_path = models_dir / "lstm_scaler.pkl"
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    print(f"[lstm] Saved model -> {model_path}")

    return metrics, {"window": window}


def train_city(city_key: str):
    data_dir = city_data_dir(city_key)
    models_dir = city_models_dir(city_key)

    df = pd.read_csv(data_dir / "processed_weather.csv", parse_dates=["date"])
    train_df, test_df = time_based_split(df)
    rf_features = rf_feature_list(df)

    metrics = {}

    lin_trend = LinearRegression()
    lin_trend.fit(train_df[TREND_FEATURES], train_df[TARGET])
    metrics["linear_trend"] = evaluate(test_df[TARGET], lin_trend.predict(test_df[TREND_FEATURES]))
    joblib.dump(lin_trend, models_dir / "linear_trend.pkl")

    lin_seasonal = LinearRegression()
    lin_seasonal.fit(train_df[SEASONAL_FEATURES], train_df[TARGET])
    metrics["linear_seasonal"] = evaluate(test_df[TARGET], lin_seasonal.predict(test_df[SEASONAL_FEATURES]))
    joblib.dump(lin_seasonal, models_dir / "linear_seasonal.pkl")

    poly_model = make_pipeline(PolynomialFeatures(degree=2, include_bias=False), LinearRegression())
    poly_model.fit(train_df[SEASONAL_FEATURES], train_df[TARGET])
    metrics["polynomial_seasonal"] = evaluate(test_df[TARGET], poly_model.predict(test_df[SEASONAL_FEATURES]))
    joblib.dump(poly_model, models_dir / "poly_seasonal.pkl")

    rf_model = RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=2, random_state=42, n_jobs=-1)
    rf_model.fit(train_df[rf_features], train_df[TARGET])
    metrics["random_forest"] = evaluate(test_df[TARGET], rf_model.predict(test_df[rf_features]))
    joblib.dump(rf_model, models_dir / "random_forest.pkl")
    feature_importances = dict(zip(rf_features, rf_model.feature_importances_.round(4).tolist()))

    lstm_metrics, lstm_meta = train_lstm(train_df, test_df, models_dir)
    if lstm_metrics is not None:
        metrics["lstm"] = lstm_metrics

    series = df.set_index("date")[TARGET].asfreq("D").interpolate()
    period = 365 if len(series) >= 730 else max(2, len(series) // 2)
    decomposition = seasonal_decompose(series, model="additive", period=period)
    decomp_df = pd.DataFrame(
        {
            "date": series.index,
            "observed": decomposition.observed.values,
            "trend": decomposition.trend.values,
            "seasonal": decomposition.seasonal.values,
            "residual": decomposition.resid.values,
        }
    )
    decomp_df.to_csv(data_dir / "seasonal_decomposition.csv", index=False)

    metadata = {
        "city_key": city_key,
        "city_label": CITIES[city_key]["label"],
        "target": TARGET,
        "trend_features": TREND_FEATURES,
        "seasonal_features": SEASONAL_FEATURES,
        "rf_features": rf_features,
        "lstm_available": lstm_metrics is not None,
        "lstm_meta": lstm_meta,
        "test_fraction": TEST_FRACTION,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "metrics": metrics,
        "feature_importances": feature_importances,
        "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
    }
    (models_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"\n[{city_key}] " + json.dumps(metrics, indent=2))
    print(f"[{city_key}] Best model by RMSE:", min(metrics, key=lambda k: metrics[k]["rmse"]))
    return metrics


def main(city_keys=None):
    city_keys = city_keys or list(CITIES.keys())
    for city_key in city_keys:
        train_city(city_key)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--city", action="append", help="City key to train (repeatable). Default: all configured cities.")
    args = parser.parse_args()
    main(args.city)
