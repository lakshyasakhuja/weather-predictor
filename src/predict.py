"""
predict.py
Future-forecasting utilities built on top of the trained models, for a
given city.

- The seasonal linear/polynomial models only need a future date (they use
  the time trend + cyclical day-of-year), so they can forecast arbitrarily
  far into the future in one shot.
- The Random Forest and LSTM models also use recent lag values of the
  target, so forecasting more than one step ahead requires a recursive
  walk-forward loop where each day's prediction becomes an input for the
  next day.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import city_data_dir, city_models_dir  # noqa: E402


def load_artifacts(city_key: str):
    models_dir = city_models_dir(city_key)
    data_dir = city_data_dir(city_key)

    metadata = json.loads((models_dir / "metadata.json").read_text())
    models = {
        "linear_trend": joblib.load(models_dir / "linear_trend.pkl"),
        "linear_seasonal": joblib.load(models_dir / "linear_seasonal.pkl"),
        "polynomial_seasonal": joblib.load(models_dir / "poly_seasonal.pkl"),
        "random_forest": joblib.load(models_dir / "random_forest.pkl"),
    }
    if metadata.get("lstm_available"):
        import tensorflow as tf

        models["lstm"] = tf.keras.models.load_model(models_dir / "lstm_model.keras")
        models["_lstm_scaler"] = joblib.load(models_dir / "lstm_scaler.pkl")

    processed = pd.read_csv(data_dir / "processed_weather.csv", parse_dates=["date"])
    return models, metadata, processed


def _calendar_features_for_dates(dates: pd.DatetimeIndex, date_min: pd.Timestamp) -> pd.DataFrame:
    df = pd.DataFrame({"date": dates})
    df["time_index"] = (df["date"] - date_min).dt.days
    df["day_of_year"] = df["date"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    return df


def forecast_seasonal_model(model, model_name, metadata, processed, horizon_days) -> pd.DataFrame:
    """One-shot forecast for the two linear models / polynomial model."""
    date_min = processed["date"].min()
    last_date = processed["date"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon_days, freq="D")
    feats = _calendar_features_for_dates(future_dates, date_min)

    feature_cols = metadata["seasonal_features"] if model_name != "linear_trend" else metadata["trend_features"]
    preds = model.predict(feats[feature_cols])
    return pd.DataFrame({"date": future_dates, "predicted_temp_mean": preds, "model": model_name})


def forecast_random_forest(model, metadata, processed, horizon_days) -> pd.DataFrame:
    """Recursive multi-step forecast using lag/rolling features."""
    date_min = processed["date"].min()
    history = processed[["date", "temp_mean"]].copy()
    rf_features = metadata["rf_features"]
    # humidity/pressure (if present in rf_features) aren't forecastable
    # themselves, so hold them at their most recent observed value.
    static_optional = {}
    for col in ("humidity_mean", "pressure_mean"):
        if col in rf_features and col in processed.columns:
            static_optional[col] = processed[col].iloc[-1]

    last_date = processed["date"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon_days, freq="D")

    predictions = []
    for current_date in future_dates:
        calendar = _calendar_features_for_dates(pd.DatetimeIndex([current_date]), date_min).iloc[0]
        recent = history["temp_mean"].values
        row = {
            "time_index": calendar["time_index"],
            "doy_sin": calendar["doy_sin"],
            "doy_cos": calendar["doy_cos"],
            "month": calendar["month"],
            "day_of_week": calendar["day_of_week"],
            "temp_lag_1": recent[-1],
            "temp_lag_2": recent[-2],
            "temp_lag_3": recent[-3],
            "temp_lag_7": recent[-7],
            "temp_lag_14": recent[-14],
            "temp_roll_mean_7": recent[-7:].mean(),
            "temp_roll_std_7": recent[-7:].std(ddof=0),
            "temp_roll_mean_14": recent[-14:].mean(),
            "temp_roll_std_14": recent[-14:].std(ddof=0),
            "temp_roll_mean_30": recent[-30:].mean(),
            "temp_roll_std_30": recent[-30:].std(ddof=0),
            **static_optional,
        }
        x = pd.DataFrame([row])[rf_features]
        pred_temp = model.predict(x)[0]

        predictions.append({"date": current_date, "predicted_temp_mean": pred_temp, "model": "random_forest"})
        history = pd.concat([history, pd.DataFrame({"date": [current_date], "temp_mean": [pred_temp]})], ignore_index=True)

    return pd.DataFrame(predictions)


def forecast_lstm(model, scaler, metadata, processed, horizon_days) -> pd.DataFrame:
    """Recursive multi-step forecast for the LSTM sliding-window model."""
    window = metadata["lstm_meta"]["window"]
    history = processed["temp_mean"].values[-window:].tolist()
    last_date = processed["date"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon_days, freq="D")

    predictions = []
    for current_date in future_dates:
        recent_window = np.array(history[-window:]).reshape(-1, 1)
        scaled_window = scaler.transform(recent_window).flatten().reshape(1, window, 1)
        pred_scaled = model.predict(scaled_window, verbose=0).flatten()[0]
        pred_temp = scaler.inverse_transform([[pred_scaled]])[0][0]

        predictions.append({"date": current_date, "predicted_temp_mean": pred_temp, "model": "lstm"})
        history.append(pred_temp)

    return pd.DataFrame(predictions)


def forecast(city_key: str, model_name: str, horizon_days: int) -> pd.DataFrame:
    models, metadata, processed = load_artifacts(city_key)
    if model_name == "random_forest":
        return forecast_random_forest(models["random_forest"], metadata, processed, horizon_days)
    if model_name == "lstm":
        if "lstm" not in models:
            raise ValueError("LSTM model isn't available for this city (TensorFlow wasn't installed when it was trained).")
        return forecast_lstm(models["lstm"], models["_lstm_scaler"], metadata, processed, horizon_days)
    return forecast_seasonal_model(models[model_name], model_name, metadata, processed, horizon_days)


if __name__ == "__main__":
    for name in ["linear_trend", "linear_seasonal", "polynomial_seasonal", "random_forest", "lstm"]:
        try:
            out = forecast("new_delhi", name, horizon_days=10)
            print(f"\n{name}:")
            print(out.to_string(index=False))
        except Exception as e:
            print(f"\n{name}: skipped ({e})")
