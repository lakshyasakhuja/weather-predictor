"""
predict.py
Future-forecasting utilities built on top of the trained models.

- The seasonal linear/polynomial models only need a future date (they use
  the time trend + cyclical day-of-year), so they can forecast arbitrarily
  far into the future in one shot.
- The Random Forest model also uses lag/rolling features of the target, so
  forecasting more than one step ahead requires a recursive walk-forward
  loop where each day's prediction becomes an input lag for the next day.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"


def load_artifacts():
    metadata = json.loads((MODELS_DIR / "metadata.json").read_text())
    models = {
        "linear_trend": joblib.load(MODELS_DIR / "linear_trend.pkl"),
        "linear_seasonal": joblib.load(MODELS_DIR / "linear_seasonal.pkl"),
        "polynomial_seasonal": joblib.load(MODELS_DIR / "poly_seasonal.pkl"),
        "random_forest": joblib.load(MODELS_DIR / "random_forest.pkl"),
    }
    processed = pd.read_csv(DATA_DIR / "processed_weather.csv", parse_dates=["date"])
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


def forecast_seasonal_model(model, model_name: str, metadata: dict, processed: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    """One-shot forecast for the two linear models / polynomial model."""
    date_min = processed["date"].min()
    last_date = processed["date"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon_days, freq="D")
    feats = _calendar_features_for_dates(future_dates, date_min)

    feature_cols = metadata["seasonal_features"] if model_name != "linear_trend" else metadata["trend_features"]
    preds = model.predict(feats[feature_cols])
    return pd.DataFrame({"date": future_dates, "predicted_temp_mean": preds, "model": model_name})


def forecast_random_forest(model, metadata: dict, processed: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    """
    Recursive multi-step forecast: predict one day, append it to the
    history, recompute lag/rolling features, predict the next day, etc.
    Best suited to short horizons (a few days to a couple of weeks) since
    errors compound the further out you forecast.
    """
    date_min = processed["date"].min()
    history = processed[["date", "temp_mean"]].copy()
    rf_features = metadata["rf_features"]

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
        }
        x = pd.DataFrame([row])[rf_features]
        pred_temp = model.predict(x)[0]

        predictions.append({"date": current_date, "predicted_temp_mean": pred_temp, "model": "random_forest"})
        history = pd.concat(
            [history, pd.DataFrame({"date": [current_date], "temp_mean": [pred_temp]})],
            ignore_index=True,
        )

    return pd.DataFrame(predictions)


def forecast(model_name: str, horizon_days: int) -> pd.DataFrame:
    models, metadata, processed = load_artifacts()
    model = models[model_name]
    if model_name == "random_forest":
        return forecast_random_forest(model, metadata, processed, horizon_days)
    return forecast_seasonal_model(model, model_name, metadata, processed, horizon_days)


if __name__ == "__main__":
    for name in ["linear_trend", "linear_seasonal", "polynomial_seasonal", "random_forest"]:
        out = forecast(name, horizon_days=10)
        print(f"\n{name}:")
        print(out.to_string(index=False))
