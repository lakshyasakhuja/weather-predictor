"""
app.py — Weather Data Analysis & Prediction Dashboard

A Streamlit web app for a final-year AI/ML project: explores historical
daily weather data, visualizes seasonal trend decomposition, compares
regression / ML models, and forecasts future temperatures.

Run with:  streamlit run app.py
"""

import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
SRC_DIR = BASE_DIR / "src"

sys.path.insert(0, str(SRC_DIR))

st.set_page_config(
    page_title="Weather Data Analysis & Prediction",
    page_icon="⛅",
    layout="wide",
)

MODEL_LABELS = {
    "linear_trend": "Linear Regression (trend only)",
    "linear_seasonal": "Linear Regression (trend + seasonality)",
    "polynomial_seasonal": "Polynomial Regression (degree 2)",
    "random_forest": "Random Forest (ML)",
}


# --------------------------------------------------------------------------
# Data / model loading helpers (cached so the app stays fast on rerun)
# --------------------------------------------------------------------------

def artifacts_exist() -> bool:
    required = [
        DATA_DIR / "historical_weather.csv",
        DATA_DIR / "processed_weather.csv",
        DATA_DIR / "seasonal_decomposition.csv",
        MODELS_DIR / "metadata.json",
    ]
    return all(p.exists() for p in required)


def run_pipeline():
    """Runs fetch -> preprocess -> train as subprocesses, streaming logs."""
    steps = [
        ("Fetching historical weather data...", "fetch_data.py"),
        ("Cleaning data & engineering features...", "preprocess.py"),
        ("Training models & evaluating...", "train_models.py"),
    ]
    progress = st.progress(0.0, text="Starting pipeline...")
    log_box = st.expander("Pipeline logs", expanded=False)
    for i, (label, script) in enumerate(steps):
        progress.progress(i / len(steps), text=label)
        result = subprocess.run(
            [sys.executable, str(SRC_DIR / script)],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
        )
        with log_box:
            st.text(f"--- {script} ---\n{result.stdout}\n{result.stderr}")
        if result.returncode != 0:
            st.error(f"{script} failed. See logs above.")
            st.stop()
    progress.progress(1.0, text="Done!")


@st.cache_data
def load_processed():
    return pd.read_csv(DATA_DIR / "processed_weather.csv", parse_dates=["date"])


@st.cache_data
def load_decomposition():
    return pd.read_csv(DATA_DIR / "seasonal_decomposition.csv", parse_dates=["date"])


@st.cache_data
def load_metadata():
    return json.loads((MODELS_DIR / "metadata.json").read_text())


@st.cache_resource
def load_models():
    return {
        "linear_trend": joblib.load(MODELS_DIR / "linear_trend.pkl"),
        "linear_seasonal": joblib.load(MODELS_DIR / "linear_seasonal.pkl"),
        "polynomial_seasonal": joblib.load(MODELS_DIR / "poly_seasonal.pkl"),
        "random_forest": joblib.load(MODELS_DIR / "random_forest.pkl"),
    }


@st.cache_data
def load_city_meta():
    meta_path = DATA_DIR / "meta.txt"
    if not meta_path.exists():
        return {"city": "Unknown"}
    lines = meta_path.read_text().splitlines()
    return dict(line.split("=", 1) for line in lines if "=" in line)


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

st.sidebar.title("⛅ Weather Predictor")
st.sidebar.caption("Final-year AI/ML project — historical analysis & forecasting")

if not artifacts_exist():
    st.sidebar.warning("No data/models found yet.")
    if st.sidebar.button("Run pipeline now (fetch → process → train)", type="primary"):
        run_pipeline()
        st.rerun()
    st.stop()

st.sidebar.success("Data & models are ready.")
if st.sidebar.button("🔄 Re-fetch data & retrain models"):
    load_processed.clear()
    load_decomposition.clear()
    load_metadata.clear()
    load_models.clear()
    load_city_meta.clear()
    run_pipeline()
    st.rerun()

page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Historical Trends", "Seasonal Decomposition", "Model Comparison", "Forecast"],
)

processed = load_processed()
decomposition = load_decomposition()
metadata = load_metadata()
models = load_models()
city_meta = load_city_meta()


# --------------------------------------------------------------------------
# Page: Overview
# --------------------------------------------------------------------------

if page == "Overview":
    st.title("Weather Data Analysis & Prediction")
    st.markdown(
        "This project analyzes historical daily weather data and predicts "
        "future temperature trends using classical regression and machine "
        "learning techniques."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Location", city_meta.get("city", "Unknown"))
    col2.metric("Records", f"{len(processed):,} days")
    col3.metric("Date range", f"{metadata['date_min']} → {metadata['date_max']}")
    best_model = min(metadata["metrics"], key=lambda k: metadata["metrics"][k]["rmse"])
    col4.metric("Best model (test RMSE)", MODEL_LABELS[best_model])

    st.subheader("Sample of the processed dataset")
    st.dataframe(
        processed[["date", "temp_max", "temp_min", "temp_mean", "precipitation", "wind_speed_max"]].tail(15),
        use_container_width=True,
    )

    st.subheader("How it works")
    st.markdown(
        """
1. **Fetch** — daily weather data is pulled from a public API (Open-Meteo),
   with an automatic offline fallback to a bundled historical dataset if
   no internet connection is available.
2. **Preprocess** — missing days are interpolated, and calendar, lag, and
   rolling-average features are engineered.
3. **Train** — four models are trained on a chronological train/test split:
   two linear regressions, a polynomial regression, and a Random Forest.
4. **Decompose** — the temperature series is split into trend, seasonal,
   and residual components using classical time-series decomposition.
5. **Forecast** — any trained model can project temperatures forward,
   from a few days to several years.
        """
    )

# --------------------------------------------------------------------------
# Page: Historical Trends
# --------------------------------------------------------------------------

elif page == "Historical Trends":
    st.title("Historical Weather Trends")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=processed["date"], y=processed["temp_max"], name="Max Temp", line=dict(color="#e74c3c", width=1)))
    fig.add_trace(go.Scatter(x=processed["date"], y=processed["temp_mean"], name="Mean Temp", line=dict(color="#2c3e50", width=1.5)))
    fig.add_trace(go.Scatter(x=processed["date"], y=processed["temp_min"], name="Min Temp", line=dict(color="#3498db", width=1)))
    fig.update_layout(title="Daily Temperature Range Over Time", xaxis_title="Date", yaxis_title="Temperature (°C)", height=450)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        monthly = processed.groupby("month")[["temp_max", "temp_min", "temp_mean"]].mean().reset_index()
        monthly["month_name"] = pd.to_datetime(monthly["month"], format="%m").dt.strftime("%b")
        fig2 = px.bar(monthly, x="month_name", y="temp_mean", title="Average Temperature by Month (Climatology)", labels={"temp_mean": "Avg Mean Temp (°C)", "month_name": "Month"})
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        yearly = processed.groupby("year")["temp_mean"].mean().reset_index()
        fig3 = px.bar(yearly, x="year", y="temp_mean", title="Average Temperature by Year", labels={"temp_mean": "Avg Mean Temp (°C)"})
        st.plotly_chart(fig3, use_container_width=True)

    if "precipitation" in processed.columns:
        fig4 = px.bar(processed, x="date", y="precipitation", title="Daily Precipitation")
        fig4.update_layout(height=350)
        st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Correlation between weather variables")
    corr_cols = [c for c in ["temp_max", "temp_min", "temp_mean", "precipitation", "wind_speed_max"] if c in processed.columns]
    corr = processed[corr_cols].corr()
    fig5 = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", title="Correlation Matrix")
    st.plotly_chart(fig5, use_container_width=True)

# --------------------------------------------------------------------------
# Page: Seasonal Decomposition
# --------------------------------------------------------------------------

elif page == "Seasonal Decomposition":
    st.title("Seasonal-Trend Decomposition")
    st.markdown(
        "The daily mean temperature series is decomposed into three additive "
        "components using classical decomposition (`statsmodels.seasonal_decompose`, "
        "annual period = 365 days): **Observed = Trend + Seasonal + Residual**."
    )

    for comp, color in [("observed", "#2c3e50"), ("trend", "#e67e22"), ("seasonal", "#27ae60"), ("residual", "#c0392b")]:
        fig = px.line(decomposition, x="date", y=comp, title=comp.capitalize())
        fig.update_traces(line_color=color)
        fig.update_layout(height=280, margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    st.info(
        "The **trend** component reveals the long-run climate signal once "
        "seasonality is removed. The **seasonal** component repeats every "
        "year and captures the expected temperature cycle from winter to "
        "summer. The **residual** is day-to-day weather noise the model "
        "can't explain from calendar effects alone."
    )

# --------------------------------------------------------------------------
# Page: Model Comparison
# --------------------------------------------------------------------------

elif page == "Model Comparison":
    st.title("Model Comparison")
    st.markdown(f"Evaluated on a chronological hold-out test set of the most recent **{metadata['n_test']}** days (trained on the preceding **{metadata['n_train']}** days).")

    metrics_df = pd.DataFrame(metadata["metrics"]).T.reset_index().rename(columns={"index": "model"})
    metrics_df["model_label"] = metrics_df["model"].map(MODEL_LABELS)

    col1, col2, col3 = st.columns(3)
    with col1:
        fig = px.bar(metrics_df, x="model_label", y="rmse", title="RMSE (lower is better)", color="model_label")
        fig.update_layout(showlegend=False, xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.bar(metrics_df, x="model_label", y="mae", title="MAE (lower is better)", color="model_label")
        fig.update_layout(showlegend=False, xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
    with col3:
        fig = px.bar(metrics_df, x="model_label", y="r2", title="R² (higher is better)", color="model_label")
        fig.update_layout(showlegend=False, xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        metrics_df[["model_label", "rmse", "mae", "r2"]].rename(
            columns={"model_label": "Model", "rmse": "RMSE (°C)", "mae": "MAE (°C)", "r2": "R²"}
        ).round(3),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Random Forest — feature importance")
    fi = pd.Series(metadata["feature_importances"]).sort_values(ascending=True)
    fig_fi = px.bar(fi, orientation="h", title="Which features drive the Random Forest predictions?")
    fig_fi.update_layout(showlegend=False, xaxis_title="Importance", yaxis_title="")
    st.plotly_chart(fig_fi, use_container_width=True)

    st.subheader("Predicted vs. Actual (test period)")
    test_df = processed.iloc[len(processed) - metadata["n_test"]:].copy()
    chosen = st.selectbox("Model", list(MODEL_LABELS.keys()), format_func=lambda k: MODEL_LABELS[k], key="cmp_model")
    feature_cols = metadata["trend_features"] if chosen == "linear_trend" else (metadata["rf_features"] if chosen == "random_forest" else metadata["seasonal_features"])
    test_df["predicted"] = models[chosen].predict(test_df[feature_cols])

    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(x=test_df["date"], y=test_df["temp_mean"], name="Actual", line=dict(color="#2c3e50")))
    fig6.add_trace(go.Scatter(x=test_df["date"], y=test_df["predicted"], name="Predicted", line=dict(color="#e74c3c", dash="dash")))
    fig6.update_layout(title=f"{MODEL_LABELS[chosen]}: Predicted vs. Actual", height=450)
    st.plotly_chart(fig6, use_container_width=True)

# --------------------------------------------------------------------------
# Page: Forecast
# --------------------------------------------------------------------------

elif page == "Forecast":
    st.title("Forecast Future Temperatures")

    from predict import forecast  # local import (src/ added to sys.path above)

    col1, col2 = st.columns(2)
    with col1:
        model_choice = st.selectbox("Model", list(MODEL_LABELS.keys()), format_func=lambda k: MODEL_LABELS[k])
    with col2:
        default_horizon = 14 if model_choice == "random_forest" else 90
        max_horizon = 60 if model_choice == "random_forest" else 3650
        horizon = st.slider("Days to forecast ahead", min_value=1, max_value=max_horizon, value=default_horizon)

    if model_choice == "random_forest" and horizon > 30:
        st.warning(
            "The Random Forest model forecasts recursively using its own "
            "predictions as future lag features, so accuracy degrades "
            "noticeably beyond ~2-4 weeks. For long-range forecasts, use "
            "one of the seasonal regression models instead."
        )

    with st.spinner("Generating forecast..."):
        forecast_df = forecast(model_choice, horizon)

    recent_history = processed[["date", "temp_mean"]].tail(120).rename(columns={"temp_mean": "value"})
    recent_history["type"] = "Historical"
    future = forecast_df.rename(columns={"predicted_temp_mean": "value"})[["date", "value"]]
    future["type"] = "Forecast"
    combined = pd.concat([recent_history, future], ignore_index=True)

    fig = px.line(combined, x="date", y="value", color="type", title=f"{MODEL_LABELS[model_choice]} — {horizon}-day Forecast", color_discrete_map={"Historical": "#2c3e50", "Forecast": "#e74c3c"})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Forecast values")
    st.dataframe(forecast_df.rename(columns={"predicted_temp_mean": "Predicted Mean Temp (°C)", "date": "Date"})[["Date", "Predicted Mean Temp (°C)"]].round(2), use_container_width=True, hide_index=True)

    csv = forecast_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download forecast as CSV", csv, file_name=f"forecast_{model_choice}_{horizon}days.csv", mime="text/csv")

    st.subheader("Predict a single date")
    pick_date = st.date_input("Pick a future date", value=processed["date"].max() + pd.Timedelta(days=7))
    target_date = pd.Timestamp(pick_date)
    last_date = processed["date"].max()
    if target_date <= last_date:
        st.error("Please pick a date after the historical data range.")
    else:
        needed_horizon = (target_date - last_date).days
        if model_choice == "random_forest" and needed_horizon > 60:
            st.error("Random Forest forecasts are only supported up to 60 days ahead. Pick a closer date or use a seasonal regression model.")
        else:
            single_forecast = forecast(model_choice, needed_horizon)
            value = single_forecast.iloc[-1]["predicted_temp_mean"]
            st.metric(f"Predicted mean temperature on {target_date.date()}", f"{value:.1f} °C")
