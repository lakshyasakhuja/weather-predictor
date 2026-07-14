# Weather Data Analysis and Prediction

A final-year AI/ML project that analyzes historical daily weather data and
predicts future temperature trends using classical regression and machine
learning techniques, delivered as an interactive web dashboard.

## Project overview

**Problem statement:** Given historical daily weather observations, can we
model and forecast future temperature trends? This project builds a
complete pipeline — data acquisition, preprocessing, feature engineering,
model training/evaluation, and a forecasting web interface — around that
question.

**What it demonstrates:**
- Working with real-world time series data
- Feature engineering for time series (lag features, rolling statistics,
  cyclical calendar encoding)
- Classical statistical decomposition (trend / seasonality / residual)
- Comparing regression techniques (Linear, Polynomial) against an ML model
  (Random Forest)
- Proper time-based train/test evaluation (no data leakage from shuffling)
- Deploying a model behind an interactive web UI

## Architecture

```
weather-predictor/
├── app.py                     # Streamlit web dashboard (the "website")
├── requirements.txt
├── data/
│   ├── historical_weather.csv       # raw fetched data
│   ├── processed_weather.csv        # cleaned + feature-engineered data
│   ├── seasonal_decomposition.csv   # trend/seasonal/residual components
│   └── meta.txt                     # which data source was used
├── models/
│   ├── linear_trend.pkl
│   ├── linear_seasonal.pkl
│   ├── poly_seasonal.pkl
│   ├── random_forest.pkl
│   └── metadata.json                # feature lists + evaluation metrics
└── src/
    ├── fetch_data.py           # Step 1: acquire data
    ├── preprocess.py           # Step 2: clean + engineer features
    ├── train_models.py         # Step 3: train + evaluate models
    └── predict.py              # Step 4: forecasting utilities
```

## Data source

Data is pulled from the [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)
(free, no API key), configured by default for **New Delhi, India** with
~15 years of daily history (max/min/mean temperature, precipitation, wind
speed, humidity).

If no internet connection is available, `fetch_data.py` automatically
falls back to a bundled, genuine historical dataset — NOAA's daily Seattle
weather record (2012–2015, via the `vega_datasets` package) — so the
project always runs end-to-end, even offline. Whichever source was used is
recorded in `data/meta.txt` and shown on the dashboard's Overview page.

To point the project at a different city, edit `LATITUDE` / `LONGITUDE` /
`CITY_NAME` at the top of `src/fetch_data.py`.

## Methodology

### 1. Preprocessing (`src/preprocess.py`)
- Reindexes to a continuous daily calendar and interpolates any gaps
- Engineers calendar features: year, month, day-of-week, day-of-year
- Cyclical encoding of day-of-year (`sin`/`cos`) so the model understands
  that December 31st is adjacent to January 1st
- Lag features (previous 1, 2, 3, 7, 14 days' temperature)
- Rolling mean/std over 7, 14, and 30-day windows

### 2. Models (`src/train_models.py`)
Four models are trained on a **chronological** train/test split (last 15%
of days held out — never shuffled, since shuffling a time series leaks
future information into training):

| Model | Features used | Purpose |
|---|---|---|
| Linear Regression (trend only) | day index | isolates the raw long-term trend |
| Linear Regression (trend + seasonality) | day index, sin/cos of day-of-year | adds yearly seasonal cycle |
| Polynomial Regression (degree 2) | same as above, polynomial-expanded | captures curvature in the seasonal cycle |
| Random Forest Regressor | all of the above + lags + rolling stats | captures short-term autocorrelation |

Each model is evaluated with **RMSE**, **MAE**, and **R²** on the held-out
test period; results are saved to `models/metadata.json` and shown on the
dashboard's Model Comparison page along with Random Forest feature
importances.

### 3. Seasonal decomposition
`statsmodels.tsa.seasonal.seasonal_decompose` (additive, 365-day period)
splits the temperature series into **Observed = Trend + Seasonal +
Residual**, visualized on the Seasonal Decomposition page.

### 4. Forecasting (`src/predict.py`)
- The linear/polynomial models only depend on calendar features, so they
  can forecast in one shot arbitrarily far into the future (days to years)
  — well suited to long-range trend projection.
- The Random Forest model also depends on its own recent lag/rolling
  values, so it forecasts **recursively**: each predicted day is fed back
  in as the lag input for the next day. This is more accurate short-term
  (days to ~2-4 weeks) but compounds error further out, so the app caps
  its horizon at 60 days.

## The website (Streamlit dashboard)

`app.py` provides five pages, navigable from the sidebar:

1. **Overview** — dataset summary, best-performing model, sample data
2. **Historical Trends** — interactive temperature/precipitation charts,
   monthly climatology, yearly averages, correlation matrix
3. **Seasonal Decomposition** — trend / seasonal / residual plots
4. **Model Comparison** — RMSE/MAE/R² bar charts, feature importances,
   predicted-vs-actual chart on the test period
5. **Forecast** — pick a model and horizon (or a specific future date) and
   get an interactive forecast chart plus a downloadable CSV

The sidebar also has a **"Re-fetch data & retrain models"** button that
reruns the entire pipeline (fetch → preprocess → train) live from the UI.

## How to run it

```bash
cd weather-predictor
pip install -r requirements.txt

# (optional) run the pipeline manually first — the app will also do this
# automatically on first launch if no data/models are found yet:
python src/fetch_data.py
python src/preprocess.py
python src/train_models.py

# launch the website
streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`) in
your browser.

## Results (on the bundled offline fallback dataset, 2012–2015 daily data)

| Model | RMSE (°C) | MAE (°C) | R² |
|---|---|---|---|
| Linear Regression (trend only) | ~6.9 | ~6.0 | ~-0.13 |
| Linear Regression (trend + seasonality) | ~3.1 | ~2.5 | ~0.77 |
| Polynomial Regression (degree 2) | ~2.9 | ~2.3 | ~0.80 |
| Random Forest | ~2.1 | ~1.7 | ~0.90 |

**Interpretation:** a pure linear trend line explains almost nothing about
daily temperature (confirming that temperature is dominated by seasonality,
not a simple upward/downward drift over a few years). Adding seasonal
(cyclical) features dramatically improves fit. The Random Forest, which
also sees recent lag/rolling temperature values, performs best — it's
capturing short-term weather persistence in addition to the seasonal cycle.
Exact numbers will differ when run against live New Delhi data with 15
years of history.

## Possible extensions

- Add an LSTM/GRU deep learning model for sequence forecasting
- Multi-city comparison
- Incorporate additional predictors (humidity, pressure, ENSO indices)
- Deploy the Streamlit app to Streamlit Community Cloud for a public link
