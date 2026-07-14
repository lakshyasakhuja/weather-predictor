# Weather Data Analysis and Prediction

A final-year AI/ML project that analyzes historical daily weather data and
predicts future temperature trends using classical regression, machine
learning, and deep learning techniques, delivered as an interactive
multi-city web dashboard.

## Project overview

**Problem statement:** Given historical daily weather observations, can we
model and forecast future temperature trends? This project builds a
complete pipeline — data acquisition, preprocessing, feature engineering,
model training/evaluation, and a forecasting web interface — around that
question, across five cities.

**What it demonstrates:**
- Working with real-world time series data across multiple locations
- Feature engineering for time series (lag features, rolling statistics,
  cyclical calendar encoding, weather predictors like humidity/pressure)
- Classical statistical decomposition (trend / seasonality / residual)
- Comparing regression techniques (Linear, Polynomial) against an ML model
  (Random Forest) and a deep learning model (LSTM)
- Proper time-based train/test evaluation (no data leakage from shuffling)
- Deploying a model behind an interactive, deployable web UI

## Architecture

```
weather-predictor/
├── app.py                     # Streamlit web dashboard (the "website")
├── requirements.txt
├── data/
│   └── <city_key>/
│       ├── historical_weather.csv       # raw fetched data
│       ├── processed_weather.csv        # cleaned + feature-engineered data
│       ├── seasonal_decomposition.csv   # trend/seasonal/residual components
│       └── meta.txt                     # which data source was used
├── models/
│   └── <city_key>/
│       ├── linear_trend.pkl
│       ├── linear_seasonal.pkl
│       ├── poly_seasonal.pkl
│       ├── random_forest.pkl
│       ├── lstm_model.keras             # if TensorFlow was installed when trained
│       ├── lstm_scaler.pkl
│       └── metadata.json                # feature lists + evaluation metrics
└── src/
    ├── config.py                # city registry (lat/lon, offline fallback source)
    ├── fetch_data.py            # Step 1: acquire data (per city)
    ├── preprocess.py            # Step 2: clean + engineer features (per city)
    ├── train_models.py          # Step 3: train + evaluate models (per city)
    └── predict.py               # Step 4: forecasting utilities (per city)
```

## Cities

| City key | Location | Live source | Offline fallback |
|---|---|---|---|
| `new_delhi` | New Delhi, India | Open-Meteo API | Real NOAA Seattle daily record (via `vega_datasets`) — clearly labeled as a stand-in, not actually New Delhi |
| `mumbai` | Mumbai, India | Open-Meteo API | Synthetic seasonal model (clearly labeled, not real observations) |
| `bengaluru` | Bengaluru, India | Open-Meteo API | Synthetic seasonal model |
| `san_francisco` | San Francisco, USA | Open-Meteo API | Real NOAA San Francisco hourly-temp record (via `vega_datasets`) |
| `london` | London, UK | Open-Meteo API | Synthetic seasonal model |

Data is pulled from the [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)
(free, no API key) — ~15 years of daily max/min/mean temperature,
precipitation, wind speed, humidity, and pressure — for whichever cities
you select.

**If no internet connection is available**, `fetch_data.py` automatically
falls back per city as shown in the table above, so the project always
runs end-to-end even offline. The dashboard sidebar always shows exactly
which source is in use, and flags synthetic data with a warning banner so
results are never mistaken for real observations. Add or edit cities in
`src/config.py` (set `lat`/`lon`/`label`, and optionally a
`climate_normal_mean`/`climate_normal_amplitude` used only for the
synthetic fallback).

## Methodology

### 1. Preprocessing (`src/preprocess.py`)
- Reindexes to a continuous daily calendar and interpolates any gaps
- Engineers calendar features: year, month, day-of-week, day-of-year
- Cyclical encoding of day-of-year (`sin`/`cos`) so the model understands
  that December 31st is adjacent to January 1st
- Lag features (previous 1, 2, 3, 7, 14 days' temperature)
- Rolling mean/std over 7, 14, and 30-day windows
- Same-day humidity and surface pressure as additional predictors, when
  the data source provides them

### 2. Models (`src/train_models.py`)
Five models are trained on a **chronological** train/test split (last 15%
of days held out — never shuffled, since shuffling a time series leaks
future information into training):

| Model | Features used | Purpose |
|---|---|---|
| Linear Regression (trend only) | day index | isolates the raw long-term trend |
| Linear Regression (trend + seasonality) | day index, sin/cos of day-of-year | adds yearly seasonal cycle |
| Polynomial Regression (degree 2) | same as above, polynomial-expanded | captures curvature in the seasonal cycle |
| Random Forest Regressor | all of the above + lags + rolling stats + humidity/pressure | captures short-term autocorrelation |
| LSTM (deep learning) | sliding 30-day window of past temperatures | sequence-based short-term forecasting |

Each model is evaluated with **RMSE**, **MAE**, and **R²** on the held-out
test period; results are saved to `models/<city>/metadata.json` and shown
on the dashboard's Model Comparison page along with Random Forest feature
importances. The LSTM step lazily imports TensorFlow — if it isn't
installed, LSTM is skipped and the rest of the pipeline still runs.

### 3. Seasonal decomposition
`statsmodels.tsa.seasonal.seasonal_decompose` (additive, annual period)
splits the temperature series into **Observed = Trend + Seasonal +
Residual**, visualized on the Seasonal Decomposition page.

### 4. Forecasting (`src/predict.py`)
- The linear/polynomial models only depend on calendar features, so they
  can forecast in one shot arbitrarily far into the future (days to years)
  — well suited to long-range trend projection.
- The Random Forest and LSTM models also depend on their own recent
  values, so they forecast **recursively**: each predicted day is fed back
  in as the input for the next day. This is more accurate short-term
  (days to ~2-4 weeks) but compounds error further out, so the app caps
  their horizon at 60 days.

## The website (Streamlit dashboard)

`app.py` provides six pages, navigable from the sidebar, plus a city
selector at the top:

1. **Overview** — dataset summary, best-performing model, sample data,
   data source transparency
2. **Historical Trends** — interactive temperature/precipitation charts,
   monthly climatology, yearly averages, correlation matrix
3. **Seasonal Decomposition** — trend / seasonal / residual plots
4. **Model Comparison** — RMSE/MAE/R² bar charts, feature importances,
   predicted-vs-actual chart on the test period (works for LSTM too)
5. **Forecast** — pick a model and horizon (or a specific future date) and
   get an interactive forecast chart plus a downloadable CSV
6. **Multi-City Comparison** — side-by-side climatology and best-model
   summary across every city that's been prepared, with a clear warning
   for any cities still running on synthetic offline data

The sidebar has a **"Prepare more cities / retrain"** panel that reruns
the pipeline (fetch → preprocess → train) live from the UI for whichever
cities you select.

## How to run it locally

```bash
cd weather-predictor
pip install -r requirements.txt

# (optional) run the pipeline manually first for one or more cities — the
# app will also do this automatically on first launch if no data is found:
python src/fetch_data.py --city new_delhi
python src/preprocess.py --city new_delhi
python src/train_models.py --city new_delhi
# omit --city to run all 5 configured cities

# launch the website
streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`) in
your browser.

> Note: `tensorflow` is a large dependency (a few hundred MB). If you'd
> rather skip the LSTM model, just don't install it — everything else
> (Linear/Polynomial/Random Forest, all dashboard pages) works fine
> without it; the app detects its absence and hides the LSTM option.

## Deploying to Streamlit Community Cloud

This repo is ready to deploy as-is — `app.py` at the root and
`requirements.txt` are exactly what Streamlit Cloud expects.

1. Push this repo to GitHub (see the main project chat for the exact git
   commands if you haven't already).
2. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in
   with your GitHub account.
3. Click **"Create app"** → **"Deploy a public app from GitHub"**.
4. Fill in:
   - **Repository:** `lakshyasakhuja/weather-predictor` (or your repo path)
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. Click **Advanced settings** and set the **Python version** to match
   what you tested locally (3.10+ recommended), so TensorFlow installs
   cleanly.
6. Click **Deploy**. The first deploy will take a few minutes since it
   installs everything in `requirements.txt` (TensorFlow is the slowest
   part). Streamlit Cloud has full internet access, so `fetch_data.py`
   will pull **genuine live data** for every city — no offline fallback
   needed once deployed.
7. Once live, use the sidebar's **"Prepare more cities / retrain"** panel
   from the deployed app itself to fetch and train each city (this avoids
   committing large model files to git, though committing `data/` and
   `models/` also works if you'd rather ship pre-trained artifacts).
8. You'll get a public URL like `https://<your-app-name>.streamlit.app`
   you can share or put on your resume/report.

**If you'd rather ship pre-trained models instead of retraining on first
load:** commit the `data/` and `models/` folders to git before pushing
(they're not excluded by `.gitignore`), so the deployed app has them
immediately. Keep in mind GitHub has a 100 MB per-file limit — the
`random_forest.pkl` and `lstm_model.keras` files here are well under that,
but double-check if you increase `n_estimators` or model complexity.

## Results (offline fallback data, illustrative — not final numbers)

Random Forest and LSTM consistently outperform the linear/polynomial
models, since they see recent lag values in addition to the seasonal
cycle. A pure linear trend line explains almost nothing about daily
temperature on its own (confirming temperature is dominated by
seasonality, not a simple drift). San Francisco's numbers are noisier
because its offline fallback is only one year of data — too short to
reliably fit a multi-year trend, a good illustration of why time series
models need enough history to generalize. Exact numbers will differ once
run against live data with the full ~15-year history for each city.

## Possible extensions

- Multi-city comparison and LSTM are already implemented — see above
- Add live climate indices (e.g. ENSO/ONI) as predictors — not currently
  implemented, since it requires an additional external data feed at
  monthly resolution; the RF feature list in `train_models.py` is
  structured to make adding a new predictor straightforward
- GRU or Transformer-based sequence models as an LSTM alternative
- Auto-refresh scheduling (e.g. a daily cron job re-running the pipeline)
