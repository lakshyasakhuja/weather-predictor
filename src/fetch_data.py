"""
fetch_data.py
Downloads historical daily weather data from the Open-Meteo Archive API
(free, no API key required) for a chosen city and saves it as a CSV.

Data source: https://open-meteo.com/en/docs/historical-weather-api
"""

import requests
import pandas as pd
from pathlib import Path
from datetime import date, timedelta

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# City configuration — change these to build the project for a different location.
CITY_NAME = "New Delhi, India"
LATITUDE = 28.6139
LONGITUDE = 77.2090

# Pull ~15 years of daily history up to yesterday (archive API has a short lag).
END_DATE = date.today() - timedelta(days=5)
START_DATE = END_DATE.replace(year=END_DATE.year - 15)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_historical_weather(
    lat: float = LATITUDE,
    lon: float = LONGITUDE,
    start: date = START_DATE,
    end: date = END_DATE,
) -> pd.DataFrame:
    """Fetch daily historical weather data and return it as a DataFrame."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "temperature_2m_mean",
            "precipitation_sum",
            "windspeed_10m_max",
            "relative_humidity_2m_mean",
        ],
        "timezone": "auto",
    }

    print(f"Requesting data for {CITY_NAME} from {start} to {end} ...")
    response = requests.get(ARCHIVE_URL, params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()

    daily = payload["daily"]
    df = pd.DataFrame(daily)
    df.rename(
        columns={
            "time": "date",
            "temperature_2m_max": "temp_max",
            "temperature_2m_min": "temp_min",
            "temperature_2m_mean": "temp_mean",
            "precipitation_sum": "precipitation",
            "windspeed_10m_max": "wind_speed_max",
            "relative_humidity_2m_mean": "humidity_mean",
        },
        inplace=True,
    )
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def fetch_fallback_dataset() -> pd.DataFrame:
    """
    Fallback used when the live Open-Meteo API is unreachable (e.g. no
    internet access, firewalled environment). Uses the real, publicly
    published NOAA Seattle daily weather dataset bundled with the
    `vega_datasets` package (2012-01-01 to 2015-12-31), so the project
    still runs end-to-end on genuine historical data.
    """
    from vega_datasets import data as vega_data

    print("Live weather API unreachable — falling back to the bundled "
          "NOAA Seattle daily weather dataset (vega_datasets, 2012-2015).")
    df = vega_data.seattle_weather()
    df = df.rename(columns={"wind": "wind_speed_max"})
    df["temp_mean"] = (df["temp_max"] + df["temp_min"]) / 2
    df["humidity_mean"] = pd.NA
    df["date"] = pd.to_datetime(df["date"])
    df = df[
        [
            "date",
            "temp_max",
            "temp_min",
            "temp_mean",
            "precipitation",
            "wind_speed_max",
            "humidity_mean",
            "weather",
        ]
    ]
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def main():
    global CITY_NAME
    try:
        df = fetch_historical_weather()
    except Exception as exc:
        print(f"\n[warning] Could not reach Open-Meteo API: {exc}\n")
        df = fetch_fallback_dataset()
        CITY_NAME = "Seattle, USA (NOAA historical dataset)"

    out_path = DATA_DIR / "historical_weather.csv"
    df.to_csv(out_path, index=False)

    meta_path = DATA_DIR / "meta.txt"
    meta_path.write_text(f"city={CITY_NAME}\nrows={len(df)}\n")

    print(f"Saved {len(df)} rows to {out_path} for {CITY_NAME}")
    print(df.head())
    print(df.tail())
    print("\nMissing values per column:")
    print(df.isna().sum())


if __name__ == "__main__":
    main()
