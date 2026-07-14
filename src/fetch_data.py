"""
fetch_data.py
Downloads historical daily weather data from the Open-Meteo Archive API
(free, no API key required) for one or all configured cities and saves
each as a CSV under data/<city_key>/historical_weather.csv.

Data source: https://open-meteo.com/en/docs/historical-weather-api

If the live API is unreachable (no internet / firewalled environment),
each city falls back to an offline dataset so the project still runs
end-to-end:
  - "seattle" / "sf_temps": genuine, publicly published NOAA daily/hourly
    records bundled with the `vega_datasets` package.
  - "synthetic": a clearly labeled seasonal sine-wave + noise model tuned
    to rough climate normals for that city. This is NOT real observed
    data — it exists purely so the multi-city comparison still works with
    no internet connection. It is labeled as such everywhere it's used.
"""

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CITIES, city_data_dir  # noqa: E402

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "windspeed_10m_max",
    "relative_humidity_2m_mean",
    "surface_pressure_mean",
]

# Pull ~15 years of daily history up to a few days ago (archive API has a
# short data-availability lag).
END_DATE = date.today() - timedelta(days=5)
START_DATE = END_DATE.replace(year=END_DATE.year - 15)


def fetch_live(lat: float, lon: float, start: date = START_DATE, end: date = END_DATE) -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": DAILY_VARS,
        "timezone": "auto",
    }
    response = requests.get(ARCHIVE_URL, params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()

    df = pd.DataFrame(payload["daily"])
    df.rename(
        columns={
            "time": "date",
            "temperature_2m_max": "temp_max",
            "temperature_2m_min": "temp_min",
            "temperature_2m_mean": "temp_mean",
            "precipitation_sum": "precipitation",
            "windspeed_10m_max": "wind_speed_max",
            "relative_humidity_2m_mean": "humidity_mean",
            "surface_pressure_mean": "pressure_mean",
        },
        inplace=True,
    )
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def fetch_fallback_seattle() -> pd.DataFrame:
    from vega_datasets import data as vega_data

    df = vega_data.seattle_weather().rename(columns={"wind": "wind_speed_max"})
    df["temp_mean"] = (df["temp_max"] + df["temp_min"]) / 2
    df["humidity_mean"] = pd.NA
    df["pressure_mean"] = pd.NA
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "temp_max", "temp_min", "temp_mean", "precipitation", "wind_speed_max", "humidity_mean", "pressure_mean"]]


def fetch_fallback_sf_temps() -> pd.DataFrame:
    from vega_datasets import data as vega_data

    hourly = vega_data.sf_temps()
    hourly["date"] = pd.to_datetime(hourly["date"])
    daily = hourly.set_index("date")["temp"].resample("D").agg(["max", "min", "mean"])
    daily.columns = ["temp_max", "temp_min", "temp_mean"]
    daily = daily.reset_index()
    daily["precipitation"] = 0.0  # not available in this bundled dataset
    daily["wind_speed_max"] = pd.NA
    daily["humidity_mean"] = pd.NA
    daily["pressure_mean"] = pd.NA
    return daily


def fetch_fallback_synthetic(city_cfg: dict, start: date = START_DATE, end: date = END_DATE, seed: int = 42) -> pd.DataFrame:
    """
    Deterministic, clearly-labeled synthetic seasonal weather generator.
    Only used offline, for cities without a bundled real dataset. NOT real
    observed data.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    doy = dates.dayofyear.to_numpy()

    mean = city_cfg["climate_normal_mean"]
    amplitude = city_cfg["climate_normal_amplitude"]
    peak_doy = 200  # mid-July peak, appropriate for all-Northern-Hemisphere city list

    seasonal = mean + amplitude * np.cos(2 * np.pi * (doy - peak_doy) / 365.25)

    # AR(1) noise for realistic day-to-day autocorrelation instead of pure white noise.
    noise = np.zeros(len(dates))
    phi = 0.7
    sigma = 1.2
    for i in range(1, len(noise)):
        noise[i] = phi * noise[i - 1] + rng.normal(0, sigma)

    temp_mean = seasonal + noise
    daily_spread = 4 + rng.normal(0, 0.5, size=len(dates)).clip(min=1)
    temp_max = temp_mean + daily_spread
    temp_min = temp_mean - daily_spread

    precipitation = rng.exponential(scale=2.0, size=len(dates))
    precipitation[rng.random(len(dates)) > 0.35] = 0.0  # most days dry
    wind_speed_max = np.abs(rng.normal(15, 5, size=len(dates)))

    return pd.DataFrame(
        {
            "date": dates,
            "temp_max": temp_max,
            "temp_min": temp_min,
            "temp_mean": temp_mean,
            "precipitation": precipitation,
            "wind_speed_max": wind_speed_max,
            "humidity_mean": pd.NA,
            "pressure_mean": pd.NA,
        }
    )


def fetch_city(city_key: str) -> tuple[pd.DataFrame, str]:
    """Returns (dataframe, source_label)."""
    cfg = CITIES[city_key]
    try:
        df = fetch_live(cfg["lat"], cfg["lon"])
        return df, f"{cfg['label']} — live Open-Meteo API"
    except Exception as exc:
        print(f"[warning] {city_key}: live API unreachable ({exc}); using offline fallback.")
        fallback = cfg["offline_fallback"]
        if fallback == "seattle":
            return fetch_fallback_seattle(), f"{cfg['label']} — OFFLINE FALLBACK: real NOAA Seattle data (not actually {cfg['label']})"
        if fallback == "sf_temps":
            return fetch_fallback_sf_temps(), f"{cfg['label']} — offline fallback: real NOAA San Francisco hourly-temp data"
        return fetch_fallback_synthetic(cfg), f"{cfg['label']} — SYNTHETIC offline demo data (not real observations)"


def main(city_keys=None):
    city_keys = city_keys or list(CITIES.keys())
    for city_key in city_keys:
        print(f"\n=== Fetching data for {city_key} ===")
        df, source_label = fetch_city(city_key)

        out_dir = city_data_dir(city_key)
        df.to_csv(out_dir / "historical_weather.csv", index=False)
        (out_dir / "meta.txt").write_text(f"city_key={city_key}\nsource={source_label}\nrows={len(df)}\n")

        print(f"Saved {len(df)} rows -> {out_dir / 'historical_weather.csv'}")
        print(f"Source: {source_label}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--city", action="append", help="City key to fetch (repeatable). Default: all configured cities.")
    args = parser.parse_args()
    main(args.city)
