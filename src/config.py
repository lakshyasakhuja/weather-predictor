"""
config.py
Central registry of cities supported by the project, and shared paths.

Each city fetches real, live historical data from the Open-Meteo Archive
API when internet access is available. Two cities additionally have a
genuine bundled offline dataset (used automatically if the live API can't
be reached): New Delhi falls back to NOAA's Seattle daily weather record,
and San Francisco falls back to its own real NOAA hourly record — both via
the `vega_datasets` package. The remaining cities fall back to a clearly
labeled *synthetic* seasonal model (not real observations) so the
multi-city comparison still works fully offline; this is only ever used
when there is no internet connection.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

DEFAULT_CITY = "new_delhi"

# climate_normal_mean / climate_normal_amplitude are only used to generate
# the synthetic offline fallback (rough annual mean °C and half the
# summer-winter swing °C) — NOT used at all when live data is available.
CITIES = {
    "new_delhi": {
        "label": "New Delhi, India",
        "lat": 28.6139,
        "lon": 77.2090,
        "climate_normal_mean": 25.0,
        "climate_normal_amplitude": 9.0,
        "offline_fallback": "seattle",  # real dataset, but a different city — clearly labeled in the UI
    },
    "mumbai": {
        "label": "Mumbai, India",
        "lat": 19.0760,
        "lon": 72.8777,
        "climate_normal_mean": 27.0,
        "climate_normal_amplitude": 3.0,
        "offline_fallback": "synthetic",
    },
    "bengaluru": {
        "label": "Bengaluru, India",
        "lat": 12.9716,
        "lon": 77.5946,
        "climate_normal_mean": 23.5,
        "climate_normal_amplitude": 3.0,
        "offline_fallback": "synthetic",
    },
    "san_francisco": {
        "label": "San Francisco, USA",
        "lat": 37.7749,
        "lon": -122.4194,
        "climate_normal_mean": 14.0,
        "climate_normal_amplitude": 3.5,
        "offline_fallback": "sf_temps",  # real dataset for this exact city
    },
    "london": {
        "label": "London, UK",
        "lat": 51.5074,
        "lon": -0.1278,
        "climate_normal_mean": 11.0,
        "climate_normal_amplitude": 7.0,
        "offline_fallback": "synthetic",
    },
}


def city_data_dir(city_key: str) -> Path:
    d = DATA_DIR / city_key
    d.mkdir(parents=True, exist_ok=True)
    return d


def city_models_dir(city_key: str) -> Path:
    d = MODELS_DIR / city_key
    d.mkdir(parents=True, exist_ok=True)
    return d
