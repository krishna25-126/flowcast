"""Step 1 smoke tests: config loads, raw data present, schemas match the dictionary."""
import pandas as pd

from src.utils import load_config, resolve

TRAFFIC_COLS = {
    "road_id", "road_name", "latitude", "longitude", "weather_station_id", "date",
    "time", "traffic_volume", "vehicle_count", "vehicle_type_dist", "avg_speed",
    "occupancy", "congestion_level", "travel_time", "accident_count",
    "signal_timing", "road_capacity",
}
WEATHER_COLS = {"station_id", "date", "time", "weather_condition", "temperature",
                "rainfall", "visibility"}
CALENDAR_COLS = {"date", "public_holiday", "holiday_name", "event_flag",
                 "event_name", "roadwork_flag"}


def _load(key):
    cfg = load_config()
    return pd.read_csv(resolve(cfg["paths"]["raw_dir"]) / cfg["raw_files"][key])


def test_config_has_seed():
    assert load_config()["project"]["seed"] == 42


def test_traffic_schema():
    df = _load("traffic")
    assert TRAFFIC_COLS <= set(df.columns)
    assert df["road_id"].nunique() == 25


def test_weather_schema():
    assert WEATHER_COLS <= set(_load("weather").columns)


def test_calendar_schema():
    assert CALENDAR_COLS <= set(_load("calendar").columns)
