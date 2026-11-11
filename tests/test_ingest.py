"""Step 2 tests: M1 ingestion & validation, using deliberately dirty synthetic rows."""
import json

import pandas as pd
import pytest

from src.ingest import ingest_all, validate_calendar, validate_traffic, validate_weather
from src.utils import load_config, resolve

CFG = load_config()


def traffic_row(**kw):
    row = dict(
        road_id="NL-001", road_name="Main St", latitude=28.6, longitude=77.2,
        weather_station_id="WS-NORTH", date="2025-02-01", time="08:00",
        traffic_volume="500", vehicle_count="500",
        vehicle_type_dist=json.dumps({"2W": 0.4, "Car": 0.4, "LCV": 0.1, "HCV": 0.1}),
        avg_speed="40", occupancy="30", congestion_level="Moderate", travel_time="3.1",
        accident_count="0", signal_timing="45", road_capacity="2000",
    )
    row.update(kw)
    return row


def make(rows):
    return pd.DataFrame(rows).astype(object)


# ------------------------------------------------------------------ traffic
def test_clean_row_passes_untouched():
    out, q, stats = validate_traffic(make([traffic_row()]), CFG)
    assert len(out) == 1 and q.empty
    assert out.loc[0, "share_Car"] == pytest.approx(0.4)
    assert out.loc[0, "timestamp"] == pd.Timestamp("2025-02-01 08:00")


def test_impossible_values_are_nulled_not_dropped():
    rows = [
        traffic_row(time="08:00", traffic_volume="-59"),
        traffic_row(time="08:30", avg_speed="319.6"),
        traffic_row(time="09:00", occupancy="259.9"),
    ]
    out, q, _ = validate_traffic(make(rows), CFG)
    assert len(out) == 3  # rows kept
    assert pd.isna(out.loc[0, "traffic_volume"])
    assert pd.isna(out.loc[1, "avg_speed"])
    assert pd.isna(out.loc[2, "occupancy"])
    assert set(q["action"]) == {"value_nulled"} and len(q) == 3
    assert q.loc[q.field == "traffic_volume", "original_value"].iloc[0] == "-59"


def test_structural_defects_remove_row_and_are_logged():
    rows = [
        traffic_row(time="07:00"),                           # good
        traffic_row(time="07:30", road_id="BAD-1"),          # malformed id
        traffic_row(time="08:00", date="31/01/2025"),        # wrong date format
        traffic_row(time="08:15"),                           # off the 30-min grid
        traffic_row(time="09:00", date="2024-12-31"),        # outside study period
        traffic_row(time="09:30", weather_station_id="WS-X"),  # unknown station
        traffic_row(time="10:00", road_capacity=None),       # missing capacity
    ]
    out, q, stats = validate_traffic(make(rows), CFG)
    assert len(out) == 1
    assert stats["rows_in"] == 7
    assert set(q.loc[q.action == "row_removed", "src_line"]) == {3, 4, 5, 6, 7, 8}  # nothing silent


def test_inconsistent_segment_metadata_is_rejected():
    rows = [traffic_row(time="07:00"), traffic_row(time="07:30"), traffic_row(time="08:00"),
            traffic_row(time="08:30", road_capacity="9999")]
    out, q, _ = validate_traffic(make(rows), CFG)
    assert len(out) == 3
    assert "road_capacity" in set(q["field"])


def test_bad_congestion_label_and_vehicle_mix_are_nulled():
    rows = [
        traffic_row(time="07:00", congestion_level="Gridlock"),
        traffic_row(time="07:30", vehicle_type_dist="not json"),
        traffic_row(time="08:00", vehicle_type_dist=json.dumps({"2W": 0.9, "Car": 0.9, "LCV": 0.1, "HCV": 0.1})),
    ]
    out, q, _ = validate_traffic(make(rows), CFG)
    assert len(out) == 3
    assert pd.isna(out.loc[0, "congestion_level"])
    assert out["share_2W"].iloc[1:].isna().all()
    assert len(q) == 3


def test_blank_congestion_is_missing_not_quarantined():
    out, q, _ = validate_traffic(make([traffic_row(congestion_level=None)]), CFG)
    assert q.empty and pd.isna(out.loc[0, "congestion_level"])


def test_duplicates_and_missing_windows_are_reported():
    rows = [traffic_row(time="07:00"), traffic_row(time="07:00"), traffic_row(time="08:00")]
    out, _, stats = validate_traffic(make(rows), CFG)
    assert stats["duplicate_keys"] == 1 and len(out) == 3  # de-dup happens in Step 3
    assert stats["missing_windows"] == stats["expected_windows"] - 2


# ------------------------------------------------------------------ weather
def weather_row(**kw):
    row = dict(station_id="WS-NORTH", date="01/02/2025", time="08:00",
               weather_condition="RAIN ", temperature="12.5", rainfall="1.2", visibility="4000")
    row.update(kw)
    return row


def test_weather_parses_ddmmyyyy_and_keeps_label_for_step3():
    out, q, _ = validate_weather(make([weather_row()]), CFG)
    assert out.loc[0, "timestamp"] == pd.Timestamp("2025-02-01 08:00")  # DD/MM/YYYY, not MM/DD
    assert out.loc[0, "weather_condition"] == "RAIN"  # trimmed only; normalised in Step 3
    assert q.empty


def test_weather_bad_rows_and_values():
    rows = [weather_row(time="08:30"), weather_row(station_id="WS-MARS"),
            weather_row(time="09:00", temperature="900"), weather_row(time="10:00", rainfall="-4")]
    out, q, _ = validate_weather(make(rows), CFG)
    assert len(out) == 2
    assert out["temperature"].isna().sum() == 1 and out["rainfall"].isna().sum() == 1


# ----------------------------------------------------------------- calendar
def test_calendar_rejects_duplicate_dates_and_bad_flags():
    rows = [
        dict(date="2025-01-01", public_holiday="1", holiday_name="New Year", event_flag="0", event_name=None, roadwork_flag="0"),
        dict(date="2025-01-01", public_holiday="1", holiday_name="New Year", event_flag="0", event_name=None, roadwork_flag="0"),
        dict(date="2025-01-02", public_holiday="2", holiday_name=None, event_flag="0", event_name=None, roadwork_flag="0"),
    ]
    out, q, _ = validate_calendar(make(rows), CFG)
    assert len(out) == 1 and len(q) == 2


# ------------------------------------------------- integration on the real files
RAW = resolve(CFG["paths"]["raw_dir"]) / CFG["raw_files"]["traffic"]


@pytest.mark.skipif(not RAW.exists(), reason="raw data not present")
def test_real_data_counts_match_data_dictionary():
    s = ingest_all(CFG, write=False)
    t = s["traffic"]
    assert t["rows_in"] == 178468 and t["rows_removed"] == 0
    by_field = {r["field"]: r["count"] for r in t["quarantine_breakdown"]}
    assert by_field == {"traffic_volume": 241, "avg_speed": 237, "occupancy": 234}
    assert t["duplicate_keys"] == 1767
    assert t["segments"] == 25
    assert s["weather"]["rows_out"] == 10872 and s["calendar"]["missing_dates"] == 0
