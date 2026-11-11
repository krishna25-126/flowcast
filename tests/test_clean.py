"""Step 3 tests: M2 cleaning, on small synthetic corridors with known defects."""
import copy

import numpy as np
import pandas as pd
import pytest

from src.clean import clean_all, clean_calendar, clean_traffic, clean_weather, derive_congestion
from src.utils import load_config, resolve, split_boundaries

CFG = load_config()
# 4-day study period: days 1-2 = train, days 3-4 = test (val gets 0 days)
SMALL = copy.deepcopy(CFG)
SMALL["data"]["date_start"], SMALL["data"]["date_end"] = "2025-02-03", "2025-02-06"


def base_traffic():
    """One segment, 4 days, perfectly regular values so every expectation is exact."""
    ts = pd.date_range("2025-02-03", periods=192, freq="30min")
    slot = np.asarray(ts.hour * 2 + ts.minute // 30)
    day = np.asarray((ts.normalize() - ts[0].normalize()).days)
    vol = 120 + 6 * slot + 4 * day
    speed = np.round(40 - 0.1 * slot + 0.2 * day, 1)
    df = pd.DataFrame({
        "road_id": "NL-001", "road_name": "Main St", "latitude": 28.6, "longitude": 77.2,
        "weather_station_id": "WS-NORTH", "timestamp": ts, "traffic_volume": vol.astype(float),
        "vehicle_count": pd.array(vol, dtype="Int64"),
        "share_2W": 0.4, "share_Car": 0.4, "share_LCV": 0.1, "share_HCV": 0.1,
        "avg_speed": speed, "occupancy": 30 + 0.2 * slot + 0.5 * day,
        "congestion_level": derive_congestion(pd.Series(vol), pd.Series(600, index=range(192)), CFG).to_numpy(),
        "travel_time": np.round(2.0 / speed * 60, 2), "accident_count": 0, "signal_timing": 45,
        "road_capacity": 600, "src_line": np.arange(192) + 2,
    })
    return df


def calendar(event_date=None):
    d = pd.date_range("2025-02-03", "2025-02-06")
    return pd.DataFrame({"date": d, "public_holiday": 0, "holiday_name": pd.NA,
                         "event_flag": (d == pd.Timestamp(event_date)).astype(int) if event_date else 0,
                         "event_name": pd.NA, "roadwork_flag": 0, "src_line": range(4)})


def run(df, cal=None):
    out, stats = clean_traffic(df, cal if cal is not None else calendar(), SMALL)
    return out.set_index("timestamp"), stats


def at(day, slot):
    return pd.Timestamp("2025-02-03") + pd.Timedelta(days=day, minutes=30 * slot)


# ---------------------------------------------------------------------- traffic
def test_clean_input_passes_through_with_full_grid():
    out, stats = run(base_traffic())
    assert len(out) == 192 and stats["windows_reconstructed"] == 0
    assert (out["volume_fill"] == "observed").all()


def test_duplicates_removed_keeping_most_complete_record():
    df = base_traffic()
    dup = df.iloc[[10]].copy()
    dup["avg_speed"] = np.nan  # the retry is less complete
    out, stats = run(pd.concat([dup, df], ignore_index=True))
    assert stats["dedup"]["duplicates_removed"] == 1
    assert out.loc[df.timestamp[10], "avg_speed"] == df.avg_speed[10]


def test_volume_recovered_from_vehicle_count():
    df = base_traffic()
    truth = df.loc[50, "traffic_volume"]
    df.loc[50, "traffic_volume"] = np.nan
    out, stats = run(df)
    assert out.loc[df.timestamp[50], "traffic_volume"] == truth
    assert out.loc[df.timestamp[50], "volume_fill"] == "recovered"
    assert stats["recovered"]["traffic_volume_from_vehicle_count"] == 1


def test_speed_recovered_from_travel_time():
    df = base_traffic()
    truth = df.loc[60, "avg_speed"]
    df.loc[60, "avg_speed"] = np.nan
    out, _ = run(df)
    assert abs(out.loc[df.timestamp[60], "avg_speed"] - truth) <= 0.15
    assert out.loc[df.timestamp[60], "speed_fill"] == "recovered"


def test_absent_window_is_rebuilt_and_interpolated():
    df = base_traffic()
    gone = at(1, 20)
    out, stats = run(df[df.timestamp != gone])
    row = out.loc[gone]
    assert stats["windows_reconstructed"] == 1 and row["is_reconstructed"]
    assert row["traffic_volume"] == 244                      # exact linear midpoint of 238 and 250
    assert row["volume_fill"] == "interpolated"
    assert pd.isna(row["accident_count"])                    # unknown, not invented as 0
    assert row["road_name"] == "Main St" and row["road_capacity"] == 600
    assert row["travel_time"] == pytest.approx(2.0 / row["avg_speed"] * 60, abs=0.1)


def test_long_gap_uses_train_period_median_not_future_data():
    df = base_traffic()
    gap = [at(3, s) for s in range(30, 40)]                  # 10 windows > 4-window limit, in the TEST period
    out, _ = run(df[~df.timestamp.isin(gap)])
    for s in range(30, 40):
        r = out.loc[at(3, s)]
        assert r["volume_fill"] == "median"
        assert r["traffic_volume"] == 120 + 6 * s + 2        # median of TRAIN days only (offsets 0 and 4)


def test_extreme_spike_on_ordinary_day_is_treated_as_fault():
    df = base_traffic()
    df.loc[df.timestamp == at(3, 12), ["traffic_volume", "vehicle_count"]] = [5000.0, 5000]
    out, stats = run(df)
    r = out.loc[at(3, 12)]
    assert r["traffic_volume"] < 400 and r["volume_fill"] == "interpolated"
    assert stats["statistical_outliers"]["traffic_volume"]["extreme_nulled"] == 1


def test_extreme_value_on_event_day_is_real_and_kept():
    df = base_traffic()
    df.loc[df.timestamp == at(3, 12), ["traffic_volume", "vehicle_count"]] = [5000.0, 5000]
    out, stats = run(df, calendar(event_date="2025-02-06"))
    r = out.loc[at(3, 12)]
    assert r["traffic_volume"] == 5000 and r["stat_outlier"]
    assert stats["statistical_outliers"]["traffic_volume"]["extreme_kept_calendar_day"] == 1


def test_blank_congestion_is_derived_from_volume_to_capacity():
    df = base_traffic()
    df["congestion_level"] = None
    out, stats = run(df)
    assert out["congestion_derived"].all() and stats["congestion_derived"] == 192
    expected = derive_congestion(out["traffic_volume"], out["road_capacity"], CFG)
    assert (out["congestion_level"].astype(object) == expected).all()
    assert set(out["congestion_level"].astype(str)) == {"Free-flow", "Moderate", "Heavy", "Severe"}
    assert (out["congestion_code"] == out["congestion_level"].cat.codes).all()


def test_congestion_band_boundaries():
    vol = pd.Series([499, 500, 799, 800, 999, 1000, np.nan])
    lab = derive_congestion(vol, pd.Series(2000, index=vol.index), CFG)       # capacity/2 = 1000 per window
    assert list(lab[:6]) == ["Free-flow", "Moderate", "Moderate", "Heavy", "Heavy", "Severe"]
    assert pd.isna(lab[6])


def test_final_table_has_no_nulls_in_modelling_columns():
    df = base_traffic()
    df.loc[[5, 6, 7], ["traffic_volume", "avg_speed", "occupancy"]] = np.nan
    out, _ = run(df[df.timestamp != at(2, 3)])
    assert out[["traffic_volume", "avg_speed", "occupancy", "congestion_level", "travel_time"]].notna().all().all()


# ---------------------------------------------------------------------- weather
def test_weather_labels_harmonised_and_gaps_filled():
    ts = pd.date_range("2025-02-03", periods=96, freq="h")
    labels = np.array(["Clear", "clear", "CLEAR", "Overcast", "rainy", "RAIN", "foggy", "cloudy"] * 12, dtype=object)
    w = pd.DataFrame({"station_id": "WS-NORTH", "timestamp": ts, "weather_condition": labels,
                      "temperature": np.linspace(10, 20, 96), "rainfall": 0.0, "visibility": 10000.0,
                      "src_line": range(96)})
    w.loc[[10, 11], "temperature"] = np.nan            # short gap -> interpolated
    w.loc[[70, 71, 72, 73, 74], "temperature"] = np.nan  # 5 h > 3 h limit -> train median
    w.loc[20, "visibility"] = np.nan
    out, stats = clean_weather(w, SMALL)
    assert set(out["weather_condition"].astype(str)) == {"Clear", "Cloudy", "Rain", "Fog"}
    assert stats["labels_before"] == 8 and stats["labels_after"] == 4
    assert out[["temperature", "rainfall", "visibility"]].notna().all().all()
    assert out.loc[10, "temperature"] == pytest.approx(w.temperature[9] + (w.temperature[12] - w.temperature[9]) / 3, abs=0.01)


def test_calendar_is_sorted_and_unique():
    cal = clean_calendar(calendar().iloc[::-1])
    assert cal["date"].is_monotonic_increasing and cal["date"].is_unique


def test_split_boundaries_are_chronological():
    b = split_boundaries(CFG)
    assert b["n_train_days"] + b["n_val_days"] + b["n_test_days"] == 151
    assert b["train_end"] < b["val_end"]


# ------------------------------------------------ integration on the real files
RAW = resolve(CFG["paths"]["raw_dir"]) / CFG["raw_files"]["traffic"]


@pytest.mark.skipif(not RAW.exists(), reason="raw data not present")
def test_real_data_end_to_end():
    s = clean_all(CFG, write=False)["traffic"]
    assert s["rows_out"] == 25 * 151 * 48 == 181200
    assert s["dedup"]["duplicates_removed"] == 1767
    assert s["windows_reconstructed"] == 4499
    assert s["recovered"] == {"traffic_volume_from_vehicle_count": 4587, "avg_speed_from_travel_time": 4579}
    agree = s["congestion_label_agreement_with_VC_rule"]
    assert agree["observed_volume_rows"] == 1.0 and agree["recovered_volume_rows"] == 1.0
    assert s["statistical_outliers"]["traffic_volume"]["extreme_nulled"] == 0   # real event/holiday peaks kept
