"""M2 Cleaning & wrangling (FR-03), PRD section 9 stages 2-5.

Input : validated tables from ``src.ingest`` (data/interim/*_validated.parquet)
Output: data/interim/{traffic,weather,calendar}_clean.parquet + reports/clean_summary.json

Pipeline (traffic), in order:
  1. de-duplicate on road_id + timestamp, keeping the most complete record
  2. recover values the ingest step nulled from redundant channels
       volume <- vehicle_count (identical on every healthy row)
       speed  <- segment_length / travel_time (travel_time is derived from speed)
  3. flag statistical outliers (robust z-score + IQR per segment x slot x day-type);
       flag-and-keep by default; only extreme values on ordinary days are treated as
       faults (nulled + re-imputed) - holiday/event/roadwork days shift demand legitimately
  4. rebuild the complete segment x 30-min grid (absent windows become explicit rows)
  5. fill gaps: linear interpolation up to ``max_interp_gap_windows``, otherwise the
       segment x slot x weekday/weekend median fitted on the TRAIN period only
  6. derive congestion_level from V/C where blank; ordinal-encode it
Every filled value carries a ``*_fill`` provenance label, so models can exclude or
down-weight imputed targets. Weather labels are mapped to a controlled vocabulary.

Run:  python -m src.clean
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.ingest import VEHICLE_CLASSES, ingest_all
from src.utils import get_logger, load_config, resolve, set_seed, split_boundaries

log = get_logger("clean")

CONGESTION_ORDER = ["Free-flow", "Moderate", "Heavy", "Severe"]
FILL_LABELS = ["observed", "recovered", "interpolated", "median"]
ROAD_META = ["road_name", "latitude", "longitude", "weather_station_id", "road_capacity"]
SHARE_COLS = [f"share_{c}" for c in VEHICLE_CLASSES]


# --------------------------------------------------------------------------- helpers
def _add_time_keys(df: pd.DataFrame) -> pd.DataFrame:
    df["slot"] = (df["timestamp"].dt.hour * 2 + df["timestamp"].dt.minute // 30).astype("int16")
    df["weekend"] = (df["timestamp"].dt.day_of_week >= 5).astype("int8")
    return df


def _gap_lengths(isna: np.ndarray) -> np.ndarray:
    """For each NaN position, the length of the consecutive-NaN run it belongs to (else 0)."""
    out = np.zeros(len(isna), dtype=int)
    edges = np.diff(np.concatenate(([0], isna.astype(int), [0])))
    for s, e in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
        out[s:e] = e - s
    return out


def derive_congestion(volume: pd.Series, capacity: pd.Series, cfg: dict) -> pd.Series:
    """Band V/C = volume / (capacity / 2) as in the data dictionary."""
    vc = volume / (capacity / 2.0)
    b = cfg["congestion_bands"]
    lab = np.select([vc < b["free_flow_max"], vc < b["moderate_max"], vc < b["heavy_max"]],
                    CONGESTION_ORDER[:3], default=CONGESTION_ORDER[3])
    return pd.Series(lab, index=volume.index, dtype=object).where(vc.notna())


def _drop_duplicates(df: pd.DataFrame, keys: list[str], value_cols: list[str]):
    n0 = len(df)
    exact = int(df.drop(columns=[c for c in ["src_line"] if c in df]).duplicated().sum())
    df = df.assign(_nn=df[value_cols].notna().sum(axis=1))
    df = (df.sort_values(keys + ["_nn"], ascending=[True] * len(keys) + [False], kind="stable")
            .drop_duplicates(keys, keep="first").drop(columns="_nn").reset_index(drop=True))
    return df, {"duplicates_removed": n0 - len(df), "of_which_exact": exact}


def _train_medians(df: pd.DataFrame, col: str, keys: list[str], train_mask: pd.Series) -> pd.Series:
    """Median of ``col`` per key, fitted on the training period only (no leakage)."""
    return df.loc[train_mask].groupby(keys)[col].median()


def _fill_gaps(df, col, group_col, limit, median_keys, train_mask):
    """Interpolate short gaps, median-fill the rest. Returns provenance labels (np array)."""
    isna = df[col].isna()
    label = np.where(isna, "", "observed").astype(object)
    medians = _train_medians(df, col, median_keys, train_mask)  # fitted before any filling

    interp = df.groupby(group_col)[col].transform(lambda s: s.interpolate(method="linear", limit_area="inside"))
    gaplen = np.zeros(len(df), dtype=int)
    arr = isna.to_numpy()
    for _, ix in df.groupby(group_col).indices.items():
        gaplen[ix] = _gap_lengths(arr[ix])
    use = isna.to_numpy() & interp.notna().to_numpy() & (gaplen <= limit)
    df.loc[use, col] = interp[use]
    label[use] = "interpolated"

    still = df[col].isna()
    if still.any():
        fallback = pd.MultiIndex.from_frame(df.loc[still, median_keys]).map(medians) if len(median_keys) > 1 \
            else df.loc[still, median_keys[0]].map(medians)
        df.loc[still, col] = np.asarray(fallback, dtype=float)
        label[still.to_numpy()] = "median"
    return label


def _provenance(label: np.ndarray, recovered: pd.Series) -> pd.Categorical:
    label = np.where(recovered.to_numpy() & (label == "observed"), "recovered", label)
    return pd.Categorical(label, categories=FILL_LABELS)


# ------------------------------------------------------------------------ calendar
def clean_calendar(cal: pd.DataFrame) -> pd.DataFrame:
    """Calendar is already unique-per-date after validation; just order it."""
    return cal.sort_values("date").reset_index(drop=True)


# ------------------------------------------------------------------------- weather
def clean_weather(w: pd.DataFrame, cfg: dict):
    stats: dict = {}
    split = split_boundaries(cfg)
    w, stats["dedup"] = _drop_duplicates(w, ["station_id", "timestamp"],
                                         ["weather_condition", "temperature", "rainfall", "visibility"])

    # categorical harmonisation: 'RAIN ', 'rainy' -> Rain, etc.
    raw_labels = w["weather_condition"].astype("string").str.strip()  # 'RAIN ' vs 'RAIN' already trimmed in ingest
    lab = raw_labels.str.lower()
    mapped = lab.map(cfg["cleaning"]["weather_label_map"])
    unknown = lab.notna() & mapped.isna()
    stats["labels_before"] = int(raw_labels.nunique())
    stats["labels_after"] = int(mapped.nunique())
    stats["unknown_labels_nulled"] = int(unknown.sum())
    w["weather_condition"] = mapped.astype(object)

    # complete hourly grid per station
    start, end = pd.Timestamp(cfg["data"]["date_start"]), pd.Timestamp(cfg["data"]["date_end"])
    full = pd.date_range(start, end + pd.Timedelta(hours=23), freq="h")
    idx = pd.MultiIndex.from_product([sorted(w["station_id"].unique()), full], names=["station_id", "timestamp"])
    w = w.set_index(["station_id", "timestamp"]).reindex(idx).reset_index()
    w["is_reconstructed"] = w["src_line"].isna()
    stats["hours_reconstructed"] = int(w["is_reconstructed"].sum())
    w["hour"] = w["timestamp"].dt.hour.astype("int8")
    train = w["timestamp"] < split["train_end"]

    limit = cfg["cleaning"]["max_interp_gap_hours_weather"]
    stats["imputed"] = {}
    for col, keys in [("temperature", ["station_id", "hour"]), ("rainfall", ["station_id", "hour"])]:
        before = int(w[col].isna().sum())
        _fill_gaps(w, col, "station_id", limit, keys, train)
        stats["imputed"][col] = before
    # visibility depends on the condition (fog, rain) -> fall back to the condition median
    before = int(w["visibility"].isna().sum())
    w["weather_condition"] = w.groupby("station_id")["weather_condition"].transform(lambda s: s.ffill(limit=limit).bfill(limit=limit))
    _fill_gaps(w, "visibility", "station_id", limit, ["weather_condition"], train)
    stats["imputed"]["visibility"] = before

    w["weather_condition"] = pd.Categorical(w["weather_condition"], categories=sorted(set(cfg["cleaning"]["weather_label_map"].values())))
    w["src_line"] = w["src_line"].astype("Int64")
    out = w[["station_id", "timestamp", "weather_condition", "temperature", "rainfall",
             "visibility", "is_reconstructed", "src_line"]].copy()
    assert out[["weather_condition", "temperature", "rainfall", "visibility"]].notna().all().all(), "weather nulls remain"
    return out, stats


# -------------------------------------------------------------------------- traffic
def _statistical_outliers(df, cal, cfg, train_mask):
    """Robust z-score (median/MAD) and Tukey-IQR per segment x slot x day-type, train-fitted."""
    oc = cfg["cleaning"]["outlier"]
    keys = ["road_id", "slot", "weekend"]
    report, any_flag = {}, pd.Series(False, index=df.index)
    event_days = set(cal.loc[(cal.public_holiday == 1) | (cal.event_flag == 1) | (cal.roadwork_flag == 1), "date"])
    on_cal = df["timestamp"].dt.normalize().isin(event_days)
    for col in ["traffic_volume", "avg_speed", "occupancy"]:
        tr = df.loc[train_mask, keys + [col]].dropna(subset=[col])
        med = tr.groupby(keys)[col].transform("median")
        tab = pd.DataFrame({"med": med.groupby([tr[k] for k in keys]).first(),
                            "mad": (tr[col] - med).abs().groupby([tr[k] for k in keys]).median(),
                            "q1": tr.groupby(keys)[col].quantile(0.25),
                            "q3": tr.groupby(keys)[col].quantile(0.75)})
        j = df[keys].join(tab, on=keys)
        rz = 0.6745 * (df[col] - j["med"]) / j["mad"].where(j["mad"] > 0)
        iqr = j["q3"] - j["q1"]
        z_flag = rz.abs() > oc["robust_z_flag"]
        # extreme values on holiday / event / roadwork days are real demand shifts, not faults
        hard = (rz.abs() > oc["robust_z_null"]) & ~on_cal
        iqr_flag = (df[col] < j["q1"] - oc["iqr_k"] * iqr) | (df[col] > j["q3"] + oc["iqr_k"] * iqr)
        report[col] = {"robust_z_flagged": int(z_flag.sum()), "iqr_flagged": int(iqr_flag.sum()),
                       "extreme_nulled": int(hard.sum()),
                       "extreme_kept_calendar_day": int(((rz.abs() > oc["robust_z_null"]) & on_cal).sum()),
                       "flagged_on_holiday_event_roadwork_days_pct":
                           round(100 * float(on_cal[z_flag].mean()), 1) if z_flag.any() else None}
        df.loc[hard, col] = np.nan
        any_flag |= z_flag.fillna(False) | iqr_flag.fillna(False)
    return any_flag, report


def clean_traffic(t: pd.DataFrame, cal: pd.DataFrame, cfg: dict):
    stats: dict = {}
    split = split_boundaries(cfg)
    df = t.copy()

    # 1. duplicates
    df, stats["dedup"] = _drop_duplicates(df, ["road_id", "timestamp"],
                                          ["traffic_volume", "avg_speed", "occupancy", "vehicle_count"])
    df = _add_time_keys(df)

    # 2. recover faulty values from redundant channels
    vol_rec = df["traffic_volume"].isna() & df["vehicle_count"].notna()
    df.loc[vol_rec, "traffic_volume"] = df.loc[vol_rec, "vehicle_count"].astype(float)
    ok = df["avg_speed"].notna() & (df["avg_speed"] > 0) & df["travel_time"].notna() & (df["travel_time"] > 0)
    length = (df.loc[ok, "travel_time"] * df.loc[ok, "avg_speed"] / 60.0).groupby(df.loc[ok, "road_id"]).median()
    df["segment_length_km"] = df["road_id"].map(length)
    spd_rec = df["avg_speed"].isna() & df["travel_time"].notna() & (df["travel_time"] > 0)
    implied = (df["segment_length_km"] / df["travel_time"] * 60.0).round(1)
    spd_rec &= implied.between(0, cfg["validation"]["traffic"]["avg_speed"]["max"])
    df.loc[spd_rec, "avg_speed"] = implied[spd_rec]
    df["_vol_rec"], df["_spd_rec"] = vol_rec, spd_rec
    stats["recovered"] = {"traffic_volume_from_vehicle_count": int(vol_rec.sum()),
                          "avg_speed_from_travel_time": int(spd_rec.sum())}

    # 3. statistical outliers (train-fitted)
    train = df["timestamp"] < split["train_end"]
    df["stat_outlier"], stats["statistical_outliers"] = _statistical_outliers(df, cal, cfg, train)

    # 4. full segment x window grid
    start, end = pd.Timestamp(cfg["data"]["date_start"]), pd.Timestamp(cfg["data"]["date_end"])
    full = pd.date_range(start, end + pd.Timedelta(hours=23, minutes=30), freq=f"{cfg['data']['window_minutes']}min")
    idx = pd.MultiIndex.from_product([sorted(df["road_id"].unique()), full], names=["road_id", "timestamp"])
    meta = df.groupby("road_id")[ROAD_META + ["segment_length_km"]].first()
    df = df.set_index(["road_id", "timestamp"]).reindex(idx).reset_index()
    df["is_reconstructed"] = df["src_line"].isna()
    for c in ROAD_META + ["segment_length_km"]:
        df[c] = df[c].where(df[c].notna(), df["road_id"].map(meta[c]))
    for c in ["_vol_rec", "_spd_rec", "stat_outlier"]:
        df[c] = df[c].fillna(False).astype(bool)
    df = _add_time_keys(df)
    train = df["timestamp"] < split["train_end"]
    stats["windows_reconstructed"] = int(df["is_reconstructed"].sum())

    # 5. fill remaining gaps
    limit = cfg["cleaning"]["max_interp_gap_windows"]
    mk = ["road_id", "slot", "weekend"]
    lab = {}
    for col in ["traffic_volume", "avg_speed", "occupancy"]:
        lab[col] = _fill_gaps(df, col, "road_id", limit, mk, train)
    for col in ["signal_timing", *SHARE_COLS]:
        _fill_gaps(df, col, "road_id", limit, mk, train)
    df["traffic_volume"] = df["traffic_volume"].round().astype("int64")
    df["signal_timing"] = df["signal_timing"].round().astype("int64")
    df["occupancy"] = df["occupancy"].clip(0, 100).round(1)
    df["avg_speed"] = df["avg_speed"].round(1)
    share_sum = df[SHARE_COLS].sum(axis=1)
    df[SHARE_COLS] = df[SHARE_COLS].div(share_sum, axis=0).round(4)
    df["volume_fill"] = _provenance(lab["traffic_volume"], df["_vol_rec"])
    df["speed_fill"] = _provenance(lab["avg_speed"], df["_spd_rec"])
    df["occupancy_fill"] = _provenance(lab["occupancy"], pd.Series(False, index=df.index))
    stats["fill_methods"] = {n: {k: int(v) for k, v in df[f"{n}_fill"].value_counts().items() if v}
                             for n in ["volume", "speed", "occupancy"]}

    # travel time: keep recorded values; recompute from length / speed where speed was filled
    recompute = df["speed_fill"].isin(["interpolated", "median"]) | df["travel_time"].isna()
    df.loc[recompute, "travel_time"] = (df.loc[recompute, "segment_length_km"] / df.loc[recompute, "avg_speed"] * 60).round(2)
    stats["travel_time_recomputed"] = int(recompute.sum())

    # 6. congestion level
    derived = derive_congestion(df["traffic_volume"], df["road_capacity"], cfg)
    has = df["congestion_level"].notna()
    check = has & df["volume_fill"].isin(["observed", "recovered"])
    stats["congestion_label_agreement_with_VC_rule"] = {
        "observed_volume_rows": round(float((df.loc[check & (df.volume_fill == "observed"), "congestion_level"].astype(object)
                                             == derived[check & (df.volume_fill == "observed")]).mean()), 4),
        "recovered_volume_rows": round(float((df.loc[check & (df.volume_fill == "recovered"), "congestion_level"].astype(object)
                                              == derived[check & (df.volume_fill == "recovered")]).mean()), 4),
    }
    df["congestion_derived"] = ~has
    df["congestion_level"] = pd.Categorical(df["congestion_level"].astype(object).where(has, derived),
                                            categories=CONGESTION_ORDER, ordered=True)
    df["congestion_code"] = df["congestion_level"].cat.codes.astype("int8")
    stats["congestion_derived"] = int((~has).sum())

    # accident_count is genuinely unknown for reconstructed windows -> keep missing (do not invent zeros)
    df["accident_count"] = df["accident_count"].astype("Int64")
    df["vehicle_count"] = df["vehicle_count"].astype("Int64")
    df["src_line"] = df["src_line"].astype("Int64")

    # 7. final checks (stage 8 lite)
    must = ["traffic_volume", "avg_speed", "occupancy", "congestion_level", "travel_time", "signal_timing", *SHARE_COLS]
    assert df[must].notna().all().all(), "nulls remain in modelling columns"
    assert not df.duplicated(["road_id", "timestamp"]).any(), "duplicate keys remain"
    assert len(df) == len(idx), "grid incomplete"
    cols = ["road_id", "road_name", "latitude", "longitude", "weather_station_id", "segment_length_km",
            "timestamp", "traffic_volume", "vehicle_count", *SHARE_COLS, "avg_speed", "occupancy",
            "congestion_level", "congestion_code", "congestion_derived", "travel_time", "accident_count",
            "signal_timing", "road_capacity", "volume_fill", "speed_fill", "occupancy_fill",
            "is_reconstructed", "stat_outlier", "src_line"]
    stats["rows_out"] = len(df)
    return df[cols], stats


# -------------------------------------------------------------------------- runner
def clean_all(cfg: dict | None = None, write: bool = True) -> dict:
    cfg = cfg or load_config()
    set_seed(cfg["project"]["seed"])
    interim = resolve(cfg["paths"]["interim_dir"])
    if not all((interim / f"{n}_validated.parquet").exists() for n in ("traffic", "weather", "calendar")):
        log.info("validated tables not found - running ingest first")
        ingest_all(cfg, write=True)
    cal = clean_calendar(pd.read_parquet(interim / "calendar_validated.parquet"))
    weather, w_stats = clean_weather(pd.read_parquet(interim / "weather_validated.parquet"), cfg)
    log.info("weather cleaned: %s", w_stats["imputed"])
    traffic, t_stats = clean_traffic(pd.read_parquet(interim / "traffic_validated.parquet"), cal, cfg)
    log.info("traffic cleaned: %d rows | dedup %s | recovered %s | reconstructed %d windows",
             len(traffic), t_stats["dedup"], t_stats["recovered"], t_stats["windows_reconstructed"])
    summary = {"split": {k: str(v) for k, v in split_boundaries(cfg).items()},
               "calendar": {"rows": len(cal)}, "weather": w_stats, "traffic": t_stats}
    if write:
        interim.mkdir(parents=True, exist_ok=True)
        cal.to_parquet(interim / "calendar_clean.parquet", index=False)
        weather.to_parquet(interim / "weather_clean.parquet", index=False)
        traffic.to_parquet(interim / "traffic_clean.parquet", index=False)
        rep = resolve(cfg["paths"]["reports_dir"])
        (rep / "clean_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        log.info("wrote cleaned tables and %s", rep / "clean_summary.json")
    return summary


if __name__ == "__main__":
    clean_all()
