"""M1 Ingestion & validation (FR-01, FR-02).

Loads the three raw tables, enforces schema / types / ranges and writes a validated
copy of each to ``data/interim``. Nothing is dropped silently: every rejection is
recorded in a quarantine log.

Two kinds of quarantine action:

* ``row_removed``  - a *key or structural* field is unusable (bad road_id, unparseable
  timestamp, impossible coordinates ...). The row cannot be placed in the dataset.
* ``value_nulled`` - a *measurement* is impossible (negative volume, speed > 200 km/h,
  occupancy > 100 %). The faulty value is set to NaN and the row is kept, so Step 3 can
  impute it from neighbouring windows instead of losing the whole window.

Run:  python -m src.ingest
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import get_logger, load_config, resolve, set_seed

log = get_logger("ingest")

CONGESTION_LABELS = ("Free-flow", "Moderate", "Heavy", "Severe")
VEHICLE_CLASSES = ("2W", "Car", "LCV", "HCV")
ROAD_META = ("road_name", "latitude", "longitude", "weather_station_id", "road_capacity")
QUARANTINE_COLS = ["source", "src_line", "field", "reason", "original_value", "action"]


# --------------------------------------------------------------------------- helpers
class _Quarantine:
    """Accumulates rejected rows / values for one source table."""

    def __init__(self, source: str):
        self.source = source
        self._frames: list[pd.DataFrame] = []

    def add(self, idx, field: str, reason: str, values, action: str) -> None:
        idx = pd.Index(idx)
        if len(idx) == 0:
            return
        self._frames.append(
            pd.DataFrame(
                {
                    "source": self.source,
                    "src_line": idx + 2,  # +1 for header, +1 for 1-based line numbers
                    "field": field,
                    "reason": reason,
                    "original_value": pd.Series(values).astype(str).to_numpy(),
                    "action": action,
                }
            )
        )

    def frame(self) -> pd.DataFrame:
        if not self._frames:
            return pd.DataFrame(columns=QUARANTINE_COLS)
        return pd.concat(self._frames, ignore_index=True)


class _Validator:
    """Small helper bundling a frame, its quarantine log and the rejected-row mask."""

    def __init__(self, raw: pd.DataFrame, source: str):
        self.df = raw.copy()
        self.q = _Quarantine(source)
        self.rejected = pd.Series(False, index=self.df.index)

    def reject(self, mask, field: str, reason: str) -> None:
        mask = pd.Series(mask, index=self.df.index).fillna(False).astype(bool)
        if field in self.df.columns:
            vals = self.df.loc[mask, field]
        else:
            vals = pd.Series([""] * int(mask.sum()))
        self.q.add(self.df.index[mask], field, reason, vals, "row_removed")
        self.rejected |= mask

    def null_out(self, col: str, mask, reason: str, original: pd.Series) -> None:
        mask = pd.Series(mask, index=self.df.index).fillna(False).astype(bool) & ~self.rejected
        self.q.add(self.df.index[mask], col, reason, original[mask], "value_nulled")
        self.df.loc[mask, col] = np.nan

    def check_ranges(self, rules: dict, required: bool) -> None:
        """Coerce columns to numeric and apply min/max rules from the config."""
        for col, rule in rules.items():
            original = self.df[col]
            num = pd.to_numeric(original, errors="coerce")
            lo, hi = rule.get("min", -np.inf), rule.get("max", np.inf)
            out_of_range = (num < lo) | (num > hi)
            if required:
                self.reject(
                    num.isna() | out_of_range,
                    col,
                    f"missing, non-numeric or outside [{lo}, {hi}]",
                )
                self.df[col] = num
            else:
                self.df[col] = num
                non_numeric = original.notna() & num.isna()
                self.null_out(col, non_numeric, "non-numeric value", original)
                self.null_out(col, out_of_range, f"outside [{lo}, {hi}]", original)

    def check_timestamp(self, fmt: str, cfg: dict, step_minutes: int) -> pd.Series:
        date_s = self.df["date"].astype("string").str.strip()
        time_s = self.df["time"].astype("string").str.strip()
        ts = pd.to_datetime(date_s + " " + time_s, format=fmt, errors="coerce")
        self.reject(ts.isna(), "date/time", "unparseable date or time")
        off_grid = ts.notna() & (ts.dt.minute % step_minutes != 0)
        self.reject(off_grid, "time", f"not on the {step_minutes}-minute grid")
        start = pd.Timestamp(cfg["data"]["date_start"])
        end = pd.Timestamp(cfg["data"]["date_end"])
        day = ts.dt.normalize()
        self.reject(ts.notna() & ((day < start) | (day > end)), "date", "outside study period")
        return ts


def _expected_days(cfg: dict) -> int:
    return (pd.Timestamp(cfg["data"]["date_end"]) - pd.Timestamp(cfg["data"]["date_start"])).days + 1


def _quarantine_breakdown(qdf: pd.DataFrame) -> list[dict]:
    if qdf.empty:
        return []
    g = qdf.groupby(["field", "reason", "action"]).size().reset_index(name="count")
    return g.sort_values("count", ascending=False).to_dict("records")


# ------------------------------------------------------------------------ calendar
def validate_calendar(raw: pd.DataFrame, cfg: dict):
    v = _Validator(raw, "calendar")
    df = v.df
    date = pd.to_datetime(df["date"].astype("string").str.strip(), format="%Y-%m-%d", errors="coerce")
    v.reject(date.isna(), "date", "unparseable date")
    start, end = pd.Timestamp(cfg["data"]["date_start"]), pd.Timestamp(cfg["data"]["date_end"])
    v.reject(date.notna() & ((date < start) | (date > end)), "date", "outside study period")
    v.reject(date.notna() & date.duplicated(keep="first"), "date", "duplicate date (join key must be unique)")
    for flag in ("public_holiday", "event_flag", "roadwork_flag"):
        num = pd.to_numeric(df[flag], errors="coerce")
        v.reject(~num.isin([0, 1]), flag, "flag must be 0 or 1")
        df[flag] = num
    for name in ("holiday_name", "event_name"):
        s = df[name].astype("string").str.strip()
        df[name] = s.where(s != "", other=pd.NA)
    df["date"] = date
    df["src_line"] = df.index + 2
    out = df.loc[~v.rejected, ["date", "public_holiday", "holiday_name", "event_flag",
                               "event_name", "roadwork_flag", "src_line"]].copy()
    for flag in ("public_holiday", "event_flag", "roadwork_flag"):
        out[flag] = out[flag].astype("int8")
    out = out.sort_values("date").reset_index(drop=True)
    missing_dates = _expected_days(cfg) - out["date"].nunique()
    stats = {"rows_in": len(raw), "rows_out": len(out), "missing_dates": int(missing_dates)}
    return out, v.q.frame(), stats


# ------------------------------------------------------------------------- weather
def validate_weather(raw: pd.DataFrame, cfg: dict):
    v = _Validator(raw, "weather")
    df = v.df
    station = df["station_id"].astype("string").str.strip()
    v.reject(~station.isin(cfg["validation"]["weather_stations"]), "station_id", "unknown weather station")
    ts = v.check_timestamp("%d/%m/%Y %H:%M", cfg, step_minutes=60)  # weather uses DD/MM/YYYY
    v.check_ranges(cfg["validation"]["weather"], required=False)
    cond = df["weather_condition"].astype("string").str.strip()
    df["weather_condition"] = cond.where(cond != "", other=pd.NA)  # labels harmonised in Step 3
    df["station_id"] = station
    df["timestamp"] = ts
    df["src_line"] = df.index + 2
    out = df.loc[~v.rejected, ["station_id", "timestamp", "weather_condition", "temperature",
                               "rainfall", "visibility", "src_line"]]
    out = out.sort_values(["station_id", "timestamp"]).reset_index(drop=True)
    keys = out[["station_id", "timestamp"]]
    expected = len(cfg["validation"]["weather_stations"]) * _expected_days(cfg) * 24
    stats = {
        "rows_in": len(raw), "rows_out": len(out),
        "duplicate_keys": int(keys.duplicated().sum()),
        "expected_hours": int(expected),
        "missing_hours": int(expected - len(keys.drop_duplicates())),
        "nulls_remaining": {c: int(out[c].isna().sum()) for c in
                            ["weather_condition", "temperature", "rainfall", "visibility"]},
    }
    return out, v.q.frame(), stats


# -------------------------------------------------------------------------- traffic
def _parse_vehicle_mix(s, tol: float):
    """Return the 4 class shares from a JSON string, or None if it is invalid."""
    try:
        d = json.loads(s)
        if not isinstance(d, dict) or set(d) != set(VEHICLE_CLASSES):
            return None
        vals = [float(d[k]) for k in VEHICLE_CLASSES]
    except (TypeError, ValueError):
        return None
    if any(x < 0 for x in vals) or abs(sum(vals) - 1.0) > tol:
        return None
    return vals


def validate_traffic(raw: pd.DataFrame, cfg: dict):
    v = _Validator(raw, "traffic")
    df, val = v.df, cfg["validation"]

    # 1. identifiers
    rid = df["road_id"].astype("string").str.strip()
    v.reject(rid.isna() | ~rid.str.fullmatch(val["road_id_pattern"]).fillna(False),
             "road_id", "missing or malformed road_id")
    df["road_id"] = rid
    station = df["weather_station_id"].astype("string").str.strip()
    v.reject(~station.isin(val["weather_stations"]), "weather_station_id", "unknown weather station")
    df["weather_station_id"] = station

    # 2. timestamp (YYYY-MM-DD + HH:MM, 30-minute grid, inside study period)
    ts = v.check_timestamp("%Y-%m-%d %H:%M", cfg, step_minutes=cfg["data"]["window_minutes"])

    # 3. required numerics (row removed) and optional measurements (value nulled)
    v.check_ranges(val["traffic_required"], required=True)
    v.check_ranges(val["traffic"], required=False)

    # 4. each segment must have one consistent name / location / station / capacity
    sub = df.loc[~v.rejected]
    for col in ROAD_META:
        usual = sub.groupby("road_id")[col].agg(lambda s: s.mode().iloc[0])
        mismatch = pd.Series(False, index=df.index)
        mismatch.loc[sub.index] = sub[col] != sub["road_id"].map(usual)
        v.reject(mismatch, col, "inconsistent with the segment's usual value")

    # 5. congestion label (blank is fine - derived in Step 3; unknown labels are nulled)
    cong = df["congestion_level"].astype("string").str.strip()
    v.null_out("congestion_level", cong.notna() & ~cong.isin(CONGESTION_LABELS),
               "unknown congestion label", df["congestion_level"])
    df["congestion_level"] = cong.where(cong.isin(CONGESTION_LABELS), other=pd.NA)

    # 6. nested vehicle mix JSON -> four share columns
    tol = val["vehicle_share_tolerance"]
    mix_raw = df["vehicle_type_dist"]
    parsed = [(_parse_vehicle_mix(s, tol) if isinstance(s, str) else None) for s in mix_raw]
    shares = np.array([p if p is not None else [np.nan] * 4 for p in parsed], dtype=float)
    invalid = pd.Series([p is None and isinstance(s, str) for p, s in zip(parsed, mix_raw)], index=df.index)
    v.null_out("vehicle_type_dist", invalid, "invalid JSON, classes or shares", mix_raw)
    for i, cls in enumerate(VEHICLE_CLASSES):
        df[f"share_{cls}"] = shares[:, i]

    # 7. assemble output
    df["timestamp"] = ts
    df["src_line"] = df.index + 2
    cols = ["road_id", "road_name", "latitude", "longitude", "weather_station_id", "timestamp",
            "traffic_volume", "vehicle_count", *[f"share_{c}" for c in VEHICLE_CLASSES],
            "avg_speed", "occupancy", "congestion_level", "travel_time", "accident_count",
            "signal_timing", "road_capacity", "src_line"]
    out = df.loc[~v.rejected, cols].copy()
    for c in ("accident_count", "signal_timing", "road_capacity"):
        out[c] = out[c].astype("int64")
    out["vehicle_count"] = out["vehicle_count"].astype("Int64")
    out = out.sort_values(["road_id", "timestamp"], kind="stable").reset_index(drop=True)

    keys = out[["road_id", "timestamp"]]
    n_roads = out["road_id"].nunique()
    if n_roads != cfg["data"]["n_segments"]:
        log.warning("expected %d segments but found %d", cfg["data"]["n_segments"], n_roads)
    expected = n_roads * _expected_days(cfg) * (24 * 60 // cfg["data"]["window_minutes"])
    present = len(keys.drop_duplicates())
    stats = {
        "rows_in": len(raw), "rows_out": len(out), "segments": int(n_roads),
        "duplicate_keys": int(keys.duplicated().sum()),
        "expected_windows": int(expected), "missing_windows": int(expected - present),
        "nulls_remaining": {c: int(out[c].isna().sum()) for c in
                            ["traffic_volume", "avg_speed", "occupancy", "congestion_level"]},
    }
    return out, v.q.frame(), stats


# -------------------------------------------------------------------------- runner
def _read_raw(path: Path) -> pd.DataFrame:
    # Everything as text: validation, not pandas guessing, decides what each value is.
    return pd.read_csv(path, dtype=str)


def ingest_all(cfg: dict | None = None, write: bool = True) -> dict:
    """Load + validate calendar, weather and traffic (smallest first, as the dictionary advises)."""
    cfg = cfg or load_config()
    set_seed(cfg["project"]["seed"])
    raw_dir, interim = resolve(cfg["paths"]["raw_dir"]), resolve(cfg["paths"]["interim_dir"])
    validators = {"calendar": validate_calendar, "weather": validate_weather, "traffic": validate_traffic}
    summary: dict = {}
    for name, fn in validators.items():
        raw = _read_raw(raw_dir / cfg["raw_files"][name])
        log.info("%-8s loaded %d rows", name, len(raw))
        out, qdf, stats = fn(raw, cfg)
        stats["quarantined_records"] = int(len(qdf))
        stats["rows_removed"] = int(stats["rows_in"] - stats["rows_out"])
        stats["quarantine_breakdown"] = _quarantine_breakdown(qdf)
        summary[name] = stats
        log.info("%-8s validated: %d kept, %d removed, %d quarantine records",
                 name, stats["rows_out"], stats["rows_removed"], len(qdf))
        if write:
            interim.mkdir(parents=True, exist_ok=True)
            (interim / "quarantine").mkdir(exist_ok=True)
            out.to_parquet(interim / f"{name}_validated.parquet", index=False)
            qdf.to_csv(interim / "quarantine" / f"{name}_quarantine.csv", index=False)
    if write:
        rep = resolve(cfg["paths"]["reports_dir"])
        rep.mkdir(parents=True, exist_ok=True)
        (rep / "ingest_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        log.info("wrote %s", rep / "ingest_summary.json")
    return summary


if __name__ == "__main__":
    ingest_all()
