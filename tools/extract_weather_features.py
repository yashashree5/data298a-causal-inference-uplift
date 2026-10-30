"""Historical weather and sun-position features for nuPlan scenarios, at simulation start.

Weather is an estimate for the scenario's city and hour from the Open-Meteo Historical Weather
API (https://open-meteo.com/en/docs/historical-weather-api; data CC BY 4.0, free for
non-commercial research). It is contextual information only: nuPlan replays recorded traffic,
so weather does not change the simulation itself.

Time alignment (no future information):
    t0         = window_start_us = anchor - 3 s, the scenario's simulation start
    hour H     = t0 floored to the UTC hour, so H <= t0
    instant    temperature, humidity, cloud cover, wind speed, weather code: value at H
    preceding  precipitation, rain, wind gusts: Open-Meteo's sum/max over (H - 1 h, H]
    prev 3 h   precipitation summed over the labels H, H-1, H-2, i.e. (H - 3 h, H]
    sun        solar elevation at t0 exactly (NOAA equations; no API)

Location: one point per city, the median of the first ego pose of every train log, converted
from the log's UTM zone (ego_pose.epsg) to WGS 84. Train logs in each city span <= 7 km, below
the 9 km model grid.

Usage (from the repository root):
    python -m tools.extract_weather_features fetch    # 4 API calls, cached; refuses to refetch
    python -m tools.extract_weather_features build    # offline: candidates + mini_68 features
    python -m tools.extract_weather_features check    # validate, write docs/data reports
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import statistics
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

WEATHER_VERSION = "weather-v1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path("D:/nuplan/dataset")
DEFAULT_PROCESSED = Path("D:/nuplan/processed")
DEFAULT_REPORT_DIR = PROJECT_ROOT / "docs" / "data"
MINI_MANIFEST = PROJECT_ROOT / "configs" / "scenarios" / "idm_mini_sample.csv"
API_URL = "https://archive-api.open-meteo.com/v1/archive"
# ECMWF IFS at ~9 km is Open-Meteo's archive model for 2017 onward. ERA5 (0.25 deg) puts the
# Singapore grid point about 22 km from one-north; IFS lies within 1 km of every city point.
WEATHER_MODEL = "ecmwf_ifs"
CITY_BY_MAP = {"us-ma-boston": "boston", "us-nv-las-vegas-strip": "las_vegas",
               "us-pa-pittsburgh-hazelwood": "pittsburgh", "sg-one-north": "singapore"}
# Open-Meteo variable -> (feature column, aggregation the API applies to the hour label)
HOURLY = {
    "temperature_2m": ("weather_temperature_c", "instant"),
    "relative_humidity_2m": ("weather_relative_humidity_pct", "instant"),
    "cloud_cover": ("weather_cloud_cover_pct", "instant"),
    "wind_speed_10m": ("weather_wind_speed_kmh", "instant"),
    "weather_code": ("weather_code", "instant"),
    "precipitation": ("weather_precipitation_mm", "preceding hour sum"),
    "rain": ("weather_rain_mm", "preceding hour sum"),
    "wind_gusts_10m": ("weather_wind_gusts_kmh", "preceding hour max"),
}
PREV_3H = "weather_precipitation_prev_3h_mm"
SUN = "sun_elevation_deg"
FEATURES = [col for col, _ in HOURLY.values()] + [PREV_3H, SUN]
KEYS = ["scenario_id", "log_name", "scenario_token"]
HOUR_US = 3_600_000_000


# ---------------------------------------------------------------------------------- sun position --

def sun_elevation_deg(timestamp_us, lat_deg: float, lon_deg: float):
    """Solar elevation angle (degrees, no refraction) from the NOAA solar equations.

    Accepts a scalar or array of UTC microsecond timestamps. Accurate to well under 1 degree,
    enough to separate day, twilight and night.
    """
    ts = np.asarray(timestamp_us, dtype="float64")
    jd = ts / 86_400e6 + 2440587.5            # Julian day from the Unix epoch
    t = (jd - 2451545.0) / 36525.0            # Julian centuries since J2000
    l0 = np.radians((280.46646 + t * (36000.76983 + 0.0003032 * t)) % 360)
    m = np.radians(357.52911 + t * (35999.05029 - 0.0001537 * t))
    e = 0.016708634 - t * (0.000042037 + 0.0000001267 * t)
    c = np.radians((1.914602 - t * (0.004817 + 0.000014 * t)) * np.sin(m)
                   + (0.019993 - 0.000101 * t) * np.sin(2 * m) + 0.000289 * np.sin(3 * m))
    omega = np.radians(125.04 - 1934.136 * t)
    lam = l0 + c - np.radians(0.00569 + 0.00478 * np.sin(omega))
    eps = np.radians(23 + (26 + (21.448 - t * (46.815 + t * (0.00059 - t * 0.001813))) / 60) / 60
                     + 0.00256 * np.cos(omega))
    decl = np.arcsin(np.sin(eps) * np.sin(lam))
    y = np.tan(eps / 2) ** 2
    eq_time_min = 4 * np.degrees(y * np.sin(2 * l0) - 2 * e * np.sin(m) + 4 * e * y * np.sin(m) * np.cos(2 * l0)
                                 - 0.5 * y * y * np.sin(4 * l0) - 1.25 * e * e * np.sin(2 * m))
    minutes_utc = (ts / 60e6) % 1440
    hour_angle = np.radians((minutes_utc + eq_time_min + 4 * lon_deg) / 4 - 180)
    lat = np.radians(lat_deg)
    cos_zenith = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(hour_angle)
    return np.degrees(np.arcsin(np.clip(cos_zenith, -1, 1)))


# ------------------------------------------------------------------------------- city locations --

def city_points(data_root: Path) -> dict:
    """Median WGS 84 position of the first ego pose of every train log, per city."""
    from pyproj import Transformer

    meta = json.loads((data_root / "nuplan-v1.1" / "trainval_extraction.json").read_text(encoding="utf-8"))
    db_dir = data_root / "nuplan-v1.1" / "trainval"
    by_city: dict[str, list] = {}
    for log in meta["logs"]:
        conn = sqlite3.connect(f"file:{(db_dir / (log['log_name'] + '.db')).as_posix()}?mode=ro", uri=True)
        try:
            row = conn.execute("SELECT x, y, epsg FROM ego_pose WHERE rowid = (SELECT MIN(rowid) FROM ego_pose)").fetchone()
        finally:
            conn.close()
        if row:
            by_city.setdefault(log["city"], []).append(row)
    points = {}
    for city, rows in sorted(by_city.items()):
        epsgs = {int(r[2]) for r in rows}
        if len(epsgs) != 1:
            raise ValueError(f"{city}: logs use several EPSG codes {sorted(epsgs)}")
        epsg = epsgs.pop()
        to_wgs84 = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
        lonlat = [to_wgs84.transform(x, y) for x, y, _ in rows]
        lats, lons = [p[1] for p in lonlat], [p[0] for p in lonlat]
        lat, lon = statistics.median(lats), statistics.median(lons)
        points[city] = {
            "latitude": round(lat, 5), "longitude": round(lon, 5), "source_epsg": epsg, "logs": len(rows),
            "span_km_north_south": round((max(lats) - min(lats)) * 111.2, 1),
            "span_km_east_west": round((max(lons) - min(lons)) * 111.2 * math.cos(math.radians(lat)), 1),
        }
    return points


# ---------------------------------------------------------------------------------------- fetch --

def scenario_frame(processed: Path) -> pd.DataFrame:
    """Benchmark candidates from the eligibility table plus the 68 mini scenarios."""
    elig = pd.read_parquet(processed / "scenario_eligibility.parquet",
                           columns=KEYS + ["source_city", "window_start_us"],
                           filters=[("benchmark_candidate", "==", True)])
    elig["subset"] = "trainval_candidates"
    with MINI_MANIFEST.open(newline="", encoding="utf-8") as stream:
        mini = pd.DataFrame(list(csv.DictReader(stream)))
    mini = pd.DataFrame({
        "scenario_id": mini["scenario_id"], "log_name": mini["log_name"], "scenario_token": mini["scenario_token"],
        "source_city": mini["map_name"].map(CITY_BY_MAP), "window_start_us": mini["window_start_us"].astype("int64"),
        "subset": "mini_68"})
    if mini["source_city"].isna().any():
        raise ValueError("mini manifest has an unknown map_name")
    return pd.concat([elig, mini], ignore_index=True)


def request_params(lat: float, lon: float, start: str, end: str) -> dict:
    return {"latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
            "hourly": ",".join(HOURLY), "timezone": "GMT", "models": WEATHER_MODEL}


def fetch(args) -> int:
    import requests

    out_dir = args.processed / "weather"
    raw = out_dir / "open_meteo_raw.json"
    if raw.exists():
        print(f"{raw} exists; refusing to refetch (delete it deliberately to fetch again).")
        return 0
    points = city_points(args.data_root)
    scen = scenario_frame(args.processed)
    starts = pd.to_datetime(scen["window_start_us"], unit="us", utc=True)
    responses = {}
    for city, point in points.items():
        mask = scen["source_city"].eq(city)
        if not mask.any():
            continue
        # One day of margin before the first scenario covers the 3-hour precipitation lookback.
        first = (starts[mask].min() - timedelta(days=1)).strftime("%Y-%m-%d")
        last = starts[mask].max().strftime("%Y-%m-%d")
        params = request_params(point["latitude"], point["longitude"], first, last)
        resp = requests.get(API_URL, params=params, timeout=120)
        resp.raise_for_status()
        body = resp.json()
        if "hourly" not in body:
            raise RuntimeError(f"{city}: unexpected API response {body}")
        responses[city] = {"request": params, "point": point, "response": body,
                           "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        print(f"{city}: {len(body['hourly']['time'])} hours {first}..{last}, "
              f"grid {body['latitude']}, {body['longitude']}")
    out_dir.mkdir(parents=True, exist_ok=True)
    raw.write_text(json.dumps({"api": API_URL, "model": WEATHER_MODEL, "weather_version": WEATHER_VERSION,
                               "cities": responses}, indent=2), encoding="utf-8")
    print(f"Wrote {raw}")
    return 0


# ---------------------------------------------------------------------------------------- build --

def hourly_frame(raw: dict) -> pd.DataFrame:
    """Long hourly table, one row per city and UTC hour label, with the 3-hour lookback sum."""
    frames = []
    for city, entry in raw["cities"].items():
        hourly = entry["response"]["hourly"]
        df = pd.DataFrame({col: hourly[var] for var, (col, _) in HOURLY.items()})
        df.insert(0, "hour_utc", pd.to_datetime(hourly["time"], utc=True))
        df.insert(0, "source_city", city)
        df = df.sort_values("hour_utc").reset_index(drop=True)
        if not df["hour_utc"].diff().dropna().eq(pd.Timedelta(hours=1)).all():
            raise ValueError(f"{city}: hourly series has gaps or duplicates")
        # Labels H, H-1, H-2 cover (H-3h, H]; min_periods=3 leaves the first two hours null.
        df[PREV_3H] = df["weather_precipitation_mm"].rolling(3, min_periods=3).sum().round(3)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def scenario_weather(scen: pd.DataFrame, hourly: pd.DataFrame, points: dict) -> pd.DataFrame:
    """Join each scenario to the hour at or before its simulation start and add sun elevation."""
    df = scen.copy()
    df["hour_utc"] = pd.to_datetime(df["window_start_us"] // HOUR_US * HOUR_US, unit="us", utc=True)
    df = df.merge(hourly, on=["source_city", "hour_utc"], how="left", validate="many_to_one")
    lat = df["source_city"].map(lambda c: points[c]["latitude"])
    lon = df["source_city"].map(lambda c: points[c]["longitude"])
    df[SUN] = np.round(sun_elevation_deg(df["window_start_us"].to_numpy(), lat.to_numpy(), lon.to_numpy()), 3)
    df["weather_code"] = df["weather_code"].astype("Int64")
    for col in FEATURES:
        if df[col].isna().any():
            df[f"{col}_missing"] = df[col].isna()
    return df


def write_table(df: pd.DataFrame, path: Path, meta: dict) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pandas(df, preserve_index=False)
    table = table.replace_schema_metadata({**(table.schema.metadata or {}),
                                           b"weather_provenance": json.dumps(meta).encode()})
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(table, tmp)
    tmp.replace(path)


def provenance(raw: dict) -> dict:
    return {
        "weather_version": WEATHER_VERSION, "api": raw["api"], "model": raw["model"], "license": "CC BY 4.0",
        "time_rule": "hour label floor(window_start_us); instant vars at H, accumulations over (H-1h, H]",
        "cities": {c: {"point": e["point"], "grid": [e["response"]["latitude"], e["response"]["longitude"]],
                       "grid_elevation_m": e["response"].get("elevation"), "fetched_utc": e["fetched_utc"],
                       "request": e["request"]} for c, e in raw["cities"].items()},
    }


def build(args) -> int:
    out_dir = args.processed / "weather"
    raw = json.loads((out_dir / "open_meteo_raw.json").read_text(encoding="utf-8"))
    points = {c: e["point"] for c, e in raw["cities"].items()}
    hourly = hourly_frame(raw)
    meta = provenance(raw)
    write_table(hourly, out_dir / "weather_hourly.parquet", meta)
    joined = scenario_weather(scenario_frame(args.processed), hourly, points)
    features = [c for c in joined.columns if c in FEATURES or c.endswith("_missing")]
    for subset, name in (("trainval_candidates", "scenario_weather.parquet"),
                         ("mini_68", "scenario_weather_mini_68.parquet")):
        part = joined.loc[joined["subset"].eq(subset), KEYS + features].reset_index(drop=True)
        write_table(part, out_dir / name, meta)
        print(f"Wrote {out_dir / name}: {len(part):,} scenarios")
    # Audit table: which hour and city each scenario used (not a model input).
    audit = joined[KEYS + ["subset", "source_city", "window_start_us", "hour_utc"]]
    write_table(audit, out_dir / "scenario_weather_alignment.parquet", meta)
    return 0


# ---------------------------------------------------------------------------------------- check --

def check(args) -> int:
    out_dir = args.processed / "weather"
    raw = json.loads((out_dir / "open_meteo_raw.json").read_text(encoding="utf-8"))
    meta = provenance(raw)
    hourly = pd.read_parquet(out_dir / "weather_hourly.parquet")
    cand = pd.read_parquet(out_dir / "scenario_weather.parquet")
    mini = pd.read_parquet(out_dir / "scenario_weather_mini_68.parquet")
    align = pd.read_parquet(out_dir / "scenario_weather_alignment.parquet")
    expected = scenario_frame(args.processed)
    results = {}
    results["one row per benchmark candidate"] = len(cand) == int(expected["subset"].eq("trainval_candidates").sum())
    results["68 mini_68 rows"] = len(mini) == 68
    results["scenario_id unique in each table"] = bool(cand["scenario_id"].is_unique and mini["scenario_id"].is_unique)
    results["no missing weather values"] = not any(c.endswith("_missing") for c in list(cand.columns) + list(mini.columns))
    # Microseconds since the epoch, independent of the stored datetime resolution (ns or us).
    hour_us = (align["hour_utc"] - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(microseconds=1)
    lag_us = align["window_start_us"] - hour_us
    results["weather hour is at or before simulation start (lag 0 to <1 h)"] = bool(
        ((lag_us >= 0) & (lag_us < HOUR_US)).all())
    both = pd.concat([cand, mini])
    ranges = {"weather_temperature_c": (-40, 50), "weather_relative_humidity_pct": (0, 100),
              "weather_cloud_cover_pct": (0, 100), "weather_precipitation_mm": (0, 200),
              "weather_rain_mm": (0, 200), PREV_3H: (0, 600), "weather_wind_speed_kmh": (0, 200),
              "weather_wind_gusts_kmh": (0, 300), SUN: (-90, 90), "weather_code": (0, 99)}
    results["values within physical ranges"] = all(both[c].between(lo, hi).all() for c, (lo, hi) in ranges.items())
    results["rain never exceeds precipitation"] = bool((both["weather_rain_mm"] <= both["weather_precipitation_mm"] + 1e-9).all())
    results["3-hour precipitation >= current hour"] = bool((both[PREV_3H] + 1e-9 >= both["weather_precipitation_mm"]).all())
    grid_km = {c: math.dist((v["point"]["latitude"], v["point"]["longitude"] * math.cos(math.radians(v["point"]["latitude"]))),
                            (v["grid"][0], v["grid"][1] * math.cos(math.radians(v["grid"][0])))) * 111.2
               for c, v in meta["cities"].items()}
    results["model grid point within 10 km of every city point"] = all(d <= 10 for d in grid_km.values())
    for name, ok in results.items():
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")

    cities = sorted(meta["cities"])
    cand_city = align.loc[align["subset"].eq("trainval_candidates"), ["scenario_id", "source_city", "hour_utc"]]
    merged = cand.merge(cand_city, on="scenario_id")
    by_city = {}
    for city in cities:
        part = merged[merged["source_city"].eq(city)]
        hours = hourly[hourly["source_city"].eq(city) & hourly["hour_utc"].isin(part["hour_utc"].unique())]
        by_city[city] = {
            "candidates": int(len(part)), "distinct_hours": int(part["hour_utc"].nunique()),
            "distinct_days": int(part["hour_utc"].dt.date.nunique()),
            "first_hour_utc": str(part["hour_utc"].min()), "last_hour_utc": str(part["hour_utc"].max()),
            "temperature_c_min_median_max": [float(hours["weather_temperature_c"].min()),
                                             float(hours["weather_temperature_c"].median()),
                                             float(hours["weather_temperature_c"].max())],
            "wet_hours_precip_gt_0_1mm": int((hours["weather_precipitation_mm"] > 0.1).sum()),
            "scenarios_in_wet_hours": int((part["weather_precipitation_mm"] > 0.1).sum()),
            "scenarios_at_night_sun_below_0": int((part[SUN] < 0).sum()),
            "grid_distance_km": round(grid_km[city], 2),
        }
    summary = {"weather_version": WEATHER_VERSION, "provenance": meta, "features": FEATURES, "by_city": by_city,
               "mini_68": {"scenarios": int(len(mini)), "wet": int((mini["weather_precipitation_mm"] > 0.1).sum()),
                           "night": int((mini[SUN] < 0).sum())},
               "checks": {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                          "results": results, "complete": all(results.values())}}
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "weather_features.json").write_text(json.dumps(summary, indent=2) + "\n",
                                                           encoding="utf-8", newline="\n")
    (args.report_dir / "weather_features.md").write_text(markdown_report(summary), encoding="utf-8", newline="\n")
    print(f"Reports written to {args.report_dir}. Status: {'COMPLETE' if summary['checks']['complete'] else 'INCOMPLETE'}")
    return 0 if summary["checks"]["complete"] else 1


def markdown_report(summary: dict) -> str:
    prov, cities = summary["provenance"], sorted(summary["by_city"])
    lines = [
        "# Weather features",
        "",
        f"Generated by `python -m tools.extract_weather_features check` ({summary['weather_version']}, "
        f"{summary['checks']['generated_utc']}).",
        f"Source: [Open-Meteo Historical Weather API]({prov['api'].replace('archive-api.', '').replace('/v1/archive', '/en/docs/historical-weather-api')}), "
        f"model `{prov['model']}` (~9 km), licence {prov['license']}.",
        "Tables stay outside Git in `D:\\nuplan\\processed\\weather\\`; this report and the code are committed.",
        "",
        "## Time alignment",
        "",
        "Each scenario uses the UTC hour **at or before** its simulation start (anchor - 3 s). Instant values "
        "(temperature, humidity, cloud cover, wind speed, weather code) are taken at that hour; precipitation, "
        "rain and gusts are Open-Meteo totals/maxima over the preceding hour, and the 3-hour precipitation sums "
        "the three hours ending there. Sun elevation is computed at the exact start time. No value describes "
        "time after simulation start.",
        "",
        "## Features",
        "",
        "| Column | Meaning |",
        "|---|---|",
        "| `weather_temperature_c` | Air temperature at 2 m |",
        "| `weather_relative_humidity_pct` | Relative humidity at 2 m |",
        "| `weather_cloud_cover_pct` | Total cloud cover |",
        "| `weather_wind_speed_kmh` | Wind speed at 10 m |",
        "| `weather_code` | WMO weather code (categorical) |",
        "| `weather_precipitation_mm` | Precipitation in the hour before start |",
        "| `weather_rain_mm` | Rain in the hour before start |",
        f"| `{PREV_3H}` | Precipitation in the 3 hours before start (wet road proxy) |",
        "| `weather_wind_gusts_kmh` | Maximum gust in the hour before start |",
        f"| `{SUN}` | Solar elevation at start; below 0 is night, 0 to -6 civil twilight |",
        "",
        "## Coverage by city (benchmark candidates)",
        "",
        "| City | Point (lat, lon) | Grid distance | Candidates | Hours | Days | Temperature min/median/max | Wet hours | Candidates in wet hours | Candidates at night |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for c in cities:
        r, p = summary["by_city"][c], prov["cities"][c]["point"]
        t = r["temperature_c_min_median_max"]
        lines.append(f"| {c} | {p['latitude']}, {p['longitude']} | {r['grid_distance_km']} km | {r['candidates']:,} "
                     f"| {r['distinct_hours']} | {r['distinct_days']} | {t[0]:.1f} / {t[1]:.1f} / {t[2]:.1f} °C "
                     f"| {r['wet_hours_precip_gt_0_1mm']} | {r['scenarios_in_wet_hours']:,} | {r['scenarios_at_night_sun_below_0']:,} |")
    m = summary["mini_68"]
    lines += [
        "",
        f"mini_68: {m['scenarios']} scenarios, {m['wet']} in wet hours, {m['night']} at night.",
        "",
        "## Limits",
        "",
        "- These are model estimates for a ~9 km grid cell, not measurements at the vehicle. Visibility is not "
        "provided by the historical API and is not included.",
        "- Every benchmark candidate was recorded in daylight (about 07:00-17:00 local time), so there are no night "
        "scenarios and sun elevation varies only between morning, midday and afternoon.",
        "- Las Vegas has no wet hours at all in its recording period; rain information comes from the other three cities.",
        "- Weather does not alter the nuPlan simulation; it can only explain planner differences through the "
        "recorded traffic situations it co-occurs with.",
        "- Variation is limited and tied to city and drive: scenarios from one drive share an hour, and each city "
        "was recorded over a few weeks of one season. Report weather effects per city and treat them as weak evidence.",
        "- A different archive model (for example ERA5) gives noticeably different hourly precipitation; the model "
        "is pinned and recorded so results are reproducible.",
        "",
        "## Validation",
        "",
    ]
    lines += [f"- {'PASS' if ok else 'FAIL'}: {name}" for name, ok in summary["checks"]["results"].items()]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["fetch", "build", "check"])
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--processed", type=Path, default=DEFAULT_PROCESSED)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args(argv)
    return {"fetch": fetch, "build": build, "check": check}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
