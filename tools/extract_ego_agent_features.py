"""Ego-vehicle and surrounding-agent features at simulation start, for nuPlan scenarios.

Simulation start (iteration 0) follows the pinned devkit (a581fbc):
    nuplan_scenario_mapping.yaml gives [15.0 s, -3.0 s] for all 14 challenge types, and
    extract_sensor_tokens_as_scenario selects lidar_pc rows with timestamp >= anchor - 3 s
    (inclusive), ordered by time. Iteration 0 is the first of them.

At that frame, mirroring get_ego_state_for_lidarpc_token_from_db and
get_tracked_objects_for_lidarpc_token_from_db:
    ego_x, ego_y            rear-axle position (map lookups only, not model inputs)
    ego_heading             yaw from the ego_pose quaternion (global frame, not a model input)
    ego_speed               |(vx, vy)|, m/s
    ego_acceleration        |(acceleration_x, acceleration_y)|, m/s^2
    ego_yaw_rate            ego_pose.angular_rate_z, rad/s. The devkit's initial ego state sets
                            angular velocity to 0, so the raw log value is used instead.
    num_tracked_agents      every lidar_box (agents and static objects), as the devkit returns
    num_vehicles, num_pedestrians
    nearest_agent_distance_m         centre distance from the ego centre (rear axle + 1.461 m,
                                     Pacifica parameters) to the nearest vehicle/pedestrian/bicycle
    nearest_agent_relative_speed_mps closing speed to that agent; positive = approaching
    agents_within_50m       vehicles, pedestrians and bicycles with centre within 50 m

Ego velocity is stored in the ego frame and rotated by the heading; lidar_box velocity is global.
Both were verified against position differences on sampled logs (see docs/data/ego_agent_features.md).

Usage (from the repository root):
    python -m tools.extract_ego_agent_features build          # resumable, one shard per log
    python -m tools.extract_ego_agent_features build-mini     # the 68 mini_68 scenarios
    python -m tools.extract_ego_agent_features verify-mini    # recompute mini_68 with the devkit SQL
    python -m tools.extract_ego_agent_features check          # validate, combine, reports
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sqlite3
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_VERSION = "ego-agent-v1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path("D:/nuplan/dataset")
DEFAULT_PROCESSED = Path("D:/nuplan/processed")
DEFAULT_MINI_ZIP = Path("D:/nuplan/downloads/nuplan-v1.1_mini.zip")
DEFAULT_SCRATCH = Path("D:/nuplan/scratch")
DEFAULT_REPORT_DIR = PROJECT_ROOT / "docs" / "data"
MINI_MANIFEST = PROJECT_ROOT / "configs" / "scenarios" / "idm_mini_sample.csv"

EXTRACTION_OFFSET_US = -3_000_000          # nuplan_scenario_mapping.yaml, all 14 challenge types
REAR_AXLE_TO_CENTER_M = 4.049 / 2 + 1.127 / 2 - 1.127   # Pacifica: half length - rear length = 1.461
AGENT_CATEGORIES = ("vehicle", "pedestrian", "bicycle")  # devkit AGENT_TYPES
NEAR_RADIUS_M = 50.0
CITY_BY_MAP = {"us-ma-boston": "boston", "us-nv-las-vegas-strip": "las_vegas",
               "us-pa-pittsburgh-hazelwood": "pittsburgh", "sg-one-north": "singapore"}

EGO = ["ego_x", "ego_y", "ego_heading", "ego_speed", "ego_acceleration", "ego_yaw_rate"]
AGENTS = ["num_tracked_agents", "num_vehicles", "num_pedestrians", "nearest_agent_distance_m",
          "nearest_agent_relative_speed_mps", "agents_within_50m"]
FEATURES = EGO + AGENTS
KEYS = ["scenario_id", "log_name", "scenario_token"]
AUDIT = KEYS + ["source_city", "log_token", "anchor_timestamp_us", "start_lidar_token", "start_timestamp_us",
                "start_offset_us", "ego_pose_lag_us"]


# ------------------------------------------------------------------------------- core extraction --

def heading_from_quaternion(qw, qx, qy, qz):
    """Yaw (rotation about z) of a unit quaternion; equals pyquaternion yaw_pitch_roll[0]."""
    return np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def start_frames(lidar_ts: np.ndarray, anchor_ts: np.ndarray) -> np.ndarray:
    """Index of the first lidar_pc with timestamp >= anchor - 3 s (lidar_ts sorted ascending)."""
    return np.searchsorted(lidar_ts, anchor_ts + EXTRACTION_OFFSET_US, side="left")


def agent_features(ego: pd.DataFrame, boxes: pd.DataFrame) -> pd.DataFrame:
    """Per-frame agent features.

    ego: index frame_idx, columns ego_x, ego_y, ego_heading, ego_vx_local, ego_vy_local.
    boxes: columns frame_idx, category, x, y, vx, vy (global frame).
    """
    out = pd.DataFrame(index=ego.index)
    counts = boxes.groupby("frame_idx").size()
    out["num_tracked_agents"] = counts.reindex(out.index, fill_value=0)
    for col, cat in (("num_vehicles", "vehicle"), ("num_pedestrians", "pedestrian")):
        out[col] = boxes[boxes["category"].eq(cat)].groupby("frame_idx").size().reindex(out.index, fill_value=0)
    agents = boxes[boxes["category"].isin(AGENT_CATEGORIES)].merge(
        ego.reset_index(), on="frame_idx", how="inner")
    cos_h, sin_h = np.cos(agents["ego_heading"]), np.sin(agents["ego_heading"])
    cx = agents["ego_x"] + REAR_AXLE_TO_CENTER_M * cos_h
    cy = agents["ego_y"] + REAR_AXLE_TO_CENTER_M * sin_h
    ego_vx = agents["ego_vx_local"] * cos_h - agents["ego_vy_local"] * sin_h
    ego_vy = agents["ego_vx_local"] * sin_h + agents["ego_vy_local"] * cos_h
    dx, dy = agents["x"] - cx, agents["y"] - cy
    dist = np.hypot(dx, dy)
    # Closing speed: -d|p|/dt = -(p . v_rel) / |p|; undefined (NaN) for coincident centres.
    closing = -(dx * (agents["vx"] - ego_vx) + dy * (agents["vy"] - ego_vy)) / dist.where(dist > 0)
    agents = agents.assign(dist=dist, closing=closing)
    out["agents_within_50m"] = agents[agents["dist"] <= NEAR_RADIUS_M].groupby("frame_idx").size().reindex(
        out.index, fill_value=0)
    nearest = agents.sort_values(["frame_idx", "dist"]).drop_duplicates("frame_idx").set_index("frame_idx")
    out["nearest_agent_distance_m"] = nearest["dist"].reindex(out.index)
    out["nearest_agent_relative_speed_mps"] = nearest["closing"].reindex(out.index)
    for col in ("num_tracked_agents", "num_vehicles", "num_pedestrians", "agents_within_50m"):
        out[col] = out[col].astype("int64")
    return out


def lidar_channel_token(conn: sqlite3.Connection) -> bytes:
    rows = conn.execute("SELECT token, channel FROM lidar").fetchall()
    merged = [r for r in rows if r[1] == "MergedPointCloud"]
    if len(rows) == 1:
        return rows[0][0]
    if len(merged) == 1:
        return merged[0][0]
    raise ValueError(f"cannot identify the lidar channel among {[r[1] for r in rows]}")


def extract_log(db_path: Path, scenarios: pd.DataFrame, city: str) -> pd.DataFrame:
    """Features and audit columns for the given scenarios (scenario_id, scenario_token, anchor_timestamp_us)."""
    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    try:
        log_token = conn.execute("SELECT token FROM log LIMIT 1").fetchone()[0].hex()
        lidar = lidar_channel_token(conn)
        frames = pd.read_sql_query(
            "SELECT token, timestamp, ego_pose_token FROM lidar_pc WHERE lidar_token = ? ORDER BY timestamp",
            conn, params=(lidar,))
        lidar_ts = frames["timestamp"].to_numpy("int64")
        idx = start_frames(lidar_ts, scenarios["anchor_timestamp_us"].to_numpy("int64"))
        valid = idx < len(frames)
        sel = np.unique(idx[valid])
        conn.execute("CREATE TEMP TABLE sel (token BLOB PRIMARY KEY, frame_idx INTEGER)")
        conn.executemany("INSERT INTO sel VALUES (?, ?)",
                         ((frames.at[i, "token"], int(i)) for i in sel))
        ego = pd.read_sql_query("""
            SELECT sel.frame_idx, ep.x, ep.y, ep.qw, ep.qx, ep.qy, ep.qz, ep.vx, ep.vy,
                   ep.acceleration_x, ep.acceleration_y, ep.angular_rate_z, ep.timestamp AS ego_ts
            FROM sel JOIN lidar_pc AS lp ON lp.token = sel.token
            JOIN ego_pose AS ep ON ep.token = lp.ego_pose_token""", conn).set_index("frame_idx")
        boxes = pd.read_sql_query("""
            SELECT sel.frame_idx, c.name AS category, lb.x, lb.y, lb.vx, lb.vy
            FROM sel JOIN lidar_box AS lb ON lb.lidar_pc_token = sel.token
            JOIN track AS t ON t.token = lb.track_token
            JOIN category AS c ON c.token = t.category_token""", conn)
    finally:
        conn.close()
    ego_f = pd.DataFrame(index=ego.index)
    ego_f["ego_x"], ego_f["ego_y"] = ego["x"], ego["y"]
    ego_f["ego_heading"] = heading_from_quaternion(ego["qw"], ego["qx"], ego["qy"], ego["qz"])
    ego_f["ego_speed"] = np.hypot(ego["vx"], ego["vy"])
    ego_f["ego_acceleration"] = np.hypot(ego["acceleration_x"], ego["acceleration_y"])
    ego_f["ego_yaw_rate"] = ego["angular_rate_z"]
    ego_f["ego_vx_local"], ego_f["ego_vy_local"] = ego["vx"], ego["vy"]
    per_frame = ego_f.join(agent_features(ego_f, boxes))

    out = scenarios[["scenario_id", "scenario_token", "anchor_timestamp_us"]].copy()
    out.insert(1, "log_name", db_path.stem)
    out["source_city"] = city
    out["log_token"] = log_token
    out["frame_idx"] = np.where(valid, idx, -1)
    out = out.join(per_frame, on="frame_idx")
    safe = np.where(valid, idx, 0)
    out["start_lidar_token"] = np.where(valid, frames["token"].map(lambda b: b.hex()).to_numpy()[safe], None)
    out["start_timestamp_us"] = pd.Series(lidar_ts[safe], index=out.index, dtype="Int64").where(valid)
    out["start_offset_us"] = out["start_timestamp_us"] - out["anchor_timestamp_us"]
    out["ego_pose_lag_us"] = (out["frame_idx"].map(ego["ego_ts"]) - out["start_timestamp_us"]).astype("Int64")
    return out[AUDIT + FEATURES]


# ------------------------------------------------------------------------------------- jobs --

def candidates_by_log(processed: Path) -> pd.DataFrame:
    return pd.read_parquet(processed / "scenario_eligibility.parquet",
                           columns=["scenario_id", "log_name", "scenario_token", "anchor_timestamp_us", "source_city"],
                           filters=[("benchmark_candidate", "==", True)])


def process_log(db_file: str, scen: pd.DataFrame, city: str, out_file: str) -> dict:
    out, started = Path(out_file), time.time()
    try:
        df = extract_log(Path(db_file), scen, city)
        tmp = out.with_name(out.name + ".tmp")
        df.to_parquet(tmp, index=False)
        os.replace(tmp, out)
        return {"ok": True, "log_name": out.stem, "rows": len(df), "seconds": round(time.time() - started, 2)}
    except Exception as exc:
        return {"ok": False, "log_name": out.stem, "city": city, "error": f"{type(exc).__name__}: {exc}"}


def build(args) -> int:
    cand = candidates_by_log(args.processed)
    db_dir = args.data_root / "nuplan-v1.1" / "trainval"
    root = args.processed / "features" / "ego_agent"
    jobs = []
    for log_name, scen in cand.groupby("log_name", sort=True):
        city = scen["source_city"].iat[0]
        if args.city and city not in args.city:
            continue
        out = root / city / f"{log_name}.parquet"
        if out.exists():
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        jobs.append((str(db_dir / f"{log_name}.db"), scen.reset_index(drop=True), city, str(out)))
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"{cand['log_name'].nunique()} logs with candidates, {len(jobs)} to process with {args.workers} workers")
    failures, done, rows, t0 = [], 0, 0, time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for fut in as_completed([pool.submit(process_log, *job) for job in jobs]):
            res = fut.result()
            done += 1
            if res["ok"]:
                rows += res["rows"]
            else:
                failures.append(res)
                print(f"  FAIL {res['log_name']}: {res['error']}", flush=True)
            if done % 250 == 0 or done == len(jobs):
                rate = done / max(time.time() - t0, 1e-9)
                print(f"  {done}/{len(jobs)} logs, {rows:,} scenarios, {len(failures)} failures, "
                      f"{rate:.1f} logs/s, ETA {(len(jobs) - done) / rate / 60 if rate else 0:.1f} min", flush=True)
    root.mkdir(parents=True, exist_ok=True)
    (root / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    print(f"Finished: {done - len(failures)} written, {len(failures)} failed. Re-run to retry failures.")
    return 1 if failures else 0


def build_mini(args) -> int:
    with MINI_MANIFEST.open(newline="", encoding="utf-8") as stream:
        mini = pd.DataFrame(list(csv.DictReader(stream)))
    mini["anchor_timestamp_us"] = mini["anchor_timestamp_us"].astype("int64")
    scratch = args.scratch / "ego_agent_mini"
    scratch.mkdir(parents=True, exist_ok=True)
    frames = []
    with zipfile.ZipFile(args.mini_zip) as z:
        entries = {Path(i.filename).stem: i for i in z.infolist() if i.filename.lower().endswith(".db")}
        for log_name, scen in mini.groupby("log_name", sort=True):
            db = scratch / f"{log_name}.db"
            try:
                with z.open(entries[log_name]) as src, open(db, "wb") as dst:
                    shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
                frames.append(extract_log(db, scen.reset_index(drop=True), CITY_BY_MAP[scen["map_name"].iat[0]]))
            finally:
                db.unlink(missing_ok=True)
    df = pd.concat(frames, ignore_index=True).sort_values("scenario_id").reset_index(drop=True)
    out = args.processed / "features" / "ego_agent_mini_68.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"Wrote {out}: {len(df)} scenarios from {df['log_name'].nunique()} logs")
    return 0


# Verbatim from nuplan/database/nuplan_db/nuplan_scenario_queries.py (devkit a581fbc).
DEVKIT_WINDOW_SQL = """
    WITH numbered AS
    (
        SELECT token, timestamp, ROW_NUMBER() OVER (ORDER BY timestamp ASC) AS row_num
        FROM lidar_pc
        WHERE timestamp >= ?
        AND timestamp <= ?
        AND lidar_token == ?
    )
    SELECT token
    FROM numbered
    WHERE ((row_num - 1) % ?) = 0
    ORDER BY timestamp ASC;
"""
DEVKIT_EGO_SQL = """
        SELECT  ep.x, ep.y, ep.qw, ep.qx, ep.qy, ep.qz, lp.timestamp, ep.vx, ep.vy,
                ep.acceleration_x, ep.acceleration_y, ep.angular_rate_z
        FROM ego_pose AS ep
        INNER JOIN lidar_pc AS lp
            ON lp.ego_pose_token = ep.token
        WHERE lp.token = ?
"""
DEVKIT_OBJECTS_SQL = """
        SELECT  c.name AS category_name, lb.x, lb.y, lb.vx, lb.vy
        FROM lidar_box AS lb
        INNER JOIN track AS t
            ON t.token = lb.track_token
        INNER JOIN category AS c
            ON c.token = t.category_token
        INNER JOIN lidar_pc AS lp
            ON lp.token = lb.lidar_pc_token
        WHERE lp.token = ?
"""


def reference_features(db_path: Path, anchor_ts: int) -> dict:
    """Slow per-scenario recomputation with the devkit's own queries and plain Python maths."""
    import math

    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    try:
        lidar = lidar_channel_token(conn)
        start = anchor_ts + EXTRACTION_OFFSET_US
        # subsample_ratio_override 0.5 -> every 2nd row; iteration 0 is the first row either way.
        token = conn.execute(DEVKIT_WINDOW_SQL, (start, start + 15_000_000, lidar, 2)).fetchone()[0]
        e = conn.execute(DEVKIT_EGO_SQL, (token,)).fetchone()
        objs = conn.execute(DEVKIT_OBJECTS_SQL, (token,)).fetchall()
    finally:
        conn.close()
    x, y, qw, qx, qy, qz, ts, vx, vy, ax, ay, wz = e
    yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    cx, cy = x + REAR_AXLE_TO_CENTER_M * math.cos(yaw), y + REAR_AXLE_TO_CENTER_M * math.sin(yaw)
    gvx, gvy = vx * math.cos(yaw) - vy * math.sin(yaw), vx * math.sin(yaw) + vy * math.cos(yaw)
    agents = [(math.hypot(ox - cx, oy - cy), ox - cx, oy - cy, ovx - gvx, ovy - gvy)
              for cat, ox, oy, ovx, ovy in objs if cat in AGENT_CATEGORIES]
    nearest = min(agents) if agents else None
    return {
        "start_lidar_token": token.hex(), "ego_x": x, "ego_y": y, "ego_heading": yaw,
        "ego_speed": math.hypot(vx, vy), "ego_acceleration": math.hypot(ax, ay), "ego_yaw_rate": wz,
        "num_tracked_agents": len(objs), "num_vehicles": sum(o[0] == "vehicle" for o in objs),
        "num_pedestrians": sum(o[0] == "pedestrian" for o in objs),
        "agents_within_50m": sum(a[0] <= NEAR_RADIUS_M for a in agents),
        "nearest_agent_distance_m": nearest[0] if nearest else float("nan"),
        "nearest_agent_relative_speed_mps": -(nearest[1] * nearest[3] + nearest[2] * nearest[4]) / nearest[0]
        if nearest else float("nan"),
    }


def verify_mini(args) -> int:
    """Compare the vectorized mini_68 table with reference_features() for every scenario."""
    table = pd.read_parquet(args.processed / "features" / "ego_agent_mini_68.parquet").set_index("scenario_id")
    scratch = args.scratch / "ego_agent_verify"
    scratch.mkdir(parents=True, exist_ok=True)
    worst: dict[str, float] = {}
    token_mismatch = 0
    with zipfile.ZipFile(args.mini_zip) as z:
        entries = {Path(i.filename).stem: i for i in z.infolist() if i.filename.lower().endswith(".db")}
        for log_name, rows in table.groupby("log_name"):
            db = scratch / f"{log_name}.db"
            try:
                with z.open(entries[log_name]) as src, open(db, "wb") as dst:
                    shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
                for sid, row in rows.iterrows():
                    ref = reference_features(db, int(row["anchor_timestamp_us"]))
                    token_mismatch += ref.pop("start_lidar_token") != row["start_lidar_token"]
                    for k, v in ref.items():
                        diff = 0.0 if (pd.isna(v) and pd.isna(row[k])) else abs(float(v) - float(row[k]))
                        worst[k] = max(worst.get(k, 0.0), diff)
            finally:
                db.unlink(missing_ok=True)
    ok = token_mismatch == 0 and all(d < 1e-6 for d in worst.values())
    print(f"[{'PASS' if token_mismatch == 0 else 'FAIL'}] start frame equals the devkit window query's first token "
          f"({token_mismatch} mismatches of {len(table)})")
    print(f"[{'PASS' if ok else 'FAIL'}] every feature equals the per-scenario devkit-query recomputation "
          f"(max abs difference {max(worst.values()):.2e})")
    return 0 if ok else 1


# ------------------------------------------------------------------------------------- check --

def feature_table(df: pd.DataFrame) -> pd.DataFrame:
    """Join keys + features, with <name>_missing flags only for columns that have nulls."""
    out = df[KEYS + FEATURES].copy()
    for col in FEATURES:
        if out[col].isna().any():
            out[f"{col}_missing"] = out[col].isna()
    return out


def write_parquet(df: pd.DataFrame, path: Path, meta: dict) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pandas(df, preserve_index=False)
    table = table.replace_schema_metadata({**(table.schema.metadata or {}),
                                           b"ego_agent_provenance": json.dumps(meta).encode()})
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(table, tmp)
    tmp.replace(path)


def check(args) -> int:
    root = args.processed / "features"
    cand = candidates_by_log(args.processed)
    shards = sorted((root / "ego_agent").glob("*/*.parquet"))
    missing_logs = sorted(set(cand["log_name"]) - {p.stem for p in shards})
    df = pd.concat([pd.read_parquet(p) for p in shards], ignore_index=True)
    mini = pd.read_parquet(root / "ego_agent_mini_68.parquet")
    with MINI_MANIFEST.open(newline="", encoding="utf-8") as stream:
        manifest_ids = {r["scenario_id"] for r in csv.DictReader(stream)}
    both = pd.concat([df, mini], ignore_index=True)
    failures = json.loads((root / "ego_agent" / "failures.json").read_text())
    results = {
        "one shard per log with candidates": not missing_logs,
        "no failed logs": not failures,
        "one row per benchmark candidate": len(df) == len(cand) and set(df["scenario_id"]) == set(cand["scenario_id"]),
        "68 mini_68 rows matching the manifest": set(mini["scenario_id"]) == manifest_ids and len(mini) == 68,
        "scenario_id unique": bool(df["scenario_id"].is_unique and mini["scenario_id"].is_unique),
        "every scenario has a start frame and ego pose": bool(both["start_timestamp_us"].notna().all()
                                                              and both[EGO].notna().all().all()),
        "start frame within [anchor - 3 s, anchor - 3 s + 0.1 s)": bool(
            both["start_offset_us"].between(EXTRACTION_OFFSET_US, EXTRACTION_OFFSET_US + 100_000 - 1).all()),
        "ego_pose matched within 0.1 s of the lidar frame": bool(both["ego_pose_lag_us"].abs().max() < 100_000),
        "physical ranges (speed < 45 m/s, |accel| < 20, |yaw rate| < 3)": bool(
            both["ego_speed"].between(0, 45).all() and both["ego_acceleration"].between(0, 20).all()
            and both["ego_yaw_rate"].abs().lt(3).all()),
        "counts consistent (vehicles + pedestrians <= tracked; within-50m <= tracked)": bool(
            (both["num_vehicles"] + both["num_pedestrians"] <= both["num_tracked_agents"]).all()
            and (both["agents_within_50m"] <= both["num_tracked_agents"]).all()),
        "nearest distance <= 50 m whenever an agent is within 50 m": bool(
            (both.loc[both["agents_within_50m"] > 0, "nearest_agent_distance_m"] <= NEAR_RADIUS_M).all()),
        "nearest-agent values null only when no agent is present": bool(
            (both["nearest_agent_distance_m"].isna() == both["nearest_agent_relative_speed_mps"].isna()).all()),
    }
    for name, ok in results.items():
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")

    elig = pd.read_parquet(args.processed / "scenario_eligibility.parquet",
                           columns=["scenario_id", "benchmark_scenario_type"],
                           filters=[("benchmark_candidate", "==", True)])
    typed = df.merge(elig, on="scenario_id")
    by_type = typed.groupby("benchmark_scenario_type").agg(
        scenarios=("scenario_id", "size"), ego_speed_median=("ego_speed", "median"),
        ego_speed_p90=("ego_speed", lambda s: s.quantile(0.9)), vehicles_median=("num_vehicles", "median"),
        pedestrians_median=("num_pedestrians", "median"), within_50m_median=("agents_within_50m", "median"),
        nearest_median_m=("nearest_agent_distance_m", "median")).round(2)
    by_city = df.groupby("source_city").agg(
        scenarios=("scenario_id", "size"), ego_speed_median=("ego_speed", "median"),
        tracked_median=("num_tracked_agents", "median"), within_50m_median=("agents_within_50m", "median"),
        no_agent_scenarios=("nearest_agent_distance_m", lambda s: int(s.isna().sum()))).round(2)
    meta = {"feature_version": FEATURE_VERSION, "devkit_commit": "a581fbcab373db99a13f191f94521c75c256e080",
            "extraction_offset_us": EXTRACTION_OFFSET_US, "rear_axle_to_center_m": round(REAR_AXLE_TO_CENTER_M, 3),
            "near_radius_m": NEAR_RADIUS_M}
    write_parquet(feature_table(df), root / "scenario_ego_agent.parquet", meta)
    write_parquet(feature_table(mini), root / "scenario_ego_agent_mini_68.parquet", meta)
    write_parquet(pd.concat([df[AUDIT], mini[AUDIT]], keys=["trainval_candidates", "mini_68"])
                  .reset_index(level=0).rename(columns={"level_0": "subset"}),
                  root / "scenario_ego_agent_alignment.parquet", meta)
    summary = {"feature_version": FEATURE_VERSION, "provenance": meta,
               "rows": {"trainval_candidates": int(len(df)), "mini_68": int(len(mini))},
               "null_counts": {c: int(both[c].isna().sum()) for c in FEATURES if both[c].isna().any()},
               "by_city": by_city.to_dict(orient="index"), "by_type": by_type.to_dict(orient="index"),
               "checks": {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                          "results": results, "complete": all(results.values())}}
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "ego_agent_features.json").write_text(
        json.dumps(summary, indent=2, default=float) + "\n", encoding="utf-8", newline="\n")
    (args.report_dir / "ego_agent_features.md").write_text(markdown_report(summary), encoding="utf-8", newline="\n")
    print(f"Wrote feature tables in {root} and reports in {args.report_dir}. "
          f"Status: {'COMPLETE' if summary['checks']['complete'] else 'INCOMPLETE'}")
    return 0 if summary["checks"]["complete"] else 1


def markdown_report(s: dict) -> str:
    lines = [
        "# Ego and surrounding-agent features",
        "",
        f"Generated by `python -m tools.extract_ego_agent_features check` ({s['feature_version']}, "
        f"{s['checks']['generated_utc']}). Tables stay outside Git in `D:\\nuplan\\processed\\features\\`.",
        "",
        "## Simulation start",
        "",
        "Features are taken at iteration 0 of the devkit scenario (commit `a581fbc`): the first `lidar_pc` "
        "with timestamp >= anchor - 3.0 s. `nuplan_scenario_mapping.yaml` gives `[15.0, -3.0]` for all 14 "
        "challenge types and the window query is inclusive. No later frame is read.",
        "",
        "## Definitions",
        "",
        "| Column | Definition | Model input |",
        "|---|---|---|",
        "| `ego_x`, `ego_y` | Rear-axle position (UTM, m) | No: identifies location; for map lookups |",
        "| `ego_heading` | Yaw from the ego_pose quaternion (rad) | No: global-frame angle |",
        "| `ego_speed` | Magnitude of (vx, vy), m/s | Yes |",
        "| `ego_acceleration` | Magnitude of (acceleration_x, acceleration_y), m/s^2 | Yes |",
        "| `ego_yaw_rate` | `ego_pose.angular_rate_z`, rad/s | Yes |",
        "| `num_tracked_agents` | All lidar boxes, including static objects | Yes |",
        "| `num_vehicles`, `num_pedestrians` | Boxes by category | Yes |",
        "| `nearest_agent_distance_m` | Ego centre to nearest vehicle/pedestrian/bicycle centre | Yes |",
        "| `nearest_agent_relative_speed_mps` | Closing speed to that agent; positive = approaching | Yes |",
        "| `agents_within_50m` | Vehicles, pedestrians and bicycles within 50 m | Yes |",
        "",
        "Verified conventions: ego_pose velocity is in the ego frame (rotating it by the heading matches "
        "position differences to 0.19 m/s median, versus 8.9 m/s if read as global); lidar_box velocity is "
        "global (0.83 m/s versus 10.8 m/s); `angular_rate_z` correlates 0.996 with the heading derivative. "
        "The devkit's `get_ego_state_for_lidarpc_token_from_db` does not pass angular velocity (it is 0), so "
        "`ego_yaw_rate` uses the raw log value. Ego centre = rear axle + 1.461 m (Pacifica parameters).",
        "",
        "## By city (benchmark candidates)",
        "",
        "| City | Scenarios | Median speed (m/s) | Median tracked objects | Median agents within 50 m | Scenarios with no agent |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for c, r in sorted(s["by_city"].items()):
        lines.append(f"| {c} | {int(r['scenarios']):,} | {r['ego_speed_median']:.2f} | {r['tracked_median']:.0f} "
                     f"| {r['within_50m_median']:.0f} | {int(r['no_agent_scenarios']):,} |")
    lines += ["", "## By category (sanity check)", "",
              "| Category | Scenarios | Median speed | 90th pct speed | Median vehicles | Median pedestrians | Median within 50 m | Median nearest (m) |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for t, r in sorted(s["by_type"].items()):
        lines.append(f"| {t} | {int(r['scenarios']):,} | {r['ego_speed_median']:.2f} | {r['ego_speed_p90']:.2f} "
                     f"| {r['vehicles_median']:.0f} | {r['pedestrians_median']:.0f} | {r['within_50m_median']:.0f} "
                     f"| {r['nearest_median_m']:.1f} |")
    nulls = ", ".join(f"`{k}`: {v:,}" for k, v in s["null_counts"].items()) or "none"
    lines += ["", f"Rows: {s['rows']['trainval_candidates']:,} benchmark candidates, {s['rows']['mini_68']} mini_68. "
              f"Null values (scenarios with no vehicle, pedestrian or bicycle; stored with a `_missing` flag): {nulls}.",
              "", "## Not included", "",
              "Map and traffic-light fields (`lane_type`, `in_intersection`, `traffic_light_state`) are a separate "
              "task. Scenario tags are not used: they describe the logged driver's behaviour after the start.",
              "", "## Validation", ""]
    lines += [f"- {'PASS' if ok else 'FAIL'}: {n}" for n, ok in s["checks"]["results"].items()]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["build", "build-mini", "verify-mini", "check"])
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--processed", type=Path, default=DEFAULT_PROCESSED)
    parser.add_argument("--mini-zip", type=Path, default=DEFAULT_MINI_ZIP)
    parser.add_argument("--scratch", type=Path, default=DEFAULT_SCRATCH)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 4))
    parser.add_argument("--city", action="append", choices=["boston", "las_vegas", "pittsburgh", "singapore"])
    parser.add_argument("--limit", type=int, help="Process at most N logs (smoke test)")
    args = parser.parse_args(argv)
    return {"build": build, "build-mini": build_mini, "verify-mini": verify_mini, "check": check}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
