"""Flag which tagged nuPlan anchors can be simulated, for the extracted train-city logs.

Applies the selection rules of the mini_68 experiment (configs/scenarios/README.md) to every
distinct tagged anchor in nuplan-v1.1/trainval. Anchors are flagged, never dropped:

    in_valid_scene        not in the first two or last two scenes of the log (devkit discovery rule)
    valid_mission_goal    the scene's goal_ego_pose_token resolves to an ego_pose row
    has_route_roadblocks  the scene's roadblock_ids is non-empty
    complete_window       [anchor - 3 s, anchor + 12 s] lies inside the log's lidar_pc time range
    simulation_ready      all four of the above
    benchmark_scenario_type  alphabetically last of the 14 challenge tags on the anchor (MAX(type) rule)
    shares_mini_recording the log comes from a drive (source_recording) that also has a mini log
    benchmark_candidate   simulation_ready, has a benchmark type, and not shares_mini_recording

source_recording is the drive a log segment was cut from, "<date>_veh-<id>" (log name without
its two trailing segment numbers). Several logs share one recording, so recordings, not logs,
are the unit for overlap checks and train/val/test splits.

Usage (from the repository root):
    python -m tools.build_scenario_eligibility build            # resumable, one shard per log
    python -m tools.build_scenario_eligibility check            # validate, combine, write reports
    python -m tools.build_scenario_eligibility verify-mini      # reproduce the mini_68 numbers
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sqlite3
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ELIGIBILITY_VERSION = "eligibility-v1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path("D:/nuplan/dataset")
DEFAULT_PROCESSED = Path("D:/nuplan/processed")
DEFAULT_MINI_ZIP = Path("D:/nuplan/downloads/nuplan-v1.1_mini.zip")
DEFAULT_SCRATCH = Path("D:/nuplan/scratch")
DEFAULT_REPORT_DIR = PROJECT_ROOT / "docs" / "data"
MINI_MANIFEST = PROJECT_ROOT / "configs" / "scenarios" / "idm_mini_sample.csv"

WINDOW_OFFSET_US = -3_000_000
WINDOW_DURATION_US = 15_000_000
BENCHMARK_TYPES = (
    "behind_long_vehicle", "changing_lane", "following_lane_with_lead", "high_lateral_acceleration",
    "high_magnitude_speed", "low_magnitude_speed", "near_multiple_vehicles", "starting_left_turn",
    "starting_right_turn", "starting_straight_traffic_light_intersection_traversal",
    "stationary_in_traffic", "stopping_with_lead", "traversing_pickup_dropoff",
    "waiting_for_pedestrian_to_cross",
)
LOG_NAME = re.compile(r"^(?P<recording>\d{4}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2}_veh-\d+)_\d+_\d+$")
FLAGS = ["in_valid_scene", "valid_mission_goal", "has_route_roadblocks", "complete_window"]
COLUMNS = [
    "scenario_id", "log_name", "db_file", "scenario_token", "anchor_timestamp_us", "source_city",
    "map_name", "source_recording", "all_scenario_tags", "tag_count", "tag_rows", "benchmark_scenario_type",
    "scene_index", "scene_count", "in_valid_scene", "valid_mission_goal", "has_route_roadblocks",
    "log_start_us", "log_end_us", "window_start_us", "window_end_us", "complete_window",
    "simulation_ready", "in_mini_split", "shares_mini_recording", "benchmark_candidate", "eligibility_version",
]

# One row per distinct tagged anchor (lidar_pc). Same scene/goal/roadblock rules as the mini EDA
# notebook, but computed as flags instead of INNER JOIN filters so nothing is dropped.
ANCHOR_SQL = """
WITH ordered AS (SELECT token, ROW_NUMBER() OVER (ORDER BY name ASC) AS idx FROM scene)
SELECT
    LOWER(HEX(lp.token)) AS scenario_token,
    lp.timestamp AS anchor_timestamp_us,
    o.idx AS scene_index,
    COUNT(*) AS tag_rows,
    GROUP_CONCAT(DISTINCT st.type) AS tags,
    CASE WHEN s.goal_ego_pose_token IS NOT NULL AND goal.token IS NOT NULL THEN 1 ELSE 0 END
        AS valid_mission_goal,
    CASE WHEN s.roadblock_ids IS NOT NULL AND TRIM(s.roadblock_ids) NOT IN ('', '[]') THEN 1 ELSE 0 END
        AS has_route_roadblocks
FROM scenario_tag AS st
JOIN lidar_pc AS lp ON lp.token = st.lidar_pc_token
JOIN scene AS s ON s.token = lp.scene_token
JOIN ordered AS o ON o.token = s.token
LEFT JOIN ego_pose AS goal ON goal.token = s.goal_ego_pose_token
GROUP BY lp.token
ORDER BY lp.timestamp, scenario_token
"""


def source_recording(log_name: str) -> str:
    match = LOG_NAME.match(log_name)
    if not match:
        raise ValueError(f"Unexpected nuPlan log name: {log_name}")
    return match["recording"]


def benchmark_type(tags: list[str]) -> str | None:
    matching = set(tags) & set(BENCHMARK_TYPES)
    return max(matching) if matching else None


def anchor_frame(db_path: Path, city: str, mini_logs: set[str], mini_recordings: set[str]) -> pd.DataFrame:
    """Return one eligibility row per tagged anchor of one log database."""
    log_name = db_path.stem
    recording = source_recording(log_name)
    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    try:
        df = pd.read_sql_query(ANCHOR_SQL, conn)
        log_start, log_end = conn.execute("SELECT MIN(timestamp), MAX(timestamp) FROM lidar_pc").fetchone()
        scene_count = conn.execute("SELECT COUNT(*) FROM scene").fetchone()[0]
        map_row = conn.execute("SELECT map_version FROM log LIMIT 1").fetchone()
    finally:
        conn.close()
    tags = df.pop("tags").fillna("").map(lambda v: sorted(set(filter(None, v.split(",")))))
    df.insert(0, "scenario_id", log_name + ":" + df["scenario_token"])
    df.insert(1, "log_name", log_name)
    df.insert(2, "db_file", db_path.name)
    df["source_city"] = city
    df["map_name"] = map_row[0] if map_row else None
    df["source_recording"] = recording
    df["all_scenario_tags"] = tags.map("|".join)
    df["tag_count"] = tags.map(len).astype("int64")
    df["benchmark_scenario_type"] = tags.map(benchmark_type)
    df["scene_count"] = int(scene_count)
    df["in_valid_scene"] = (df["scene_index"] >= 3) & (df["scene_index"] < scene_count - 1)
    df["valid_mission_goal"] = df["valid_mission_goal"].astype(bool)
    df["has_route_roadblocks"] = df["has_route_roadblocks"].astype(bool)
    df["log_start_us"] = log_start
    df["log_end_us"] = log_end
    df["window_start_us"] = df["anchor_timestamp_us"] + WINDOW_OFFSET_US
    df["window_end_us"] = df["window_start_us"] + WINDOW_DURATION_US
    df["complete_window"] = (df["window_start_us"] >= log_start) & (df["window_end_us"] <= log_end)
    df["simulation_ready"] = df[FLAGS].all(axis=1)
    df["in_mini_split"] = log_name in mini_logs
    df["shares_mini_recording"] = recording in mini_recordings
    df["benchmark_candidate"] = (
        df["simulation_ready"] & df["benchmark_scenario_type"].notna() & ~df["shares_mini_recording"]
    )
    df["eligibility_version"] = ELIGIBILITY_VERSION
    for col in ("anchor_timestamp_us", "scene_index", "tag_rows", "scene_count", "log_start_us",
                "log_end_us", "window_start_us", "window_end_us"):
        df[col] = df[col].astype("int64")
    return df[COLUMNS]


def mini_reference(mini_zip: Path) -> tuple[set[str], set[str]]:
    """Mini log names and their source recordings, read from the mini archive's entry list."""
    with zipfile.ZipFile(mini_zip) as z:
        logs = {Path(i.filename).stem for i in z.infolist() if i.filename.lower().endswith(".db")}
    if not logs:
        raise ValueError(f"No .db entries in {mini_zip}")
    return logs, {source_recording(n) for n in logs}


def load_log_cities(data_root: Path) -> dict[str, str]:
    """log_name -> city from the extraction provenance written by tools.extract_nuplan_logs check."""
    provenance = data_root / "nuplan-v1.1" / "trainval_extraction.json"
    if not provenance.is_file():
        raise FileNotFoundError(f"{provenance} missing; run `python -m tools.extract_nuplan_logs check` first")
    meta = json.loads(provenance.read_text(encoding="utf-8"))
    if not meta.get("complete"):
        raise ValueError(f"{provenance} reports an incomplete extraction")
    return {log["log_name"]: log["city"] for log in meta["logs"]}


def process_log(db_file: str, city: str, out_file: str, mini_logs: set[str], mini_recordings: set[str]) -> dict:
    out = Path(out_file)
    started = time.time()
    try:
        df = anchor_frame(Path(db_file), city, mini_logs, mini_recordings)
        tmp = out.with_name(out.name + ".tmp")
        df.to_parquet(tmp, index=False)
        os.replace(tmp, out)
        return {"ok": True, "log_name": out.stem, "rows": len(df), "seconds": round(time.time() - started, 2)}
    except Exception as exc:
        return {"ok": False, "log_name": out.stem, "city": city, "error": f"{type(exc).__name__}: {exc}"}


def build(args) -> int:
    cities = load_log_cities(args.data_root)
    mini_logs, mini_recordings = mini_reference(args.mini_zip)
    db_dir = args.data_root / "nuplan-v1.1" / "trainval"
    shard_root = args.processed / "eligibility"
    jobs = []
    for log_name, city in sorted(cities.items()):
        if args.city and city not in args.city:
            continue
        out = shard_root / city / f"{log_name}.parquet"
        if out.exists():
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        jobs.append((str(db_dir / f"{log_name}.db"), city, str(out)))
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"{len(cities)} logs, {len(jobs)} to process with {args.workers} workers -> {shard_root}")
    failures, done, rows, t0 = [], 0, 0, time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_log, *job, mini_logs, mini_recordings) for job in jobs]
        for fut in as_completed(futures):
            res = fut.result()
            done += 1
            if res["ok"]:
                rows += res["rows"]
            else:
                failures.append(res)
                print(f"  FAIL {res['log_name']}: {res['error']}", flush=True)
            if done % 500 == 0 or done == len(jobs):
                rate = done / max(time.time() - t0, 1e-9)
                print(f"  {done}/{len(jobs)} logs, {rows:,} anchors, {len(failures)} failures, "
                      f"{rate:.0f} logs/s, ETA {(len(jobs) - done) / rate / 60 if rate else 0:.1f} min", flush=True)
    shard_root.mkdir(parents=True, exist_ok=True)
    (shard_root / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    print(f"Finished: {done - len(failures)} written, {len(failures)} failed. Re-run to retry failures.")
    return 1 if failures else 0


def funnel(df: pd.DataFrame) -> list[dict]:
    """Sequential filter counts, in the order the rules are documented."""
    steps = [
        ("tagged anchors", pd.Series(True, index=df.index)),
        ("in valid scene (not first/last two scenes)", df["in_valid_scene"]),
        ("valid mission goal", df["valid_mission_goal"]),
        ("route roadblocks present", df["has_route_roadblocks"]),
        ("complete 15 s window", df["complete_window"]),
        ("has one of the 14 benchmark tags", df["benchmark_scenario_type"].notna()),
        ("recording not shared with mini", ~df["shares_mini_recording"]),
    ]
    mask, out = pd.Series(True, index=df.index), []
    for label, step in steps:
        mask &= step
        out.append({"step": label, "anchors": int(mask.sum()), "logs": int(df.loc[mask, "log_name"].nunique()),
                    "recordings": int(df.loc[mask, "source_recording"].nunique())})
    return out


def summarize(df: pd.DataFrame) -> dict:
    cand = df[df["benchmark_candidate"]]
    by_city_type = (cand.groupby(["benchmark_scenario_type", "source_city"]).size()
                    .unstack(fill_value=0).reindex(index=list(BENCHMARK_TYPES), fill_value=0))
    by_city = df.groupby("source_city").agg(
        logs=("log_name", "nunique"), recordings=("source_recording", "nunique"), anchors=("scenario_id", "size"),
        simulation_ready=("simulation_ready", "sum"), benchmark_candidates=("benchmark_candidate", "sum"))
    cand_by_city = cand.groupby("source_city").agg(
        candidate_logs=("log_name", "nunique"), candidate_recordings=("source_recording", "nunique"))
    by_city = by_city.join(cand_by_city).fillna(0).astype("int64")
    return {
        "eligibility_version": ELIGIBILITY_VERSION,
        "rules": {
            "scene_rule": "exclude anchors in the first two and last two scenes (ordered by scene.name)",
            "window_us": [WINDOW_OFFSET_US, WINDOW_OFFSET_US + WINDOW_DURATION_US],
            "benchmark_types": list(BENCHMARK_TYPES),
            "benchmark_label": "alphabetically last matching benchmark tag (MAX(type))",
            "mini_exclusion": "whole source_recording excluded if it also has a mini log",
        },
        "funnel": funnel(df),
        "by_city": {city: row.to_dict() for city, row in by_city.iterrows()},
        "benchmark_candidates_by_type_and_city": {t: {c: int(v) for c, v in row.items()}
                                                  for t, row in by_city_type.iterrows()},
        "mini_overlap": {"logs_in_mini_split": int(df.loc[df["in_mini_split"], "log_name"].nunique()),
                         "logs_sharing_mini_recording": int(df.loc[df["shares_mini_recording"], "log_name"].nunique())},
    }


def markdown_report(summary: dict, checks: dict) -> str:
    cities = sorted(summary["by_city"])
    lines = [
        "# Train-city scenario eligibility",
        "",
        "Generated by `python -m tools.build_scenario_eligibility check` "
        f"({summary['eligibility_version']}, {checks['generated_utc']}).",
        "The per-anchor table stays outside Git at `D:\\nuplan\\processed\\scenario_eligibility.parquet`.",
        "Rules are the `mini_68` selection rules; see the module docstring and `configs/scenarios/README.md`.",
        "",
        "## Funnel",
        "",
        "Each row applies one more rule. Counts are distinct tagged anchors (one anchor can carry several tags).",
        "",
        "| Step | Anchors | Logs | Recordings |",
        "|---|---:|---:|---:|",
    ]
    lines += [f"| {s['step']} | {s['anchors']:,} | {s['logs']:,} | {s['recordings']:,} |" for s in summary["funnel"]]
    lines += ["", "## By city", "",
              "| City | Logs | Recordings | Anchors | Simulation-ready | Benchmark candidates | Candidate recordings |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for c in cities:
        r = summary["by_city"][c]
        lines.append(f"| {c} | {r['logs']:,} | {r['recordings']:,} | {r['anchors']:,} | {r['simulation_ready']:,} "
                     f"| {r['benchmark_candidates']:,} | {r['candidate_recordings']:,} |")
    lines += ["", "## Benchmark candidates by category and city", "",
              "| Category | " + " | ".join(cities) + " | Total |", "|---|" + "---:|" * (len(cities) + 1)]
    for t, row in summary["benchmark_candidates_by_type_and_city"].items():
        vals = [row.get(c, 0) for c in cities]
        lines.append(f"| {t} | " + " | ".join(f"{v:,}" for v in vals) + f" | {sum(vals):,} |")
    mo = summary["mini_overlap"]
    lines += ["", "## Notes", "",
              f"- {mo['logs_sharing_mini_recording']} train logs come from a recording that also has a mini log; "
              f"{mo['logs_in_mini_split']} of them are the identical log. All are excluded from benchmark candidates "
              "so the mini_68 results stay independent.",
              "- Candidate anchors within one recording overlap heavily in time. Counts are an upper bound on "
              "independent scenarios; selection must still reject overlapping windows per recording.",
              "- Most anchors are lost at the mission-goal rule because the source scenes have no goal: in a "
              "400-log sample, 54-72% of scenes per city had a NULL `goal_ego_pose_token` and none pointed to a "
              "missing ego_pose. The simulation runs with `remove_invalid_goals=true`, so these cannot be simulated.",
              "- After the goal rule, the roadblock and window rules remove no further anchors in this data.",
              "- A cell of 0 means that category cannot be sampled in that city "
              "(`traversing_pickup_dropoff` exists only in Las Vegas). Small cells limit equal per-city quotas.",
              "", "## Validation", ""]
    lines += [f"- {'PASS' if ok else 'FAIL'}: {name}" for name, ok in checks["results"].items()]
    return "\n".join(lines) + "\n"


def check(args) -> int:
    cities = load_log_cities(args.data_root)
    shard_root = args.processed / "eligibility"
    results, problems = {}, []
    present = {p.stem: p for city in set(cities.values()) for p in (shard_root / city).glob("*.parquet")}
    missing = sorted(set(cities) - set(present))
    results[f"one shard per extracted log ({len(cities):,})"] = not missing
    if missing:
        problems.append(f"{len(missing)} logs without a shard, e.g. {missing[0]}")
    failures = json.loads((shard_root / "failures.json").read_text()) if (shard_root / "failures.json").exists() else []
    results["no failed logs in failures.json"] = not failures
    df = pd.concat([pd.read_parquet(present[name]) for name in sorted(set(cities) & set(present))], ignore_index=True)

    results["scenario_id unique"] = bool(df["scenario_id"].is_unique)
    key_nulls = {c: int(df[c].isna().sum()) for c in COLUMNS if c != "benchmark_scenario_type"}
    results["no nulls outside benchmark_scenario_type"] = not any(key_nulls.values())
    # Cross-check against the independent catalog: tag rows per log must match its shard row count.
    catalog_dir = args.processed / "catalog"
    tag_rows = df.groupby("log_name")["tag_rows"].sum()
    mismatched = []
    for log_name, city in cities.items():
        shard = catalog_dir / city / f"{log_name}.parquet"
        if shard.exists():
            n = len(pd.read_parquet(shard, columns=["scenario_tag_token"]))
            if n != int(tag_rows.get(log_name, 0)):
                mismatched.append(log_name)
        else:
            mismatched.append(log_name)
    results["tag rows per log equal the scenario catalog"] = not mismatched
    if mismatched:
        problems.append(f"{len(mismatched)} logs differ from the catalog, e.g. {mismatched[0]}")
    ready = df["simulation_ready"]
    results["simulation_ready implies every rule"] = bool(df.loc[ready, FLAGS].all().all())
    results["benchmark candidates exclude mini recordings"] = not bool(
        (df["benchmark_candidate"] & df["shares_mini_recording"]).any())
    for name, ok in results.items():
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    summary = summarize(df)
    checks = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "results": results, "problems": problems, "complete": all(results.values())}
    out = args.processed / "scenario_eligibility.parquet"
    df.to_parquet(out, index=False)
    (args.processed / "scenario_eligibility_summary.json").write_text(
        json.dumps({**summary, "checks": checks}, indent=2), encoding="utf-8")
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "trainval_eligibility.json").write_text(
        json.dumps({**summary, "checks": checks}, indent=2) + "\n", encoding="utf-8", newline="\n")
    (args.report_dir / "trainval_eligibility.md").write_text(
        markdown_report(summary, checks), encoding="utf-8", newline="\n")
    print(json.dumps(summary["funnel"], indent=2))
    print(f"Wrote {out} ({len(df):,} anchors) and reports in {args.report_dir}. "
          f"Status: {'COMPLETE' if checks['complete'] else 'INCOMPLETE'}")
    return 0 if checks["complete"] else 1


def verify_mini(args) -> int:
    """Run the same rules on the 64 mini DBs and compare with the documented mini_68 numbers."""
    mini_logs, mini_recordings = mini_reference(args.mini_zip)
    scratch = args.scratch / "verify_mini"  # keeps the original file name: the stem is the log name
    scratch.mkdir(parents=True, exist_ok=True)
    frames = []
    with zipfile.ZipFile(args.mini_zip) as z:
        for info in sorted((i for i in z.infolist() if i.filename.lower().endswith(".db")), key=lambda i: i.filename):
            db = scratch / Path(info.filename).name
            try:
                with z.open(info) as src, open(db, "wb") as dst:
                    shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
                frames.append(anchor_frame(db, "mini", mini_logs, mini_recordings))
            finally:
                db.unlink(missing_ok=True)
    df = pd.concat(frames, ignore_index=True)
    valid = df[df["in_valid_scene"]]
    with MINI_MANIFEST.open(newline="", encoding="utf-8") as stream:
        manifest = list(csv.DictReader(stream))
    by_id = df.set_index("scenario_id")
    manifest_ok = all(
        r["scenario_id"] in by_id.index
        and bool(by_id.at[r["scenario_id"], "simulation_ready"])
        and by_id.at[r["scenario_id"], "benchmark_scenario_type"] == r["scenario_type"]
        and int(by_id.at[r["scenario_id"], "window_start_us"]) == int(r["window_start_us"])
        and by_id.at[r["scenario_id"], "all_scenario_tags"] == r["all_scenario_tags"]
        for r in manifest
    )
    results = {
        # One mini log (2021.06.23.20.43.31_veh-16_03607_04007) has no scenario_tag rows.
        "64 mini DBs read (63 with tagged anchors)": len(frames) == 64 and df["log_name"].nunique() == 63,
        "334,753 anchors after the scene rule (configs/scenarios/README.md)": len(valid) == 334_753,
        "57,992 benchmark anchors with valid mission goals": int(
            (valid["benchmark_scenario_type"].notna() & valid["valid_mission_goal"]).sum()) == 57_992,
        "68 manifest rows simulation-ready with identical type, tags and window": manifest_ok,
    }
    for name, ok in results.items():
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    return 0 if all(results.values()) else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["build", "check", "verify-mini"])
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--processed", type=Path, default=DEFAULT_PROCESSED)
    parser.add_argument("--mini-zip", type=Path, default=DEFAULT_MINI_ZIP)
    parser.add_argument("--scratch", type=Path, default=DEFAULT_SCRATCH)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 4))
    parser.add_argument("--city", action="append", choices=["boston", "las_vegas", "pittsburgh", "singapore"])
    parser.add_argument("--limit", type=int, help="Process at most N logs (smoke test)")
    args = parser.parse_args(argv)
    return {"build": build, "check": check, "verify-mini": verify_mini}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
