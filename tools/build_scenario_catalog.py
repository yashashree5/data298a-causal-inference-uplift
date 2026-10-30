"""Build the multi-city nuPlan scenario catalog from the training ZIP archives.

Local replacement for the Colab catalog cell. Each SQLite log is streamed out of
its ZIP into a scratch file, queried, and written as one Parquet shard:

    <output>/catalog/<city>/<log_name>.parquet

Completed shards are skipped, so the job can be stopped and resumed. Logs are
processed in parallel. Failures are written to <output>/catalog/failures.json.

Usage (from the repository root):

    python -m tools.build_scenario_catalog build
    python -m tools.build_scenario_catalog build --archives-dir D:/nuplan/downloads --workers 16
    python -m tools.build_scenario_catalog check      # validate shards, write scenario_catalog.parquet

Tagged anchors are candidates, not simulation-ready scenarios; eligibility is
decided in a later stage.
"""

from __future__ import annotations

import argparse
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

EXTRACTION_VERSION = "catalog-v1"

DEFAULT_ARCHIVE_DIRS = [Path.home() / "Downloads", Path("D:/nuplan/downloads")]
DEFAULT_OUTPUT = Path("D:/nuplan/processed")
DEFAULT_SCRATCH = Path("D:/nuplan/scratch")

# Matches nuplan-v1.1_train_boston.zip, nuplan-v1.1_train_boston (2).zip,
# nuplan-v1.1_train_vegas_3.zip, ...
ARCHIVE_PATTERN = re.compile(r"^nuplan-v1\.1_train_(?P<city>[a-z]+)(?:_(?P<part>\d+))?(?: \(\d+\))?\.zip$")
CITY_NAMES = {"boston": "boston", "pittsburgh": "pittsburgh", "singapore": "singapore", "vegas": "las_vegas"}

SQL = """
SELECT
    hex(s.token) AS scenario_tag_token,
    hex(s.lidar_pc_token) AS scenario_token,
    s.type AS scenario_type,
    l.timestamp AS anchor_timestamp_us,
    hex(l.ego_pose_token) AS ego_pose_token
FROM scenario_tag AS s
JOIN lidar_pc AS l ON s.lidar_pc_token = l.token
ORDER BY l.timestamp
"""

KEY = ["log_name", "scenario_tag_token"]


def discover_archives(dirs: list[Path]) -> list[tuple[str, Path]]:
    """Return (city, path) for each training archive found. The first directory wins on duplicates."""
    found: dict[str, tuple[str, Path]] = {}
    for d in dirs:
        if not d.is_dir():
            continue
        for path in sorted(d.glob("nuplan-v1.1_train_*.zip")):
            m = ARCHIVE_PATTERN.match(path.name)
            if not m or m["city"] not in CITY_NAMES:
                continue
            key = f"{m['city']}_{m['part'] or ''}"
            found.setdefault(key, (CITY_NAMES[m["city"]], path))
    return sorted(found.values(), key=lambda x: x[1].name)


def process_entry(archive: str, city: str, entry_name: str, out_file: str, scratch: str) -> dict:
    """Extract one DB, query it, write its shard atomically. Runs in a worker process."""
    out_path = Path(out_file)
    db_path = Path(scratch) / f"{os.getpid()}_{Path(entry_name).name}"
    started = time.time()
    try:
        with zipfile.ZipFile(archive) as z, z.open(entry_name) as src, open(db_path, "wb") as dst:
            shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
        conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
        try:
            df = pd.read_sql_query(SQL, conn)
            log_cols = {row[1] for row in conn.execute("PRAGMA table_info(log)")}
            wanted = [c for c in ("map_version", "location", "vehicle_name", "logfile") if c in log_cols]
            log_row = dict(zip(wanted, conn.execute(f"SELECT {', '.join(wanted)} FROM log LIMIT 1").fetchone() or [])) if wanted else {}
        finally:
            conn.close()
        log_name = Path(entry_name).stem
        df["log_name"] = log_name
        df["source_city"] = city
        df["map_name"] = log_row.get("map_version")
        df["log_location"] = log_row.get("location")
        df["vehicle_name"] = log_row.get("vehicle_name")
        df["source_archive"] = Path(archive).name
        df["source_db_entry"] = entry_name
        df["extraction_version"] = EXTRACTION_VERSION
        tmp = out_path.with_suffix(".tmp")
        df.to_parquet(tmp, index=False)
        os.replace(tmp, out_path)
        return {"ok": True, "log_name": log_name, "rows": len(df), "seconds": round(time.time() - started, 1)}
    except Exception as exc:  # recorded and retried on the next run
        return {"ok": False, "city": city, "archive": Path(archive).name, "entry": entry_name,
                "log_name": Path(entry_name).stem, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        db_path.unlink(missing_ok=True)


def build(args) -> int:
    archives = discover_archives(args.archives_dir)
    if not archives:
        print("No nuplan-v1.1_train_*.zip archives found in:", *args.archives_dir, sep="\n  ")
        return 2
    catalog_dir = args.output / "catalog"
    args.scratch.mkdir(parents=True, exist_ok=True)

    jobs = []
    for city, archive in archives:
        if args.city and city not in args.city:
            continue
        try:
            with zipfile.ZipFile(archive) as z:
                entries = [i.filename for i in z.infolist() if i.filename.lower().endswith(".db")]
        except zipfile.BadZipFile as exc:
            print(f"SKIP {archive.name}: unreadable ZIP ({exc}); is the download complete?")
            continue
        city_dir = catalog_dir / city
        city_dir.mkdir(parents=True, exist_ok=True)
        todo = [e for e in entries if not (city_dir / f"{Path(e).stem}.parquet").exists()]
        print(f"{archive.name}: {len(entries)} DBs, {len(entries) - len(todo)} done, {len(todo)} to process")
        jobs += [(str(archive), city, e, str(city_dir / f"{Path(e).stem}.parquet")) for e in todo]

    if args.limit:
        jobs = jobs[: args.limit]
    print(f"\nProcessing {len(jobs)} logs with {args.workers} workers -> {catalog_dir}")

    failures, done, rows, t0 = [], 0, 0, time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_entry, *job, str(args.scratch)) for job in jobs]
        for fut in as_completed(futures):
            res = fut.result()
            done += 1
            if res["ok"]:
                rows += res["rows"]
            else:
                failures.append(res)
                print(f"  FAIL {res['log_name']}: {res['error']}")
            if done % 100 == 0 or done == len(jobs):
                rate = done / max(time.time() - t0, 1e-9)
                eta = (len(jobs) - done) / rate if rate else 0
                print(f"  {done}/{len(jobs)} logs, {rows:,} anchors, {len(failures)} failures, "
                      f"{rate:.1f} logs/s, ETA {eta / 60:.1f} min", flush=True)

    catalog_dir.mkdir(parents=True, exist_ok=True)
    (catalog_dir / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    print(f"\nFinished: {done - len(failures)} written, {len(failures)} failed "
          f"(see {catalog_dir / 'failures.json'}). Re-run to retry failures.")
    return 1 if failures else 0


def check(args) -> int:
    """Validate shards against archive contents and write the combined scenario_catalog.parquet."""
    catalog_dir = args.output / "catalog"
    problems: list[str] = []
    expected: dict[str, set[str]] = {}
    archive_counts = {}
    for city, archive in discover_archives(args.archives_dir):
        try:
            with zipfile.ZipFile(archive) as z:
                logs = {Path(i.filename).stem for i in z.infolist() if i.filename.lower().endswith(".db")}
        except zipfile.BadZipFile:
            problems.append(f"unreadable archive {archive.name}")
            continue
        expected.setdefault(city, set()).update(logs)
        archive_counts[archive.name] = len(logs)

    frames, coverage = [], {}
    for city, logs in sorted(expected.items()):
        present = {p.stem for p in (catalog_dir / city).glob("*.parquet")}
        missing = logs - present
        coverage[city] = {"expected_logs": len(logs), "shards": len(present & logs), "missing": len(missing)}
        if missing:
            problems.append(f"{city}: {len(missing)} logs without a shard (e.g. {sorted(missing)[0]})")
        frames += [pd.read_parquet(catalog_dir / city / f"{log}.parquet") for log in sorted(present & logs)]

    if not frames:
        print("No shards found.")
        return 2
    cat = pd.concat(frames, ignore_index=True)
    dupes = int(cat.duplicated(KEY).sum())
    if dupes:
        problems.append(f"{dupes} duplicate (log_name, scenario_tag_token) rows")
    for col in ("scenario_token", "scenario_type", "anchor_timestamp_us", "map_name"):
        n = int(cat[col].isna().sum())
        if n:
            problems.append(f"{n} rows with missing {col}")
    failures_file = catalog_dir / "failures.json"
    n_fail = len(json.loads(failures_file.read_text())) if failures_file.exists() else 0
    if n_fail:
        problems.append(f"failures.json lists {n_fail} failed logs")

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "extraction_versions": sorted(cat["extraction_version"].dropna().unique().tolist()) if "extraction_version" in cat else [],
        "archives": archive_counts,
        "coverage": coverage,
        "rows": len(cat),
        "logs": int(cat["log_name"].nunique()),
        "rows_by_city": cat["source_city"].value_counts().to_dict(),
        "rows_by_map": cat["map_name"].value_counts().to_dict(),
        "scenario_types": int(cat["scenario_type"].nunique()),
        "duplicate_keys": dupes,
        "problems": problems,
        "complete": not problems,
    }
    out = args.output / "scenario_catalog.parquet"
    cat.to_parquet(out, index=False)
    (args.output / "scenario_catalog_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows_by_map"}, indent=2))
    print(f"\nWrote {out} ({len(cat):,} rows). Status: {'COMPLETE' if not problems else 'INCOMPLETE'}")
    return 0 if not problems else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["build", "check"])
    parser.add_argument("--archives-dir", type=Path, action="append",
                        help="Directory containing nuplan-v1.1_train_*.zip (repeatable). "
                             f"Default: {', '.join(map(str, DEFAULT_ARCHIVE_DIRS))}")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scratch", type=Path, default=DEFAULT_SCRATCH, help="Temporary DB extraction directory")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 4))
    parser.add_argument("--city", action="append", choices=sorted(set(CITY_NAMES.values())))
    parser.add_argument("--limit", type=int, help="Process at most N logs (smoke test)")
    args = parser.parse_args(argv)
    args.archives_dir = args.archives_dir or DEFAULT_ARCHIVE_DIRS
    return build(args) if args.command == "build" else check(args)


if __name__ == "__main__":
    sys.exit(main())
