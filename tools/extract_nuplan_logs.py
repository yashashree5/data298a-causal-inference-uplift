"""Extract nuPlan v1.1 training log databases from the ZIP archives into one directory.

Layout written (matches the pinned devkit's `nuplan` scenario builder,
data_root = ${NUPLAN_DATA_ROOT}/nuplan-v1.1/trainval):

    <data-root>/nuplan-v1.1/trainval/<log_name>.db
    <data-root>/nuplan-v1.1/trainval_extraction.json   # per-log provenance, written by `check`

The archives store DBs as data/cache/train_<city>/<log_name>.db; the city folder is
dropped because log names are unique across cities. Each DB is written to a temporary
file and renamed only after the full read succeeds, so zipfile's CRC-32 check has passed.
Existing files with the expected size are skipped, so the job can be stopped and resumed.
Archives are never modified or deleted.

Usage (from the repository root):
    python -m tools.extract_nuplan_logs extract                  # all train archives
    python -m tools.extract_nuplan_logs extract --city boston --workers 4
    python -m tools.extract_nuplan_logs check                    # verify and write provenance
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

EXTRACTION_VERSION = "extract-v1"
DEFAULT_ARCHIVES_DIR = Path("D:/nuplan/downloads")
DEFAULT_DATA_ROOT = Path("D:/nuplan/dataset")
ARCHIVE_PATTERN = re.compile(r"^nuplan-v1\.1_train_(?P<city>[a-z]+)(?:_(?P<part>\d+))?(?: \(\d+\))?\.zip$")
CITY_NAMES = {"boston": "boston", "pittsburgh": "pittsburgh", "singapore": "singapore", "vegas": "las_vegas"}


def discover_archives(archives_dir: Path) -> list[tuple[str, Path]]:
    found = []
    for path in sorted(archives_dir.glob("nuplan-v1.1_train_*.zip")):
        m = ARCHIVE_PATTERN.match(path.name)
        if m and m["city"] in CITY_NAMES:
            found.append((CITY_NAMES[m["city"]], path))
    return found


def list_jobs(archives_dir: Path, out_dir: Path, cities: list[str] | None) -> tuple[list[dict], list[str]]:
    """Return one job per DB entry and any problems (unreadable archives, duplicate log names)."""
    jobs, problems, seen = [], [], {}
    for city, archive in discover_archives(archives_dir):
        if cities and city not in cities:
            continue
        try:
            with zipfile.ZipFile(archive) as z:
                infos = [i for i in z.infolist() if i.filename.lower().endswith(".db")]
        except zipfile.BadZipFile as exc:
            problems.append(f"unreadable archive {archive.name}: {exc}")
            continue
        for info in infos:
            name = Path(info.filename).name
            if name in seen:
                problems.append(f"{name} appears in both {seen[name]} and {archive.name}")
                continue
            seen[name] = archive.name
            jobs.append({"archive": str(archive), "city": city, "entry": info.filename,
                         "size": info.file_size, "out": str(out_dir / name)})
    return jobs, problems


def extract_one(job: dict) -> dict:
    out = Path(job["out"])
    tmp = out.with_name(out.name + ".partial")
    started = time.time()
    try:
        with zipfile.ZipFile(job["archive"]) as z, z.open(job["entry"]) as src, open(tmp, "wb") as dst:
            shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)  # raises BadZipFile on CRC mismatch
        if tmp.stat().st_size != job["size"]:
            raise OSError(f"size {tmp.stat().st_size} != expected {job['size']}")
        os.replace(tmp, out)
        return {"ok": True, "name": out.name, "bytes": job["size"], "seconds": round(time.time() - started, 1)}
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        return {"ok": False, "name": out.name, "archive": Path(job["archive"]).name,
                "entry": job["entry"], "error": f"{type(exc).__name__}: {exc}"}


def extract(args) -> int:
    out_dir = args.data_root / "nuplan-v1.1" / "trainval"
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*.partial"):  # left behind by an interrupted run
        stale.unlink()
    jobs, problems = list_jobs(args.archives_dir, out_dir, args.city)
    for p in problems:
        print("PROBLEM", p)
    todo = [j for j in jobs if not (Path(j["out"]).is_file() and Path(j["out"]).stat().st_size == j["size"])]
    need = sum(j["size"] for j in todo)
    free = shutil.disk_usage(out_dir).free
    print(f"{len(jobs)} DBs found, {len(jobs) - len(todo)} already extracted, {len(todo)} to extract "
          f"({need / 1e9:.1f} GB; {free / 1e9:.1f} GB free) -> {out_dir}")
    if need > free:
        print("Not enough free disk space.")
        return 2
    failures, done, written, t0 = [], 0, 0, time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for fut in as_completed([pool.submit(extract_one, j) for j in todo]):
            res = fut.result()
            done += 1
            if res["ok"]:
                written += res["bytes"]
            else:
                failures.append(res)
                print(f"  FAIL {res['name']}: {res['error']}", flush=True)
            if done % 100 == 0 or done == len(todo):
                elapsed = max(time.time() - t0, 1e-9)
                rate = written / elapsed
                eta = (need - written) / rate / 60 if rate else 0
                print(f"  {done}/{len(todo)} DBs, {written / 1e9:.1f}/{need / 1e9:.1f} GB, "
                      f"{rate / 1e6:.0f} MB/s, {len(failures)} failures, ETA {eta:.0f} min", flush=True)
    (out_dir.parent / "trainval_extraction_failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    print(f"Finished: {done - len(failures)} extracted, {len(failures)} failed. Re-run to retry failures.")
    return 1 if failures or problems else 0


def check(args) -> int:
    out_dir = args.data_root / "nuplan-v1.1" / "trainval"
    jobs, problems = list_jobs(args.archives_dir, out_dir, args.city)
    expected = {Path(j["out"]).name for j in jobs}
    logs, by_city = [], {}
    for j in jobs:
        out = Path(j["out"])
        actual = out.stat().st_size if out.is_file() else None
        if actual != j["size"]:
            detail = "missing" if actual is None else f"size {actual} != {j['size']}"
            problems.append(f"{out.name}: {detail}")
        logs.append({"log_name": out.stem, "city": j["city"], "source_archive": Path(j["archive"]).name,
                     "source_db_entry": j["entry"], "bytes": j["size"]})
        c = by_city.setdefault(j["city"], {"dbs": 0, "bytes": 0})
        c["dbs"] += 1
        c["bytes"] += j["size"]
    extra = sorted(p.name for p in out_dir.glob("*.db") if p.name not in expected) if out_dir.is_dir() else []
    if extra and not args.city:
        problems.append(f"{len(extra)} DB files not in any archive (e.g. {extra[0]})")
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "extraction_version": EXTRACTION_VERSION,
        "data_dir": out_dir.as_posix(),
        "archives": sorted({l["source_archive"] for l in logs}),
        "by_city": by_city,
        "dbs": len(logs),
        "bytes": sum(l["bytes"] for l in logs),
        "problems": problems[:50],
        "problem_count": len(problems),
        "complete": not problems,
        "logs": logs,
    }
    if not args.city:
        (out_dir.parent / "trainval_extraction.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "logs"}, indent=2))
    print(f"Status: {'COMPLETE' if not problems else 'INCOMPLETE'}")
    return 0 if not problems else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["extract", "check"])
    parser.add_argument("--archives-dir", type=Path, default=DEFAULT_ARCHIVES_DIR)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT,
                        help="Directory that contains maps/ and receives nuplan-v1.1/trainval/")
    parser.add_argument("--city", action="append", choices=sorted(set(CITY_NAMES.values())))
    parser.add_argument("--workers", type=int, default=6,
                        help="Parallel DBs; disk-bound, so more workers than ~8 rarely helps")
    args = parser.parse_args(argv)
    return extract(args) if args.command == "extract" else check(args)


if __name__ == "__main__":
    sys.exit(main())
