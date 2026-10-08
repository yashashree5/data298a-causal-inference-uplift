"""Freeze the approved seed-7 Pittsburgh-only 65-scenario cohort."""

from bisect import bisect_left
from collections import Counter, defaultdict
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sqlite3


FIELDS = ["scenario_id", "log_name", "db_file", "scenario_token",
          "anchor_timestamp_us", "scenario_type", "all_scenario_tags", "map_name",
          "source_recording", "window_start_us", "window_end_us"]


def select(catalog, quota=5):
    connection = sqlite3.connect(catalog.resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.create_function(
        "sample_rank", 1,
        lambda scenario_id: hashlib.sha256(("7:" + scenario_id).encode()).hexdigest()[:16],
    )
    counts = dict(connection.execute("""SELECT scenario_type, COUNT(*) FROM candidates
        WHERE overlaps_mini_recording=0 GROUP BY scenario_type"""))
    occupied, selected = defaultdict(list), []
    for kind in sorted(counts, key=lambda value: (counts[value], value)):
        candidates = connection.execute("""SELECT * FROM candidates
            WHERE scenario_type=? AND overlaps_mini_recording=0
            ORDER BY sample_rank(scenario_id), scenario_id""", (kind,))
        accepted = 0
        for candidate in candidates:
            row = dict(candidate)
            intervals = occupied[row["source_recording"]]
            interval = (row["window_start_us"], row["window_end_us"])
            index = bisect_left(intervals, interval)
            if index and intervals[index - 1][1] > interval[0]:
                continue
            if index < len(intervals) and intervals[index][0] < interval[1]:
                continue
            intervals.insert(index, interval)
            selected.append(row)
            accepted += 1
            if accepted == quota:
                break
        if accepted != quota:
            raise ValueError(f"Only selected {accepted}/{quota} for {kind}")
    connection.close()
    category_counts = Counter(row["scenario_type"] for row in selected)
    if len(category_counts) != 13 or set(category_counts.values()) != {quota}:
        raise ValueError(f"Expected 5 each across 13 categories, got {category_counts}")
    return sorted(selected, key=lambda row: (row["scenario_type"], row["scenario_id"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.provenance.exists():
        raise FileExistsError("Frozen cohort or provenance already exists; nothing overwritten")
    rows = select(args.catalog)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({field: row[field] for field in FIELDS} for row in rows)
    provenance = {
        "scenario_count": len(rows),
        "category_counts": dict(Counter(row["scenario_type"] for row in rows)),
        "source_recordings": len({row["source_recording"] for row in rows}),
        "seed": 7,
        "quota_per_available_category": 5,
        "excluded_mini_source_recordings": True,
        "overlap_group": "source_recording",
        "sample_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "catalog_sha256": hashlib.sha256(args.catalog.read_bytes()).hexdigest(),
    }
    args.provenance.write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
