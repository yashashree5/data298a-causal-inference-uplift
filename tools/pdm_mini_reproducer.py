"""Run PDM-Closed or matched IDM on the existing mini manifest; validate results."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import re
from pathlib import Path
import shlex
import subprocess
import sys

from tools.idm_mini_reproducer import (
    DEFAULT_SAMPLE, PROJECT_ROOT, build_command as build_idm_command, load_scenarios,
)

PLANNERS = {"pdm-closed": ("pdm_closed_planner", "PDMClosedPlanner"), "idm": ("idm_planner", "IDMPlanner")}
SEARCH_PATH = (
    "hydra.searchpath=[pkg://nuplan.planning.script.config.common,"
    "pkg://nuplan.planning.script.experiments,"
    "pkg://tuplan_garage.planning.script.config.common,"
    "pkg://tuplan_garage.planning.script.config.simulation]"
)


def build_command(scenarios, devkit_root, python_executable, run_root, experiment_uid, planner="pdm-closed"):
    """Reuse the IDM settings; change only the planner and output experiment name."""
    command = build_idm_command(scenarios, devkit_root, python_executable, run_root, experiment_uid)
    command[command.index("planner=idm_planner")] = f"planner={PLANNERS[planner][0]}"
    command[command.index("experiment_name=idm_mini_reproduction")] = "experiment_name=planner_comparison"
    return command + [SEARCH_PATH]


def validate_results(scenarios, reports, scores, final_rows, planner_name):
    """Check exact log/token membership, categories, planner, failures and aggregate."""
    expected = {(row["log_name"], row["scenario_token"]): row["scenario_type"] for row in scenarios}
    errors = []
    for label, rows, token_column in [("runner", reports, "scenario_name"), ("scores", scores, "scenario")]:
        keys = [(row["log_name"], row[token_column]) for row in rows]
        missing = sorted(set(expected) - set(keys))
        extra = sorted(set(keys) - set(expected))
        duplicates = sorted(key for key, count in Counter(keys).items() if count > 1)
        if missing or extra or duplicates:
            errors.append(f"{label}: missing={missing}, unexpected={extra}, duplicates={duplicates}")
        if any(row["planner_name"] != planner_name for row in rows):
            errors.append(f"{label}: unexpected planner")

    failed = sum(row["succeeded"] != True for row in reports)
    if failed:
        errors.append(f"{failed} simulation(s) failed; inspect runner_report.parquet")
    for row in scores:
        key = (row["log_name"], row["scenario"])
        if key in expected and row["scenario_type"] != expected[key]:
            errors.append(f"Category mismatch for {key}")
        if not math.isfinite(float(row["score"])) or not 0 <= float(row["score"]) <= 1:
            errors.append(f"Invalid score for {key}")

    final_score = None
    if len(final_rows) != 1:
        errors.append("Expected exactly one final_score row")
    else:
        final = final_rows[0]
        value = float(final["score"])
        if math.isfinite(value):
            final_score = value
        if final["planner_name"] != planner_name or final["num_scenarios"] != len(scenarios):
            errors.append("Aggregate planner or scenario count differs from the request")
        mean_score = sum(float(row["score"]) for row in scores) / len(scores) if scores else float("nan")
        if not math.isfinite(value) or not math.isclose(value, mean_score, abs_tol=1e-10):
            errors.append("Official aggregate does not match the mean of per-scenario scores")

    return {
        "valid": not errors,
        "requested": len(scenarios),
        "executed": len(reports),
        "scored": len(scores),
        "failed": failed,
        "planner": planner_name,
        "official_score": final_score,
        "errors": errors,
    }


def summarize_run(run_root, scenarios, planner_name):
    """Read nuPlan outputs only after simulation; pandas is supplied by Docker."""
    import pandas as pd

    reports = list(run_root.rglob("runner_report.parquet"))
    aggregates = list(run_root.rglob("aggregator_metric/*.parquet"))
    if len(reports) != 1 or len(aggregates) != 1:
        raise RuntimeError(f"Expected one runner report and aggregate; found {len(reports)} and {len(aggregates)}")
    report = pd.read_parquet(reports[0])
    aggregate = pd.read_parquet(aggregates[0])
    scores = aggregate[aggregate["log_name"].notna()]
    final = aggregate[aggregate["scenario"] == "final_score"]
    summary = validate_results(scenarios, report.to_dict("records"), scores.to_dict("records"),
                               final.to_dict("records"), planner_name)
    summary["runner_report"] = str(reports[0].relative_to(run_root))
    summary["aggregate"] = str(aggregates[0].relative_to(run_root))
    (run_root / "result.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    scores.to_csv(run_root / "scenario_scores.csv", index=False)
    print(json.dumps(summary, indent=2))
    if not summary["valid"]:
        raise RuntimeError("Run failed result validation; see result.json")
    return summary


def checkout_revision(root):
    """Record the actual upstream commit, rejecting locally edited upstream code."""
    git = ["git", "-c", f"safe.directory={root}", "-C", str(root)]
    if subprocess.check_output(git + ["status", "--porcelain", "--untracked-files=no"], text=True).strip():
        raise RuntimeError(f"Upstream checkout has modifications: {root}")
    return subprocess.check_output(git + ["rev-parse", "HEAD"], text=True).strip()


def experiment_run_root(output_root, experiment, planner):
    """Keep each planner under a named experiment, never a timestamped outer folder."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", experiment):
        raise ValueError("Experiment name must contain only letters, numbers, underscores or hyphens")
    return output_root / experiment / planner.replace("-", "_")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--planner", choices=PLANNERS, default="pdm-closed")
    parser.add_argument("--limit", type=int, help="First N saved rows; omit to run all 68")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--experiment", help="Experiment folder name; defaults to mini_<selected count>")
    parser.add_argument("--output-root", type=Path,
                        default=Path(os.getenv("PLANNER_OUTPUT_ROOT", PROJECT_ROOT / "artifacts")))
    args = parser.parse_args()
    sample = args.sample.expanduser().resolve()
    scenarios = load_scenarios(sample, args.limit)
    keys = {(row["log_name"], row["scenario_token"]) for row in scenarios}
    if len(keys) != len(scenarios):
        raise ValueError("Duplicate log/token pairs in manifest")
    devkit = Path(os.getenv("NUPLAN_DEVKIT_ROOT", PROJECT_ROOT / "external" / "nuplan-devkit")).resolve()
    garage = Path(os.getenv("TUPLAN_GARAGE_ROOT", PROJECT_ROOT / "external" / "tuplan_garage")).resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    uid = f"{args.planner}_mini_{stamp}"
    experiment = args.experiment or f"mini_{len(scenarios)}"
    run_root = experiment_run_root(args.output_root.expanduser().resolve(), experiment, args.planner)
    command = build_command(scenarios, devkit, sys.executable, run_root, uid, args.planner)
    print(f"Validated {len(scenarios)} scenarios; planner={args.planner}", flush=True)
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return 0

    if run_root.exists():
        raise FileExistsError(f"Results already exist at {run_root}. Use a new --experiment name; nothing was overwritten.")

    for name in ["NUPLAN_DATA_ROOT", "NUPLAN_MAPS_ROOT"]:
        if not os.getenv(name) or not Path(os.environ[name]).is_dir():
            raise RuntimeError(f"Set {name} to an existing directory (use the Docker service)")
    if not Path(command[1]).is_file():
        raise FileNotFoundError(command[1])
    for row in scenarios:
        db = Path(os.environ["NUPLAN_DATA_ROOT"]) / "nuplan-v1.1" / "splits" / "mini" / row["db_file"]
        if not db.is_file():
            raise FileNotFoundError(db)
    manifest = {
        "created_utc": stamp,
        "run_id": uid,
        "experiment": experiment,
        "planner": args.planner,
        "sample": str(sample),
        "sample_sha256": hashlib.sha256(sample.read_bytes()).hexdigest(),
        "scenario_count": len(scenarios),
        "selected_scenarios": scenarios,
        "nuplan_commit": checkout_revision(devkit),
        "tuplan_garage_commit": checkout_revision(garage),
        "command": command,
        "data_root": os.environ["NUPLAN_DATA_ROOT"],
        "maps_root": os.environ["NUPLAN_MAPS_ROOT"],
        "python": sys.version,
    }
    run_root.mkdir(parents=True, exist_ok=False)
    (run_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    packages = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (run_root / "environment.txt").write_text(packages)
    environment = os.environ.copy()
    environment["NUPLAN_EXP_ROOT"] = str(run_root / "nuplan")
    print(f"Artifacts: {run_root}", flush=True)
    # nuPlan's resolved configuration is large; keep the full console output on disk.
    print(f"Simulation log: {run_root / 'console.log'}", flush=True)
    with (run_root / "console.log").open("w") as log:
        completed = subprocess.run(command, cwd=devkit, env=environment, check=False,
                                   stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode:
        print(f"Simulation exited with {completed.returncode}; inspect {run_root}", file=sys.stderr)
        return completed.returncode
    summarize_run(run_root, scenarios, PLANNERS[args.planner][1])
    print(f"Validated run artifacts: {run_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
