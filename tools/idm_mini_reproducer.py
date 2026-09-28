#!/usr/bin/env python3
"""Run the nuPlan IDM planner on the committed mini scenario manifest."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLE = PROJECT_ROOT / "configs" / "scenarios" / "idm_mini_sample.csv"
DEFAULT_DEVKIT_ROOT = PROJECT_ROOT / "external" / "nuplan-devkit"
DEFAULT_OUTPUT_ROOT = Path(os.getenv("IDM_OUTPUT_ROOT", PROJECT_ROOT / "artifacts" / "idm_mini"))

REQUIRED_COLUMNS = {
    "scenario_id",
    "log_name",
    "db_file",
    "scenario_token",
    "scenario_type",
    "map_name",
}
TOKEN_PATTERN = re.compile(r"^[0-9a-f]{16}$")


def load_scenarios(sample_path: Path, limit: Optional[int] = None) -> List[Dict[str, str]]:
    """Load and validate scenario rows from the committed manifest."""
    with sample_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        available_columns = set(reader.fieldnames or [])
        missing_columns = REQUIRED_COLUMNS - available_columns
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Scenario manifest is missing required columns: {missing}")
        scenarios = list(reader)

    if not scenarios:
        raise ValueError("Scenario manifest contains no scenarios")

    seen_ids = set()
    for row in scenarios:
        scenario_id = row["scenario_id"]
        token = row["scenario_token"].lower()
        if scenario_id in seen_ids:
            raise ValueError(f"Duplicate scenario_id: {scenario_id}")
        if not TOKEN_PATTERN.fullmatch(token):
            raise ValueError(f"Invalid scenario token for {scenario_id}: {token}")
        if row["db_file"] != f"{row['log_name']}.db":
            raise ValueError(f"Database and log name do not match for {scenario_id}")
        seen_ids.add(scenario_id)
        row["scenario_token"] = token

    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        scenarios = scenarios[:limit]

    return scenarios


def hydra_list(values: Sequence[str]) -> str:
    """Format strings as a Hydra list override."""
    return "[" + ",".join(json.dumps(value) for value in values) + "]"


def build_command(
    scenarios: Sequence[Dict[str, str]],
    devkit_root: Path,
    python_executable: str,
    run_root: Path,
    experiment_uid: str,
) -> List[str]:
    """Build the pinned nuPlan closed-loop IDM simulation command."""
    scenario_tokens = [row["scenario_token"] for row in scenarios]
    log_names = sorted({row["log_name"] for row in scenarios})
    simulation_script = devkit_root / "nuplan" / "planning" / "script" / "run_simulation.py"

    return [
        python_executable,
        str(simulation_script),
        "+simulation=closed_loop_reactive_agents",
        "planner=idm_planner",
        "scenario_builder=nuplan_mini",
        "scenario_filter=nuplan_challenge_scenarios",
        f"scenario_filter.scenario_tokens={hydra_list(scenario_tokens)}",
        f"scenario_filter.log_names={hydra_list(log_names)}",
        "scenario_filter.remove_invalid_goals=true",
        "worker=sequential",
        "seed=0",
        "experiment_name=idm_mini_reproduction",
        f"experiment_uid={experiment_uid}",
        f"group={run_root}",
        "log_config=true",
        "verbose=true",
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE, help="Scenario manifest CSV")
    parser.add_argument(
        "--devkit-root",
        type=Path,
        default=Path(os.getenv("NUPLAN_DEVKIT_ROOT", DEFAULT_DEVKIT_ROOT)),
        help="Pinned nuPlan devkit checkout",
    )
    parser.add_argument(
        "--python",
        default=os.getenv("NUPLAN_PYTHON", sys.executable),
        help="Python executable containing nuPlan dependencies",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--limit", type=int, help="Run only the first N manifest rows")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the command without running it")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sample_path = args.sample.expanduser().resolve()
    devkit_root = args.devkit_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    scenarios = load_scenarios(sample_path, args.limit)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    experiment_uid = f"idm_mini_{timestamp}"
    run_root = output_root / experiment_uid
    command = build_command(scenarios, devkit_root, args.python, run_root, experiment_uid)

    print(f"Validated {len(scenarios)} scenarios from {sample_path}")
    print(f"Scenario types: {len({row['scenario_type'] for row in scenarios})}")
    print(f"Driving logs: {len({row['log_name'] for row in scenarios})}")

    if args.dry_run:
        print(shlex.join(command))
        return 0

    simulation_script = Path(command[1])
    if not simulation_script.is_file():
        raise FileNotFoundError(f"nuPlan simulation entrypoint not found: {simulation_script}")
    if not os.getenv("NUPLAN_DATA_ROOT"):
        raise RuntimeError("NUPLAN_DATA_ROOT must point to the mounted nuPlan dataset")
    if not os.getenv("NUPLAN_MAPS_ROOT"):
        raise RuntimeError("NUPLAN_MAPS_ROOT must point to the mounted nuPlan maps")

    run_root.mkdir(parents=True, exist_ok=False)
    manifest = {
        "created_utc": timestamp,
        "sample": str(sample_path),
        "scenario_count": len(scenarios),
        "scenario_types": sorted({row["scenario_type"] for row in scenarios}),
        "log_count": len({row["log_name"] for row in scenarios}),
        "command": command,
    }
    (run_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    environment = os.environ.copy()
    environment["NUPLAN_EXP_ROOT"] = str(run_root / "nuplan")
    completed = subprocess.run(command, cwd=devkit_root, env=environment, check=False)
    if completed.returncode != 0:
        print(f"IDM mini reproduction failed with exit code {completed.returncode}", file=sys.stderr)
        return completed.returncode

    print(f"IDM mini reproduction artifacts: {run_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
