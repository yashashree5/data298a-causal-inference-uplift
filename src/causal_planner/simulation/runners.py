"""Construct and execute the existing IDM and matched PDM/IDM commands."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Dict, List, Sequence

from causal_planner.data.scenarios import DEFAULT_SAMPLE, PROJECT_ROOT, load_scenarios
from causal_planner.simulation.validation import summarize_run

DEFAULT_DEVKIT_ROOT = PROJECT_ROOT / "external" / "nuplan-devkit"
DEFAULT_OUTPUT_ROOT = Path(os.getenv("IDM_OUTPUT_ROOT", PROJECT_ROOT / "artifacts" / "idm_mini"))

PLANNERS = {"pdm-closed": ("pdm_closed_planner", "PDMClosedPlanner"), "idm": ("idm_planner", "IDMPlanner")}
SEARCH_PATH = (
    "hydra.searchpath=[pkg://nuplan.planning.script.config.common,"
    "pkg://nuplan.planning.script.experiments,"
    "pkg://tuplan_garage.planning.script.config.common,"
    "pkg://tuplan_garage.planning.script.config.simulation]"
)


def hydra_list(values: Sequence[str]) -> str:
    """Format strings as a Hydra list override."""
    return "[" + ",".join(json.dumps(value) for value in values) + "]"


def build_idm_command(
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


def build_pdm_command(scenarios, devkit_root, python_executable, run_root, experiment_uid, planner="pdm-closed"):
    """Reuse the IDM settings; change only the planner and output experiment name."""
    command = build_idm_command(scenarios, devkit_root, python_executable, run_root, experiment_uid)
    command[command.index("planner=idm_planner")] = f"planner={PLANNERS[planner][0]}"
    command[command.index("experiment_name=idm_mini_reproduction")] = "experiment_name=planner_comparison"
    return command + [SEARCH_PATH]

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


def parse_idm_args() -> argparse.Namespace:
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


def idm_main() -> int:
    args = parse_idm_args()
    sample_path = args.sample.expanduser().resolve()
    devkit_root = args.devkit_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    scenarios = load_scenarios(sample_path, args.limit)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    experiment_uid = f"idm_mini_{timestamp}"
    run_root = output_root / experiment_uid
    command = build_idm_command(scenarios, devkit_root, args.python, run_root, experiment_uid)

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


def pdm_main():
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
    command = build_pdm_command(scenarios, devkit, sys.executable, run_root, uid, args.planner)
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
