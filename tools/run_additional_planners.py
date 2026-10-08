"""Run new planners sequentially, then assemble five complete matched runs.

Run in a detached, resource-limited container. Reference IDM/PDM outputs must
be mounted read-only; all writes go to the new output root. Failed stages stop
the batch and are recorded in batch_status.json.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from tools._bootstrap import bootstrap

bootstrap()

from causal_planner.data.scenarios import DEFAULT_SAMPLE, load_scenarios
from causal_planner.data.outcomes import wait_for_runs
from causal_planner.simulation.runners import experiment_run_root

PLANNERS = ["pdm-hybrid", "urban-driver", "gc-pgp"]


def execute(command):
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("/artifacts"))
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--smoke-experiment", required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    args = parser.parse_args()
    # Validate names before constructing directories.
    experiment_run_root(args.output_root, args.experiment, "idm")
    experiment_run_root(args.output_root, args.smoke_experiment, "idm")
    root = args.output_root / args.experiment
    root.mkdir(parents=True, exist_ok=False)
    status_path = root / "batch_status.json"

    def status(stage, **fields):
        record = dict(stage=stage, updated_utc=datetime.now(timezone.utc).isoformat(), **fields)
        temporary = status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.replace(status_path)
        print(json.dumps(record), flush=True)

    stage = "starting"
    try:
        stage = "waiting_for_baselines"
        status(stage)
        wait_for_runs([args.reference_root / 'idm', args.reference_root / 'pdm_closed'], 86400)
        for planner in PLANNERS:
            for limit, experiment in [(1, args.smoke_experiment), (None, args.experiment)]:
                stage = f"{planner}:{'smoke' if limit else 'full'}"
                status(stage)
                command = [sys.executable, "-m", "tools.pdm_mini_reproducer",
                           "--planner", planner, "--sample", str(args.sample),
                           "--output-root", str(args.output_root), "--experiment", experiment]
                if limit:
                    command += ["--limit", str(limit)]
                execute(command)
        stage = "assembly"
        status(stage)
        command = [sys.executable, "-m", "tools.build_planner_outcomes",
                   "--sample", str(args.sample), "--wait-seconds", "86400",
                   "--run", f"idm={args.reference_root / 'idm'}",
                   "--run", f"pdm-closed={args.reference_root / 'pdm_closed'}"]
        for planner in PLANNERS:
            command += ["--run", f"{planner}={experiment_run_root(args.output_root, args.experiment, planner)}"]
        command += ["--output", str(root / "planner_outcomes.parquet"),
                    "--csv", str(root / "planner_outcomes.csv")]
        execute(command)
        status("complete", expected_rows=len(load_scenarios(args.sample)) * 5)
    except Exception as error:
        status("failed", failed_stage=stage, error=str(error))
        raise


if __name__ == "__main__":
    main()
