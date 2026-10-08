"""Smoke-test and run all five planners sequentially, then assemble outcomes."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from tools._bootstrap import bootstrap
bootstrap()

from causal_planner.data.outcomes import build_table_for_runs
from causal_planner.data.scenarios import load_scenarios
from causal_planner.simulation.runners import PLANNERS, experiment_run_root


ORDER = ["idm", "pdm-closed", "pdm-hybrid", "urban-driver", "gc-pgp"]


def execute(command):
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("/artifacts"))
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--smoke-experiment", required=True)
    parser.add_argument("--dataset-split", choices=["train_pittsburgh"], required=True)
    args = parser.parse_args()
    scenarios = load_scenarios(args.sample)
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
        for planner in ORDER:
            for limit, experiment in [(1, args.smoke_experiment), (None, args.experiment)]:
                stage = f"{planner}:{'smoke' if limit else 'full'}"
                status(stage)
                command = [sys.executable, "-m", "tools.pdm_mini_reproducer",
                           "--planner", planner, "--sample", str(args.sample),
                           "--dataset-split", args.dataset_split,
                           "--output-root", str(args.output_root), "--experiment", experiment]
                if limit:
                    command += ["--limit", "1"]
                execute(command)
        stage = "assembly"
        status(stage)
        runs = {planner: experiment_run_root(args.output_root, args.experiment, planner)
                for planner in ORDER}
        table, sources = build_table_for_runs(runs, args.sample)
        parquet = root / "planner_outcomes.parquet"
        csv_path = root / "planner_outcomes.csv"
        table.to_parquet(parquet, index=False)
        table.to_csv(csv_path, index=False)
        if len(table) != len(scenarios) * len(ORDER):
            raise ValueError("Combined table has an unexpected row count")
        (root / "experiment.json").write_text(json.dumps({
            "sample": str(args.sample), "rows": len(table), "sources": sources,
        }, indent=2) + "\n")
        status("complete", expected_rows=len(scenarios) * len(ORDER))
    except Exception as error:
        status("failed", failed_stage=stage, error=str(error))
        raise


if __name__ == "__main__":
    main()
