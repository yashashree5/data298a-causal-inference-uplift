"""Combine two completed matched runs into a long-format Parquet outcome table.

This is experiment data, not a spreadsheet workbook. Initial-condition features
are not extracted here; scenario categories remain retrospective metadata.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from tools.idm_mini_reproducer import DEFAULT_SAMPLE, load_scenarios
from tools.pdm_mini_reproducer import PLANNERS, validate_results


OUTCOMES = [
    "score", "no_ego_at_fault_collisions", "drivable_area_compliance",
    "ego_is_making_progress", "driving_direction_compliance",
    "ego_progress_along_expert_route", "time_to_collision_within_bound",
    "speed_limit_compliance", "ego_is_comfortable",
]
MATCHED_CONFIG = [
    "scenario_builder", "scenario_filter", "observation", "ego_controller",
    "simulation_time_controller", "simulation_history_buffer_duration",
    "simulation_metric", "seed", "worker", "run_metric",
]


def validate_pair(manifests, configs, environments):
    """Reject different samples, revisions, packages or evaluation settings."""
    for key in ["sample_sha256", "selected_scenarios", "scenario_count", "nuplan_commit",
                "tuplan_garage_commit", "data_root", "maps_root", "python"]:
        if manifests[0][key] != manifests[1][key]:
            raise ValueError(f"Run mismatch: {key}")
    if environments[0] != environments[1]:
        raise ValueError("Run mismatch: installed packages")
    for key in MATCHED_CONFIG:
        if configs[0][key] != configs[1][key]:
            raise ValueError(f"Run mismatch: simulation setting {key}")
    # Aggregate filenames contain a run timestamp, not a scientific parameter.
    aggregates = [{name: {key: value for key, value in config.items() if key != "file_name"}
                   for name, config in item["metric_aggregator"].items()} for item in configs]
    if aggregates[0] != aggregates[1]:
        raise ValueError("Run mismatch: metric aggregation")


def exactly_one(root, pattern):
    files = list(root.rglob(pattern))
    if len(files) != 1:
        raise ValueError(f"Expected exactly one {pattern} in {root}, found {len(files)}")
    return files[0]


def recorded_run_id(manifest):
    """Use saved provenance, not the directory name, so results can be relocated."""
    if manifest.get("run_id"):
        return manifest["run_id"]
    ids = [arg.split("=", 1)[1] for arg in manifest["command"] if arg.startswith("experiment_uid=")]
    if len(ids) != 1:
        raise ValueError("Expected one recorded experiment_uid in the run manifest")
    return ids[0]


def wait_for_runs(runs, seconds):
    """Allow assembly to follow ongoing simulations without resubmitting them."""
    if seconds < 0:
        raise ValueError("Wait duration cannot be negative")
    deadline = time.monotonic() + seconds
    while True:
        ready = 0
        for run in runs:
            try:
                result = json.loads((run / "result.json").read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                continue  # A run may still be writing its final report.
            if not result.get("valid"):
                raise ValueError(f"Run validation failed; inspect {run / 'result.json'}")
            ready += 1
        if ready == len(runs):
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Runs are not both ready; no combined table was written")
        time.sleep(min(5, remaining))


def build_table(idm_run, pdm_run, sample):
    import pandas as pd
    import yaml

    scenarios = load_scenarios(sample)
    sample_hash = hashlib.sha256(sample.read_bytes()).hexdigest()
    frames, manifests, configs, environments = [], [], [], []
    sources = {}
    for planner, run in [("idm", idm_run), ("pdm-closed", pdm_run)]:
        manifest = json.loads((run / "run_manifest.json").read_text())
        result = json.loads((run / "result.json").read_text())
        if manifest["planner"] != planner or not result["valid"]:
            raise ValueError(f"Wrong planner or invalid/incomplete run: {run}")
        if manifest["sample_sha256"] != sample_hash or manifest["selected_scenarios"] != scenarios:
            raise ValueError(f"Run does not contain the entire requested sample: {run}")
        if manifest["scenario_count"] != len(scenarios):
            raise ValueError(f"Scenario count mismatch: {run}")
        report_file = exactly_one(run, "runner_report.parquet")
        aggregate_file = exactly_one(run, "aggregator_metric/*.parquet")
        report = pd.read_parquet(report_file)
        aggregate = pd.read_parquet(aggregate_file)
        scores = aggregate[aggregate["log_name"].notna()]
        final = aggregate[aggregate["scenario"] == "final_score"]
        verified = validate_results(scenarios, report.to_dict("records"), scores.to_dict("records"),
                                    final.to_dict("records"), PLANNERS[planner][1])
        if not verified["valid"]:
            raise ValueError(f"Raw artifact validation failed: {verified['errors']}")
        for column in OUTCOMES:
            if not pd.to_numeric(scores[column], errors="coerce").between(0, 1).all():
                raise ValueError(f"Missing or invalid outcome {column}: {run}")
        context = pd.DataFrame(scenarios)[[
            "scenario_id", "log_name", "scenario_token", "scenario_type", "map_name",
            "anchor_timestamp_us", "window_start_us", "window_end_us",
        ]]
        for column in ["anchor_timestamp_us", "window_start_us", "window_end_us"]:
            context[column] = context[column].astype("int64")
        outcomes = scores[["log_name", "scenario"] + OUTCOMES].rename(columns={"scenario": "scenario_token"})
        frame = context.merge(outcomes, on=["log_name", "scenario_token"], validate="one_to_one")
        status = report[["log_name", "scenario_name", "succeeded", "error_message"]].rename(
            columns={"scenario_name": "scenario_token"})
        frame = frame.merge(status, on=["log_name", "scenario_token"], validate="one_to_one")
        frame["planner"] = planner
        run_id = recorded_run_id(manifest)
        frame["run_id"] = run_id
        frame["sample_sha256"] = sample_hash
        frame["nuplan_commit"] = manifest["nuplan_commit"]
        frame["tuplan_garage_commit"] = manifest["tuplan_garage_commit"]
        frames.append(frame)
        manifests.append(manifest)
        configs.append(yaml.safe_load(exactly_one(run, "code/hydra/config.yaml").read_text()))
        environments.append((run / "environment.txt").read_text())
        sources[planner] = {
            "run_id": run_id, "official_score": verified["official_score"],
            "report_sha256": hashlib.sha256(report_file.read_bytes()).hexdigest(),
            "aggregate_sha256": hashlib.sha256(aggregate_file.read_bytes()).hexdigest(),
        }

    validate_pair(manifests, configs, environments)
    table = pd.concat(frames, ignore_index=True).sort_values(["scenario_id", "planner"]).reset_index(drop=True)
    if len(table) != 2 * len(scenarios) or table.duplicated(["scenario_id", "planner"]).any():
        raise ValueError("Expected exactly two unique planner rows per scenario")
    if not (table.groupby("scenario_id")["planner"].nunique() == 2).all():
        raise ValueError("Incomplete planner pairs")
    return table, sources


def save_csv(table, output):
    """Save a local inspection copy without an index or overwriting an existing file."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False, mode="x", encoding="utf-8")
    return output


def experiment_metadata(table, sources, output, idm_run, pdm_run, csv_output=None):
    """Describe the setup with relative paths; run IDs stay in provenance only."""
    parent = output.resolve().parent
    manifest = json.loads((idm_run / "run_manifest.json").read_text())
    settings = dict(arg.split("=", 1) for arg in manifest["command"] if "=" in arg)
    metadata = {
        "experiment": parent.name,
        "scenario_count": int(table.scenario_id.nunique()),
        "row_count": len(table),
        "sample_sha256": table.sample_sha256.iloc[0],
        "nuplan_commit": table.nuplan_commit.iloc[0],
        "tuplan_garage_commit": table.tuplan_garage_commit.iloc[0],
        "simulation": settings["+simulation"],
        "simulation_seed": int(settings["seed"]),
        "scenario_builder": settings["scenario_builder"],
        "outcome_table": output.name,
        "csv_copy": os.path.relpath(csv_output.resolve(), parent) if csv_output else None,
        "conditions_status": "Not extracted; category and map columns are metadata, not model features.",
        "source_runs": {},
    }
    for planner, root in [("idm", idm_run), ("pdm-closed", pdm_run)]:
        metadata["source_runs"][planner] = {
            **sources[planner], "directory": os.path.relpath(root.resolve(), parent),
        }
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--idm-run", type=Path, required=True)
    parser.add_argument("--pdm-run", type=Path, required=True)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv", type=Path, help="Also save a CSV inspection copy at this path")
    parser.add_argument("--wait-seconds", type=int, default=0,
                        help="Wait up to N seconds for ongoing runs; default requires finished runs")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.csv and (args.csv.exists() or args.csv.resolve() == args.output.resolve()):
        raise ValueError("CSV output must be a new file distinct from the Parquet output")
    metadata_path = args.output.parent / "experiment.json"
    if metadata_path.exists():
        raise FileExistsError(metadata_path)
    if args.wait_seconds:
        print(f"Waiting up to {args.wait_seconds}s for both validated runs; output: {args.output}", flush=True)
    wait_for_runs([args.idm_run, args.pdm_run], args.wait_seconds)
    table, sources = build_table(args.idm_run, args.pdm_run, args.sample)
    import pyarrow as pa
    import pyarrow.parquet as pq

    # Parquet retains numeric types, identifiers and provenance in one data file.
    data = pa.Table.from_pandas(table, preserve_index=False)
    metadata = dict(data.schema.metadata or {})
    metadata[b"experiment_sources"] = json.dumps(sources).encode()
    metadata[b"conditions_status"] = b"Not extracted; category and map columns are metadata, not model features."
    data = data.replace_schema_metadata(metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        pq.write_table(data, stream)
    reread = pq.read_table(args.output)
    if not reread.equals(data):
        raise RuntimeError("Saved Parquet differs from assembled table")
    if args.csv:
        save_csv(table, args.csv)
        print(f"Saved CSV inspection copy to {args.csv}")
    metadata = experiment_metadata(table, sources, args.output, args.idm_run, args.pdm_run, args.csv)
    with metadata_path.open("x") as stream:
        json.dump(metadata, stream, indent=2)
        stream.write("\n")
    print(f"Saved {len(table)} rows / {table.scenario_id.nunique()} scenarios to {args.output}")
    print(json.dumps(sources, indent=2))


if __name__ == "__main__":
    main()
