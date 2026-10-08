"""Combine completed matched planner runs into a long-format Parquet table.

This is experiment data, not a spreadsheet workbook. Initial-condition features
are not extracted here; scenario categories remain retrospective metadata.
"""

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import time
from typing import Dict, Mapping, Optional, Sequence

from causal_planner.data.scenarios import DEFAULT_SAMPLE, load_scenarios
from causal_planner.simulation.runners import CHECKPOINT_FILES, PLANNERS, PLANNER_MODEL_CONFIGS
from causal_planner.simulation.validation import MATCHED_CONFIG, validate_pair, validate_results


OUTCOMES = [
    "score", "no_ego_at_fault_collisions", "drivable_area_compliance",
    "ego_is_making_progress", "driving_direction_compliance",
    "ego_progress_along_expert_route", "time_to_collision_within_bound",
    "speed_limit_compliance", "ego_is_comfortable",
]


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


def validate_planner_runs(planner_runs: Mapping[str, Path]) -> Dict[str, Path]:
    """Validate logical planner names and normalize run directories."""
    runs = {planner: Path(run) for planner, run in planner_runs.items()}
    unknown = sorted(set(runs) - set(PLANNERS))
    if unknown:
        raise ValueError(f"Unsupported planner(s): {', '.join(unknown)}")
    if len(runs) < 2:
        raise ValueError("At least two planner runs are required")
    return runs


def parse_planner_runs(
    values: Sequence[str], idm_run: Optional[Path] = None, pdm_run: Optional[Path] = None
) -> Dict[str, Path]:
    """Parse repeatable PLANNER=RUN arguments while retaining the legacy pair flags."""
    runs: Dict[str, Path] = {}
    if bool(idm_run) != bool(pdm_run):
        raise ValueError("--idm-run and --pdm-run must be provided together")
    if idm_run and pdm_run:
        runs.update({"idm": idm_run, "pdm-closed": pdm_run})
    for value in values:
        planner, separator, directory = value.partition("=")
        if not separator or not planner or not directory:
            raise ValueError(f"Expected PLANNER=RUN, got: {value}")
        if planner in runs:
            raise ValueError(f"Duplicate planner run: {planner}")
        runs[planner] = Path(directory)
    return validate_planner_runs(runs)


def build_table_for_runs(planner_runs: Mapping[str, Path], sample: Path):
    import pandas as pd
    import yaml

    planner_runs = validate_planner_runs(planner_runs)
    scenarios = load_scenarios(sample)
    sample_hash = hashlib.sha256(sample.read_bytes()).hexdigest()
    frames, manifests, configs, environments = [], [], [], []
    sources = {}
    for planner, run in planner_runs.items():
        manifest = json.loads((run / "run_manifest.json").read_text())
        result = json.loads((run / "result.json").read_text())
        if manifest["planner"] != planner or not result["valid"]:
            raise ValueError(f"Wrong planner or invalid/incomplete run: {run}")
        if manifest["sample_sha256"] != sample_hash or manifest["selected_scenarios"] != scenarios:
            raise ValueError(f"Run does not contain the entire requested sample: {run}")
        if manifest["scenario_count"] != len(scenarios):
            raise ValueError(f"Scenario count mismatch: {run}")
        command = manifest.get("command", [])
        if f"planner={PLANNERS[planner][0]}" not in command:
            raise ValueError(f"Run manifest does not select {planner}: {run}")
        model_config = PLANNER_MODEL_CONFIGS.get(planner)
        if model_config and f"model={model_config}" not in command:
            raise ValueError(f"Run manifest does not select the {planner} model: {run}")
        checkpoint = manifest.get("checkpoint")
        if planner in CHECKPOINT_FILES:
            if not checkpoint or checkpoint.get("filename") != CHECKPOINT_FILES[planner]:
                raise ValueError(f"Missing or unexpected checkpoint provenance: {run}")
            if not re.fullmatch(r"[0-9a-f]{64}", checkpoint.get("sha256", "")):
                raise ValueError(f"Invalid checkpoint hash: {run}")
        elif checkpoint:
            raise ValueError(f"Rule-based planner unexpectedly records a checkpoint: {run}")
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
        checkpoint = checkpoint or {}
        frame["checkpoint_sha256"] = pd.Series(
            [checkpoint.get("sha256")] * len(frame), dtype="string"
        )
        frames.append(frame)
        manifests.append(manifest)
        config = yaml.safe_load(exactly_one(run, "code/hydra/config.yaml").read_text())
        validate_model_identity(planner, config)
        configs.append(config)
        environments.append((run / "environment.txt").read_text())
        sources[planner] = {
            "run_id": run_id, "official_score": verified["official_score"],
            "report_sha256": hashlib.sha256(report_file.read_bytes()).hexdigest(),
            "aggregate_sha256": hashlib.sha256(aggregate_file.read_bytes()).hexdigest(),
            "checkpoint": checkpoint or None,
        }

    validate_pair(manifests, configs, environments)
    table = pd.concat(frames, ignore_index=True).sort_values(["scenario_id", "planner"]).reset_index(drop=True)
    planner_count = len(planner_runs)
    if len(table) != planner_count * len(scenarios) or table.duplicated(["scenario_id", "planner"]).any():
        raise ValueError(f"Expected exactly {planner_count} unique planner rows per scenario")
    if not (table.groupby("scenario_id")["planner"].nunique() == planner_count).all():
        raise ValueError("Incomplete matched planner set")
    return table, sources


def validate_model_identity(planner, config):
    """MLPlanner's report name alone cannot identify its underlying learned model."""
    targets = {
        "urban-driver": "nuplan.planning.training.modeling.models.urban_driver_open_loop_model.UrbanDriverOpenLoopModel",
        "gc-pgp": "tuplan_garage.planning.training.modeling.models.pgp.pgp_model.PGPModel",
    }
    if planner in targets and config.get("model", {}).get("_target_") != targets[planner]:
        raise ValueError(f"Resolved model configuration differs from {planner}")


def build_table(idm_run, pdm_run, sample):
    """Backward-compatible two-planner outcome builder."""
    return build_table_for_runs({"idm": idm_run, "pdm-closed": pdm_run}, sample)


def save_csv(table, output):
    """Save a local inspection copy without an index or overwriting an existing file."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False, mode="x", encoding="utf-8")
    return output


def experiment_metadata_for_runs(table, sources, output, planner_runs, csv_output=None):
    """Describe the setup with relative paths; run IDs stay in provenance only."""
    planner_runs = validate_planner_runs(planner_runs)
    parent = output.resolve().parent
    first_run = next(iter(planner_runs.values()))
    manifest = json.loads((first_run / "run_manifest.json").read_text())
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
    for planner, root in planner_runs.items():
        metadata["source_runs"][planner] = {
            **sources[planner], "directory": os.path.relpath(root.resolve(), parent),
        }
    return metadata


def experiment_metadata(table, sources, output, idm_run, pdm_run, csv_output=None):
    """Backward-compatible two-planner metadata builder."""
    return experiment_metadata_for_runs(
        table, sources, output, {"idm": idm_run, "pdm-closed": pdm_run}, csv_output
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--idm-run", type=Path, help="Legacy IDM run flag; use with --pdm-run")
    parser.add_argument("--pdm-run", type=Path, help="Legacy PDM-Closed run flag; use with --idm-run")
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="PLANNER=DIR",
        help="Matched planner run; repeat for each planner (2 or more)",
    )
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv", type=Path, help="Also save a CSV inspection copy at this path")
    parser.add_argument("--wait-seconds", type=int, default=0,
                        help="Wait up to N seconds for ongoing runs; default requires finished runs")
    args = parser.parse_args()
    try:
        planner_runs = parse_planner_runs(args.run, args.idm_run, args.pdm_run)
    except ValueError as error:
        parser.error(str(error))
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.csv and (args.csv.exists() or args.csv.resolve() == args.output.resolve()):
        raise ValueError("CSV output must be a new file distinct from the Parquet output")
    metadata_path = args.output.parent / "experiment.json"
    if metadata_path.exists():
        raise FileExistsError(metadata_path)
    if args.wait_seconds:
        print(
            f"Waiting up to {args.wait_seconds}s for {len(planner_runs)} validated runs; "
            f"output: {args.output}",
            flush=True,
        )
    wait_for_runs(list(planner_runs.values()), args.wait_seconds)
    table, sources = build_table_for_runs(planner_runs, args.sample)
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
    metadata = experiment_metadata_for_runs(table, sources, args.output, planner_runs, args.csv)
    with metadata_path.open("x") as stream:
        json.dump(metadata, stream, indent=2)
        stream.write("\n")
    print(f"Saved {len(table)} rows / {table.scenario_id.nunique()} scenarios to {args.output}")
    print(json.dumps(sources, indent=2))


if __name__ == "__main__":
    main()
