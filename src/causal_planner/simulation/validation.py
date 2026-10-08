"""Validate planner reports and compatibility of matched experiments."""

from collections import Counter
import json
import math

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
