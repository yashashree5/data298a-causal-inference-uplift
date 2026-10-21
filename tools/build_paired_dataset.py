"""Clean, preprocess and quality-check the mini_68 planner outcomes for modeling.

Stages, all driven by configs/features/first_experiment_schema.yaml:
  1. extract     load the committed manifest, outcome table and experiment metadata
                 (the simulation outputs that the runners extracted from the nuPlan mini DBs)
  2. clean       type coercion, range checks, duplicate handling and the schema's missing-data
                 policies; a failed or incomplete scenario is excluded for BOTH planners
  3. preprocess  one row per scenario (matched pair), derived outcomes, model-feature table
                 and deterministic log-grouped folds
  4. quality     completeness, uniqueness, validity, consistency, leakage and split checks

Usage, from the repository root:
    python3 -m tools.build_paired_dataset
    python3 -m tools.build_paired_dataset --output-dir /tmp/mini_68_processed --overwrite

Outputs go to artifacts/processed/mini_68/ (ignored by Git). Existing outputs are not
overwritten unless --overwrite is given. Exit status: 0 when every quality check passes,
1 when a check fails, 2 when an input is missing.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.validate_scenario_subset import (  # noqa: E402
    DEFAULT_SCHEMA, PROJECT_ROOT, InputError, assign_log_folds, blank, check_feature_table_columns, check_leakage,
    check_log_split, load_schema, model_features, read_csv, resolve, schema_fields, sha256_file, validate,
)

PLANNERS = ("idm", "pdm-closed")
PREFIX = {"idm": "idm", "pdm-closed": "pdm"}
KEYS = ("scenario_id", "log_name", "scenario_token")
# nuPlan closed-loop score = product(multipliers) * weighted mean(weighted metrics).
MULTIPLIERS = ("no_ego_at_fault_collisions", "drivable_area_compliance",
               "driving_direction_compliance", "ego_is_making_progress")
WEIGHTED = {"ego_progress_along_expert_route": 5, "time_to_collision_within_bound": 5,
            "speed_limit_compliance": 4, "ego_is_comfortable": 2}
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "processed" / "mini_68"
OUTPUT_FILES = ("paired_outcomes.csv", "model_features.csv", "splits.csv",
                "cleaning_log.json", "quality_report.json", "quality_report.md")
PASS, FAIL, WARN = "PASS", "FAIL", "WARN"


# ------------------------------------------------------------------------------------ extract --

def extract(schema):
    """Load the committed sources and record their hashes and sizes."""
    subset = schema["subset"]
    paths = {name: resolve(subset[name]) for name in ("manifest", "outcomes_csv", "experiment_metadata")}
    for path in paths.values():
        if not path.is_file():
            raise InputError(f"Source not found: {path}")
    manifest_columns, manifest_rows = read_csv(paths["manifest"])
    outcome_columns, outcome_rows = read_csv(paths["outcomes_csv"])
    return {
        "manifest_rows": manifest_rows,
        "manifest_columns": manifest_columns,
        "outcome_rows": outcome_rows,
        "outcome_columns": outcome_columns,
        "experiment": json.loads(paths["experiment_metadata"].read_text(encoding="utf-8")),
        "sources": {str(path.relative_to(PROJECT_ROOT)): {"sha256": sha256_file(path), "bytes": path.stat().st_size}
                    for path in paths.values()},
    }


# -------------------------------------------------------------------------------------- clean --

def column_specs(schema, artifact):
    return {item["column"]: item for item in schema_fields(schema)
            if item.get("column") and artifact in (item.get("artifacts") or [])}


def parse_value(value, dtype):
    """Parse a CSV string into the schema dtype; blank -> None; invalid -> ValueError."""
    if blank(value):
        return None
    text = str(value).strip()
    if dtype == "int64":
        return int(text)
    if dtype == "float64":
        number = float(text)
        if not math.isfinite(number):
            raise ValueError(f"non-finite value {text!r}")
        return number
    if dtype == "bool":
        if text.lower() in ("true", "1"):
            return True
        if text.lower() in ("false", "0"):
            return False
        raise ValueError(f"not a boolean: {text!r}")
    return text


class CleanResult:
    def __init__(self):
        self.rows = []
        self.steps = []
        self.excluded = defaultdict(list)
        self.fatal = []

    def step(self, name, rows_affected, detail=""):
        self.steps.append({"step": name, "rows_affected": rows_affected, "detail": detail})

    def exclude(self, scenario_id, reason):
        if reason not in self.excluded[scenario_id]:
            self.excluded[scenario_id].append(reason)

    def to_json(self):
        return {"steps": self.steps,
                "excluded_scenarios": {sid: reasons for sid, reasons in sorted(self.excluded.items())},
                "fatal_errors": self.fatal, "rows_out": len(self.rows)}


def clean(manifest_rows, outcome_rows, schema, planners=PLANNERS):
    """Apply the schema's typing, range and missing-data rules to the long outcome table."""
    result = CleanResult()
    specs = column_specs(schema, "outcomes")
    manifest = {row["scenario_id"]: row for row in manifest_rows}

    undeclared = sorted({col for row in outcome_rows for col in row} - set(specs))
    invalid = Counter()
    typed = []
    for row in outcome_rows:
        out = {}
        for col, spec in specs.items():
            try:
                out[col] = parse_value(row.get(col), spec["dtype"])
            except ValueError:
                out[col] = None
                invalid[col] += 1
        typed.append(out)
    result.step("drop undeclared columns", 0, ", ".join(undeclared) or "none")
    result.step("coerce types from schema", sum(invalid.values()),
                json.dumps(dict(sorted(invalid.items()))) if invalid else "all values parsed")

    out_of_range = Counter()
    metrics = [col for col, spec in specs.items() if spec["role"] == "outcome" and spec["dtype"] == "float64"]
    for row in typed:
        for col in metrics:
            if row[col] is not None and not 0.0 <= row[col] <= 1.0:
                row[col] = None
                out_of_range[col] += 1
    result.step("set metrics outside [0, 1] to missing", sum(out_of_range.values()),
                json.dumps(dict(sorted(out_of_range.items()))) if out_of_range else "none")

    unknown = [r for r in typed if r["scenario_id"] not in manifest or r["planner"] not in planners]
    typed = [r for r in typed if r["scenario_id"] in manifest and r["planner"] in planners]
    result.step("drop rows for unknown scenarios or planners", len(unknown),
                ", ".join(sorted({f"{r['scenario_id']} [{r['planner']}]" for r in unknown})[:5]) or "none")

    groups = defaultdict(list)
    for row in typed:
        groups[(row["scenario_id"], row["planner"])].append(row)
    exact = conflicting = 0
    deduped = []
    for (sid, planner), rows in sorted(groups.items()):
        distinct = {json.dumps(r, sort_keys=True) for r in rows}
        if len(distinct) > 1:
            conflicting += len(rows)
            result.exclude(sid, f"{planner}: conflicting duplicate rows")
        exact += len(rows) - len(distinct) if len(distinct) == 1 else 0
        deduped.append(rows[0])
    conflicted = sorted(sid for sid, reasons in result.excluded.items() if any("conflicting" in r for r in reasons))
    result.step("remove exact duplicate rows", exact, f"{exact} identical copies" if exact else "none")
    result.step("conflicting duplicates (pair excluded)", conflicting, ", ".join(conflicted[:5]) or "none")

    shared = [c for c in specs if c in manifest_rows[0] and c not in ("scenario_id",)] if manifest_rows else []
    missing_by_policy = Counter()
    for row in deduped:
        sid, planner = row["scenario_id"], row["planner"]
        for col in shared:
            if row[col] is not None and str(row[col]) != manifest[sid][col]:
                result.exclude(sid, f"{planner}: {col} differs from manifest")
        if row["succeeded"] is False:
            result.exclude(sid, f"{planner}: simulation failed ({row.get('error_message') or 'no message'})")
        for col, spec in specs.items():
            if row[col] is not None:
                continue
            policy = spec["missing_policy"]
            if policy == "exclude_pair":
                result.exclude(sid, f"{planner}: missing {col}")
                missing_by_policy[policy] += 1
            elif policy == "fail_validation":
                result.fatal.append(f"{sid} [{planner}]: required {col} is missing")
                missing_by_policy[policy] += 1
    result.step("apply missing-data policy", sum(missing_by_policy.values()),
                json.dumps(dict(sorted(missing_by_policy.items()))) if missing_by_policy else "no missing values")

    present = defaultdict(set)
    for row in deduped:
        present[row["scenario_id"]].add(row["planner"])
    for sid in manifest:
        missing = [p for p in planners if p not in present.get(sid, set())]
        if missing:
            result.exclude(sid, f"no outcome row for {', '.join(missing)}")
    result.rows = sorted((r for r in deduped if r["scenario_id"] not in result.excluded),
                         key=lambda r: (r["scenario_id"], r["planner"]))
    result.step("exclude incomplete or failed pairs (both planners)", len(deduped) - len(result.rows),
                f"{len(result.excluded)} scenario(s): {', '.join(sorted(result.excluded)[:5])}"
                if result.excluded else "none")
    return result


# --------------------------------------------------------------------------------- preprocess --

def outcome_metrics(schema):
    """Float outcome columns in schema order (the primary score first)."""
    columns = [i["column"] for i in schema_fields(schema)
               if i["role"] == "outcome" and i.get("column") and i["dtype"] == "float64"]
    return ["score"] + [c for c in columns if c != "score"]


def preprocess(manifest_rows, clean_rows, schema):
    """Build one row per matched pair, the model-feature table and the log-grouped folds."""
    by_scenario = defaultdict(dict)
    for row in clean_rows:
        by_scenario[row["scenario_id"]][row["planner"]] = row
    cities = schema.get("city_by_map_name") or {}
    metrics = outcome_metrics(schema)
    paired = []
    for meta in sorted(manifest_rows, key=lambda r: r["scenario_id"]):
        runs = by_scenario.get(meta["scenario_id"])
        if not runs or set(runs) != set(PLANNERS):
            continue
        row = {key: meta[key] for key in KEYS}
        row.update(scenario_type=meta["scenario_type"], all_scenario_tags=meta["all_scenario_tags"],
                   map_name=meta["map_name"], city=cities.get(meta["map_name"]))
        for planner in PLANNERS:
            run, prefix = runs[planner], PREFIX[planner]
            for col in metrics:
                row[f"{prefix}_{col}"] = run[col]
            row[f"{prefix}_gate_failed"] = any(run[c] <= 0 for c in MULTIPLIERS)
            row[f"{prefix}_at_fault_collision"] = run["no_ego_at_fault_collisions"] < 1
        row["score_delta"] = row["pdm_score"] - row["idm_score"]
        row["pair_outcome"] = ("pdm_better" if row["score_delta"] > 1e-9 else
                               "idm_better" if row["score_delta"] < -1e-9 else "tie")
        paired.append(row)

    split = schema.get("split") or {}
    assignment = assign_log_folds(paired, int(split.get("folds", 5)), int(split.get("seed", 0))) if paired else {}
    for row in paired:
        row["fold"] = assignment[row["log_name"]]

    manifest_columns = set(manifest_rows[0]) if manifest_rows else set()
    feature_columns = [i["column"] for i in model_features(schema)
                       if i["status"] == "available" and i.get("column") in manifest_columns]
    manifest = {r["scenario_id"]: r for r in manifest_rows}
    features = [{**{k: row[k] for k in KEYS}, **{c: manifest[row["scenario_id"]][c] for c in feature_columns}}
                for row in paired]
    splits = [{"scenario_id": r["scenario_id"], "log_name": r["log_name"], "fold": r["fold"]} for r in paired]
    return paired, features, splits, assignment


# ------------------------------------------------------------------------------------ quality --

def reconstruct_score(run):
    product = math.prod(run[c] for c in MULTIPLIERS)
    return product * sum(run[c] * w for c, w in WEIGHTED.items()) / sum(WEIGHTED.values())


def assess_quality(schema, sources, cleaned, paired, features, assignment):
    """Return the quality report: named checks with status and the key dataset metrics."""
    checks = []

    def check(name, errors_or_status, detail=""):
        if isinstance(errors_or_status, list):
            status = FAIL if errors_or_status else PASS
            detail = "; ".join(errors_or_status[:5]) if errors_or_status else (detail or "ok")
        else:
            status = errors_or_status
        checks.append({"check": name, "status": status, "detail": detail})

    manifest_rows, outcome_rows = sources["manifest_rows"], sources["outcome_rows"]
    expected = schema["subset"]["expected"]
    manifest_hash = sources["sources"][schema["subset"]["manifest"]]["sha256"]

    check("cleaning: no fatal errors", cleaned.fatal)
    check("cleaning: excluded scenarios", WARN if cleaned.excluded else PASS,
          f"{len(cleaned.excluded)} excluded" + (f": {sorted(cleaned.excluded)[:5]}" if cleaned.excluded else ""))

    columns = list(paired[0]) if paired else []
    nulls = {c: sum(r[c] is None for r in paired) for c in columns if c != "city"}
    check("completeness: no nulls in the paired table", [f"{c}: {n}" for c, n in nulls.items() if n],
          f"{len(columns)} columns x {len(paired)} rows")
    check("completeness: every map has a city", [r["scenario_id"] for r in paired if r["city"] is None],
          "via city_by_map_name")

    ids = [r["scenario_id"] for r in paired]
    tokens = [r["scenario_token"] for r in paired]
    check("uniqueness: one row per scenario and token",
          [] if len(set(ids)) == len(ids) == len(set(tokens)) else ["duplicate scenario_id or scenario_token"],
          f"{len(ids)} unique scenarios")

    metrics = outcome_metrics(schema)
    bad = [f"{r['scenario_id']}: {PREFIX[p]}_{c}" for r in paired for p in PLANNERS for c in metrics
           if not 0.0 <= r[f"{PREFIX[p]}_{c}"] <= 1.0]
    check("validity: every score and metric in [0, 1]", bad, f"{len(metrics)} metrics x {len(PLANNERS)} planners")

    errors = [abs(reconstruct_score(r) - r["score"]) for r in cleaned.rows]
    max_error = max(errors, default=0.0)
    check("consistency: score = multipliers x weighted metrics", [] if max_error < 1e-9 else [f"max error {max_error:.3g}"],
          f"max abs error {max_error:.1e}")

    provenance = sorted({r["sample_sha256"] for r in cleaned.rows}, key=str)
    check("consistency: outcomes come from the frozen manifest",
          [] if provenance == [manifest_hash] else [f"sample_sha256 {provenance} != manifest {manifest_hash}"],
          manifest_hash[:16])

    means = {p: sum(r[f"{PREFIX[p]}_score"] for r in paired) / len(paired) for p in PLANNERS} if paired else {}
    official = {p: (sources["experiment"].get("source_runs") or {}).get(p, {}).get("official_score") for p in PLANNERS}
    if cleaned.excluded:
        check("consistency: planner means equal official aggregates", WARN, "not comparable after exclusions")
    else:
        check("consistency: planner means equal official aggregates",
              [f"{p}: {means[p]} != {official[p]}" for p in PLANNERS
               if official[p] is None or not math.isclose(means[p], official[p], abs_tol=1e-9)],
              ", ".join(f"{p} {means[p]:.4f}" for p in PLANNERS))

    observed = {"scenarios": len(paired), "scenario_types": len({r["scenario_type"] for r in paired}),
                "logs": len({r["log_name"] for r in paired}), "maps": len({r["map_name"] for r in paired})}
    short = [f"{k}: {observed[k]} < {expected[k]}" for k in observed if observed[k] < expected[k]]
    check("coverage: subset size after cleaning", WARN if short else PASS, "; ".join(short) or json.dumps(observed))

    check("leakage: schema keeps outcomes out of model features", check_leakage(schema))
    feature_columns = list(features[0]) if features else []
    check("leakage: model-feature table has no outcome or treatment columns",
          check_feature_table_columns(schema, feature_columns), ", ".join(feature_columns))

    check("splits: no log in more than one fold", check_log_split(paired, assignment) if paired else ["no rows"],
          f"{len(set(assignment.values()))} folds")

    statuses = Counter(i["status"] for i in schema_fields(schema))
    unavailable = [i["name"] for i in schema_fields(schema) if i["status"] == "unavailable" and i["required"]]
    check("feature availability: required pre-simulation fields extracted", WARN if unavailable else PASS,
          f"{len(unavailable)} required fields need the nuPlan DBs: {', '.join(unavailable)}" if unavailable else "all")

    fold_sizes = defaultdict(lambda: {"logs": 0, "scenarios": 0})
    for row in paired:
        fold_sizes[row["fold"]]["scenarios"] += 1
    for log, fold in assignment.items():
        fold_sizes[fold]["logs"] += 1
    metrics_summary = {
        "rows": {"outcome_rows_in": len(outcome_rows), "outcome_rows_clean": len(cleaned.rows),
                 "manifest_scenarios": len(manifest_rows), "pairs": len(paired),
                 "excluded_scenarios": len(cleaned.excluded)},
        "coverage": observed,
        "field_status": dict(sorted(statuses.items())),
        "planner_mean_score": means,
        "planner_zero_scores": {p: sum(r[f"{PREFIX[p]}_score"] <= 0 for r in paired) for p in PLANNERS},
        "mean_score_delta": (sum(r["score_delta"] for r in paired) / len(paired)) if paired else None,
        "pair_outcomes": dict(sorted(Counter(r["pair_outcome"] for r in paired).items())),
        "fold_sizes": {str(k): v for k, v in sorted(fold_sizes.items())},
        "model_features_available": [c for c in feature_columns if c not in KEYS],
    }
    passed = not any(c["status"] == FAIL for c in checks)
    return {"passed": passed, "checks": checks, "metrics": metrics_summary, "sources": sources["sources"]}


def render_markdown(report, cleaned):
    lines = ["# mini_68 data quality report", "",
             f"**Result: {'PASS' if report['passed'] else 'FAIL'}**", "",
             "## Sources", "", "| File | Bytes | SHA-256 |", "| --- | ---: | --- |"]
    lines += [f"| `{path}` | {info['bytes']:,} | `{info['sha256'][:16]}` |" for path, info in report["sources"].items()]
    lines += ["", "## Cleaning steps", "", "| Step | Rows affected | Detail |", "| --- | ---: | --- |"]
    lines += [f"| {s['step']} | {s['rows_affected']} | {s['detail']} |" for s in cleaned.steps]
    if cleaned.excluded:
        lines += ["", "Excluded scenarios:", ""]
        lines += [f"- `{sid}`: {'; '.join(reasons)}" for sid, reasons in sorted(cleaned.excluded.items())]
    lines += ["", "## Quality checks", "", "| Check | Status | Detail |", "| --- | --- | --- |"]
    lines += [f"| {c['check']} | {c['status']} | {c['detail']} |" for c in report["checks"]]
    lines += ["", "## Dataset metrics", "", "```json", json.dumps(report["metrics"], indent=2), "```", ""]
    return "\n".join(lines)


# ------------------------------------------------------------------------------------- output --

def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: "" if v is None else v for k, v in row.items()} for row in rows)


def write_outputs(output_dir, paired, features, splits, cleaned, report, overwrite=False):
    output_dir = Path(output_dir)
    existing = [name for name in OUTPUT_FILES + ("paired_outcomes.parquet",) if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(f"{output_dir} already contains {existing}; use --overwrite or another --output-dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "paired_outcomes.csv", paired)
    write_csv(output_dir / "model_features.csv", features)
    write_csv(output_dir / "splits.csv", splits)
    (output_dir / "cleaning_log.json").write_text(json.dumps(cleaned.to_json(), indent=2) + "\n", encoding="utf-8")
    (output_dir / "quality_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output_dir / "quality_report.md").write_text(render_markdown(report, cleaned), encoding="utf-8")
    written = list(OUTPUT_FILES)
    try:
        import pandas as pd
        pd.DataFrame(paired).to_parquet(output_dir / "paired_outcomes.parquet", index=False)
        written.append("paired_outcomes.parquet")
    except ImportError:
        pass
    return written


def run(schema_path=DEFAULT_SCHEMA):
    """Run the four stages in memory and return their results."""
    schema = load_schema(schema_path)
    sources = extract(schema)
    cleaned = clean(sources["manifest_rows"], sources["outcome_rows"], schema)
    paired, features, splits, assignment = preprocess(sources["manifest_rows"], cleaned.rows, schema)
    report = assess_quality(schema, sources, cleaned, paired, features, assignment)
    report["subset_validator_failures"] = [c["check"] for c in validate(schema_path).failed]
    if report["subset_validator_failures"]:
        report["passed"] = False
    return sources, cleaned, paired, features, splits, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        sources, cleaned, paired, features, splits, report = run(args.schema)
        written = write_outputs(args.output_dir, paired, features, splits, cleaned, report, args.overwrite)
    except (InputError, FileExistsError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"1. extract     {len(sources['manifest_rows'])} scenarios, {len(sources['outcome_rows'])} outcome rows")
    print(f"2. clean       {len(cleaned.rows)} rows kept, {len(cleaned.excluded)} scenarios excluded, "
          f"{len(cleaned.fatal)} fatal errors")
    print(f"3. preprocess  {len(paired)} matched pairs, {len(features[0]) - len(KEYS) if features else 0} "
          f"model feature(s) available, {len(set(r['fold'] for r in paired))} log-grouped folds")
    print("4. quality")
    for c in report["checks"]:
        print(f"   [{c['status']:4s}] {c['check']}: {c['detail']}")
    print(f"   subset validator failures: {len(report['subset_validator_failures'])}")
    print(f"Wrote {', '.join(written)} to {args.output_dir}")
    print("RESULT:", "PASS" if report["passed"] else "FAIL")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
