"""Validate the mini_68 scenario subset, matched planner outcomes and first-experiment schema.

Checks the committed manifest, the long-format outcome table, experiment metadata and the
feature schema (including leakage rules). Reads committed files only; no nuPlan data or Docker.

Usage, from the repository root:
    python3 tools/validate_scenario_subset.py
    python3 -m tools.validate_scenario_subset --json /tmp/subset_report.json
    python3 tools/validate_scenario_subset.py --features path/to/feature_table.csv

Exit status: 0 when every required check passes, 1 when any required check fails,
2 when an input file or PyYAML is missing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.idm_mini_reproducer import PROJECT_ROOT, load_scenarios  # noqa: E402

DEFAULT_SCHEMA = PROJECT_ROOT / "configs" / "features" / "first_experiment_schema.yaml"
REQUIRED_GROUPS = ("scenario_metadata", "ego_scene_pre_simulation", "planner_treatment", "post_simulation_outcomes")
FIELD_ATTRIBUTES = ("name", "column", "artifacts", "source", "dtype", "required",
                    "available_before_simulation", "model_feature", "status", "missing_policy")
FEATURE_ROLES = {"metadata", "covariate"}
# Identifiers a feature table may carry for joining and log-grouped splitting.
FEATURE_TABLE_KEYS = {"scenario_id", "log_name", "scenario_token"}
TOKEN_PATTERN = re.compile(r"^[0-9a-f]{16}$")

PASS, FAIL, WARN, INFO, SKIP = "PASS", "FAIL", "WARN", "INFO", "SKIP"


class InputError(Exception):
    """A required input file cannot be read."""


# ------------------------------------------------------------------------------------ loading --

def load_schema(path):
    try:
        import yaml
    except ImportError as exc:
        raise InputError("PyYAML is required to read the feature schema: python3 -m pip install pyyaml") from exc
    if not Path(path).is_file():
        raise InputError(f"Feature schema not found: {path}")
    try:
        schema = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise InputError(f"Feature schema is not valid YAML ({path}): {exc}") from exc
    if not isinstance(schema, dict):
        raise InputError(f"Feature schema is not a YAML mapping: {path}")
    return schema


def schema_fields(schema):
    """Flatten groups into field dicts annotated with their group and role."""
    fields = []
    for group, spec in (schema.get("groups") or {}).items():
        for item in spec.get("fields") or []:
            fields.append({**item, "group": group, "role": spec.get("role")})
    return fields


def read_csv(path):
    if not Path(path).is_file():
        raise InputError(f"CSV not found: {path}")
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stable_hash(text):
    """Same construction as the EDA sampler: first 16 hex digits of SHA-256."""
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def blank(value):
    return value is None or str(value).strip() == ""


# ------------------------------------------------------------------------------ schema checks --

def check_schema_structure(schema):
    errors = []
    vocab = schema.get("vocabularies") or {}
    allowed = {key: set(vocab.get(key) or []) for key in ("dtype", "status", "role", "missing_policy")}
    groups = schema.get("groups") or {}
    for group in REQUIRED_GROUPS:
        if group not in groups:
            errors.append(f"missing schema group '{group}'")
    for group, spec in groups.items():
        if spec.get("role") not in allowed["role"]:
            errors.append(f"group '{group}' has unknown role {spec.get('role')!r}")
    names = Counter()
    for item in schema_fields(schema):
        label = item.get("name", "<unnamed>")
        names[label] += 1
        missing = [attr for attr in FIELD_ATTRIBUTES if attr not in item]
        if missing:
            errors.append(f"field '{label}' is missing attributes {missing}")
            continue
        for attr in ("dtype", "status", "missing_policy"):
            if item[attr] not in allowed[attr]:
                errors.append(f"field '{label}' has unknown {attr} {item[attr]!r}")
        for attr in ("required", "available_before_simulation", "model_feature"):
            if not isinstance(item[attr], bool):
                errors.append(f"field '{label}' attribute {attr} must be true/false")
        if item["status"] == "available" and (not item["column"] or not item["artifacts"]):
            errors.append(f"field '{label}' is 'available' but has no committed column/artifact")
        if item["status"] != "available" and item["artifacts"]:
            errors.append(f"field '{label}' lists artifacts but status is {item['status']!r}")
    errors += [f"duplicate field name '{name}'" for name, count in names.items() if count > 1]
    return errors


def model_features(schema):
    return [item for item in schema_fields(schema) if item.get("model_feature") is True]


def check_leakage(schema):
    """Outcomes and post-simulation values must never be model inputs."""
    errors = []
    patterns = [p.lower() for p in schema.get("forbidden_feature_patterns") or []]
    for item in schema_fields(schema):
        label = item.get("name")
        if item.get("role") == "outcome":
            if item.get("available_before_simulation"):
                errors.append(f"outcome field '{label}' is marked available before simulation")
            if item.get("model_feature"):
                errors.append(f"outcome field '{label}' is marked as a model feature")
        if item.get("model_feature"):
            if not item.get("available_before_simulation"):
                errors.append(f"model feature '{label}' is not available before simulation")
            if item.get("role") not in FEATURE_ROLES:
                errors.append(f"model feature '{label}' belongs to role '{item.get('role')}'")
            for text in filter(None, (label, item.get("column"))):
                hits = [p for p in patterns if p in str(text).lower()]
                if hits:
                    errors.append(f"model feature '{label}' matches forbidden pattern(s) {hits}")
    return errors


def declared_columns(schema, artifact):
    return {item["column"] for item in schema_fields(schema)
            if item.get("column") and artifact in (item.get("artifacts") or [])}


def check_declared_columns(schema, artifact, columns):
    """Every committed column must be classified, and every declared column must exist."""
    declared = declared_columns(schema, artifact)
    errors = [f"{artifact}: column '{c}' is not declared in the schema" for c in columns if c not in declared]
    errors += [f"{artifact}: declared column '{c}' is absent" for c in sorted(declared - set(columns))]
    return errors


def check_feature_table_columns(schema, columns):
    """A model feature table may contain only join keys and declared model features."""
    by_column = {}
    for item in schema_fields(schema):
        by_column[item["name"]] = item
        if item.get("column"):
            by_column[item["column"]] = item
    allowed = FEATURE_TABLE_KEYS | {i["name"] for i in model_features(schema)} | {
        i["column"] for i in model_features(schema) if i.get("column")}
    errors = []
    for column in columns:
        base = column[:-len("_missing")] if column.endswith("_missing") else column
        if base in allowed:
            continue
        item = by_column.get(base)
        reason = f"role '{item['role']}', model_feature={item['model_feature']}" if item else "undeclared"
        errors.append(f"feature table column '{column}' is not an allowed model input ({reason})")
    return errors


# ---------------------------------------------------------------------------- manifest checks --

def find_duplicates(rows, keys):
    counts = Counter(tuple(row.get(k) for k in keys) for row in rows)
    return sorted(key for key, count in counts.items() if count > 1)


def missing_values(rows, columns):
    return {c: sum(blank(row.get(c)) for row in rows) for c in columns if any(blank(row.get(c)) for row in rows)}


def required_columns(schema, artifact):
    return [item["column"] for item in schema_fields(schema)
            if item.get("required") and item.get("status") == "available"
            and artifact in (item.get("artifacts") or [])
            and item.get("missing_policy") != "not_applicable_when_succeeded"]


def check_manifest(rows, schema):
    errors = []
    window = (schema.get("subset") or {}).get("scenario_window") or {}
    offset_us = int(float(window.get("offset_s", -3.0)) * 1_000_000)
    duration_us = int(float(window.get("duration_s", 15.0)) * 1_000_000)
    for keys in (["scenario_id"], ["scenario_token"], ["log_name", "scenario_token"]):
        dupes = find_duplicates(rows, keys)
        if dupes:
            errors.append(f"duplicate {'+'.join(keys)}: {dupes[:5]}")
    for col, count in missing_values(rows, required_columns(schema, "manifest")).items():
        errors.append(f"required column '{col}' is blank in {count} row(s)")
    windows = defaultdict(list)
    for row in rows:
        sid = row.get("scenario_id")
        if blank(row.get("log_name")) or blank(row.get("scenario_type")) or blank(row.get("scenario_token")):
            continue  # already reported as missing
        if not TOKEN_PATTERN.fullmatch(row["scenario_token"]):
            errors.append(f"{sid}: scenario_token is not 16 lowercase hex characters")
        if sid != f"{row['log_name']}:{row['scenario_token']}":
            errors.append(f"{sid}: scenario_id does not equal log_name:scenario_token")
        try:
            anchor, start, end = (int(row[c]) for c in ("anchor_timestamp_us", "window_start_us", "window_end_us"))
        except (KeyError, TypeError, ValueError):
            errors.append(f"{sid}: timestamps are not integers")
            continue
        if start != anchor + offset_us or end != start + duration_us:
            errors.append(f"{sid}: window is not [anchor{offset_us / 1e6:+.0f} s, +{duration_us / 1e6:.0f} s]")
        windows[row["log_name"]].append((start, end, sid))
    for log, spans in windows.items():
        spans.sort()
        for (s0, e0, a), (s1, e1, b) in zip(spans, spans[1:]):
            if s1 < e0:
                errors.append(f"{log}: windows of {a} and {b} overlap")
    return errors


# ----------------------------------------------------------------------------- outcome checks --

def check_pairs(manifest_rows, outcome_rows, planners):
    """Each manifest scenario must have exactly one row for every planner, and nothing else."""
    errors = []
    expected = {row["scenario_id"] for row in manifest_rows}
    observed = sorted({row.get("planner") for row in outcome_rows})
    if observed != sorted(planners):
        errors.append(f"planners {observed} != expected {sorted(planners)}")
    by_planner = defaultdict(list)
    for row in outcome_rows:
        by_planner[row.get("planner")].append(row["scenario_id"])
    for planner in planners:
        ids = by_planner.get(planner, [])
        missing, extra = sorted(expected - set(ids)), sorted(set(ids) - expected)
        if missing:
            errors.append(f"{planner}: no outcome row for {len(missing)} scenario(s), e.g. {missing[:3]}")
        if extra:
            errors.append(f"{planner}: {len(extra)} outcome row(s) for scenarios not in the manifest, e.g. {extra[:3]}")
    dupes = find_duplicates(outcome_rows, ["scenario_id", "planner"])
    if dupes:
        errors.append(f"duplicate scenario-planner rows: {dupes[:5]}")
    return errors


def check_outcome_consistency(manifest_rows, outcome_rows, schema):
    """Shared metadata must match the manifest; outcome metrics must be finite scores in [0, 1]."""
    errors = []
    shared = sorted(declared_columns(schema, "manifest") & declared_columns(schema, "outcomes"))
    manifest = {row["scenario_id"]: row for row in manifest_rows}
    for row in outcome_rows:
        ref = manifest.get(row["scenario_id"])
        if ref is None:
            continue
        for col in shared:
            if row.get(col) != ref.get(col):
                errors.append(f"{row['scenario_id']} [{row.get('planner')}]: {col} differs from manifest")
    metrics = [i["column"] for i in schema_fields(schema)
               if i.get("role") == "outcome" and i.get("column") and i.get("dtype") == "float64"]
    for row in outcome_rows:
        succeeded = row.get("succeeded") == "True"
        if row.get("succeeded") not in ("True", "False"):
            errors.append(f"{row['scenario_id']} [{row.get('planner')}]: succeeded={row.get('succeeded')!r}")
        if not succeeded:
            continue  # handled by the exclude_pair policy and reported separately
        for col in metrics:
            try:
                value = float(row[col])
            except (KeyError, TypeError, ValueError):
                errors.append(f"{row['scenario_id']} [{row.get('planner')}]: {col} missing or non-numeric")
                continue
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                errors.append(f"{row['scenario_id']} [{row.get('planner')}]: {col}={value} outside [0, 1]")
    return errors


def planner_means(outcome_rows):
    totals = defaultdict(list)
    for row in outcome_rows:
        totals[row["planner"]].append(float(row["score"]))
    return {planner: sum(values) / len(values) for planner, values in sorted(totals.items())}


# ----------------------------------------------------------------------------------- splits --

def assign_log_folds(rows, folds=5, seed=0):
    """Deterministically assign whole logs to folds, balancing scenario counts.

    Logs are ordered by size (largest first) and then by a seeded stable hash; each log goes to
    the fold with the fewest scenarios so far (lowest index on ties). Input order is irrelevant.
    """
    if folds < 2:
        raise ValueError("folds must be at least 2")
    sizes = Counter(row["log_name"] for row in {r["scenario_id"]: r for r in rows}.values())
    ordered = sorted(sizes, key=lambda log: (-sizes[log], stable_hash(f"{seed}:{log}"), log))
    load = [0] * folds
    assignment = {}
    for log in ordered:
        fold = min(range(folds), key=lambda k: (load[k], k))
        assignment[log] = fold
        load[fold] += sizes[log]
    return dict(sorted(assignment.items()))


def check_log_split(rows, assignment):
    """No log in more than one fold; both planner rows of a scenario share a fold."""
    errors = []
    folds_per_scenario = defaultdict(set)
    for row in rows:
        if row["log_name"] not in assignment:
            errors.append(f"log {row['log_name']} has no fold")
            continue
        folds_per_scenario[row["scenario_id"]].add(assignment[row["log_name"]])
    errors += [f"{sid} spans folds {sorted(f)}" for sid, f in folds_per_scenario.items() if len(f) > 1]
    for fold in sorted(set(assignment.values())):
        test_logs = {log for log, k in assignment.items() if k == fold}
        train_logs = {row["log_name"] for row in rows if assignment.get(row["log_name"]) != fold}
        if test_logs & train_logs:
            errors.append(f"fold {fold}: logs in both train and test {sorted(test_logs & train_logs)}")
    return errors


def canonical_order(rows, keys):
    return sorted(rows, key=lambda row: tuple(row.get(k) or "" for k in keys))


# ------------------------------------------------------------------------------ orchestration --

class Report:
    def __init__(self):
        self.checks = []
        self.facts = {}

    def add(self, name, errors_or_status, detail="", required=True):
        if isinstance(errors_or_status, list):
            status = FAIL if errors_or_status else PASS
            detail = detail if not errors_or_status else "; ".join(errors_or_status[:6]) + (
                f" (+{len(errors_or_status) - 6} more)" if len(errors_or_status) > 6 else "")
        else:
            status = errors_or_status
        self.checks.append({"check": name, "status": status, "detail": detail, "required": required})

    @property
    def failed(self):
        return [c for c in self.checks if c["required"] and c["status"] == FAIL]

    def print(self):
        width = max(len(c["check"]) for c in self.checks) + 2
        print("=" * 100)
        print("mini_68 scenario subset validation")
        print("=" * 100)
        for c in self.checks:
            tag = c["status"] if c["required"] or c["status"] in (INFO, SKIP) else f"{c['status']}*"
            print(f"[{tag:5s}] {c['check']:<{width}} {c['detail']}")
        print("-" * 100)
        print(f"Required failures: {len(self.failed)}   (* = informational, does not affect exit status)")
        print("RESULT:", "FAIL" if self.failed else "PASS")


def counts(rows, key):
    return dict(sorted(Counter(key(row) for row in rows).items()))


def validate(schema_path=DEFAULT_SCHEMA, features_path=None, data_root=None):
    report = Report()
    schema = load_schema(schema_path)
    subset = schema.get("subset") or {}
    expected = subset.get("expected") or {}
    manifest_path = resolve(subset["manifest"])
    outcomes_path = resolve(subset["outcomes_csv"])
    experiment_path = resolve(subset["experiment_metadata"])

    # Schema and leakage rules.
    report.add("schema structure", check_schema_structure(schema), f"{len(schema_fields(schema))} fields")
    report.add("leakage: outcomes are never model features", check_leakage(schema),
               f"{len(model_features(schema))} model features, none post-simulation")
    statuses = Counter(item["status"] for item in schema_fields(schema))
    report.add("field status counts", INFO, json.dumps(dict(sorted(statuses.items()))), required=False)
    unavailable = [i["name"] for i in schema_fields(schema) if i["status"] == "unavailable" and i["required"]]
    report.add("required fields not yet extracted", WARN if unavailable else PASS,
               f"{len(unavailable)}: {', '.join(unavailable)}" if unavailable else "none", required=False)

    # Manifest.
    manifest_columns, manifest_rows = read_csv(manifest_path)
    manifest_hash = sha256_file(manifest_path)
    report.facts["manifest_sha256"] = manifest_hash
    report.add("manifest SHA-256 matches frozen sample",
               [] if manifest_hash == subset.get("manifest_sha256") else [f"got {manifest_hash}"], manifest_hash[:16])
    try:
        load_scenarios(manifest_path)
        loader_errors = []
    except ValueError as exc:
        loader_errors = [str(exc)]
    report.add("existing runner loader accepts manifest", loader_errors, "tools.idm_mini_reproducer.load_scenarios")
    report.add("manifest columns declared in schema", check_declared_columns(schema, "manifest", manifest_columns),
               f"{len(manifest_columns)} columns")
    report.add("manifest: unique ids/tokens, windows, no overlap", check_manifest(manifest_rows, schema),
               f"{len(manifest_rows)} scenarios")
    report.add("manifest: every scenario has log_name and scenario_type",
               [f"{c} blank in {n} row(s)" for c, n in missing_values(manifest_rows, ["log_name", "scenario_type"]).items()],
               "log_name is the log key; log_token is not stored in committed artifacts")
    report.add("manifest: log_token availability", WARN,
               "unavailable in committed artifacts (read log.token from the 38 DBs at extraction)", required=False)

    cities = schema.get("city_by_map_name") or {}
    unmapped = sorted({r["map_name"] for r in manifest_rows} - set(cities))
    report.add("every map_name maps to a city", [f"unmapped: {unmapped}"] if unmapped else [], "")
    observed = {
        "scenarios": len({r["scenario_id"] for r in manifest_rows}),
        "scenario_types": len({r["scenario_type"] for r in manifest_rows}),
        "logs": len({r["log_name"] for r in manifest_rows}),
        "maps": len({r["map_name"] for r in manifest_rows}),
    }
    report.add("subset size matches definition",
               [f"{k}: {observed[k]} != {expected.get(k)}" for k in observed if observed[k] != expected.get(k)],
               json.dumps(observed))
    report.add("counts by scenario type", INFO, json.dumps(counts(manifest_rows, lambda r: r["scenario_type"])), required=False)
    report.add("counts by city", INFO, json.dumps(counts(manifest_rows, lambda r: cities.get(r["map_name"], "?"))),
               required=False)
    report.add("counts by map", INFO, json.dumps(counts(manifest_rows, lambda r: r["map_name"])), required=False)
    per_log = Counter(r["log_name"] for r in manifest_rows)
    report.facts["counts_by_log"] = dict(sorted(per_log.items()))
    logs_by_size = {f"{n} scenario(s)": logs for n, logs in sorted(Counter(per_log.values()).items())}
    report.add("counts by log", INFO, f"{len(per_log)} logs; {json.dumps(logs_by_size)} (per-log counts in --json)",
               required=False)

    # Outcomes.
    outcome_columns, outcome_rows = read_csv(outcomes_path)
    report.add("outcome columns declared in schema", check_declared_columns(schema, "outcomes", outcome_columns),
               f"{len(outcome_columns)} columns")
    planners = expected.get("planners") or []
    report.add("both planners have one row per scenario (matched pairs)",
               check_pairs(manifest_rows, outcome_rows, planners),
               f"{len(outcome_rows)} rows = {len(manifest_rows)} scenarios x {len(planners)} planners")
    report.add("duplicate scenario-planner rows",
               [f"{d}" for d in find_duplicates(outcome_rows, ["scenario_id", "planner"])], "0 duplicates")
    report.add("outcome row count", [] if len(outcome_rows) == expected.get("rows") else
               [f"{len(outcome_rows)} != {expected.get('rows')}"], str(len(outcome_rows)))
    report.add("outcome metadata matches manifest; metrics in [0, 1]",
               check_outcome_consistency(manifest_rows, outcome_rows, schema), "")
    required_missing = missing_values(outcome_rows, required_columns(schema, "outcomes"))
    report.add("no missing values in required outcome fields",
               [f"{c}: {n}" for c, n in required_missing.items()], "0 missing")
    failures = [r for r in outcome_rows if r.get("succeeded") != "True"]
    report.add("simulation success", WARN if failures else PASS,
               f"{len(outcome_rows) - len(failures)}/{len(outcome_rows)} succeeded"
               + (f"; pairs to exclude: {sorted({r['scenario_id'] for r in failures})[:5]}" if failures else ""),
               required=False)
    unexpected_errors = [r["scenario_id"] for r in outcome_rows if r.get("succeeded") == "True" and not blank(r.get("error_message"))]
    report.add("failure_reason empty when simulation succeeded", WARN if unexpected_errors else PASS,
               f"{len(unexpected_errors)} rows with a message despite success", required=False)
    provenance_errors = []
    for col, want in (("sample_sha256", manifest_hash), ("nuplan_commit", subset.get("nuplan_commit")),
                      ("tuplan_garage_commit", subset.get("tuplan_garage_commit"))):
        values = sorted({r.get(col) for r in outcome_rows})
        if values != [want]:
            provenance_errors.append(f"{col}: {values} != [{want}]")
    report.add("outcome provenance matches manifest and pinned commits", provenance_errors, "")
    report.add("canonical row order (scenario_id, planner)",
               [] if outcome_rows == canonical_order(outcome_rows, ["scenario_id", "planner"])
               else ["outcome table is not sorted by (scenario_id, planner)"], "", required=False)

    # Experiment metadata and per-planner summaries.
    if not experiment_path.is_file():
        raise InputError(f"Experiment metadata not found: {experiment_path}")
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    meta_errors = []
    for key, want in (("sample_sha256", manifest_hash), ("scenario_count", expected.get("scenarios")),
                      ("row_count", expected.get("rows")), ("simulation", subset.get("simulation")),
                      ("simulation_seed", subset.get("simulation_seed")), ("scenario_builder", subset.get("scenario_builder")),
                      ("nuplan_commit", subset.get("nuplan_commit")),
                      ("tuplan_garage_commit", subset.get("tuplan_garage_commit"))):
        if experiment.get(key) != want:
            meta_errors.append(f"{key}: {experiment.get(key)!r} != {want!r}")
    report.add("experiment.json matches subset definition", meta_errors, "seed 0, closed-loop reactive, nuplan_mini")
    means = planner_means(outcome_rows)
    score_errors = []
    for planner, mean in means.items():
        official = (experiment.get("source_runs") or {}).get(planner, {}).get("official_score")
        if official is None or not math.isclose(mean, official, abs_tol=1e-9):
            score_errors.append(f"{planner}: mean {mean} != official {official}")
        summary_dir = (experiment.get("source_runs") or {}).get(planner, {}).get("directory")
        summary_path = experiment_path.parent / summary_dir / "result.json" if summary_dir else None
        if summary_path and summary_path.is_file():
            result = json.loads(summary_path.read_text(encoding="utf-8"))
            if not (result.get("valid") and result.get("scored") == expected.get("scenarios") and result.get("failed") == 0):
                score_errors.append(f"{planner}: result.json is not a complete valid run")
        else:
            score_errors.append(f"{planner}: result.json not found")
    report.facts["overall_score"] = means
    report.add("overall_score = mean scenario_score; result.json complete", score_errors,
               ", ".join(f"{p} {m:.4f}" for p, m in means.items()))

    parquet_path = resolve(subset["outcomes_parquet"])
    try:
        import pyarrow.parquet as pq
    except ImportError:
        report.add("Parquet copy matches CSV", SKIP, "pyarrow not installed", required=False)
    else:
        table = pq.read_table(parquet_path).to_pylist()
        mismatch = [] if len(table) == len(outcome_rows) else [f"{len(table)} vs {len(outcome_rows)} rows"]
        for a, b in zip(table, outcome_rows):
            if (a["scenario_id"], a["planner"]) != (b["scenario_id"], b["planner"]) or \
                    not math.isclose(float(a["score"]), float(b["score"]), abs_tol=1e-12):
                mismatch.append(f"row {a['scenario_id']} [{a['planner']}] differs")
                break
        report.add("Parquet copy matches CSV", mismatch, "keys, order and scores identical")

    # Log-grouped splits.
    split = schema.get("split") or {}
    assignment = assign_log_folds(manifest_rows, int(split.get("folds", 5)), int(split.get("seed", 0)))
    report.add("log-grouped folds: no log in both train and test", check_log_split(outcome_rows, assignment),
               f"{split.get('folds', 5)} folds by {split.get('group', 'log_name')}")
    fold_sizes = defaultdict(lambda: {"logs": 0, "scenarios": 0})
    for log, fold in assignment.items():
        fold_sizes[fold]["logs"] += 1
        fold_sizes[fold]["scenarios"] += per_log[log]
    report.facts["log_folds"] = dict(sorted(fold_sizes.items()))
    report.add("fold sizes (logs/scenarios)", INFO,
               ", ".join(f"fold {k}: {v['logs']}/{v['scenarios']}" for k, v in sorted(fold_sizes.items())), required=False)

    # Optional feature table.
    if features_path:
        columns, _ = read_csv(features_path)
        report.add("feature table contains no outcome/treatment columns", check_feature_table_columns(schema, columns),
                   f"{len(columns)} columns in {features_path}")
    else:
        report.add("feature table leakage check", SKIP, "no pre-simulation feature table exists yet (--features)",
                   required=False)

    # Committed sizes and optional local nuPlan data.
    committed = {str(p): resolve(p).stat().st_size for p in
                 (subset["manifest"], subset["outcomes_csv"], subset["outcomes_parquet"], subset["experiment_metadata"])}
    report.facts["committed_bytes"] = committed
    report.add("committed subset artifacts (bytes)", INFO, json.dumps(committed), required=False)
    root = data_root or os.getenv("NUPLAN_DATA_ROOT") or os.getenv("NUPLAN_HOST_DATA_ROOT")
    if root:
        mini = Path(root).expanduser() / "nuplan-v1.1" / "splits" / "mini"
        files = [mini / r["db_file"] for r in manifest_rows]
        present = sorted({f for f in files if f.is_file()})
        size = sum(f.stat().st_size for f in present)
        report.facts["selected_log_db_bytes"] = size
        report.add("local nuPlan DBs for selected logs", PASS if len(present) == len(set(files)) else WARN,
                   f"{len(present)}/{len(set(files))} present, {size} bytes under {mini}", required=False)
    else:
        report.add("local nuPlan DBs for selected logs", SKIP,
                   "NUPLAN_DATA_ROOT / NUPLAN_HOST_DATA_ROOT not set; raw DB size not measured", required=False)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--features", type=Path, help="Optional pre-simulation feature table (CSV) to leakage-check")
    parser.add_argument("--data-root", help="Optional nuPlan dataset root to measure the selected log databases")
    parser.add_argument("--json", type=Path, help="Also write the checks and facts as JSON")
    args = parser.parse_args(argv)
    try:
        report = validate(args.schema, args.features, args.data_root)
    except InputError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    report.print()
    if args.json:
        args.json.write_text(json.dumps({"checks": report.checks, "facts": report.facts,
                                         "passed": not report.failed}, indent=2) + "\n", encoding="utf-8")
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
