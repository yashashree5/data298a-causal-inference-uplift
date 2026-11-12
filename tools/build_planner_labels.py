"""Build per-scenario planner labels and a modeling table from finished planner runs.

Assembles the long outcome table (one row per scenario_id x planner) with
causal_planner.data.outcomes.build_table_for_runs, pivots it to one label row per scenario
(score_<planner>, delta_<planner>_vs_idm, succeeded_<planner>, best_planner), joins the labels
with scenario_features.parquet, assigns log-grouped folds and writes a data report.

Any subset of planners that includes the IDM control works, so the tool runs on finished runs
(for example idm,pdm-closed) before every simulation is done. A sample scenario where any included
planner has no row, failed, or has a null score is excluded from the labels and the modeling table
and listed with one reason per failing planner in the report; its outcome rows are kept.

Outputs:
    <output-root>/<experiment>/planner_outcomes.parquet and .csv    long outcome table
    <out-dir>/planner_labels.parquet                                 one row per scenario
    <out-dir>/planner_modeling_table.parquet                         labels + features + fold
    <report-dir>/planner_labels.md and .json                         data report

Usage (from the repository root):
    python -m tools.build_planner_labels build --planners idm,pdm-closed
    python -m tools.build_planner_labels check
    python -m tools.build_planner_labels build --overwrite      # all five planners

Every input, output and report path is a flag (see --help). Exit status: 0 on success, 1 on a
failed run, coverage gap or failed check, 2 on a bad --planners value, missing input file or
column, or an existing output without --overwrite.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools._bootstrap import bootstrap  # noqa: E402

bootstrap()

from causal_planner.data.outcomes import build_table_for_runs  # noqa: E402
from causal_planner.simulation.runners import experiment_run_root  # noqa: E402

TOOL_VERSION = "planner-labels-v1"
PROVENANCE_KEY = b"planner_labels_provenance"
PROJECT_ROOT = Path(__file__).resolve().parents[1]

PLANNER_ORDER = ("idm", "pdm-closed", "pdm-hybrid", "urban-driver", "gc-pgp")
CONTROL = "idm"
TIE_TOLERANCE = 1e-9  # same convention as tools/build_paired_dataset.pair_outcome
KEYS = ["scenario_id", "log_name", "scenario_token"]
CONTEXT = KEYS + ["scenario_type", "map_name"]
LABEL_PREFIXES = ("score_", "delta_", "succeeded_")
BEST_PLANNER = "best_planner"
OUTCOME_COLUMNS = ["scenario_id", "planner", "score", "succeeded"]

DEFAULT_PROCESSED = Path("E:/Project/nuplan/processed")
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "additional_planners"
DEFAULT_EXPERIMENT = "pittsburgh_65_five"
DEFAULT_SAMPLE = PROJECT_ROOT / "configs" / "scenarios" / "pittsburgh_65_sample.csv"
DEFAULT_FEATURES = DEFAULT_PROCESSED / "scenario_features.parquet"
DEFAULT_OUT_DIR = DEFAULT_PROCESSED / "labels" / DEFAULT_EXPERIMENT
DEFAULT_REPORT_DIR = PROJECT_ROOT / "docs" / "data"
DEFAULT_SCHEMA = PROJECT_ROOT / "configs" / "features" / "first_experiment_schema.yaml"


class InputError(Exception):
    """Bad planner list, missing input file or missing column (exit status 2)."""


class CoverageError(Exception):
    """Label scenarios are not covered by the feature table (exit status 1).

    absent:     number of label scenario_id values missing from the feature table
    examples:   the first absent scenario_id values
    mismatched: number of rows whose identity columns differ between labels and features
    """

    def __init__(self, absent: int = 0, examples: list[str] | tuple[str, ...] = (),
                 mismatched: int = 0):
        self.absent = int(absent)
        self.examples = list(examples)
        self.mismatched = int(mismatched)
        parts = []
        if self.absent:
            parts.append(f"{self.absent} label scenario_id absent from the feature table "
                         f"(first: {', '.join(self.examples)})")
        if self.mismatched:
            parts.append(f"{self.mismatched} rows with mismatched log_name/scenario_token/map_name")
        super().__init__("coverage check failed; " + "; ".join(parts))


# ------------------------------------------------------------------------------------ planners --

def token(planner: str) -> str:
    """Planner_Token: 'pdm-closed' -> 'pdm_closed'."""
    return planner.replace("-", "_")


def parse_planners(value: str) -> list[str]:
    """Comma-separated planner names -> planners in PLANNER_ORDER.

    Raises InputError naming the offending value for an unknown or repeated planner, or when idm
    is missing.
    """
    names = [name.strip() for name in str(value).split(",")]
    seen: set[str] = set()
    for name in names:
        if name not in PLANNER_ORDER:
            raise InputError(f"--planners: unknown planner {name!r}; "
                             f"choose from {', '.join(PLANNER_ORDER)}")
        if name in seen:
            raise InputError(f"--planners: planner {name!r} is repeated")
        seen.add(name)
    if CONTROL not in seen:
        raise InputError(f"--planners: {value!r} must include the control planner {CONTROL!r}")
    return [planner for planner in PLANNER_ORDER if planner in seen]


def ordered(planners: list[str]) -> list[str]:
    """Planners in PLANNER_ORDER (unknown names raise InputError)."""
    unknown = [planner for planner in planners if planner not in PLANNER_ORDER]
    if unknown:
        raise InputError(f"unknown planner {unknown[0]!r}")
    return [planner for planner in PLANNER_ORDER if planner in planners]


def label_columns(planners: list[str]) -> list[str]:
    """score_<t> per planner, delta_<t>_vs_idm per non-idm planner, succeeded_<t> per planner,
    then best_planner."""
    planners = ordered(planners)
    columns = [f"score_{token(p)}" for p in planners]
    columns += [f"delta_{token(p)}_vs_{token(CONTROL)}" for p in planners if p != CONTROL]
    columns += [f"succeeded_{token(p)}" for p in planners]
    return columns + [BEST_PLANNER]


# ------------------------------------------------------------------------------------- labels --

def require_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise InputError(f"{name} lacks required column(s): {', '.join(missing)}")


def outcome_rows(outcomes: pd.DataFrame, planners: list[str]) -> dict[tuple[str, str], dict]:
    """{(scenario_id, planner): row} for the included planners; duplicates raise InputError."""
    require_columns(outcomes, OUTCOME_COLUMNS, "outcome table")
    subset = outcomes[outcomes["planner"].isin(planners)]
    duplicated = subset.duplicated(["scenario_id", "planner"])
    if duplicated.any():
        first = subset.loc[duplicated, ["scenario_id", "planner"]].iloc[0]
        raise InputError(f"outcome table has {int(duplicated.sum())} duplicate (scenario_id, planner) "
                         f"rows (first: {first['scenario_id']}, {first['planner']})")
    return {(row["scenario_id"], row["planner"]): row for row in subset.to_dict("records")}


def failure_reason(planner: str, row: dict | None) -> str | None:
    """Reason a planner row makes its scenario a Failed_Scenario, or None when it is usable."""
    if row is None:
        return f"{planner}: no row"
    succeeded = row.get("succeeded")
    if pd.isna(succeeded) or not bool(succeeded):
        message = row.get("error_message")
        if message is None or pd.isna(message) or str(message).strip() == "":
            message = "no message"
        return f"{planner}: simulation failed ({message})"
    if pd.isna(row.get("score")):
        return f"{planner}: missing score"
    return None


def find_failures(outcomes: pd.DataFrame, sample: pd.DataFrame,
                  planners: list[str]) -> dict[str, list[str]]:
    """{scenario_id: reasons} for every Failed_Scenario of the sample, one reason per failing
    planner: '<p>: no row', '<p>: simulation failed (<error_message or no message>)' or
    '<p>: missing score'."""
    planners = ordered(planners)
    require_columns(sample, ["scenario_id"], "sample")
    rows = outcome_rows(outcomes, planners)
    failures: dict[str, list[str]] = {}
    for scenario_id in sorted(sample["scenario_id"].astype(str)):
        reasons = [reason for planner in planners
                   if (reason := failure_reason(planner, rows.get((scenario_id, planner))))]
        if reasons:
            failures[scenario_id] = reasons
    return failures


def best_planner(scores: dict[str, float], planners: list[str]) -> str:
    """Token of the first planner in PLANNER_ORDER whose score >= max(scores) - TIE_TOLERANCE."""
    planners = ordered(planners)
    top = max(float(scores[p]) for p in planners)
    for planner in planners:
        if float(scores[planner]) >= top - TIE_TOLERANCE:
            return token(planner)
    raise ValueError("no planner scores")  # unreachable: the maximum always qualifies


def build_labels(outcomes: pd.DataFrame, sample: pd.DataFrame,
                 planners: list[str]) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Pivot the long outcome table to one row per non-failed sample scenario.

    Returns (labels, failures). Labels are sorted by scenario_id with columns
    CONTEXT + label_columns(planners); delta_<t>_vs_idm = score_<t> - score_idm (float64).
    The outcome table is not modified.
    """
    planners = ordered(planners)
    if CONTROL not in planners:
        raise InputError(f"planners must include the control planner {CONTROL!r}")
    require_columns(sample, CONTEXT, "sample")
    if sample["scenario_id"].duplicated().any():
        raise InputError("sample has duplicate scenario_id values")
    failures = find_failures(outcomes, sample, planners)
    rows = outcome_rows(outcomes, planners)

    context = sample[CONTEXT].copy()
    for column in CONTEXT:
        context[column] = context[column].astype(str)
    context = context[~context["scenario_id"].isin(failures)]
    context = context.sort_values("scenario_id").reset_index(drop=True)

    records = []
    for scenario_id in context["scenario_id"]:
        scores = {p: float(rows[(scenario_id, p)]["score"]) for p in planners}
        record = {f"score_{token(p)}": scores[p] for p in planners}
        for p in planners:
            if p != CONTROL:
                record[f"delta_{token(p)}_vs_{token(CONTROL)}"] = scores[p] - scores[CONTROL]
        for p in planners:
            record[f"succeeded_{token(p)}"] = bool(rows[(scenario_id, p)]["succeeded"])
        record[BEST_PLANNER] = best_planner(scores, planners)
        records.append(record)

    columns = label_columns(planners)
    values = pd.DataFrame.from_records(records, columns=columns)
    for column in columns:
        if column.startswith(("score_", "delta_")):
            values[column] = values[column].astype("float64")
        elif column.startswith("succeeded_"):
            values[column] = values[column].astype(bool)
        else:
            values[column] = values[column].astype(object)
    labels = pd.concat([context, values], axis=1)
    return labels[CONTEXT + columns], failures


# ------------------------------------------------------------------------------ modeling table --

IDENTITY = ["log_name", "scenario_token", "map_name"]  # must agree between labels and features
ABSENT_EXAMPLES = 5


def feature_columns(features_columns: list[str]) -> list[str]:
    """Feature_Columns: every feature table column except KEYS, in table order (map_name stays a
    feature-table column)."""
    return [column for column in features_columns if column not in KEYS]


def join_features(labels: pd.DataFrame, features: pd.DataFrame, split: dict) -> pd.DataFrame:
    """Inner-join labels with the feature table on scenario_id and add a log-grouped fold.

    Raises CoverageError when a label scenario_id is absent from the features (absent count and
    the first absent ids) or when log_name/scenario_token/map_name differ for the same
    scenario_id (mismatch count). The feature-side identity copies are dropped after the check,
    so map_name appears once (label side). fold = assign_log_folds(rows, split['folds'],
    split['seed'])[log_name]. Columns: label columns, the remaining Feature_Columns in table
    order, then fold (int64); rows keep the label order. Inputs are not modified.
    """
    from tools.validate_scenario_subset import assign_log_folds

    require_columns(labels, CONTEXT, "label table")
    require_columns(features, KEYS, "feature table")
    group = split.get("group", "log_name")
    if group != "log_name":
        raise InputError(f"schema split group must be 'log_name', got {group!r}")
    folds, seed = int(split.get("folds", 5)), int(split.get("seed", 0))

    feats = features.copy()
    feats["scenario_id"] = feats["scenario_id"].astype(str)
    duplicated = feats["scenario_id"].duplicated()
    if duplicated.any():
        raise InputError(f"feature table has {int(duplicated.sum())} duplicate scenario_id values "
                         f"(first: {feats.loc[duplicated, 'scenario_id'].iloc[0]})")

    label_ids = labels["scenario_id"].astype(str)
    present = set(feats["scenario_id"])
    absent = sorted(sid for sid in label_ids if sid not in present)
    if absent:
        raise CoverageError(absent=len(absent), examples=absent[:ABSENT_EXAMPLES])

    identity = [column for column in IDENTITY if column in feats.columns]
    side = feats.set_index("scenario_id").loc[label_ids.tolist(), identity].reset_index(drop=True)
    mismatched = pd.Series(False, index=side.index)
    for column in identity:
        mismatched |= (labels[column].astype(str).reset_index(drop=True)
                       != side[column].astype(str))
    if mismatched.any():
        raise CoverageError(mismatched=int(mismatched.sum()))

    extra = [column for column in feature_columns(list(feats.columns)) if column not in labels.columns]
    modeling = labels.reset_index(drop=True).merge(
        feats[["scenario_id"] + extra], on="scenario_id", how="inner", validate="one_to_one")
    rows = modeling[["scenario_id", "log_name"]].to_dict("records")
    assignment = assign_log_folds(rows, folds, seed)
    modeling["fold"] = modeling["log_name"].map(assignment).astype("int64")
    return modeling[list(labels.columns) + extra + ["fold"]]


# --------------------------------------------------------------------------------------- checks --

CHECK_EXAMPLES = 5  # ids or columns quoted per error message


def _first(values) -> str:
    values = [str(value) for value in values]
    return ", ".join(values[:CHECK_EXAMPLES])


def _check_unique_ids(labels: pd.DataFrame, modeling: pd.DataFrame) -> list[str]:
    errors = []
    for name, frame in (("label table", labels), ("modeling table", modeling)):
        ids = frame["scenario_id"].astype(str)
        duplicated = ids[ids.duplicated()]
        if len(duplicated):
            errors.append(f"{name}: {len(duplicated)} duplicate scenario_id "
                          f"(first: {_first(sorted(set(duplicated)))})")
    return errors


def _check_row_counts(outcomes: pd.DataFrame, labels: pd.DataFrame, modeling: pd.DataFrame,
                      sample: pd.DataFrame, planners: list[str], failed: int) -> list[str]:
    errors = []
    expected = len(sample) * len(planners)
    if len(outcomes) != expected:
        errors.append(f"outcome table has {len(outcomes)} rows, expected {expected} "
                      f"({len(sample)} sample scenarios x {len(planners)} planners)")
    expected = len(sample) - failed
    for name, frame in (("label table", labels), ("modeling table", modeling)):
        if len(frame) != expected:
            errors.append(f"{name} has {len(frame)} rows, expected {expected} "
                          f"({len(sample)} sample scenarios - {failed} failed)")
    return errors


def _check_outcome_completeness(outcomes: pd.DataFrame, sample: pd.DataFrame,
                                planners: list[str]) -> list[str]:
    present = set(zip(outcomes["scenario_id"].astype(str), outcomes["planner"].astype(str)))
    errors = []
    for planner in planners:
        missing = sorted(sid for sid in sample["scenario_id"].astype(str)
                         if (sid, planner) not in present)
        if missing:
            errors.append(f"{planner}: {len(missing)} sample scenario(s) without an outcome row "
                          f"(first: {_first(missing)})")
    return errors


def _check_delta_exact(labels: pd.DataFrame, modeling: pd.DataFrame,
                       planners: list[str]) -> list[str]:
    errors = []
    control = f"score_{token(CONTROL)}"
    for name, frame in (("label table", labels), ("modeling table", modeling)):
        for planner in planners:
            if planner == CONTROL:
                continue
            score, delta = f"score_{token(planner)}", f"delta_{token(planner)}_vs_{token(CONTROL)}"
            missing = [column for column in (control, score, delta) if column not in frame.columns]
            if missing:
                errors.append(f"{name}: missing column(s) {', '.join(missing)}")
                continue
            expected = frame[score].to_numpy(dtype="float64") - frame[control].to_numpy(dtype="float64")
            actual = frame[delta].to_numpy(dtype="float64")
            bad = ~(actual == expected)  # exact float equality; NaN never matches
            if bad.any():
                ids = frame["scenario_id"].astype(str).to_numpy()[bad]
                errors.append(f"{name}: {int(bad.sum())} {delta} value(s) differ from "
                              f"{score} - {control} (first: {_first(ids)})")
    return errors


def _check_score_range(labels: pd.DataFrame, modeling: pd.DataFrame,
                       planners: list[str]) -> list[str]:
    errors = []
    for name, frame in (("label table", labels), ("modeling table", modeling)):
        for planner in planners:
            column = f"score_{token(planner)}"
            if column not in frame.columns:
                errors.append(f"{name}: missing column {column}")
                continue
            values = pd.to_numeric(frame[column], errors="coerce")
            bad = ~values.between(0.0, 1.0)  # NaN is outside [0, 1]
            if bad.any():
                ids = frame.loc[bad, "scenario_id"].astype(str)
                errors.append(f"{name}: {int(bad.sum())} {column} value(s) outside [0, 1] "
                              f"(first: {_first(ids)})")
    return errors


def _check_feature_coverage(modeling: pd.DataFrame, feature_ids) -> list[str]:
    present = {str(sid) for sid in feature_ids}
    absent = sorted(sid for sid in modeling["scenario_id"].astype(str) if sid not in present)
    if absent:
        return [f"coverage error: {len(absent)} modeling scenario_id absent from the feature "
                f"table (first: {_first(absent)})"]
    return []


def _check_column_separation(labels: pd.DataFrame, modeling: pd.DataFrame, feature_cols,
                             provenance: dict, planners: list[str], schema: dict) -> list[str]:
    errors = []
    features = feature_columns(list(feature_cols))
    expected_labels = label_columns(planners)

    prov_labels = provenance.get("label_columns")
    prov_features = provenance.get("feature_columns")
    if prov_labels is None:
        errors.append("provenance lacks label_columns")
    elif list(prov_labels) != expected_labels:
        errors.append(f"provenance label_columns {list(prov_labels)} differ from the "
                      f"Label_Columns {expected_labels}")
    if prov_features is None:
        errors.append("provenance lacks feature_columns")
    elif list(prov_features) != features:
        extra = [c for c in prov_features if c not in features]
        missing = [c for c in features if c not in prov_features]
        errors.append(f"provenance feature_columns differ from the Feature_Columns "
                      f"(extra: [{_first(extra)}], missing: [{_first(missing)}])")
    if prov_labels is not None and prov_features is not None:
        shared = [c for c in prov_labels if c in set(prov_features)]
        if shared:
            errors.append(f"provenance label_columns and feature_columns share {_first(shared)}")

    actual_labels = [c for c in labels.columns if c not in CONTEXT]
    if actual_labels != expected_labels:
        errors.append(f"label table columns {actual_labels} differ from the Label_Columns "
                      f"{expected_labels}")
    expected_modeling = (list(labels.columns) + [c for c in features if c not in labels.columns]
                         + ["fold"])
    if list(modeling.columns) != expected_modeling:
        extra = [c for c in modeling.columns if c not in expected_modeling]
        missing = [c for c in expected_modeling if c not in modeling.columns]
        errors.append(f"modeling table columns differ from label columns + Feature_Columns + fold "
                      f"(extra: [{_first(extra)}], missing: [{_first(missing)}])")

    shared = [c for c in features if c in set(expected_labels)]
    if shared:
        errors.append(f"Feature_Columns share Label_Columns {_first(shared)}")
    prefixed = [c for c in features if c.startswith(LABEL_PREFIXES)]
    if prefixed:
        errors.append(f"Feature_Columns with a label prefix: {_first(prefixed)}")
    if BEST_PLANNER in features:
        errors.append(f"Feature_Columns include {BEST_PLANNER}")
    patterns = [str(p).lower() for p in schema.get("forbidden_feature_patterns") or []]
    for column in features:
        hits = [p for p in patterns if p in column.lower()]
        if hits:
            errors.append(f"feature column '{column}' matches forbidden pattern(s) {hits}")
    return errors


def _check_log_folds(modeling: pd.DataFrame) -> list[str]:
    from tools.validate_scenario_subset import check_log_split

    errors = []
    if modeling["fold"].isna().any():
        errors.append(f"{int(modeling['fold'].isna().sum())} modeling rows have no fold")
    frame = modeling[modeling["fold"].notna()]
    folds_per_log = frame.groupby(frame["log_name"].astype(str))["fold"].apply(
        lambda values: sorted({int(v) for v in values}))
    for log, folds in folds_per_log.items():
        if len(folds) > 1:
            errors.append(f"log {log} appears in folds {folds}")
    assignment = {log: folds[0] for log, folds in folds_per_log.items()}
    rows = [{"scenario_id": str(r["scenario_id"]), "log_name": str(r["log_name"])}
            for r in modeling[["scenario_id", "log_name"]].to_dict("records")]
    return errors + check_log_split(rows, assignment)


def check_tables(outcomes: pd.DataFrame, labels: pd.DataFrame, modeling: pd.DataFrame,
                 sample: pd.DataFrame, feature_ids, feature_cols, provenance: dict,
                 planners: list[str], schema: dict) -> list[tuple[str, list[str]]]:
    """Run the integrity checks; returns [(check name, errors)] in a fixed order.

    unique_ids            scenario_id unique in the label and modeling tables
    row_counts            outcomes = sample x planners; labels = modeling = sample - failed
    outcome_completeness  every sample scenario has an outcome row per included planner
    delta_exact           delta_<t>_vs_idm == score_<t> - score_idm (exact float equality)
    score_range           every score_<t> in [0, 1]
    feature_coverage      every modeling scenario_id is in feature_ids
    column_separation     provenance lists match label_columns(planners) and the Feature_Columns
                          (feature_cols minus KEYS), are disjoint, and no Feature_Column has a
                          label prefix, equals best_planner or contains a forbidden pattern
    log_folds             no log_name in more than one fold (check_log_split)

    Failed scenarios come from provenance['excluded'] when present, else find_failures.
    feature_cols may be the full feature-table column list (KEYS are dropped). Raises InputError
    for a missing required column. Inputs are not modified.
    """
    planners = ordered(list(planners))
    require_columns(outcomes, OUTCOME_COLUMNS, "outcome table")
    require_columns(labels, CONTEXT, "label table")
    require_columns(modeling, CONTEXT + ["fold"], "modeling table")
    require_columns(sample, ["scenario_id"], "sample")
    provenance = provenance or {}

    row_errors: list[str] = []
    if "excluded" in provenance:
        failed = len(provenance["excluded"] or {})
    else:
        try:
            failed = len(find_failures(outcomes, sample, planners))
        except InputError as error:  # duplicate (scenario_id, planner) rows
            row_errors.append(str(error))
            failed = 0
    row_errors = _check_row_counts(outcomes, labels, modeling, sample, planners, failed) + row_errors

    return [
        ("unique_ids", _check_unique_ids(labels, modeling)),
        ("row_counts", row_errors),
        ("outcome_completeness", _check_outcome_completeness(outcomes, sample, planners)),
        ("delta_exact", _check_delta_exact(labels, modeling, planners)),
        ("score_range", _check_score_range(labels, modeling, planners)),
        ("feature_coverage", _check_feature_coverage(modeling, feature_ids)),
        ("column_separation", _check_column_separation(labels, modeling, feature_cols,
                                                       provenance, planners, schema or {})),
        ("log_folds", _check_log_folds(modeling)),
    ]


# ------------------------------------------------------------------------------------------ I/O --

OUTCOMES_STEM = "planner_outcomes"
LABELS_FILE = "planner_labels.parquet"
MODELING_FILE = "planner_modeling_table.parquet"
REPORT_STEM = "planner_labels"


def run_roots(output_root: Path, experiment: str, planners: list[str]) -> dict[str, Path]:
    """{planner: Run_Root} from experiment_run_root, in PLANNER_ORDER. A bad experiment name
    raises InputError."""
    try:
        return {p: experiment_run_root(Path(output_root), experiment, p) for p in ordered(planners)}
    except ValueError as error:
        raise InputError(f"--experiment {experiment!r}: {error}") from error


def long_path(path: Path) -> Path:
    """Absolute path with the Windows extended-length prefix (\\\\?\\) so run directories deeper
    than MAX_PATH (260) are readable when LongPathsEnabled is off. Unchanged elsewhere."""
    if os.name != "nt":
        return Path(path)
    text = str(Path(path).resolve())
    if text.startswith("\\\\?\\"):
        return Path(text)
    if text.startswith("\\\\"):  # UNC share
        return Path("\\\\?\\UNC\\" + text[2:])
    return Path("\\\\?\\" + text)


def strip_long_prefix(value):
    """Remove the extended-length prefix from every string in a nested dict/list."""
    if isinstance(value, str):
        if value.startswith("\\\\?\\UNC\\"):
            return "\\\\" + value[8:]
        return value[4:] if value.startswith("\\\\?\\") else value
    if isinstance(value, dict):
        return {k: strip_long_prefix(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(strip_long_prefix(v) for v in value)
    return value


def experiment_root(output_root: Path, experiment: str) -> Path:
    """Parent of the Run_Roots (as in tools/run_five_planners.py)."""
    return run_roots(output_root, experiment, [CONTROL])[CONTROL].parent


def output_paths(args) -> dict[str, Path]:
    """Every file `build` writes: outcome table (parquet, csv), labels, modeling table, report."""
    root = experiment_root(args.output_root, args.experiment)
    out_dir, report_dir = Path(args.out_dir), Path(args.report_dir)
    return {
        "outcomes": root / f"{OUTCOMES_STEM}.parquet",
        "outcomes_csv": root / f"{OUTCOMES_STEM}.csv",
        "labels": out_dir / LABELS_FILE,
        "modeling": out_dir / MODELING_FILE,
        "report_md": report_dir / f"{REPORT_STEM}.md",
        "report_json": report_dir / f"{REPORT_STEM}.json",
    }


def require_file(path: Path) -> Path:
    path = Path(path)
    if not path.is_file():
        raise InputError(f"input file not found: {path}")
    return path


def write_parquet_atomic(df: pd.DataFrame, path: Path, metadata: dict) -> None:
    """Write `df` with PROVENANCE_KEY metadata to `<path>.tmp`, then os.replace it; the tmp is
    removed on failure. Column types follow tools.build_scenario_features.arrow_schema."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from tools.build_scenario_features import arrow_schema

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, schema=arrow_schema(df), preserve_index=False)
    table = table.replace_schema_metadata({**(table.schema.metadata or {}),
                                           PROVENANCE_KEY: json.dumps(metadata, default=str).encode()})
    tmp = path.with_name(path.name + ".tmp")
    try:
        pq.write_table(table, tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_text_atomic(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(text, encoding="utf-8", newline="\n")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def read_provenance(path: Path) -> dict:
    import pyarrow.parquet as pq

    raw = (pq.read_schema(path).metadata or {}).get(PROVENANCE_KEY)
    return json.loads(raw) if raw else {}


def read_sample(path: Path) -> pd.DataFrame:
    path = require_file(path)
    sample = pd.read_csv(path, dtype=str, keep_default_na=False)
    require_columns(sample, CONTEXT, f"sample {path}")
    return sample


def read_features(path: Path, ids: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """(feature rows whose scenario_id is in ids, full feature-table column list). Only the
    requested scenarios are loaded."""
    import pyarrow.parquet as pq

    path = require_file(path)
    schema = pq.read_schema(path)
    names = list(schema.names)
    missing = [column for column in KEYS if column not in names]
    if missing:
        raise InputError(f"feature table {path} lacks required column(s): {', '.join(missing)}")
    if not ids:
        return schema.empty_table().to_pandas(), names
    table = pq.read_table(path, filters=[("scenario_id", "in", list(ids))])
    return table.to_pandas(), names


# --------------------------------------------------------------------------------------- report --

def _number(value) -> float | None:
    if value is None:
        return None
    value = float(value)
    return None if math.isnan(value) else value


def build_report(outcomes: pd.DataFrame, labels: pd.DataFrame, modeling: pd.DataFrame,
                 sources: dict, failures: dict[str, list[str]], planners: list[str],
                 coverage: dict) -> dict:
    """Report dict: rows, planners, per-planner mean (over outcome rows) and official score,
    per-delta mean/median/std/share_positive (over label rows), best_planner counts, failed
    scenarios with reasons, fold sizes (rows and logs) and Feature_Table coverage."""
    planners = ordered(list(planners))
    sources = sources or {}
    scores = {}
    for planner in planners:
        rows = outcomes[outcomes["planner"] == planner]
        values = pd.to_numeric(rows["score"], errors="coerce")
        official = (sources.get(planner) or {}).get("official_score")
        scores[token(planner)] = {
            "rows": int(len(rows)),
            "mean": _number(values.mean()) if values.notna().any() else None,
            "label_mean": (_number(labels[f"score_{token(planner)}"].mean())
                           if len(labels) else None),
            "official": _number(official),
        }
    deltas = {}
    for planner in planners:
        if planner == CONTROL:
            continue
        column = f"delta_{token(planner)}_vs_{token(CONTROL)}"
        values = labels[column].astype("float64") if len(labels) else pd.Series(dtype="float64")
        deltas[column] = {
            "mean": _number(values.mean()) if len(values) else None,
            "median": _number(values.median()) if len(values) else None,
            "std": _number(values.std()) if len(values) > 1 else None,
            "share_positive": float((values > 0).mean()) if len(values) else None,
        }
    counts = labels[BEST_PLANNER].value_counts() if len(labels) else pd.Series(dtype=int)
    best = {token(p): int(counts.get(token(p), 0)) for p in planners}
    folds = {}
    if len(modeling):
        for fold, group in modeling.groupby("fold"):
            folds[str(int(fold))] = {"rows": int(len(group)),
                                     "logs": int(group["log_name"].nunique())}
    return {
        "rows": {"outcomes": int(len(outcomes)), "labels": int(len(labels)),
                 "modeling": int(len(modeling)), "sample": int(len(labels) + len(failures))},
        "planners": [token(p) for p in planners],
        "scores": scores,
        "deltas": deltas,
        "best_planner_counts": best,
        "failed": {sid: list(reasons) for sid, reasons in sorted(failures.items())},
        "fold_sizes": folds,
        "coverage": dict(coverage),
    }


def _fmt(value, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def markdown_report(report: dict) -> str:
    rows = report["rows"]
    lines = [
        "# Planner labels",
        "",
        f"Generated by `python -m tools.build_planner_labels build` "
        f"({report.get('tool_version', TOOL_VERSION)}, {report.get('generated_utc', '')}) for the "
        f"`{report.get('experiment', DEFAULT_EXPERIMENT)}` experiment. Tables stay outside Git; "
        "only this report is committed. Labels compare planners on scenarios where every included "
        "planner succeeded with a score; deltas are against the IDM control.",
        "",
        "## Tables",
        "",
        "| Table | Rows |",
        "|---|---:|",
        f"| Sample scenarios | {rows['sample']:,} |",
        f"| `{OUTCOMES_STEM}.parquet` (scenario x planner) | {rows['outcomes']:,} |",
        f"| `{LABELS_FILE}` | {rows['labels']:,} |",
        f"| `{MODELING_FILE}` | {rows['modeling']:,} |",
        "",
        "Included planners: " + ", ".join(f"`{p}`" for p in report["planners"]) + ".",
        "",
        "## Scores",
        "",
        "Mean is over all outcome rows of the planner; label mean is over label rows; official is "
        "the nuPlan aggregate score of the run.",
        "",
        "| Planner | Outcome rows | Mean score | Label mean | Official score |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, s in report["scores"].items():
        lines.append(f"| `{name}` | {s['rows']:,} | {_fmt(s['mean'])} | {_fmt(s['label_mean'])} | "
                     f"{_fmt(s['official'])} |")
    lines += ["", "## Deltas against IDM", ""]
    if report["deltas"]:
        lines += ["| Column | Mean | Median | Std | Share > 0 |", "|---|---:|---:|---:|---:|"]
        for column, d in report["deltas"].items():
            lines.append(f"| `{column}` | {_fmt(d['mean'])} | {_fmt(d['median'])} | {_fmt(d['std'])} | "
                         f"{_fmt(d['share_positive'], 3)} |")
    else:
        lines.append("None (IDM only).")
    lines += ["", "## Best planner", "",
              f"Ties within {TIE_TOLERANCE:g} go to the earliest planner in "
              + ", ".join(f"`{token(p)}`" for p in PLANNER_ORDER) + ".", "",
              "| Planner | Scenarios |", "|---|---:|"]
    for name, count in report["best_planner_counts"].items():
        lines.append(f"| `{name}` | {count:,} |")
    lines += ["", "## Excluded scenarios", ""]
    if report["failed"]:
        lines += ["| scenario_id | Reasons |", "|---|---|"]
        for sid, reasons in report["failed"].items():
            text = "; ".join(reasons).replace("|", "\\|").replace("\n", " ")
            lines.append(f"| `{sid}` | {text} |")
    else:
        lines.append("None.")
    lines += ["", "## Folds", "", "Log-grouped folds (`assign_log_folds`, group `log_name`).", "",
              "| Fold | Rows | Logs |", "|---:|---:|---:|"]
    for fold, f in report["fold_sizes"].items():
        lines.append(f"| {fold} | {f['rows']:,} | {f['logs']:,} |")
    c = report["coverage"]
    lines += ["", "## Feature coverage", "",
              f"{c.get('in_features', 0):,} of {c.get('label_ids', 0):,} label scenarios are in the "
              f"feature table ({c.get('absent', 0):,} absent); "
              f"{c.get('feature_columns', 0):,} feature columns joined."]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------------- commands --

def _error(message: str, status: int) -> int:
    print(f"error: {message}", file=sys.stderr)
    return status


def cmd_build(args) -> int:
    """Compute outcomes, labels, modeling table and report in memory; write only if all succeed."""
    from tools import validate_scenario_subset as validator

    try:
        planners = parse_planners(args.planners)
        runs = run_roots(args.output_root, args.experiment, planners)
        paths = output_paths(args)
        if not args.overwrite:
            existing = [path for path in paths.values() if path.exists()]
            if existing:
                raise InputError(f"output exists (use --overwrite): {existing[0]}")
        sample_path = require_file(args.sample)
        features_path = require_file(args.features)
        schema_path = require_file(args.schema)
        sample = read_sample(sample_path)
        schema = validator.load_schema(schema_path)
    except (InputError, validator.InputError) as error:
        return _error(str(error), 2)

    try:
        outcomes, sources = build_table_for_runs(
            {p: long_path(r) for p, r in runs.items()}, sample_path)
        sources = strip_long_prefix(sources)
    except Exception as error:  # incomplete or invalid run; nothing written
        print(f"FAIL outcome table: {type(error).__name__}: {error}; no output written",
              file=sys.stderr)
        return 1

    try:
        labels, failures = build_labels(outcomes, sample, planners)
        ids = labels["scenario_id"].astype(str).tolist()
        features, names = read_features(features_path, ids)
        split = dict(schema.get("split") or {})
        modeling = join_features(labels, features, split)
    except (InputError, validator.InputError) as error:
        return _error(str(error), 2)
    except CoverageError as error:
        print(f"FAIL feature_coverage: {error}; no output written", file=sys.stderr)
        return 1

    built_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    features_list = feature_columns(names)
    split_info = {"folds": int(split.get("folds", 5)), "seed": int(split.get("seed", 0)),
                  "group": split.get("group", "log_name")}
    hashes = {"sample": validator.sha256_file(sample_path),
              "features": validator.sha256_file(features_path),
              "schema": validator.sha256_file(schema_path)}
    provenance = {
        "tool_version": TOOL_VERSION, "built_utc": built_utc, "experiment": args.experiment,
        "planners": planners, "run_roots": {p: str(r) for p, r in runs.items()},
        "label_columns": label_columns(planners), "feature_columns": features_list,
        "sha256": hashes, "excluded": failures, "split": split_info,
    }
    outcome_provenance = {
        "tool_version": TOOL_VERSION, "built_utc": built_utc, "experiment": args.experiment,
        "planners": planners, "run_roots": provenance["run_roots"], "sources": sources,
        "sha256": {"sample": hashes["sample"]},
    }
    coverage = {"label_ids": len(ids), "in_features": int(len(modeling)),
                "absent": len(ids) - int(len(modeling)), "feature_columns": len(features_list)}
    report = {"tool_version": TOOL_VERSION, "generated_utc": built_utc,
              "experiment": args.experiment,
              **build_report(outcomes, labels, modeling, sources, failures, planners, coverage),
              "sha256": hashes, "split": split_info}
    markdown = markdown_report(report)

    write_parquet_atomic(outcomes, paths["outcomes"], outcome_provenance)
    write_text_atomic(paths["outcomes_csv"], outcomes.to_csv(index=False, lineterminator="\n"))
    print(f"Wrote {paths['outcomes']} and {paths['outcomes_csv'].name} ({len(outcomes):,} rows)")
    write_parquet_atomic(labels, paths["labels"], provenance)
    print(f"Wrote {paths['labels']} ({len(labels):,} rows, {len(labels.columns)} columns)")
    write_parquet_atomic(modeling, paths["modeling"], provenance)
    print(f"Wrote {paths['modeling']} ({len(modeling):,} rows, {len(modeling.columns)} columns)")
    write_text_atomic(paths["report_json"], json.dumps(report, indent=2) + "\n")
    write_text_atomic(paths["report_md"], markdown)
    print(f"Wrote {paths['report_md']} and {paths['report_json'].name}")
    if failures:
        print(f"Excluded {len(failures)} failed scenario(s); see the report")
    return 0


def cmd_check(args) -> int:
    """Run check_tables on the written tables; one PASS/FAIL line per check. Included planners
    come from the label-table provenance."""
    import pyarrow.parquet as pq

    from tools import validate_scenario_subset as validator

    try:
        paths = output_paths(args)
        for key in ("outcomes", "labels", "modeling"):
            require_file(paths[key])
        sample = read_sample(args.sample)
        schema = validator.load_schema(require_file(args.schema))
        outcomes = pd.read_parquet(paths["outcomes"])
        labels = pd.read_parquet(paths["labels"])
        modeling = pd.read_parquet(paths["modeling"])
        provenance = read_provenance(paths["labels"])
        if not provenance.get("planners"):
            raise InputError(f"{paths['labels']} has no {PROVENANCE_KEY.decode()} planners")
        planners = ordered(list(provenance["planners"]))
        require_columns(modeling, ["scenario_id"], "modeling table")
        ids = modeling["scenario_id"].astype(str).tolist()
        features, names = read_features(args.features, ids)
        results = check_tables(outcomes, labels, modeling, sample, features["scenario_id"].astype(str),
                               names, provenance, planners, schema)
    except (InputError, validator.InputError) as error:
        return _error(str(error), 2)

    ok = True
    for name, errors in results:
        if errors:
            ok = False
            shown = "; ".join(errors[:3]) + (f" (+{len(errors) - 3} more)" if len(errors) > 3 else "")
            print(f"FAIL {name}: {shown}")
        else:
            print(f"PASS {name}")
    return 0 if ok else 1


# ------------------------------------------------------------------------------------------ CLI --

def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT,
                        help="parent of the experiment directory (Run_Roots via experiment_run_root)")
    common.add_argument("--experiment", default=DEFAULT_EXPERIMENT,
                        help=f"experiment name; {OUTCOMES_STEM}.parquet/.csv go in its directory")
    common.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE, help="scenario sample CSV")
    common.add_argument("--features", type=Path, default=DEFAULT_FEATURES,
                        help="scenario_features.parquet")
    common.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help=f"directory of {LABELS_FILE} and {MODELING_FILE}")
    common.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR,
                        help=f"directory of {REPORT_STEM}.md and {REPORT_STEM}.json")
    common.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA, help="feature schema YAML")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", parents=[common], help="build outcomes, labels, modeling table and report")
    build.add_argument("--planners", default=",".join(PLANNER_ORDER),
                       help="comma-separated planners including idm (default: all five)")
    build.add_argument("--overwrite", action="store_true", help="replace existing outputs")
    build.set_defaults(func=cmd_build)
    sub.add_parser("check", parents=[common], help="validate the written tables").set_defaults(func=cmd_check)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
