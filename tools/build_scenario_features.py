"""Integrate per-scenario ego/agent, weather and map features into one model-input table.

Joins scenario_ego_agent*.parquet, scenario_weather*.parquet and scenario_map*.parquet on
scenario_id, adds map_name from the candidate list, and keeps only the model features declared in
configs/features/first_experiment_schema.yaml (selected by `column` or `name`). Every
null_with_indicator feature gets a <column>_missing flag, copied from the source table when present
and otherwise derived from the column's nullness. Location columns (ego_x, ego_y, ego_heading),
map identifiers and any undeclared source column are dropped.

Outputs (defaults under E:\\Project\\nuplan\\processed\\):
    scenario_features.parquet           benchmark candidates (scenario_eligibility.parquet)
    scenario_features_mini_68.parquet   the 68 scenarios of configs/scenarios/idm_mini_sample.csv

Usage (from the repository root):
    python -m tools.build_scenario_features build     # integrate, write tables and report
    python -m tools.build_scenario_features check     # validate the written tables
    python -m tools.build_scenario_features build --only mini --out-dir D:/tmp --report-dir D:/tmp

Every input, output and report path is a flag (see --help). Exit status: 0 on success, 1 on a
coverage gap, duplicate scenario_id or failed check, 2 on a missing input file or column.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import validate_scenario_subset as validator  # noqa: E402
from tools.validate_scenario_subset import (  # noqa: E402
    check_feature_table_columns,
    check_leakage,
    declared_columns,
    model_features,
)

TOOL_VERSION = "scenario-features-v1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]

KEYS = ["scenario_id", "log_name", "scenario_token"]
LOCATION_COLUMNS = {"ego_x", "ego_y", "ego_heading"}
MAP_NAME = "map_name"
MISSING_SUFFIX = "_missing"
INDICATOR_POLICY = "null_with_indicator"

DEFAULT_PROCESSED = Path("E:/Project/nuplan/processed")
DEFAULT_EGO = DEFAULT_PROCESSED / "features" / "scenario_ego_agent.parquet"
DEFAULT_EGO_MINI = DEFAULT_PROCESSED / "features" / "scenario_ego_agent_mini_68.parquet"
DEFAULT_WEATHER = DEFAULT_PROCESSED / "weather" / "scenario_weather.parquet"
DEFAULT_WEATHER_MINI = DEFAULT_PROCESSED / "weather" / "scenario_weather_mini_68.parquet"
DEFAULT_MAP = DEFAULT_PROCESSED / "features" / "scenario_map.parquet"
DEFAULT_MAP_MINI = DEFAULT_PROCESSED / "features" / "scenario_map_mini_68.parquet"
DEFAULT_ELIGIBILITY = DEFAULT_PROCESSED / "scenario_eligibility.parquet"
DEFAULT_OUT_DIR = DEFAULT_PROCESSED
DEFAULT_MINI_MANIFEST = PROJECT_ROOT / "configs" / "scenarios" / "idm_mini_sample.csv"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "docs" / "data"
DEFAULT_SCHEMA = PROJECT_ROOT / "configs" / "features" / "first_experiment_schema.yaml"


class InputError(Exception):
    """An input file is missing or lacks a required column (exit status 2)."""


class CoverageError(Exception):
    """Candidates are not covered exactly once by every source table (exit status 1).

    absent:     {table: number of candidate scenario_id values missing from the table}
    duplicates: {table: number of duplicated scenario_id rows}; "candidates" for the candidate list
    """

    def __init__(self, absent: dict[str, int], duplicates: dict[str, int]):
        self.absent = dict(absent)
        self.duplicates = dict(duplicates)
        parts = [f"{table}: {count} candidate scenario_id absent" for table, count in self.absent.items()]
        parts += [f"{table}: {count} duplicate scenario_id" for table, count in self.duplicates.items()]
        super().__init__("coverage check failed; " + "; ".join(parts))


# ----------------------------------------------------------------------------- column selection --

def feature_column(field: dict) -> str:
    """Column name of a Schema field: `column` when set, else `name`."""
    return field.get("column") or field["name"]


def missing_column(column: str) -> str:
    return column + MISSING_SUFFIX


def select_feature_columns(schema: dict, available: set[str]) -> tuple[list[dict], list[str]]:
    """Model-feature fields (column or name) present in `available`, in Schema order, excluding
    map_name; plus names of model features not found (reported as not integrated).

    Returned field dicts are copies with `column` set to the resolved column name."""
    selected, not_integrated = [], []
    for field in model_features(schema):
        column = feature_column(field)
        if column == MAP_NAME or field["name"] == MAP_NAME:
            continue
        if column in LOCATION_COLUMNS or column in KEYS:
            continue
        if column in available:
            selected.append({**field, "column": column})
        else:
            not_integrated.append(field["name"])
    return selected, not_integrated


# -------------------------------------------------------------------------------- integration --

def duplicate_count(frame: pd.DataFrame) -> int:
    return int(frame["scenario_id"].duplicated().sum())


def check_coverage(candidates: pd.DataFrame, sources: dict[str, pd.DataFrame]) -> None:
    """Raise CoverageError when any candidate is absent from a source or any id is duplicated."""
    duplicates, absent = {}, {}
    if (count := duplicate_count(candidates)):
        duplicates["candidates"] = count
    wanted = pd.Index(candidates["scenario_id"].unique())
    for table, frame in sources.items():
        if (count := duplicate_count(frame)):
            duplicates[table] = count
        missing = int((~wanted.isin(frame["scenario_id"])).sum())
        if missing:
            absent[table] = missing
    if absent or duplicates:
        raise CoverageError(absent, duplicates)


def integrate(candidates: pd.DataFrame, sources: dict[str, pd.DataFrame], schema: dict) -> pd.DataFrame:
    """candidates: scenario_id, map_name (+ log_name, scenario_token if present).
    Raises CoverageError listing per-table absent counts and duplicate counts.
    Output: KEYS + map_name + features + <col>_missing for null_with_indicator features
    (copied from source when present, else derived as isna). Sorted by scenario_id."""
    for column in ("scenario_id", MAP_NAME):
        if column not in candidates.columns:
            raise InputError(f"candidates lack required column '{column}'")
    for table, frame in sources.items():
        if "scenario_id" not in frame.columns:
            raise InputError(f"{table} lacks required column 'scenario_id'")
    check_coverage(candidates, sources)

    available = set().union(*(set(frame.columns) for frame in sources.values())) if sources else set()
    selected, _ = select_feature_columns(schema, available)

    key_columns = [c for c in KEYS if c in candidates.columns]
    out = candidates[key_columns + [MAP_NAME]].copy()
    for key in KEYS:
        if key in out.columns:
            continue
        owner = next((t for t, f in sources.items() if key in f.columns), None)
        if owner is None:
            raise InputError(f"no candidate or source table provides key column '{key}'")
        out = out.merge(sources[owner][["scenario_id", key]], on="scenario_id", how="left", validate="one_to_one")

    feature_columns, indicator_columns = [], []
    for table, frame in sources.items():
        taken = set(feature_columns)
        columns = [f["column"] for f in selected if f["column"] in frame.columns and f["column"] not in taken]
        indicators = [missing_column(f["column"]) for f in selected
                      if f["column"] in columns and f.get("missing_policy") == INDICATOR_POLICY
                      and missing_column(f["column"]) in frame.columns]
        if not columns:
            continue
        out = out.merge(frame[["scenario_id"] + columns + indicators], on="scenario_id",
                        how="left", validate="one_to_one")
        feature_columns += columns

    ordered_features = [f["column"] for f in selected]
    for field in selected:
        if field.get("missing_policy") != INDICATOR_POLICY:
            continue
        indicator = missing_column(field["column"])
        if indicator in out.columns:
            out[indicator] = out[indicator].astype(bool)
        else:
            out[indicator] = out[field["column"]].isna()
        indicator_columns.append(indicator)

    out = out[KEYS + [MAP_NAME] + ordered_features + indicator_columns]
    return out.sort_values("scenario_id", kind="mergesort").reset_index(drop=True)


# --------------------------------------------------------------------------------- table check --

NOT_NULL_POLICIES = {"exclude_pair", "fail_validation"}


def check_forbidden_columns(schema: dict, columns: list[str]) -> list[str]:
    """Non-key columns must not contain a forbidden pattern or be a location column."""
    patterns = [p.lower() for p in schema.get("forbidden_feature_patterns") or []]
    errors = []
    for column in columns:
        if column in KEYS:
            continue
        if column in LOCATION_COLUMNS:
            errors.append(f"column '{column}' is a location column")
        hits = [p for p in patterns if p in column.lower()]
        if hits:
            errors.append(f"column '{column}' matches forbidden pattern(s) {hits}")
    return errors


def check_unique_ids(df: pd.DataFrame) -> list[str]:
    if "scenario_id" not in df.columns:
        return ["column 'scenario_id' is absent"]
    count = duplicate_count(df)
    return [f"{count} duplicate scenario_id"] if count else []


def check_row_count(df: pd.DataFrame, expected_rows: int) -> list[str]:
    return [] if len(df) == expected_rows else [f"row count {len(df)} != expected {expected_rows}"]


def check_missing_indicators(df: pd.DataFrame, schema: dict) -> list[str]:
    """Every null_with_indicator feature present must have <column>_missing equal to its nullness;
    any other <column>_missing present must also agree with its base column."""
    errors = []
    required = {feature_column(f) for f in model_features(schema)
                if f.get("missing_policy") == INDICATOR_POLICY and feature_column(f) in df.columns}
    for column in sorted(required):
        if missing_column(column) not in df.columns:
            errors.append(f"missing indicator '{missing_column(column)}' is absent")
    for indicator in df.columns:
        if not indicator.endswith(MISSING_SUFFIX):
            continue
        base = indicator[:-len(MISSING_SUFFIX)]
        if base not in df.columns:
            continue
        flags = df[indicator]
        if flags.isna().any():
            errors.append(f"'{indicator}' contains {int(flags.isna().sum())} null value(s)")
            continue
        mismatches = int((flags.astype(bool).to_numpy() != df[base].isna().to_numpy()).sum())
        if mismatches:
            errors.append(f"'{indicator}' disagrees with nullness of '{base}' in {mismatches} row(s)")
    return errors


def check_required_not_null(df: pd.DataFrame, schema: dict) -> list[str]:
    """Model features with missing_policy exclude_pair or fail_validation must have no nulls."""
    errors = []
    for field in model_features(schema):
        column = feature_column(field)
        if field.get("missing_policy") not in NOT_NULL_POLICIES or column not in df.columns:
            continue
        nulls = int(df[column].isna().sum())
        if nulls:
            errors.append(f"'{column}' ({field['missing_policy']}) contains {nulls} null value(s)")
    return errors


def check_declared_present(df: pd.DataFrame, schema: dict) -> list[str]:
    """Every Schema field with `features` in its artifacts must have its column in the table."""
    return [f"declared features column '{c}' is absent"
            for c in sorted(declared_columns(schema, "features") - set(df.columns))]


def check_table(df: pd.DataFrame, schema: dict, expected_rows: int) -> list[tuple[str, list[str]]]:
    """(check name, errors) pairs: validator.check_leakage, validator.check_feature_table_columns,
    forbidden patterns / location columns, unique scenario_id, row count, indicator == isna,
    no nulls in exclude_pair/fail_validation features, artifacts:[features] columns present."""
    columns = list(df.columns)
    return [
        ("schema_leakage", check_leakage(schema)),
        ("feature_table_columns", check_feature_table_columns(schema, columns)),
        ("forbidden_columns", check_forbidden_columns(schema, columns)),
        ("unique_scenario_id", check_unique_ids(df)),
        ("row_count", check_row_count(df, expected_rows)),
        ("missing_indicators", check_missing_indicators(df, schema)),
        ("required_not_null", check_required_not_null(df, schema)),
        ("declared_columns_present", check_declared_present(df, schema)),
    ]


# --------------------------------------------------------------------------------------- inputs --

PROVENANCE_KEY = b"scenario_features_provenance"
VARIANTS = ("full", "mini")
OUTPUT_FILES = {"full": "scenario_features.parquet", "mini": "scenario_features_mini_68.parquet"}
REPORT_STEM = "scenario_features"
CANDIDATE_FLAG = "benchmark_candidate"
CANDIDATE_COLUMNS = KEYS + [MAP_NAME]
SOURCE_TABLES = ("ego_agent", "weather", "map")


def require_file(path: Path) -> Path:
    path = Path(path)
    if not path.is_file():
        raise InputError(f"input file not found: {path}")
    return path


def require_columns(path: Path, names: list[str], required: list[str]) -> None:
    missing = [c for c in required if c not in names]
    if missing:
        raise InputError(f"{path} lacks required column(s): {', '.join(missing)}")


def parquet_names(path: Path) -> list[str]:
    import pyarrow.parquet as pq

    return list(pq.read_schema(require_file(path)).names)


def file_info(path: Path, rows: int) -> dict:
    path = Path(path)
    return {"path": str(path), "rows": int(rows), "size_bytes": int(path.stat().st_size)}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_full_candidates(path: Path) -> tuple[pd.DataFrame, dict]:
    """Benchmark candidates of scenario_eligibility.parquet (only key, map_name and flag columns read)."""
    import pyarrow.parquet as pq

    names = parquet_names(path)
    require_columns(path, names, CANDIDATE_COLUMNS + [CANDIDATE_FLAG])
    table = pq.read_table(path, columns=CANDIDATE_COLUMNS + [CANDIDATE_FLAG],
                          filters=[(CANDIDATE_FLAG, "==", True)])
    frame = table.select(CANDIDATE_COLUMNS).to_pandas()
    info = file_info(path, pq.ParquetFile(path).metadata.num_rows)
    info["candidate_rows"] = int(len(frame))
    return frame, info


def load_mini_candidates(path: Path) -> tuple[pd.DataFrame, dict]:
    """The mini sample scenarios with their map_name from idm_mini_sample.csv."""
    path = require_file(path)
    names = list(pd.read_csv(path, nrows=0).columns)
    require_columns(path, names, CANDIDATE_COLUMNS)
    frame = pd.read_csv(path, usecols=CANDIDATE_COLUMNS, dtype=str, keep_default_na=False)[CANDIDATE_COLUMNS]
    info = file_info(path, len(frame))
    info["candidate_rows"] = int(len(frame))
    return frame, info


def load_candidates(variant: str, args) -> tuple[pd.DataFrame, dict]:
    if variant == "full":
        return load_full_candidates(args.eligibility)
    return load_mini_candidates(args.mini_manifest)


def count_candidates(variant: str, args) -> int:
    """Expected Feature_Table rows: benchmark candidates (full) or mini manifest rows (mini)."""
    if variant == "full":
        import pyarrow.parquet as pq

        path = args.eligibility
        require_columns(path, parquet_names(path), ["scenario_id", CANDIDATE_FLAG])
        return int(pq.read_table(path, columns=[CANDIDATE_FLAG],
                                 filters=[(CANDIDATE_FLAG, "==", True)]).num_rows)
    return len(load_mini_candidates(args.mini_manifest)[0])


def source_paths(variant: str, args) -> dict[str, Path]:
    if variant == "full":
        return {"ego_agent": Path(args.ego), "weather": Path(args.weather), "map": Path(args.map)}
    return {"ego_agent": Path(args.ego_mini), "weather": Path(args.weather_mini), "map": Path(args.map_mini)}


def load_sources(paths: dict[str, Path], schema: dict):
    """Read only key, selected feature and indicator columns of each source table.

    Returns (frames, arrow types of the copied columns, input infos, not-integrated feature names)."""
    import pyarrow.parquet as pq

    names = {}
    for table, path in paths.items():
        names[table] = parquet_names(path)
        require_columns(path, names[table], ["scenario_id"])
    available = set().union(*(set(n) for n in names.values()))
    selected, not_integrated = select_feature_columns(schema, available)
    wanted = set(KEYS)
    for field in selected:
        wanted.add(field["column"])
        if field.get("missing_policy") == INDICATOR_POLICY:
            wanted.add(missing_column(field["column"]))

    frames, types, infos = {}, {}, {}
    for table, path in paths.items():
        columns = [c for c in names[table] if c in wanted]
        arrow = pq.read_table(path, columns=columns)
        for field in arrow.schema:
            if field.name not in KEYS:
                types.setdefault(field.name, field.type)
        frames[table] = arrow.to_pandas()
        infos[table] = file_info(path, pq.ParquetFile(path).metadata.num_rows)
    return frames, types, infos, not_integrated


# -------------------------------------------------------------------------------------- output --

def output_path(variant: str, args) -> Path:
    return Path(args.out_dir) / OUTPUT_FILES[variant]


def arrow_schema(df: pd.DataFrame, types: dict | None = None):
    """Inferred schema with source column types restored (pandas widens nullable ints to float),
    `_missing` columns as bool and keys / map_name as string."""
    import pyarrow as pa

    inferred = pa.Schema.from_pandas(df, preserve_index=False)
    fields = []
    for field in inferred:
        name = field.name
        if name in KEYS or name == MAP_NAME:
            fields.append(pa.field(name, pa.string()))
        elif name.endswith(MISSING_SUFFIX):
            fields.append(pa.field(name, pa.bool_()))
        elif types and name in types:
            fields.append(pa.field(name, types[name]))
        else:
            fields.append(field)
    return pa.schema(fields)


def write_parquet_atomic(df: pd.DataFrame, path: Path, metadata: dict, types: dict | None = None) -> None:
    """Write `df` with `scenario_features_provenance` metadata to `<path>.tmp`, then os.replace it."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, schema=arrow_schema(df, types), preserve_index=False)
    table = table.replace_schema_metadata({**(table.schema.metadata or {}),
                                           PROVENANCE_KEY: json.dumps(metadata, default=str).encode()})
    tmp = path.with_name(path.name + ".tmp")
    try:
        pq.write_table(table, tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def read_provenance(path: Path) -> dict:
    import pyarrow.parquet as pq

    raw = (pq.read_schema(path).metadata or {}).get(PROVENANCE_KEY)
    return json.loads(raw) if raw else {}


# -------------------------------------------------------------------------------------- report --

def city_of(map_name, cities: dict) -> str:
    if map_name is None or (isinstance(map_name, float) and pd.isna(map_name)):
        return "null"
    return cities.get(str(map_name), "unmapped")


def build_report(df: pd.DataFrame, schema: dict, not_integrated: list[str]) -> dict:
    """Report section of one Feature_Table: columns, rows, per-column null counts, rows by city
    (Schema city_by_map_name) and the Schema model features that are not integrated."""
    cities = schema.get("city_by_map_name") or {}
    by_city = df[MAP_NAME].map(lambda m: city_of(m, cities)).value_counts() if len(df) else pd.Series(dtype=int)
    return {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "null_counts": {c: int(df[c].isna().sum()) for c in df.columns},
        "rows_by_city": {str(c): int(n) for c, n in sorted(by_city.items())},
        "not_integrated": list(not_integrated),
    }


def assemble_report(sections: dict[str, dict], previous: dict | None = None, meta: dict | None = None) -> dict:
    """{variant: {rows, columns, null_counts, rows_by_city, path}, not_integrated, ...}; variants
    not rebuilt in this run are kept from `previous`."""
    report = {v: previous[v] for v in VARIANTS if previous and isinstance(previous.get(v), dict)}
    not_integrated = list((previous or {}).get("not_integrated") or [])
    for variant, section in sections.items():
        section = dict(section)
        not_integrated = section.pop("not_integrated", not_integrated)
        report[variant] = section
    ordered = {v: report[v] for v in VARIANTS if v in report}
    ordered["not_integrated"] = not_integrated
    ordered.update(meta or {})
    return ordered


def markdown_report(report: dict) -> str:
    variants = [v for v in VARIANTS if v in report]
    lines = [
        "# Scenario features",
        "",
        f"Generated by `python -m tools.build_scenario_features build` ({report.get('tool_version', TOOL_VERSION)}, "
        f"{report.get('generated_utc', '')}). Tables stay outside Git in the output directory. "
        "Model features are selected from `configs/features/first_experiment_schema.yaml`; "
        "every `null_with_indicator` feature has a `<column>_missing` flag.",
        "",
        "## Tables",
        "",
        "| Variant | File | Rows | Columns |",
        "|---|---|---:|---:|",
    ]
    for v in variants:
        s = report[v]
        lines.append(f"| {v} | `{s.get('path', OUTPUT_FILES[v])}` | {s['rows']:,} | {len(s['columns'])} |")
    cities = sorted({c for v in variants for c in report[v]["rows_by_city"]})
    lines += ["", "## Rows by city", "",
              "| City | " + " | ".join(variants) + " |",
              "|---|" + "---:|" * len(variants)]
    for city in cities:
        lines.append(f"| {city} | " + " | ".join(f"{report[v]['rows_by_city'].get(city, 0):,}" for v in variants) + " |")
    columns = []
    for v in variants:
        columns += [c for c in report[v]["columns"] if c not in columns]
    lines += ["", "## Columns and null counts", "",
              "| Column | " + " | ".join(f"{v} nulls" for v in variants) + " |",
              "|---|" + "---:|" * len(variants)]
    for c in columns:
        cells = [f"{report[v]['null_counts'][c]:,}" if c in report[v]["null_counts"] else "absent" for v in variants]
        lines.append(f"| `{c}` | " + " | ".join(cells) + " |")
    lines += ["", "## Model features not integrated", ""]
    lines += [f"- `{name}`" for name in report.get("not_integrated") or []] or ["None."]
    return "\n".join(lines) + "\n"


def write_report(report_dir: Path, sections: dict[str, dict], meta: dict) -> tuple[Path, Path]:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path, md_path = report_dir / f"{REPORT_STEM}.json", report_dir / f"{REPORT_STEM}.md"
    previous = None
    if json_path.is_file():
        try:
            previous = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            previous = None
    report = assemble_report(sections, previous, meta)
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    md_path.write_text(markdown_report(report), encoding="utf-8", newline="\n")
    return md_path, json_path


# ----------------------------------------------------------------------------------- commands --

def selected_variants(args) -> list[str]:
    return list(VARIANTS) if args.only == "both" else [args.only]


def cmd_build(args) -> int:
    """Integrate every requested variant first; write tables and report only if all succeed."""
    built = {}
    variant = None
    try:
        schema = validator.load_schema(args.schema)
        schema_info = {"path": str(args.schema), "sha256": sha256_file(args.schema)}
        for variant in selected_variants(args):
            candidates, candidate_info = load_candidates(variant, args)
            frames, types, infos, not_integrated = load_sources(source_paths(variant, args), schema)
            df = integrate(candidates, frames, schema)
            inputs = {"candidates": candidate_info, **infos}
            built[variant] = (df, types, inputs, not_integrated)
    except (InputError, validator.InputError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except CoverageError as exc:
        print(f"FAIL {variant} coverage: no table written", file=sys.stderr)
        for table, count in exc.absent.items():
            print(f"  {table}: {count:,} candidate scenario_id absent", file=sys.stderr)
        for table, count in exc.duplicates.items():
            print(f"  {table}: {count:,} duplicate scenario_id", file=sys.stderr)
        return 1

    built_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    sections = {}
    for variant, (df, types, inputs, not_integrated) in built.items():
        path = output_path(variant, args)
        metadata = {"tool_version": TOOL_VERSION, "built_utc": built_utc, "variant": variant,
                    "schema": schema_info, "inputs": inputs, "rows": int(len(df)),
                    "columns": list(df.columns), "not_integrated": not_integrated}
        write_parquet_atomic(df, path, metadata, types)
        print(f"Wrote {path} ({len(df):,} rows, {len(df.columns)} columns)")
        sections[variant] = {**build_report(df, schema, not_integrated), "path": str(path)}
    md_path, json_path = write_report(args.report_dir, sections,
                                      {"tool_version": TOOL_VERSION, "generated_utc": built_utc,
                                       "schema_sha256": schema_info["sha256"]})
    print(f"Wrote {md_path} and {json_path}")
    return 0


def cmd_check(args) -> int:
    """Run check_table on every requested Feature_Table; one PASS/FAIL line per check."""
    try:
        schema = validator.load_schema(args.schema)
        loaded = []
        for variant in selected_variants(args):
            path = require_file(output_path(variant, args))
            loaded.append((variant, pd.read_parquet(path), count_candidates(variant, args)))
    except (InputError, validator.InputError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    ok = True
    for variant, df, expected in loaded:
        for name, errors in check_table(df, schema, expected):
            if errors:
                ok = False
                shown = "; ".join(errors[:3]) + (f" (+{len(errors) - 3} more)" if len(errors) > 3 else "")
                print(f"FAIL {variant}.{name}: {shown}")
            else:
                print(f"PASS {variant}.{name}")
    return 0 if ok else 1


# ------------------------------------------------------------------------------------------ CLI --

def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--ego", type=Path, default=DEFAULT_EGO, help="full ego/agent features parquet")
    common.add_argument("--ego-mini", type=Path, default=DEFAULT_EGO_MINI, help="mini ego/agent features parquet")
    common.add_argument("--weather", type=Path, default=DEFAULT_WEATHER, help="full weather features parquet")
    common.add_argument("--weather-mini", type=Path, default=DEFAULT_WEATHER_MINI,
                        help="mini weather features parquet")
    common.add_argument("--map", type=Path, default=DEFAULT_MAP, help="full map features parquet")
    common.add_argument("--map-mini", type=Path, default=DEFAULT_MAP_MINI, help="mini map features parquet")
    common.add_argument("--eligibility", type=Path, default=DEFAULT_ELIGIBILITY,
                        help=f"scenario_eligibility.parquet; rows with {CANDIDATE_FLAG} true are the full variant")
    common.add_argument("--mini-manifest", type=Path, default=DEFAULT_MINI_MANIFEST,
                        help="mini sample CSV (scenario_id, log_name, scenario_token, map_name)")
    common.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help=f"directory of {OUTPUT_FILES['full']} and {OUTPUT_FILES['mini']}")
    common.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR,
                        help=f"directory of {REPORT_STEM}.md and {REPORT_STEM}.json")
    common.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA, help="feature schema YAML")
    common.add_argument("--only", choices=["full", "mini", "both"], default="both", help="variants to process")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build", parents=[common], help="integrate features, write tables and report").set_defaults(
        func=cmd_build)
    sub.add_parser("check", parents=[common], help="validate the written feature tables").set_defaults(
        func=cmd_check)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
