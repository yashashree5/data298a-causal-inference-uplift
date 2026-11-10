"""Tests for tools/build_scenario_features.py on synthetic frames and parquet files in a temp dir."""
from pathlib import Path
import contextlib
import io
import json
import random
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
from pandas.testing import assert_series_equal  # noqa: E402

from tools import build_scenario_features as bsf  # noqa: E402
from tools import validate_scenario_subset as validator  # noqa: E402

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "configs" / "features" / "first_experiment_schema.yaml"
SCHEMA = validator.load_schema(SCHEMA_PATH)
ITERATIONS = 100
MAPS = list(SCHEMA["city_by_map_name"])

EGO_FEATURES = ["ego_speed", "ego_acceleration", "ego_yaw_rate", "num_tracked_agents", "num_vehicles",
                "num_pedestrians", "nearest_agent_distance_m", "nearest_agent_relative_speed_mps",
                "agents_within_50m"]
MAP_FEATURES = ["lane_type", "in_intersection", "traffic_light_state"]
WEATHER_FEATURES = ["weather_temperature_c", "weather_relative_humidity_pct", "weather_cloud_cover_pct",
                    "weather_wind_speed_kmh", "weather_code", "weather_precipitation_mm", "weather_rain_mm",
                    "weather_wind_gusts_kmh", "weather_precipitation_prev_3h_mm", "sun_elevation_deg"]
# Schema order of the integrated model features (ego/scene group, then weather group).
SCHEMA_ORDER = ["ego_speed", "ego_acceleration", "ego_yaw_rate", "lane_type", "in_intersection",
                "traffic_light_state", "num_tracked_agents", "num_vehicles", "num_pedestrians",
                "nearest_agent_distance_m", "nearest_agent_relative_speed_mps", "agents_within_50m",
                *WEATHER_FEATURES]
INDICATOR_FEATURES = {"lane_type", "in_intersection", "traffic_light_state", "nearest_agent_distance_m",
                      "nearest_agent_relative_speed_mps", *WEATHER_FEATURES}
NULLABLE = INDICATOR_FEATURES
INT_FEATURES = {"num_tracked_agents", "num_vehicles", "num_pedestrians", "agents_within_50m"}
EXTRA_COLUMNS = ["ego_x", "ego_y", "ego_heading", "lane_id", "score", "planner", "succeeded", "foo_undeclared",
                 "window_end_us"]


def make_candidates(rng: random.Random, n: int) -> pd.DataFrame:
    tokens = set()
    while len(tokens) < n:
        tokens.add(f"{rng.getrandbits(64):016x}")
    rows = []
    for i, token in enumerate(sorted(tokens)):
        log = f"2021.0{i % 5}.01.00.00.00_veh-{i % 3}"
        rows.append({"scenario_id": f"{log}:{token}", "log_name": log, "scenario_token": token,
                     "map_name": rng.choice(MAPS)})
    frame = pd.DataFrame(rows, columns=bsf.CANDIDATE_COLUMNS)
    return frame.sample(frac=1, random_state=rng.randrange(2**31)).reset_index(drop=True)


def feature_value(rng: random.Random, column: str, null_rate: float):
    if column in NULLABLE and rng.random() < null_rate:
        return None
    if column in INT_FEATURES or column == "weather_code":
        return rng.randint(0, 40)
    if column == "lane_type":
        return rng.choice(["lane", "lane_connector"])
    if column == "in_intersection":
        return rng.choice([True, False])
    if column == "traffic_light_state":
        return rng.choice(["green", "yellow", "red", "unknown", "none"])
    return rng.uniform(-50.0, 50.0)


def make_source(rng: random.Random, ids: list[str], features: list[str], indicators: bool,
                extras: list[str] = (), null_rate: float = 0.3) -> pd.DataFrame:
    data = {"scenario_id": list(ids)}
    for column in features:
        data[column] = [feature_value(rng, column, null_rate) for _ in ids]
        if indicators and column in INDICATOR_FEATURES:
            data[column + "_missing"] = [v is None for v in data[column]]
    for column in extras:
        data[column] = [rng.uniform(0, 1) for _ in ids]
    return pd.DataFrame(data)


def make_sources(rng: random.Random, candidates: pd.DataFrame, indicators=None, extras: bool = False,
                 null_rate: float = 0.3) -> dict[str, pd.DataFrame]:
    ids = list(candidates["scenario_id"])
    sources = {}
    for table, features in (("ego_agent", EGO_FEATURES), ("weather", WEATHER_FEATURES), ("map", MAP_FEATURES)):
        with_ind = rng.random() < 0.5 if indicators is None else indicators
        chosen = rng.sample(EXTRA_COLUMNS, rng.randint(0, 3)) if extras else []
        sources[table] = make_source(rng, ids, features, with_ind, chosen, null_rate)
    return sources


def expected_columns(features: list[str]) -> list[str]:
    ordered = [c for c in SCHEMA_ORDER if c in features]
    return bsf.KEYS + ["map_name"] + ordered + [c + "_missing" for c in ordered if c in INDICATOR_FEATURES]


def errors_by_check(df: pd.DataFrame, expected_rows: int) -> dict[str, list[str]]:
    return dict(bsf.check_table(df, SCHEMA, expected_rows))


def valid_table(rng: random.Random, n: int = None) -> pd.DataFrame:
    candidates = make_candidates(rng, n or rng.randint(1, 20))
    return bsf.integrate(candidates, make_sources(rng, candidates), SCHEMA)


class PropertyTests(unittest.TestCase):
    def test_column_contract(self):
        """Feature: feature-integration, Property 1: column contract. Validates: Requirements 1.1, 3.1"""
        rng = random.Random(101)
        for _ in range(ITERATIONS):
            candidates = make_candidates(rng, rng.randint(1, 15))
            ids = list(candidates["scenario_id"])
            present = []
            sources = {}
            for table, features in (("ego_agent", EGO_FEATURES), ("weather", WEATHER_FEATURES),
                                    ("map", MAP_FEATURES)):
                kept = [c for c in features if rng.random() < 0.8]
                present += kept
                sources[table] = make_source(rng, ids, kept, rng.random() < 0.5,
                                             rng.sample(EXTRA_COLUMNS, rng.randint(0, 4)))
            out = bsf.integrate(candidates, sources, SCHEMA)
            self.assertEqual(list(out.columns), expected_columns(present))

    def test_join_faithfulness_under_shuffle(self):
        """Feature: feature-integration, Property 2: join faithfulness. Validates: Requirements 1.2, 1.3"""
        rng = random.Random(202)
        for _ in range(ITERATIONS):
            candidates = make_candidates(rng, rng.randint(1, 20))
            pool = make_candidates(rng, rng.randint(0, 5))  # extra source rows that are not candidates
            ids = list(candidates["scenario_id"]) + list(pool["scenario_id"])
            sources = {}
            for table, features in (("ego_agent", EGO_FEATURES), ("weather", WEATHER_FEATURES),
                                    ("map", MAP_FEATURES)):
                frame = make_source(rng, ids, features, rng.random() < 0.5)
                sources[table] = frame.sample(frac=1, random_state=rng.randrange(2**31)).reset_index(drop=True)
            out = bsf.integrate(candidates, sources, SCHEMA)
            self.assertEqual(len(out), len(candidates))
            self.assertTrue(out["scenario_id"].is_unique)
            self.assertEqual(sorted(out["scenario_id"]), sorted(candidates["scenario_id"]))
            self.assertEqual(list(out["scenario_id"]), sorted(out["scenario_id"]))
            indexed = out.set_index("scenario_id")
            expected_meta = candidates.set_index("scenario_id").loc[indexed.index]
            for key in ("log_name", "scenario_token", "map_name"):
                self.assertEqual(list(indexed[key]), list(expected_meta[key]))
            for table, frame in sources.items():
                source = frame.set_index("scenario_id").loc[indexed.index]
                for column in (c for c in frame.columns if c in SCHEMA_ORDER):
                    assert_series_equal(indexed[column], source[column], check_dtype=False, check_names=False)

    def test_missing_indicator_consistency(self):
        """Feature: feature-integration, Property 3: missing indicator consistency.
        Validates: Requirements 1.4, 4.2"""
        rng = random.Random(303)
        for _ in range(ITERATIONS):
            candidates = make_candidates(rng, rng.randint(1, 20))
            sources = make_sources(rng, candidates, indicators=False, null_rate=rng.random())
            out = bsf.integrate(candidates, sources, SCHEMA)
            for column in INDICATOR_FEATURES:
                self.assertEqual(list(out[column + "_missing"]), list(out[column].isna()))
            self.assertEqual(errors_by_check(out, len(out))["missing_indicators"], [])
            flipped = out.copy()
            indicator = rng.choice(sorted(INDICATOR_FEATURES)) + "_missing"
            row = rng.randrange(len(flipped))
            flipped.loc[row, indicator] = not bool(flipped.loc[row, indicator])
            self.assertTrue(errors_by_check(flipped, len(flipped))["missing_indicators"])

    def test_coverage_gaps_are_reported(self):
        """Feature: feature-integration, Property 4: coverage gaps are reported. Validates: Requirements 2.1"""
        rng = random.Random(404)
        for _ in range(ITERATIONS):
            candidates = make_candidates(rng, rng.randint(1, 20))
            sources = make_sources(rng, candidates)
            table = rng.choice(sorted(sources))
            removed = rng.sample(list(candidates["scenario_id"]), rng.randint(1, len(candidates)))
            frame = sources[table]
            sources[table] = frame[~frame["scenario_id"].isin(removed)].reset_index(drop=True)
            with self.assertRaises(bsf.CoverageError) as caught:
                bsf.integrate(candidates, sources, SCHEMA)
            self.assertEqual(caught.exception.absent, {table: len(removed)})
            self.assertEqual(caught.exception.duplicates, {})

    def test_forbidden_columns_fail_the_check(self):
        """Feature: feature-integration, Property 5: forbidden columns fail the check.
        Validates: Requirements 3.2, 3.3"""
        rng = random.Random(505)
        patterns = SCHEMA["forbidden_feature_patterns"]
        for _ in range(ITERATIONS):
            out = valid_table(rng)
            self.assertEqual({k: v for k, v in errors_by_check(out, len(out)).items() if v}, {})
            if rng.random() < 0.3:
                name = rng.choice(sorted(bsf.LOCATION_COLUMNS))
            else:
                prefix = rng.choice(["", "ego_", "x", "Pre_"])
                suffix = rng.choice(["", "_mean", "2", "_missing"])
                pattern = rng.choice(patterns)
                name = prefix + (pattern.upper() if rng.random() < 0.3 else pattern) + suffix
            bad = out.copy()
            bad[name] = [rng.uniform(0, 1) for _ in range(len(bad))]
            self.assertTrue(errors_by_check(bad, len(bad))["forbidden_columns"], name)


class IntegrateExampleTests(unittest.TestCase):
    def setUp(self):
        self.rng = random.Random(7)
        self.candidates = make_candidates(self.rng, 6)
        self.sources = make_sources(self.rng, self.candidates)

    def test_duplicate_source_scenario_id(self):
        frame = self.sources["weather"]
        self.sources["weather"] = pd.concat([frame, frame.iloc[[2]]], ignore_index=True)
        with self.assertRaises(bsf.CoverageError) as caught:
            bsf.integrate(self.candidates, self.sources, SCHEMA)
        self.assertEqual(caught.exception.duplicates, {"weather": 1})
        self.assertEqual(caught.exception.absent, {})

    def test_duplicate_candidate_scenario_id(self):
        candidates = pd.concat([self.candidates, self.candidates.iloc[[0, 1]]], ignore_index=True)
        with self.assertRaises(bsf.CoverageError) as caught:
            bsf.integrate(candidates, self.sources, SCHEMA)
        self.assertEqual(caught.exception.duplicates, {"candidates": 2})

    def test_source_indicator_is_copied(self):
        sources = make_sources(self.rng, self.candidates, indicators=True)
        out = bsf.integrate(self.candidates, sources, SCHEMA)
        self.assertEqual(out["lane_type_missing"].dtype, bool)

    def test_not_integrated_features(self):
        available = set(EGO_FEATURES + MAP_FEATURES + WEATHER_FEATURES)
        selected, missing = bsf.select_feature_columns(SCHEMA, available)
        self.assertEqual([f["column"] for f in selected], SCHEMA_ORDER)
        self.assertEqual(sorted(missing), ["lane_curvature", "route_length_m"])


class CheckTableExampleTests(unittest.TestCase):
    def setUp(self):
        self.table = valid_table(random.Random(11), 8)

    def test_valid_table_passes_every_check(self):
        self.assertEqual({k: v for k, v in errors_by_check(self.table, 8).items() if v}, {})

    def test_exclude_pair_null_fails(self):
        self.table.loc[3, "ego_speed"] = None
        errors = errors_by_check(self.table, 8)["required_not_null"]
        self.assertEqual(len(errors), 1)
        self.assertIn("ego_speed", errors[0])

    def test_absent_declared_column_fails(self):
        table = self.table.drop(columns=["weather_code", "weather_code_missing"])
        errors = errors_by_check(table, 8)["declared_columns_present"]
        self.assertEqual(errors, ["declared features column 'weather_code' is absent"])

    def test_duplicate_id_and_row_count_fail(self):
        table = pd.concat([self.table, self.table.iloc[[0]]], ignore_index=True)
        errors = errors_by_check(table, 8)
        self.assertTrue(errors["unique_scenario_id"])
        self.assertTrue(errors["row_count"])


class CliTests(unittest.TestCase):
    """build/check on synthetic parquet; every path flag points into a temp dir."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        rng = random.Random(42)
        full = make_candidates(rng, 12)
        others = make_candidates(rng, 4)  # not benchmark candidates; absent from the sources
        eligibility = pd.concat([full.assign(benchmark_candidate=True), others.assign(benchmark_candidate=False)],
                                ignore_index=True)
        eligibility["extra"] = 1.0
        eligibility.to_parquet(self.root / "eligibility.parquet", index=False)
        mini = full.iloc[:5]
        mini.assign(scenario_type="x").to_csv(self.root / "mini.csv", index=False)
        self.full, self.mini = full, mini
        for suffix, candidates in (("", full), ("_mini", mini)):
            sources = make_sources(rng, candidates, extras=True)
            for table, frame in sources.items():
                frame.to_parquet(self.root / f"{table}{suffix}.parquet", index=False)
        self.out = self.root / "out"
        self.reports = self.root / "reports"

    def tearDown(self):
        self.tmp.cleanup()

    def flags(self, **overrides) -> list[str]:
        values = {
            "--ego": self.root / "ego_agent.parquet", "--ego-mini": self.root / "ego_agent_mini.parquet",
            "--weather": self.root / "weather.parquet", "--weather-mini": self.root / "weather_mini.parquet",
            "--map": self.root / "map.parquet", "--map-mini": self.root / "map_mini.parquet",
            "--eligibility": self.root / "eligibility.parquet", "--mini-manifest": self.root / "mini.csv",
            "--out-dir": self.out, "--report-dir": self.reports, "--schema": SCHEMA_PATH,
        }
        values.update({k.replace("_", "-") if k.startswith("--") else f"--{k.replace('_', '-')}": v
                       for k, v in overrides.items()})
        return [str(x) for pair in values.items() for x in pair]

    def run_cli(self, command: str, *extra: str, **overrides) -> tuple[int, str, str]:
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = bsf.main([command, *self.flags(**overrides), *extra])
        return code, stdout.getvalue(), stderr.getvalue()

    def test_build_then_check_pass(self):
        code, out, err = self.run_cli("build")
        self.assertEqual(code, 0, err)
        full = pd.read_parquet(self.out / "scenario_features.parquet")
        mini = pd.read_parquet(self.out / "scenario_features_mini_68.parquet")
        self.assertEqual(len(full), 12)
        self.assertEqual(len(mini), 5)
        self.assertEqual(list(full.columns), expected_columns(SCHEMA_ORDER))
        self.assertEqual(sorted(full["scenario_id"]), sorted(self.full["scenario_id"]))
        self.assertEqual(full["num_vehicles"].dtype, "int64")
        self.assertEqual(full["in_intersection_missing"].dtype, bool)

        code, out, err = self.run_cli("check")
        self.assertEqual(code, 0, out + err)
        lines = out.strip().splitlines()
        self.assertEqual(len(lines), 16)  # 8 checks x 2 variants
        self.assertTrue(all(line.startswith("PASS ") for line in lines), out)

    def test_check_fails_on_corrupted_table(self):
        self.assertEqual(self.run_cli("build")[0], 0)
        path = self.out / "scenario_features_mini_68.parquet"
        table = pd.read_parquet(path)
        table.loc[0, "lane_type_missing"] = not table.loc[0, "lane_type_missing"]
        table["ego_x"] = 0.0
        table.to_parquet(path, index=False)
        code, out, _ = self.run_cli("check", "--only", "mini")
        self.assertEqual(code, 1)
        self.assertIn("FAIL mini.missing_indicators", out)
        self.assertIn("FAIL mini.forbidden_columns", out)
        self.assertIn("PASS mini.row_count", out)

    def test_check_missing_table_exits_2(self):
        code, _, err = self.run_cli("check", "--only", "full")
        self.assertEqual(code, 2)
        self.assertIn("scenario_features.parquet", err)

    def test_missing_input_file_exits_2(self):
        missing = self.root / "nope.parquet"
        code, _, err = self.run_cli("build", "--only", "mini", ego_mini=missing)
        self.assertEqual(code, 2)
        self.assertIn(str(missing), err)
        self.assertFalse(self.out.exists())

    def test_missing_column_exits_2(self):
        self.mini.drop(columns=["map_name"]).to_csv(self.root / "bad.csv", index=False)
        code, _, err = self.run_cli("build", "--only", "mini", mini_manifest=self.root / "bad.csv")
        self.assertEqual(code, 2)
        self.assertIn("map_name", err)
        pd.read_parquet(self.root / "map.parquet").drop(columns=["scenario_id"]).to_parquet(
            self.root / "map_noid.parquet", index=False)
        code, _, err = self.run_cli("build", "--only", "full", map=self.root / "map_noid.parquet")
        self.assertEqual(code, 2)
        self.assertIn("scenario_id", err)

    def test_coverage_gap_exits_1_and_writes_nothing(self):
        weather = pd.read_parquet(self.root / "weather_mini.parquet").iloc[2:]
        weather.to_parquet(self.root / "weather_gap.parquet", index=False)
        code, _, err = self.run_cli("build", weather_mini=self.root / "weather_gap.parquet")
        self.assertEqual(code, 1)
        self.assertIn("weather: 2 candidate scenario_id absent", err)
        self.assertFalse((self.out / "scenario_features.parquet").exists())
        self.assertFalse((self.out / "scenario_features_mini_68.parquet").exists())
        self.assertFalse(self.reports.exists())

    def test_duplicate_exits_1(self):
        ego = pd.read_parquet(self.root / "ego_agent.parquet")
        pd.concat([ego, ego.iloc[[0]]]).to_parquet(self.root / "ego_dup.parquet", index=False)
        code, _, err = self.run_cli("build", "--only", "full", ego=self.root / "ego_dup.parquet")
        self.assertEqual(code, 1)
        self.assertIn("ego_agent: 1 duplicate scenario_id", err)

    def test_provenance_metadata(self):
        self.assertEqual(self.run_cli("build")[0], 0)
        meta = bsf.read_provenance(self.out / "scenario_features.parquet")
        self.assertEqual(meta["variant"], "full")
        self.assertEqual(meta["tool_version"], bsf.TOOL_VERSION)
        self.assertTrue(meta["built_utc"])
        self.assertEqual(meta["schema"]["sha256"], bsf.sha256_file(SCHEMA_PATH))
        self.assertEqual(meta["rows"], 12)
        self.assertEqual(set(meta["inputs"]), {"candidates", "ego_agent", "weather", "map"})
        ego = meta["inputs"]["ego_agent"]
        self.assertEqual(ego["path"], str(self.root / "ego_agent.parquet"))
        self.assertEqual(ego["rows"], 12)
        self.assertEqual(ego["size_bytes"], (self.root / "ego_agent.parquet").stat().st_size)
        self.assertEqual(meta["inputs"]["candidates"]["rows"], 16)
        self.assertEqual(meta["inputs"]["candidates"]["candidate_rows"], 12)
        self.assertEqual(bsf.read_provenance(self.out / "scenario_features_mini_68.parquet")["variant"], "mini")
        self.assertEqual(list(self.out.glob("*.tmp")), [])

    def test_report_files(self):
        self.assertEqual(self.run_cli("build")[0], 0)
        report = json.loads((self.reports / "scenario_features.json").read_text(encoding="utf-8"))
        full = pd.read_parquet(self.out / "scenario_features.parquet")
        self.assertEqual(report["full"]["rows"], 12)
        self.assertEqual(report["mini"]["rows"], 5)
        self.assertEqual(report["full"]["columns"], list(full.columns))
        self.assertEqual(report["full"]["null_counts"], {c: int(full[c].isna().sum()) for c in full.columns})
        cities = SCHEMA["city_by_map_name"]
        expected_cities = full["map_name"].map(cities).value_counts().to_dict()
        self.assertEqual(report["full"]["rows_by_city"], dict(sorted(expected_cities.items())))
        self.assertEqual(sorted(report["not_integrated"]), ["lane_curvature", "route_length_m"])
        markdown = (self.reports / "scenario_features.md").read_text(encoding="utf-8")
        self.assertIn("# Scenario features", markdown)
        self.assertIn("`route_length_m`", markdown)
        self.assertIn("`weather_code_missing`", markdown)

        # Rebuilding one variant keeps the other variant's section.
        self.assertEqual(self.run_cli("build", "--only", "mini")[0], 0)
        report = json.loads((self.reports / "scenario_features.json").read_text(encoding="utf-8"))
        self.assertEqual(report["full"]["rows"], 12)


class ValidatorParquetTests(unittest.TestCase):
    def test_validator_reads_parquet_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            good, bad = Path(tmp) / "features.parquet", Path(tmp) / "bad.parquet"
            table = valid_table(random.Random(3), 4)
            bsf.write_parquet_atomic(table, good, {"variant": "test"})
            table.assign(score=0.5).to_parquet(bad, index=False)
            self.assertEqual(validator.read_feature_columns(good), list(table.columns))
            self.assertEqual(validator.check_feature_table_columns(SCHEMA, validator.read_feature_columns(good)), [])

            def feature_check(path):
                report = validator.validate(SCHEMA_PATH, path)
                return next(c for c in report.checks if c["check"].startswith("feature table"))

            self.assertEqual(feature_check(good)["status"], validator.PASS)
            failed = feature_check(bad)
            self.assertEqual(failed["status"], validator.FAIL)
            self.assertIn("'score'", failed["detail"])
            with self.assertRaises(validator.InputError):
                validator.read_feature_columns(Path(tmp) / "absent.parquet")


if __name__ == "__main__":
    unittest.main()
