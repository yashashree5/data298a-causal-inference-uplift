import contextlib
import copy
import csv
import io
from pathlib import Path
import random
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.idm_mini_reproducer import DEFAULT_SAMPLE, load_scenarios  # noqa: E402
from tools.validate_scenario_subset import (  # noqa: E402
    DEFAULT_SCHEMA, FIELD_ATTRIBUTES, REQUIRED_GROUPS, assign_log_folds, canonical_order, check_declared_columns,
    check_feature_table_columns, check_leakage, check_log_split, check_manifest, check_pairs, check_schema_structure,
    find_duplicates, load_schema, main, missing_values, model_features, read_csv, required_columns, schema_fields,
    validate,
)

try:
    import yaml  # noqa: F401
except ImportError:
    yaml = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTCOMES = PROJECT_ROOT / "artifacts" / "mini_68" / "planner_outcomes.csv"
PLANNERS = ["idm", "pdm-closed"]
needs_yaml = unittest.skipUnless(yaml, "PyYAML is required to read the feature schema")


def manifest_rows():
    return read_csv(DEFAULT_SAMPLE)[1]


def outcome_rows():
    return read_csv(OUTCOMES)[1]


def schema_field(schema, name):
    for spec in schema["groups"].values():
        for item in spec["fields"]:
            if item["name"] == name:
                return item
    raise KeyError(name)


class ManifestTest(unittest.TestCase):
    def test_manifest_loads_with_existing_runner_loader(self):
        rows = load_scenarios(DEFAULT_SAMPLE)
        self.assertEqual(len(rows), 68)
        self.assertEqual(len({row["scenario_type"] for row in rows}), 14)
        self.assertEqual(len({row["log_name"] for row in rows}), 38)
        self.assertEqual(len({row["map_name"] for row in rows}), 4)

    def test_tokens_and_scenario_ids_are_unique(self):
        rows = manifest_rows()
        self.assertEqual(len({row["scenario_token"] for row in rows}), len(rows))
        self.assertEqual(len({row["scenario_id"] for row in rows}), len(rows))
        self.assertEqual(find_duplicates(rows, ["scenario_token"]), [])

    def test_every_scenario_has_log_and_type(self):
        self.assertEqual(missing_values(manifest_rows(), ["log_name", "scenario_type", "scenario_token"]), {})

    @needs_yaml
    def test_committed_manifest_passes(self):
        self.assertEqual(check_manifest(manifest_rows(), load_schema(DEFAULT_SCHEMA)), [])

    @needs_yaml
    def test_duplicate_token_is_rejected(self):
        rows = manifest_rows()
        rows.append(dict(rows[0]))
        errors = check_manifest(rows, load_schema(DEFAULT_SCHEMA))
        self.assertTrue(any("duplicate scenario_token" in e for e in errors))
        self.assertTrue(any("duplicate scenario_id" in e for e in errors))

    @needs_yaml
    def test_malformed_token_and_inconsistent_id_are_rejected(self):
        rows = manifest_rows()
        rows[0]["scenario_token"] = "NOT-A-TOKEN"
        errors = check_manifest(rows, load_schema(DEFAULT_SCHEMA))
        self.assertTrue(any("16 lowercase hex" in e for e in errors))
        self.assertTrue(any("does not equal log_name:scenario_token" in e for e in errors))

    @needs_yaml
    def test_overlapping_windows_in_one_log_are_rejected(self):
        rows = manifest_rows()
        clone = dict(rows[0], scenario_token="0123456789abcdef")
        clone["scenario_id"] = f"{clone['log_name']}:{clone['scenario_token']}"
        rows.append(clone)
        errors = check_manifest(rows, load_schema(DEFAULT_SCHEMA))
        self.assertTrue(any("overlap" in e for e in errors))


class MatchedPairTest(unittest.TestCase):
    def test_committed_outcomes_form_matched_pairs(self):
        outcomes = outcome_rows()
        self.assertEqual(len(outcomes), 136)
        self.assertEqual(check_pairs(manifest_rows(), outcomes, PLANNERS), [])

    def test_missing_planner_row_is_rejected(self):
        outcomes = [row for row in outcome_rows()
                    if not (row["planner"] == "pdm-closed" and row["scenario_id"] == manifest_rows()[0]["scenario_id"])]
        errors = check_pairs(manifest_rows(), outcomes, PLANNERS)
        self.assertTrue(any(e.startswith("pdm-closed: no outcome row") for e in errors))

    def test_extra_scenario_is_rejected(self):
        outcomes = outcome_rows()
        outcomes.append(dict(outcomes[0], scenario_id="unknown_log:0123456789abcdef"))
        errors = check_pairs(manifest_rows(), outcomes, PLANNERS)
        self.assertTrue(any("not in the manifest" in e for e in errors))

    def test_duplicate_scenario_planner_row_is_rejected(self):
        outcomes = outcome_rows()
        outcomes.append(dict(outcomes[0]))
        self.assertEqual(find_duplicates(outcomes, ["scenario_id", "planner"]),
                         [(outcomes[0]["scenario_id"], outcomes[0]["planner"])])
        self.assertTrue(any("duplicate scenario-planner" in e for e in check_pairs(manifest_rows(), outcomes, PLANNERS)))

    def test_unexpected_planner_is_rejected(self):
        outcomes = outcome_rows()
        outcomes[0] = dict(outcomes[0], planner="pdm-open")
        errors = check_pairs(manifest_rows(), outcomes, PLANNERS)
        self.assertTrue(any(e.startswith("planners ") for e in errors))


@needs_yaml
class RequiredFieldTest(unittest.TestCase):
    def setUp(self):
        self.schema = load_schema(DEFAULT_SCHEMA)

    def test_schema_has_required_groups_and_attributes(self):
        self.assertEqual(check_schema_structure(self.schema), [])
        self.assertTrue(set(REQUIRED_GROUPS) <= set(self.schema["groups"]))
        for item in schema_fields(self.schema):
            for attr in FIELD_ATTRIBUTES:
                self.assertIn(attr, item, item["name"])

    def test_field_without_attribute_is_rejected(self):
        broken = copy.deepcopy(self.schema)
        del schema_field(broken, "ego_speed")["missing_policy"]
        self.assertTrue(any("ego_speed" in e for e in check_schema_structure(broken)))

    def test_committed_columns_are_all_classified(self):
        self.assertEqual(check_declared_columns(self.schema, "manifest", read_csv(DEFAULT_SAMPLE)[0]), [])
        self.assertEqual(check_declared_columns(self.schema, "outcomes", read_csv(OUTCOMES)[0]), [])

    def test_undeclared_or_absent_column_is_rejected(self):
        columns = read_csv(OUTCOMES)[0]
        self.assertTrue(check_declared_columns(self.schema, "outcomes", columns + ["new_metric"]))
        self.assertTrue(check_declared_columns(self.schema, "outcomes", [c for c in columns if c != "score"]))

    def test_required_fields_have_no_missing_values(self):
        self.assertEqual(missing_values(manifest_rows(), required_columns(self.schema, "manifest")), {})
        self.assertEqual(missing_values(outcome_rows(), required_columns(self.schema, "outcomes")), {})

    def test_blank_required_value_is_reported(self):
        rows = manifest_rows()
        rows[3]["scenario_type"] = ""
        self.assertEqual(missing_values(rows, required_columns(self.schema, "manifest")), {"scenario_type": 1})
        self.assertTrue(any("scenario_type" in e for e in check_manifest(rows, self.schema)))

    def test_unavailable_fields_are_not_attached_to_committed_artifacts(self):
        for item in schema_fields(self.schema):
            if item["status"] == "unavailable":
                self.assertEqual(item["artifacts"], [], item["name"])


@needs_yaml
class LeakageTest(unittest.TestCase):
    def setUp(self):
        self.schema = load_schema(DEFAULT_SCHEMA)

    def test_committed_schema_has_no_leakage(self):
        self.assertEqual(check_leakage(self.schema), [])
        for item in model_features(self.schema):
            self.assertTrue(item["available_before_simulation"], item["name"])
            self.assertIn(item["role"], {"metadata", "covariate"}, item["name"])

    def test_outcomes_and_treatment_are_never_features(self):
        for group in ("post_simulation_outcomes", "planner_treatment"):
            for item in self.schema["groups"][group]["fields"]:
                self.assertFalse(item["model_feature"], item["name"])
        for item in self.schema["groups"]["post_simulation_outcomes"]["fields"]:
            self.assertFalse(item["available_before_simulation"], item["name"])

    def test_outcome_marked_as_feature_is_rejected(self):
        leaked = copy.deepcopy(self.schema)
        schema_field(leaked, "scenario_score")["model_feature"] = True
        self.assertTrue(any("scenario_score" in e for e in check_leakage(leaked)))

    def test_feature_matching_forbidden_pattern_is_rejected(self):
        leaked = copy.deepcopy(self.schema)
        schema_field(leaked, "ego_speed")["name"] = "ego_progress_speed"
        self.assertTrue(any("forbidden pattern" in e for e in check_leakage(leaked)))

    def test_feature_not_available_before_simulation_is_rejected(self):
        leaked = copy.deepcopy(self.schema)
        schema_field(leaked, "ego_speed")["available_before_simulation"] = False
        self.assertTrue(any("not available before simulation" in e for e in check_leakage(leaked)))

    def test_feature_table_rejects_outcomes_treatment_and_unknown_columns(self):
        allowed = ["scenario_id", "log_name", "map_name", "ego_speed", "ego_speed_missing", "num_vehicles"]
        self.assertEqual(check_feature_table_columns(self.schema, allowed), [])
        for leaked in ("score", "ego_is_comfortable", "planner", "succeeded", "ego_x", "mystery"):
            self.assertTrue(check_feature_table_columns(self.schema, allowed + [leaked]), leaked)


class DeterministicOrderingTest(unittest.TestCase):
    def test_log_folds_are_stable_and_input_order_independent(self):
        rows = manifest_rows()
        first = assign_log_folds(rows, folds=5, seed=0)
        shuffled = list(rows)
        random.Random(1).shuffle(shuffled)
        self.assertEqual(first, assign_log_folds(shuffled, folds=5, seed=0))
        self.assertEqual(first, assign_log_folds(outcome_rows(), folds=5, seed=0))
        self.assertEqual(list(first), sorted(first))

    def test_every_log_is_in_exactly_one_fold(self):
        rows = manifest_rows()
        assignment = assign_log_folds(rows, folds=5, seed=0)
        self.assertEqual(set(assignment), {row["log_name"] for row in rows})
        self.assertEqual(set(assignment.values()), set(range(5)))
        self.assertEqual(check_log_split(outcome_rows(), assignment), [])
        sizes = [sum(1 for row in rows if assignment[row["log_name"]] == k) for k in range(5)]
        self.assertLessEqual(max(sizes) - min(sizes), 4)

    def test_split_check_rejects_unassigned_logs(self):
        rows = manifest_rows()
        assignment = assign_log_folds(rows)
        assignment.pop(rows[0]["log_name"])
        self.assertTrue(check_log_split(rows, assignment))

    def test_fold_count_must_be_at_least_two(self):
        with self.assertRaises(ValueError):
            assign_log_folds(manifest_rows(), folds=1)

    def test_canonical_order_is_deterministic(self):
        outcomes = outcome_rows()
        self.assertEqual(outcomes, canonical_order(outcomes, ["scenario_id", "planner"]))
        shuffled = list(outcomes)
        random.Random(2).shuffle(shuffled)
        self.assertEqual(canonical_order(shuffled, ["scenario_id", "planner"]), outcomes)


@needs_yaml
class EndToEndTest(unittest.TestCase):
    def run_main(self, *args):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return main(list(args))

    def test_committed_subset_passes_every_required_check(self):
        report = validate(DEFAULT_SCHEMA)
        self.assertEqual(report.failed, [])
        self.assertEqual(self.run_main(), 0)

    def test_leaky_feature_table_fails_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "features.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                csv.writer(stream).writerow(["scenario_id", "log_name", "ego_speed", "score"])
            self.assertTrue(validate(DEFAULT_SCHEMA, features_path=path).failed)
            self.assertEqual(self.run_main("--features", str(path)), 1)

    def test_missing_schema_exits_with_input_error(self):
        self.assertEqual(self.run_main("--schema", "/nonexistent/schema.yaml"), 2)


if __name__ == "__main__":
    unittest.main()
