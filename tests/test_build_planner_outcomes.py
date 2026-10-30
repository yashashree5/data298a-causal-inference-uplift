import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from tools.build_planner_outcomes import (
    MATCHED_CONFIG, exactly_one, parse_planner_runs, recorded_run_id, save_csv,
    validate_pair, wait_for_runs,
)
from causal_planner.data.outcomes import validate_model_identity


class PlannerOutcomesTest(unittest.TestCase):
    def setUp(self):
        manifest = dict(sample_sha256="hash", selected_scenarios=[{"scenario_id": "a"}],
                        scenario_count=1, nuplan_commit="nuplan", tuplan_garage_commit="pdm",
                        data_root="/data", maps_root="/data/maps", python="3.9")
        self.manifests = [manifest, copy.deepcopy(manifest)]
        config = {key: {"same": True} for key in MATCHED_CONFIG}
        config["metric_aggregator"] = {"weighted": {"file_name": "first", "weight": 5}}
        self.configs = [config, copy.deepcopy(config)]
        self.configs[1]["metric_aggregator"]["weighted"]["file_name"] = "second"
        self.environments = ["same packages", "same packages"]

    def test_same_experiment_accepts_different_output_filenames(self):
        validate_pair(self.manifests, self.configs, self.environments)

    def test_same_experiment_accepts_more_than_two_planners(self):
        manifests = self.manifests + [copy.deepcopy(self.manifests[0])]
        configs = self.configs + [copy.deepcopy(self.configs[0])]
        configs[2]["metric_aggregator"]["weighted"]["file_name"] = "third"
        validate_pair(manifests, configs, self.environments + ["same packages"])

    def test_late_planner_mismatch_is_not_ignored(self):
        manifests = self.manifests + [copy.deepcopy(self.manifests[0])]
        manifests[2]["sample_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "sample_sha256"):
            validate_pair(manifests, self.configs + [self.configs[0]], self.environments + ["same packages"])

    def test_two_ml_planners_cannot_be_confused(self):
        config = {"model": {"_target_": "tuplan_garage.planning.training.modeling.models.pgp.pgp_model.PGPModel"}}
        validate_model_identity("gc-pgp", config)
        with self.assertRaisesRegex(ValueError, "urban-driver"):
            validate_model_identity("urban-driver", config)

    def test_parse_repeatable_planner_runs(self):
        runs = parse_planner_runs([
            "idm=/runs/idm",
            "pdm-closed=/runs/pdm",
            "pdm-hybrid=/runs/hybrid",
            "urban-driver=/runs/urban",
            "gc-pgp=/runs/gc",
        ])
        self.assertEqual(5, len(runs))
        self.assertEqual(Path("/runs/gc"), runs["gc-pgp"])

    def test_parse_planner_runs_retains_legacy_pair(self):
        runs = parse_planner_runs([], Path("/runs/idm"), Path("/runs/pdm"))
        self.assertEqual(
            {"idm": Path("/runs/idm"), "pdm-closed": Path("/runs/pdm")}, runs
        )

    def test_parse_planner_runs_rejects_invalid_or_duplicate_input(self):
        with self.assertRaisesRegex(ValueError, "At least two"):
            parse_planner_runs(["idm=/runs/idm"])
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            parse_planner_runs(["idm=/runs/idm", "unknown=/runs/unknown"])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            parse_planner_runs(
                ["idm=/runs/other"], Path("/runs/idm"), Path("/runs/pdm")
            )
        with self.assertRaisesRegex(ValueError, "provided together"):
            parse_planner_runs([], Path("/runs/idm"), None)

    def test_run_id_survives_directory_relocation(self):
        self.assertEqual("original", recorded_run_id({"command": ["python", "experiment_uid=original"]}))
        self.assertEqual("new", recorded_run_id({"run_id": "new"}))
        with self.assertRaises(ValueError):
            recorded_run_id({"command": ["python"]})

    def test_rejects_mismatched_manifest_fields(self):
        for key in self.manifests[0]:
            with self.subTest(key=key):
                altered = copy.deepcopy(self.manifests)
                altered[1][key] = "different"
                with self.assertRaisesRegex(ValueError, key):
                    validate_pair(altered, self.configs, self.environments)

    def test_rejects_mismatched_simulation_settings(self):
        for key in MATCHED_CONFIG:
            with self.subTest(key=key):
                altered = copy.deepcopy(self.configs)
                altered[1][key] = {"same": False}
                with self.assertRaisesRegex(ValueError, key):
                    validate_pair(self.manifests, altered, self.environments)

    def test_rejects_mismatched_aggregation(self):
        self.configs[1]["metric_aggregator"]["weighted"]["weight"] = 4
        with self.assertRaisesRegex(ValueError, "aggregation"):
            validate_pair(self.manifests, self.configs, self.environments)

    def test_rejects_different_packages(self):
        with self.assertRaisesRegex(ValueError, "packages"):
            validate_pair(self.manifests, self.configs, ["a", "b"])

    def test_requires_exactly_one_matching_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                exactly_one(root, "*.parquet")
            first = root / "first.parquet"
            first.touch()
            self.assertEqual(first, exactly_one(root, "*.parquet"))
            (root / "second.parquet").touch()
            with self.assertRaises(ValueError):
                exactly_one(root, "*.parquet")

    def test_wait_checks_completion_and_rejects_failed_or_missing_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(TimeoutError):
                wait_for_runs([root], 0)
            (root / "result.json").write_text(json.dumps({"valid": False}))
            with self.assertRaises(ValueError):
                wait_for_runs([root], 0)
            (root / "result.json").write_text(json.dumps({"valid": True}))
            wait_for_runs([root], 0)

    def test_csv_export_omits_index_and_creates_parent_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "outcomes.csv"
            table = Mock()
            self.assertEqual(output, save_csv(table, output))
            self.assertTrue(output.parent.is_dir())
            table.to_csv.assert_called_once_with(output, index=False, mode="x", encoding="utf-8")

    def test_csv_export_preserves_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "outcomes.csv"
            output.write_text("existing data")
            table = Mock()
            with self.assertRaises(FileExistsError):
                save_csv(table, output)
            table.to_csv.assert_not_called()
            self.assertEqual("existing data", output.read_text())


if __name__ == "__main__":
    unittest.main()
