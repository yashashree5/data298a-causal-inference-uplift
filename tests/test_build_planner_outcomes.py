import copy
import tempfile
import unittest
from pathlib import Path

from tools.build_planner_outcomes import MATCHED_CONFIG, exactly_one, validate_pair


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


if __name__ == "__main__":
    unittest.main()
