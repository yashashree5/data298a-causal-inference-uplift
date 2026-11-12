"""Sanity checks for the GitHub Actions CI workflow (298-43)."""

from pathlib import Path
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"


class CiWorkflowTest(unittest.TestCase):
    def setUp(self):
        self.doc = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
        self.job = self.doc["jobs"]["checks"]
        self.steps = {step["name"]: step for step in self.job["steps"] if "name" in step}

    def test_runs_on_linux_and_windows_for_prs_to_main(self):
        self.assertIn("main", self.doc["on"]["pull_request"]["branches"])
        self.assertIn("main", self.doc["on"]["push"]["branches"])
        self.assertEqual(["ubuntu-latest", "windows-latest"], self.job["strategy"]["matrix"]["os"])

    def test_runs_unit_tests_and_mini_68_validator(self):
        tests = self.steps["Unit tests"]["run"]
        self.assertIn("pytest", tests)
        self.assertIn("test_extract_map_features.py::AssemblyPropertyTests::test_missing_indicators_and_row_coverage",
                      tests)
        self.assertIn("test_extract_map_features.py::BuildCliTests::test_end_to_end_build", tests)
        self.assertIn("--deselect", tests)
        self.assertIn("python -m tools.validate_scenario_subset", self.steps["mini_68 validator"]["run"])

    def test_deselects_only_the_known_map_feature_failures(self):
        # Guard against silently growing the deselect list.
        deselected = [line.strip() for line in self.steps["Unit tests"]["run"].splitlines()
                      if "--deselect" in line]
        self.assertEqual(2, len(deselected), deselected)


if __name__ == "__main__":
    unittest.main()
