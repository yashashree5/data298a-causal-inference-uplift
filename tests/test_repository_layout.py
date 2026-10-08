"""Compatibility checks for the source-package migration."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class RepositoryLayoutTest(unittest.TestCase):
    def run_python(self, args, cwd=ROOT, env=None):
        return subprocess.run(
            [sys.executable, *args], cwd=cwd, env=env,
            capture_output=True, text=True, timeout=20,
        )

    def test_package_imports_without_tools_or_simulation_dependencies(self):
        env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
        script = """
import sys
from causal_planner.data.scenarios import DEFAULT_SAMPLE, load_scenarios
from causal_planner.data import outcomes, features, dataset
from causal_planner.simulation import runners, validation
assert len(load_scenarios(DEFAULT_SAMPLE)) == 68
assert not any(name == 'tools' or name.startswith('tools.') for name in sys.modules)
assert 'nuplan' not in sys.modules
assert 'pandas' not in sys.modules
"""
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_python(["-c", script], cwd=directory, env=env)
        self.assertEqual(0, result.returncode, result.stderr)

    def test_existing_direct_and_module_commands_still_have_help(self):
        for tool in ("idm_mini_reproducer", "pdm_mini_reproducer", "build_planner_outcomes"):
            for args in ([f"tools/{tool}.py", "--help"], ["-m", f"tools.{tool}", "--help"]):
                with self.subTest(args=args):
                    result = self.run_python(args)
                    self.assertEqual(0, result.returncode, result.stderr)
                    self.assertIn("--sample", result.stdout)

    def test_direct_dry_runs_resolve_manifest_outside_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            for tool, extra in (
                ("idm_mini_reproducer", []),
                ("pdm_mini_reproducer", []),
                ("pdm_mini_reproducer", ["--planner", "idm"]),
            ):
                with self.subTest(tool=tool, extra=extra):
                    result = self.run_python(
                        [str(ROOT / "tools" / f"{tool}.py"), "--dry-run", *extra],
                        cwd=directory,
                    )
                    self.assertEqual(0, result.returncode, result.stderr)
                    self.assertIn("Validated 68 scenarios", result.stdout)
                    self.assertIn("scenario_builder=nuplan_mini", result.stdout)
                    self.assertIn("seed=0", result.stdout)

    def test_scaffolds_cannot_claim_success_or_write_datasets(self):
        with tempfile.TemporaryDirectory() as directory:
            for tool in ("extract_scenario_features", "build_model_dataset"):
                with self.subTest(tool=tool):
                    command = [str(ROOT / "tools" / f"{tool}.py")]
                    help_result = self.run_python([*command, "--help"], cwd=directory)
                    self.assertEqual(0, help_result.returncode, help_result.stderr)
                    result = self.run_python(command, cwd=directory)
                    self.assertEqual(2, result.returncode)
                    self.assertIn("not implemented yet", result.stderr)
                    self.assertEqual([], list(Path(directory).iterdir()))


if __name__ == "__main__":
    unittest.main()
