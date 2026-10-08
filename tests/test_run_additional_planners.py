import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import run_additional_planners as batch


class AdditionalPlannerBatchTest(unittest.TestCase):
    def run_batch(self, root, execute, wait):
        argv = ["batch", "--output-root", str(root), "--experiment", "full",
                "--smoke-experiment", "smoke", "--reference-root", "/reference"]
        with patch("sys.argv", argv), patch.object(batch, "execute", execute), \
                patch.object(batch, "wait_for_runs", wait):
            batch.main()

    def test_wait_then_three_smoke_full_pairs_then_assembly(self):
        events = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.run_batch(root, lambda command: events.append(command),
                           lambda runs, timeout: events.append("wait"))
            self.assertEqual(events[0], "wait")
            self.assertEqual(len(events), 8)
            for index, planner in enumerate(batch.PLANNERS):
                smoke, full = events[1 + index * 2:3 + index * 2]
                self.assertEqual(smoke[smoke.index("--planner") + 1], planner)
                self.assertIn("--limit", smoke)
                self.assertEqual(full[full.index("--planner") + 1], planner)
                self.assertNotIn("--limit", full)
            self.assertEqual(events[-1].count("--run"), 5)
            status = json.loads((root / "full/batch_status.json").read_text())
            self.assertEqual(status["stage"], "complete")
            self.assertEqual(status["expected_rows"], 340)

    def test_smoke_failure_prevents_full_runs_and_assembly(self):
        commands = []

        def fail(command):
            commands.append(command)
            raise RuntimeError("smoke failed")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "smoke failed"):
                self.run_batch(root, fail, lambda runs, timeout: None)
            self.assertEqual(len(commands), 1)
            status = json.loads((root / "full/batch_status.json").read_text())
            self.assertEqual(status["stage"], "failed")
            self.assertEqual(status["failed_stage"], "pdm-hybrid:smoke")

    def test_baseline_failure_prevents_all_new_runs(self):
        def fail(runs, timeout):
            raise ValueError("baseline invalid")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            commands = []
            with self.assertRaisesRegex(ValueError, "baseline invalid"):
                self.run_batch(root, commands.append, fail)
            self.assertFalse(commands)
            status = json.loads((root / "full/batch_status.json").read_text())
            self.assertEqual(status["failed_stage"], "waiting_for_baselines")
