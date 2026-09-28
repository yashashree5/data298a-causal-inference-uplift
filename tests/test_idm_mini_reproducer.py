import tempfile
import unittest
from pathlib import Path

from tools.idm_mini_reproducer import DEFAULT_SAMPLE, build_command, load_scenarios


class IdmMiniReproducerTest(unittest.TestCase):
    def test_committed_sample_is_complete_and_unique(self) -> None:
        scenarios = load_scenarios(DEFAULT_SAMPLE)

        self.assertEqual(68, len(scenarios))
        self.assertEqual(14, len({row["scenario_type"] for row in scenarios}))
        self.assertEqual(68, len({row["scenario_id"] for row in scenarios}))

    def test_limit_returns_requested_prefix(self) -> None:
        scenarios = load_scenarios(DEFAULT_SAMPLE, limit=3)

        self.assertEqual(3, len(scenarios))

    def test_command_uses_idm_and_manifest_tokens(self) -> None:
        scenarios = load_scenarios(DEFAULT_SAMPLE, limit=2)
        command = build_command(
            scenarios=scenarios,
            devkit_root=Path("/opt/nuplan-devkit"),
            python_executable="python",
            run_root=Path("/artifacts/idm-mini/test"),
            experiment_uid="test",
        )

        self.assertIn("planner=idm_planner", command)
        self.assertIn("+simulation=closed_loop_reactive_agents", command)
        token_override = next(item for item in command if item.startswith("scenario_filter.scenario_tokens="))
        self.assertIn(scenarios[0]["scenario_token"], token_override)
        self.assertIn(scenarios[1]["scenario_token"], token_override)

    def test_invalid_manifest_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "invalid.csv"
            manifest.write_text("scenario_id,scenario_token\nexample,not-a-token\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "missing required columns"):
                load_scenarios(manifest)


if __name__ == "__main__":
    unittest.main()
