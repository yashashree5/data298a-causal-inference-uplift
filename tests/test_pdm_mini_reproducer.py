import copy
from pathlib import Path
import unittest

from tools.idm_mini_reproducer import DEFAULT_SAMPLE, load_scenarios
from tools.pdm_mini_reproducer import build_command, validate_results


class PdmMiniReproducerTest(unittest.TestCase):
    def setUp(self):
        self.scenarios = load_scenarios(DEFAULT_SAMPLE, limit=2)
        self.reports = [dict(log_name=row["log_name"], scenario_name=row["scenario_token"],
                             planner_name="PDMClosedPlanner", succeeded=True) for row in self.scenarios]
        self.scores = [dict(log_name=row["log_name"], scenario=row["scenario_token"],
                            scenario_type=row["scenario_type"], planner_name="PDMClosedPlanner", score=0.8)
                       for row in self.scenarios]
        self.final = [dict(planner_name="PDMClosedPlanner", num_scenarios=2, score=0.8)]

    def validate(self):
        return validate_results(self.scenarios, self.reports, self.scores, self.final, "PDMClosedPlanner")

    def test_planners_share_evaluation_settings_and_exact_sample(self):
        args = (self.scenarios, Path("/opt/nuplan-devkit"), "python", Path("/artifacts/run"), "run")
        pdm = build_command(*args)
        idm = build_command(*args, planner="idm")
        self.assertEqual([item for item in pdm if not item.startswith("planner=")],
                         [item for item in idm if not item.startswith("planner=")])
        self.assertIn("planner=pdm_closed_planner", pdm)
        self.assertIn("planner=idm_planner", idm)
        self.assertIn("+simulation=closed_loop_reactive_agents", pdm)
        self.assertIn("scenario_builder=nuplan_mini", pdm)
        self.assertIn("seed=0", pdm)
        self.assertIn("worker=sequential", pdm)
        token_arg = next(arg for arg in pdm if arg.startswith("scenario_filter.scenario_tokens="))
        for row in self.scenarios:
            self.assertIn(row["scenario_token"], token_arg)
        self.assertIn("pkg://tuplan_garage.planning.script.config.simulation", pdm[-1])

    def test_complete_run_passes(self):
        self.assertTrue(self.validate()["valid"])
        self.assertEqual(0.8, self.validate()["official_score"])

    def test_zero_scores_are_valid_not_simulation_failures(self):
        for row in self.scores:
            row["score"] = 0.0
        self.final[0]["score"] = 0.0
        self.assertTrue(self.validate()["valid"])

    def test_missing_and_unexpected_scenarios_fail(self):
        self.scores[0]["scenario"] = "0000000000000000"
        result = self.validate()
        self.assertFalse(result["valid"])
        self.assertIn("unexpected=", " ".join(result["errors"]))

    def test_duplicate_scores_fail(self):
        self.scores.append(copy.deepcopy(self.scores[0]))
        self.assertFalse(self.validate()["valid"])

    def test_failed_simulation_fails_even_with_scores(self):
        self.reports[0]["succeeded"] = False
        self.assertFalse(self.validate()["valid"])
        self.assertEqual(1, self.validate()["failed"])

    def test_wrong_category_and_wrong_planner_fail(self):
        self.scores[0]["scenario_type"] = "wrong"
        self.reports[0]["planner_name"] = "IDMPlanner"
        self.assertFalse(self.validate()["valid"])

    def test_invalid_score_fails(self):
        self.scores[0]["score"] = float("nan")
        self.assertFalse(self.validate()["valid"])

    def test_bad_or_missing_aggregate_fails(self):
        self.final[0]["score"] = 0.9
        self.assertFalse(self.validate()["valid"])
        self.final = []
        self.assertFalse(self.validate()["valid"])


if __name__ == "__main__":
    unittest.main()
