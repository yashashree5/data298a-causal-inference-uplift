import contextlib
import io
from pathlib import Path
import random
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.build_paired_dataset import (  # noqa: E402
    KEYS, OUTPUT_FILES, assess_quality, clean, extract, main, preprocess, reconstruct_score,
)
from tools.idm_mini_reproducer import DEFAULT_SAMPLE  # noqa: E402
from tools.validate_scenario_subset import DEFAULT_SCHEMA, check_feature_table_columns, load_schema, read_csv  # noqa: E402

try:
    import yaml  # noqa: F401
except ImportError:
    yaml = None

OUTCOMES = Path(__file__).resolve().parents[1] / "artifacts" / "mini_68" / "planner_outcomes.csv"


@unittest.skipUnless(yaml, "PyYAML is required to read the feature schema")
class PairedDatasetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = load_schema(DEFAULT_SCHEMA)

    def setUp(self):
        self.manifest = read_csv(DEFAULT_SAMPLE)[1]
        self.outcomes = read_csv(OUTCOMES)[1]
        self.sid = self.manifest[0]["scenario_id"]

    def row(self, planner="idm", sid=None):
        sid = sid or self.sid
        return next(r for r in self.outcomes if r["scenario_id"] == sid and r["planner"] == planner)

    def clean(self):
        return clean(self.manifest, self.outcomes, self.schema)

    def step(self, result, name):
        return next(s for s in result.steps if s["step"] == name)

    # ------------------------------------------------------------------------------ cleaning --

    def test_committed_data_needs_no_cleaning(self):
        result = self.clean()
        self.assertEqual(len(result.rows), 136)
        self.assertEqual(dict(result.excluded), {})
        self.assertEqual(result.fatal, [])
        self.assertIsInstance(result.rows[0]["score"], float)
        self.assertIsInstance(result.rows[0]["succeeded"], bool)

    def test_exact_duplicate_is_removed_without_exclusion(self):
        self.outcomes.append(dict(self.row()))
        result = self.clean()
        self.assertEqual(len(result.rows), 136)
        self.assertEqual(self.step(result, "remove exact duplicate rows")["rows_affected"], 1)
        self.assertEqual(dict(result.excluded), {})

    def test_conflicting_duplicate_excludes_the_pair(self):
        self.outcomes.append(dict(self.row(), score="0.123"))
        result = self.clean()
        self.assertIn(self.sid, result.excluded)
        self.assertEqual(len(result.rows), 134)

    def test_failed_simulation_excludes_both_planners(self):
        self.row("idm").update(succeeded="False", error_message="timeout")
        result = self.clean()
        self.assertTrue(any("simulation failed" in r for r in result.excluded[self.sid]))
        self.assertFalse(any(r["scenario_id"] == self.sid for r in result.rows))

    def test_missing_outcome_excludes_the_pair(self):
        self.row("pdm-closed")["score"] = ""
        result = self.clean()
        self.assertIn("pdm-closed: missing score", result.excluded[self.sid])

    def test_out_of_range_and_unparseable_values_are_treated_as_missing(self):
        self.row("idm")["score"] = "1.7"
        other = self.manifest[1]["scenario_id"]
        self.row("idm", other)["ego_is_comfortable"] = "abc"
        result = self.clean()
        self.assertEqual(self.step(result, "set metrics outside [0, 1] to missing")["rows_affected"], 1)
        self.assertEqual(self.step(result, "coerce types from schema")["rows_affected"], 1)
        self.assertEqual(set(result.excluded), {self.sid, other})

    def test_unknown_scenarios_and_planners_are_dropped(self):
        self.outcomes.append(dict(self.row(), scenario_id="unknown:0123456789abcdef"))
        self.outcomes.append(dict(self.row(), planner="pdm-open"))
        result = self.clean()
        self.assertEqual(self.step(result, "drop rows for unknown scenarios or planners")["rows_affected"], 2)
        self.assertEqual(len(result.rows), 136)

    def test_missing_planner_row_excludes_the_pair(self):
        self.outcomes.remove(self.row("pdm-closed"))
        result = self.clean()
        self.assertIn("no outcome row for pdm-closed", result.excluded[self.sid])

    def test_metadata_mismatch_excludes_the_pair(self):
        self.row("idm")["map_name"] = "us-ma-boston" if self.row()["map_name"] != "us-ma-boston" else "sg-one-north"
        self.assertTrue(any("map_name differs" in r for r in self.clean().excluded[self.sid]))

    def test_missing_required_identifier_is_fatal(self):
        self.row("idm")["run_id"] = ""
        self.assertTrue(any("run_id" in e for e in self.clean().fatal))

    # ------------------------------------------------------------------------- preprocessing --

    def test_paired_table_has_one_row_per_scenario(self):
        paired, features, splits, assignment = preprocess(self.manifest, self.clean().rows, self.schema)
        self.assertEqual(len(paired), 68)
        self.assertEqual([r["scenario_id"] for r in paired], sorted(r["scenario_id"] for r in self.manifest))
        for row in paired:
            self.assertAlmostEqual(row["score_delta"], row["pdm_score"] - row["idm_score"])
            self.assertEqual(row["idm_gate_failed"], row["idm_score"] == 0)
        outcomes = {k: sum(r["pair_outcome"] == k for r in paired) for k in ("pdm_better", "tie", "idm_better")}
        self.assertEqual(outcomes, {"pdm_better": 33, "tie": 30, "idm_better": 5})
        self.assertEqual(len(splits), 68)
        self.assertEqual(set(assignment), {r["log_name"] for r in self.manifest})

    def test_feature_table_contains_only_keys_and_model_features(self):
        _, features, _, _ = preprocess(self.manifest, self.clean().rows, self.schema)
        self.assertEqual(list(features[0]), list(KEYS) + ["map_name"])
        self.assertEqual(check_feature_table_columns(self.schema, list(features[0])), [])

    def test_preprocessing_is_deterministic(self):
        first = preprocess(self.manifest, self.clean().rows, self.schema)
        random.Random(3).shuffle(self.manifest)
        random.Random(4).shuffle(self.outcomes)
        second = preprocess(self.manifest, self.clean().rows, self.schema)
        self.assertEqual(first, second)

    def test_excluded_scenarios_are_not_in_the_paired_table(self):
        self.row("idm")["succeeded"] = "False"
        paired, _, splits, _ = preprocess(self.manifest, self.clean().rows, self.schema)
        self.assertEqual(len(paired), 67)
        self.assertNotIn(self.sid, {r["scenario_id"] for r in paired + splits})

    def test_score_reconstruction_matches_nuplan_formula(self):
        for row in self.clean().rows:
            self.assertAlmostEqual(reconstruct_score(row), row["score"], places=12)

    # ------------------------------------------------------------------------------- quality --

    def quality(self):
        sources = extract(self.schema)
        sources["manifest_rows"], sources["outcome_rows"] = self.manifest, self.outcomes
        cleaned = self.clean()
        paired, features, _, assignment = preprocess(self.manifest, cleaned.rows, self.schema)
        return assess_quality(self.schema, sources, cleaned, paired, features, assignment)

    def status(self, report, name):
        return next(c["status"] for c in report["checks"] if c["check"] == name)

    def test_committed_data_passes_quality_assessment(self):
        report = self.quality()
        self.assertTrue(report["passed"])
        self.assertNotIn("FAIL", {c["status"] for c in report["checks"]})
        self.assertEqual(report["metrics"]["rows"]["pairs"], 68)

    def test_exclusions_warn_and_fatal_errors_fail(self):
        self.row("idm")["succeeded"] = "False"
        report = self.quality()
        self.assertEqual(self.status(report, "cleaning: excluded scenarios"), "WARN")
        self.assertTrue(report["passed"])
        self.row("pdm-closed", self.manifest[2]["scenario_id"])["sample_sha256"] = ""
        self.assertFalse(self.quality()["passed"])

    # --------------------------------------------------------------------------------- output --

    def test_cli_writes_outputs_and_refuses_to_overwrite(self):
        quiet = contextlib.redirect_stdout(io.StringIO())
        with tempfile.TemporaryDirectory() as tmp, quiet, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["--output-dir", tmp]), 0)
            for name in OUTPUT_FILES:
                self.assertTrue((Path(tmp) / name).is_file(), name)
            self.assertEqual(main(["--output-dir", tmp]), 2)
            self.assertEqual(main(["--output-dir", tmp, "--overwrite"]), 0)


if __name__ == "__main__":
    unittest.main()
