"""Tests for tools/build_planner_labels.py on synthetic outcome frames, sample CSVs and feature
parquet files in temp dirs. No simulation runs; no E: or docs/data access (every CLI call passes
--output-root, --sample, --features, --out-dir and --report-dir under a temp dir)."""
from pathlib import Path
import contextlib
import io
import json
import math
import random
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tools import build_planner_labels as bpl  # noqa: E402
from tools import validate_scenario_subset as validator  # noqa: E402
from causal_planner.simulation.runners import experiment_run_root  # noqa: E402

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "configs" / "features" / "first_experiment_schema.yaml"
SCHEMA = validator.load_schema(SCHEMA_PATH)  # read-only: split and forbidden_feature_patterns
SPLIT = dict(SCHEMA["split"])
ITERATIONS = 100
PLANNERS = list(bpl.PLANNER_ORDER)
OTHERS = [p for p in PLANNERS if p != bpl.CONTROL]
MAP = "us-pa-pittsburgh-hazelwood"
TYPES = ["changing_lane", "following_lane_with_lead", "starting_left_turn", "stationary"]
BASE_FEATURES = ["ego_speed", "num_vehicles", "nearest_agent_distance_m", "nearest_agent_distance_m_missing"]
CHECK_NAMES = ["unique_ids", "row_counts", "outcome_completeness", "delta_exact", "score_range",
               "feature_coverage", "column_separation", "log_folds"]
EXPERIMENT = "exp_test"


# ------------------------------------------------------------------------------------ helpers --

def random_planners(rng: random.Random, min_size: int = 1) -> list[str]:
    k = rng.randint(min_size - 1, len(OTHERS))
    return bpl.ordered([bpl.CONTROL] + rng.sample(OTHERS, k))


def make_sample(rng: random.Random, log_sizes: list[int]) -> pd.DataFrame:
    rows, tokens = [], set()
    for i, size in enumerate(log_sizes):
        log = f"2021.06.{i:02d}.12.00.00_veh-{i % 9}"
        for _ in range(size):
            tok = f"{rng.getrandbits(64):016x}"
            while tok in tokens:
                tok = f"{rng.getrandbits(64):016x}"
            tokens.add(tok)
            rows.append({"scenario_id": f"{log}:{tok}", "log_name": log, "scenario_token": tok,
                         "scenario_type": rng.choice(TYPES), "map_name": MAP})
    frame = pd.DataFrame(rows, columns=bpl.CONTEXT)
    return frame.sample(frac=1, random_state=rng.randrange(2**31)).reset_index(drop=True)


def random_sample(rng: random.Random, max_logs: int = 8) -> pd.DataFrame:
    return make_sample(rng, [rng.randint(1, 4) for _ in range(rng.randint(1, max_logs))])


def make_outcomes(rng: random.Random, sample: pd.DataFrame, planners: list[str]) -> pd.DataFrame:
    rows = []
    for record in sample.to_dict("records"):
        for planner in planners:
            rows.append({**record, "score": rng.random(), "succeeded": True, "error_message": None,
                         "planner": planner, "run_id": f"run-{bpl.token(planner)}"})
    frame = pd.DataFrame(rows)
    return frame.sample(frac=1, random_state=rng.randrange(2**31)).reset_index(drop=True)


def make_features(rng: random.Random, sample: pd.DataFrame, extras: list[str] = ()) -> pd.DataFrame:
    n = len(sample)
    data = {column: sample[column].tolist() for column in ["scenario_id", "log_name", "scenario_token", "map_name"]}
    data["ego_speed"] = [rng.uniform(0, 20) for _ in range(n)]
    data["num_vehicles"] = [rng.randint(0, 30) for _ in range(n)]
    distance = [None if rng.random() < 0.3 else rng.uniform(1, 80) for _ in range(n)]
    data["nearest_agent_distance_m"] = distance
    data["nearest_agent_distance_m_missing"] = [v is None for v in distance]
    for column in extras:
        data[column] = [rng.uniform(-1, 1) for _ in range(n)]
    frame = pd.DataFrame(data)
    return frame.sample(frac=1, random_state=rng.randrange(2**31)).reset_index(drop=True)


def provenance_for(planners: list[str], features: pd.DataFrame, failures: dict) -> dict:
    return {"planners": list(planners), "label_columns": bpl.label_columns(planners),
            "feature_columns": bpl.feature_columns(list(features.columns)), "excluded": failures}


def run_checks(outcomes, labels, modeling, sample, features, planners, failures, provenance=None):
    provenance = provenance_for(planners, features, failures) if provenance is None else provenance
    results = bpl.check_tables(outcomes, labels, modeling, sample, features["scenario_id"],
                               list(features.columns), provenance, planners, SCHEMA)
    return dict(results)


def build_all(rng: random.Random, sample=None, planners=None):
    sample = random_sample(rng) if sample is None else sample
    planners = random_planners(rng) if planners is None else planners
    outcomes = make_outcomes(rng, sample, planners)
    features = make_features(rng, sample)
    labels, failures = bpl.build_labels(outcomes, sample, planners)
    modeling = bpl.join_features(labels, features, SPLIT)
    return sample, planners, outcomes, features, labels, failures, modeling


# --------------------------------------------------------------------------------- properties --

class PropertyTests(unittest.TestCase):

    def test_property_1_label_faithfulness_and_delta_exactness(self):
        """Feature: planner-outcome-labels, Property 1: label faithfulness and delta exactness.

        **Validates: Requirements 2.2, 2.3, 5.3**
        """
        for seed in range(ITERATIONS):
            rng = random.Random(seed)
            sample, planners, outcomes, features, labels, failures, modeling = build_all(rng)
            # rows of a planner outside the subset must be ignored
            extra = [p for p in PLANNERS if p not in planners]
            if extra:
                outcomes_all = pd.concat([outcomes, make_outcomes(rng, sample, extra[:1])], ignore_index=True)
                outcomes_all = outcomes_all.sample(frac=1, random_state=seed).reset_index(drop=True)
                labels, failures = bpl.build_labels(outcomes_all, sample, planners)
            self.assertEqual(failures, {}, seed)
            self.assertEqual(sorted(labels["scenario_id"]), sorted(sample["scenario_id"]), seed)
            source = outcomes.set_index(["scenario_id", "planner"])
            for row in labels.to_dict("records"):
                sid = row["scenario_id"]
                for p in planners:
                    t = bpl.token(p)
                    self.assertEqual(row[f"score_{t}"], source.loc[(sid, p), "score"], seed)
                    self.assertEqual(row[f"succeeded_{t}"], bool(source.loc[(sid, p), "succeeded"]), seed)
                    if p != bpl.CONTROL:
                        self.assertEqual(row[f"delta_{t}_vs_idm"], row[f"score_{t}"] - row["score_idm"], seed)
            for column in labels.columns:
                if column.startswith(("score_", "delta_")):
                    self.assertEqual(labels[column].dtype, np.float64, (seed, column))
            checks = run_checks(outcomes, labels, modeling, sample, features, planners, failures)
            self.assertEqual(checks["delta_exact"], [], seed)

            deltas = [c for c in labels.columns if c.startswith("delta_")]
            if deltas:
                column, i = rng.choice(deltas), rng.randrange(len(labels))
                bad = labels.copy()
                bad.loc[i, column] = np.nextafter(bad.loc[i, column], math.inf)
                checks = run_checks(outcomes, bad, modeling, sample, features, planners, failures)
                self.assertTrue(checks["delta_exact"], seed)
                self.assertIn(bad.loc[i, "scenario_id"], checks["delta_exact"][0], seed)

    def test_property_2_best_planner_consistency(self):
        """Feature: planner-outcome-labels, Property 2: best planner consistency.

        **Validates: Requirements 2.4**
        """
        tol = bpl.TIE_TOLERANCE
        for seed in range(ITERATIONS):
            rng = random.Random(seed)
            planners = random_planners(rng)
            scores = {p: rng.random() for p in planners}
            top = max(scores.values())
            for p in planners:  # deliberate ties: exact, inside and just outside the tolerance
                if rng.random() < 0.5:
                    scores[p] = top - rng.choice([0.0, rng.uniform(0, 0.9 * tol), rng.uniform(1.1 * tol, 10 * tol)])
            best = bpl.best_planner(scores, planners)
            by_token = {bpl.token(p): p for p in planners}
            self.assertIn(best, by_token, seed)
            limit = max(scores.values()) - tol
            self.assertGreaterEqual(scores[by_token[best]], limit, seed)
            for p in PLANNERS[:PLANNERS.index(by_token[best])]:
                if p in scores:
                    self.assertLess(scores[p], limit, (seed, p))

    def test_property_3_failed_scenarios_are_excluded(self):
        """Feature: planner-outcome-labels, Property 3: failed scenarios are excluded.

        **Validates: Requirements 2.5, 3.1, 3.2**
        """
        for seed in range(ITERATIONS):
            rng = random.Random(seed)
            sample = random_sample(rng)
            planners = random_planners(rng)
            outcomes = make_outcomes(rng, sample, planners)
            ids = sorted(sample["scenario_id"])
            failed = rng.sample(ids, rng.randint(0, len(ids)))
            expected, drop = {}, []
            for sid in failed:
                for p in rng.sample(planners, rng.randint(1, len(planners))):
                    mask = (outcomes["scenario_id"] == sid) & (outcomes["planner"] == p)
                    mode = rng.choice(["failed", "null", "drop"])
                    if mode == "failed":
                        message = rng.choice([None, "", "boom|x"])
                        outcomes.loc[mask, "succeeded"] = False
                        outcomes.loc[mask, "error_message"] = message
                        reason = f"{p}: simulation failed ({message or 'no message'})"
                    elif mode == "null":
                        outcomes.loc[mask, "score"] = math.nan
                        reason = f"{p}: missing score"
                    else:
                        drop.extend(outcomes.index[mask])
                        reason = f"{p}: no row"
                    expected.setdefault(sid, {})[p] = reason
            outcomes = outcomes.drop(index=drop).reset_index(drop=True)
            before = outcomes.copy()
            labels, failures = bpl.build_labels(outcomes, sample, planners)
            pd.testing.assert_frame_equal(outcomes, before)
            self.assertEqual(list(labels["scenario_id"]), sorted(set(ids) - set(failed)), seed)
            self.assertEqual(set(failures), set(failed), seed)
            for sid, reasons in expected.items():
                self.assertEqual(failures[sid], [reasons[p] for p in planners if p in reasons], (seed, sid))

    def test_property_4_coverage_gaps_are_reported(self):
        """Feature: planner-outcome-labels, Property 4: coverage gaps are reported.

        **Validates: Requirements 4.2, 5.4**
        """
        for seed in range(ITERATIONS):
            rng = random.Random(seed)
            sample, planners, outcomes, features, labels, failures, modeling = build_all(rng)
            ids = list(labels["scenario_id"])
            removed = rng.sample(ids, rng.randint(1, len(ids)))
            partial = features[~features["scenario_id"].isin(removed)].reset_index(drop=True)
            with self.assertRaises(bpl.CoverageError) as caught:
                bpl.join_features(labels, partial, SPLIT)
            self.assertEqual(caught.exception.absent, len(removed), seed)
            self.assertEqual(caught.exception.examples, sorted(removed)[:bpl.ABSENT_EXAMPLES], seed)
            checks = run_checks(outcomes, labels, modeling, sample, partial, planners, failures,
                                provenance_for(planners, features, failures))
            self.assertTrue(checks["feature_coverage"], seed)
            self.assertIn(f"{len(removed)} modeling scenario_id absent", checks["feature_coverage"][0], seed)

    def test_property_5_folds_are_disjoint_by_log(self):
        """Feature: planner-outcome-labels, Property 5: folds are disjoint by log.

        **Validates: Requirements 4.4, 5.6**
        """
        for seed in range(ITERATIONS):
            rng = random.Random(seed)
            sizes = [rng.randint(1, 6) for _ in range(rng.randint(2, 12))]
            sizes[rng.randrange(len(sizes))] = rng.randint(2, 6)  # at least one multi-scenario log
            sample = make_sample(rng, sizes)
            sample, planners, outcomes, features, labels, failures, modeling = build_all(rng, sample=sample)
            self.assertEqual(modeling["fold"].dtype, np.int64, seed)
            folds_per_log = modeling.groupby("log_name")["fold"].nunique()
            self.assertTrue((folds_per_log == 1).all(), seed)
            rows = modeling[["scenario_id", "log_name"]].to_dict("records")
            assignment = validator.assign_log_folds(rows, SPLIT["folds"], SPLIT["seed"])
            self.assertEqual(modeling["fold"].tolist(), modeling["log_name"].map(assignment).tolist(), seed)
            self.assertTrue(set(modeling["fold"]) <= set(range(SPLIT["folds"])), seed)
            checks = run_checks(outcomes, labels, modeling, sample, features, planners, failures)
            self.assertEqual(checks["log_folds"], [], seed)

            sizes_by_log = modeling["log_name"].value_counts()
            log = rng.choice(sorted(sizes_by_log[sizes_by_log > 1].index))
            i = rng.choice(list(modeling.index[modeling["log_name"] == log]))
            moved = modeling.copy()
            moved.loc[i, "fold"] = (moved.loc[i, "fold"] + rng.randint(1, SPLIT["folds"] - 1)) % SPLIT["folds"]
            checks = run_checks(outcomes, labels, moved, sample, features, planners, failures)
            self.assertTrue(any(log in error for error in checks["log_folds"]), seed)

    def test_property_6_label_and_feature_columns_are_separate(self):
        """Feature: planner-outcome-labels, Property 6: label and feature columns are separate.

        **Validates: Requirements 2.1, 4.1, 4.5, 5.5**
        """
        patterns = list(SCHEMA["forbidden_feature_patterns"])
        for seed in range(ITERATIONS):
            rng = random.Random(seed)
            sample = random_sample(rng)
            planners = random_planners(rng)
            outcomes = make_outcomes(rng, sample, planners)
            extras = [f"f{i}_x" for i in rng.sample(range(50), rng.randint(0, 5))]
            features = make_features(rng, sample, extras)
            labels, failures = bpl.build_labels(outcomes, sample, planners)
            modeling = bpl.join_features(labels, features, SPLIT)
            feature_cols = bpl.feature_columns(list(features.columns))
            self.assertEqual(list(labels.columns), bpl.CONTEXT + bpl.label_columns(planners), seed)
            self.assertEqual(list(modeling.columns),
                             list(labels.columns) + [c for c in feature_cols if c not in labels.columns] + ["fold"],
                             seed)
            self.assertEqual(len(modeling), len(labels), seed)
            provenance = provenance_for(planners, features, failures)
            self.assertFalse(set(provenance["label_columns"]) & set(provenance["feature_columns"]), seed)
            checks = run_checks(outcomes, labels, modeling, sample, features, planners, failures, provenance)
            self.assertEqual(checks["column_separation"], [], seed)

            kind = rng.choice(["prefix", "best", "pattern"])
            if kind == "prefix":
                bad = rng.choice(bpl.LABEL_PREFIXES) + f"thing{rng.randint(0, 9)}"
            elif kind == "best":
                bad = bpl.BEST_PLANNER
            else:
                bad = f"pre_{rng.choice(patterns)}_post"
            bad_features = features.copy()
            bad_features[bad] = [rng.random() for _ in range(len(bad_features))]
            bad_modeling = bpl.join_features(labels, bad_features, SPLIT)
            checks = run_checks(outcomes, labels, bad_modeling, sample, bad_features, planners, failures)
            self.assertTrue(checks["column_separation"], (seed, bad))
            # stale provenance (built without the new column) is reported too
            checks = run_checks(outcomes, labels, bad_modeling, sample, bad_features, planners, failures,
                                provenance)
            self.assertTrue(checks["column_separation"], (seed, bad))


# ----------------------------------------------------------------------------------- examples --

class PlannerAndCheckExamples(unittest.TestCase):

    def test_parse_planners(self):
        self.assertEqual(bpl.parse_planners("pdm-closed, idm"), ["idm", "pdm-closed"])
        self.assertEqual(bpl.parse_planners(",".join(reversed(PLANNERS))), PLANNERS)
        for value, offending in [("pdm-closed", "idm"), ("idm,foo", "'foo'"), ("idm,idm", "'idm'"),
                                 ("idm,pdm-closed,pdm-closed", "'pdm-closed'"), ("", "''")]:
            with self.assertRaises(bpl.InputError, msg=value) as caught:
                bpl.parse_planners(value)
            self.assertIn(offending, str(caught.exception))

    def test_label_columns(self):
        self.assertEqual(bpl.label_columns(["pdm-closed", "idm"]),
                         ["score_idm", "score_pdm_closed", "delta_pdm_closed_vs_idm", "succeeded_idm",
                          "succeeded_pdm_closed", "best_planner"])
        five = bpl.label_columns(PLANNERS)
        self.assertEqual(sum(c.startswith("score_") for c in five), 5)
        self.assertEqual(sum(c.startswith("delta_") for c in five), 4)
        self.assertEqual(sum(c.startswith("succeeded_") for c in five), 5)

    def test_identity_mismatch(self):
        rng = random.Random(1)
        sample, planners, outcomes, features, labels, failures, modeling = build_all(
            rng, sample=make_sample(rng, [3, 3]), planners=["idm", "pdm-closed"])
        bad = features.copy()
        bad.loc[0, "map_name"] = "us-ma-boston"
        bad.loc[1, "scenario_token"] = "0" * 16
        with self.assertRaises(bpl.CoverageError) as caught:
            bpl.join_features(labels, bad, SPLIT)
        self.assertEqual(caught.exception.mismatched, 2)
        self.assertIn("2 rows with mismatched", str(caught.exception))

    def test_duplicate_ids_and_row_counts(self):
        rng = random.Random(2)
        sample, planners, outcomes, features, labels, failures, modeling = build_all(
            rng, sample=make_sample(rng, [2, 3]), planners=["idm", "pdm-closed"])
        checks = run_checks(outcomes, labels, modeling, sample, features, planners, failures)
        self.assertEqual({name: errors for name, errors in checks.items() if errors}, {})

        dup_labels = pd.concat([labels, labels.iloc[[0]]], ignore_index=True)
        dup_modeling = pd.concat([modeling, modeling.iloc[[0]]], ignore_index=True)
        checks = run_checks(outcomes, dup_labels, dup_modeling, sample, features, planners, failures)
        self.assertEqual(len(checks["unique_ids"]), 2)
        self.assertIn(labels.loc[0, "scenario_id"], checks["unique_ids"][0])
        self.assertTrue(any("label table has 6 rows, expected 5" in e for e in checks["row_counts"]))

        sid = labels.loc[0, "scenario_id"]
        short = outcomes[~((outcomes["scenario_id"] == sid) & (outcomes["planner"] == "pdm-closed"))]
        checks = run_checks(short, labels, modeling, sample, features, planners, failures)
        self.assertTrue(any("outcome table has 9 rows, expected 10" in e for e in checks["row_counts"]))
        self.assertTrue(any("pdm-closed" in e and sid in e for e in checks["outcome_completeness"]))

        out_of_range = labels.copy()
        out_of_range.loc[0, "score_idm"] = 1.5
        checks = run_checks(outcomes, out_of_range, modeling, sample, features, planners, failures)
        self.assertTrue(any("score_idm" in e for e in checks["score_range"]))


# ------------------------------------------------------------------------------------------ CLI --

class CliTests(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.rng = random.Random(7)
        self.sample = make_sample(self.rng, [3, 2, 2, 1])
        self.sample_path = self.root / "sample.csv"
        self.sample.to_csv(self.sample_path, index=False)
        self.features = make_features(self.rng, self.sample, ["f1_x"])
        self.features_path = self.root / "scenario_features.parquet"
        self.features.to_parquet(self.features_path, index=False)
        self.output_root = self.root / "runs"
        self.out_dir = self.root / "labels"
        self.report_dir = self.root / "report"
        self.planners = ["idm", "pdm-closed"]
        self.outcomes = make_outcomes(self.rng, self.sample, self.planners)
        self.sources = {"idm": {"official_score": 0.61}, "pdm-closed": {"official_score": 0.83}}
        self.calls = []

    def tearDown(self):
        self._tmp.cleanup()

    def argv(self, command, *extra, planners="idm,pdm-closed", features=None):
        argv = [command, "--output-root", str(self.output_root), "--experiment", EXPERIMENT,
                "--sample", str(self.sample_path), "--features", str(features or self.features_path),
                "--out-dir", str(self.out_dir), "--report-dir", str(self.report_dir),
                "--schema", str(SCHEMA_PATH)]
        if command == "build" and planners is not None:
            argv += ["--planners", planners]
        return argv + list(extra)

    def fake_build_table(self, runs, sample):
        self.calls.append((dict(runs), Path(sample)))
        return self.outcomes.copy(), self.sources

    def run_main(self, argv, side_effect=None):
        out, err = io.StringIO(), io.StringIO()
        with mock.patch("tools.build_planner_labels.build_table_for_runs",
                        side_effect=side_effect or self.fake_build_table), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = bpl.main(argv)
        return code, out.getvalue(), err.getvalue()

    @property
    def experiment_dir(self):
        return self.output_root / EXPERIMENT

    def written(self):
        """Files under the output root, out dir and report dir (inputs live elsewhere)."""
        return sorted(p for d in (self.output_root, self.out_dir, self.report_dir)
                      for p in d.rglob("*") if p.is_file())

    def test_build_passes_run_roots_and_writes_outcomes_unchanged(self):
        code, out, err = self.run_main(self.argv("build", planners="pdm-closed,idm"))
        self.assertEqual(code, 0, err)
        self.assertEqual(len(self.calls), 1)
        runs, sample = self.calls[0]
        self.assertEqual(list(runs), self.planners)
        for planner, path in runs.items():
            self.assertEqual(path, bpl.long_path(
                experiment_run_root(self.output_root, EXPERIMENT, planner)))
            self.assertEqual(bpl.strip_long_prefix(str(path)), str(
                experiment_run_root(self.output_root, EXPERIMENT, planner).resolve()))
        self.assertEqual(sample, self.sample_path)
        written = pd.read_parquet(self.experiment_dir / "planner_outcomes.parquet")
        pd.testing.assert_frame_equal(written, self.outcomes, check_dtype=False)
        csv = pd.read_csv(self.experiment_dir / "planner_outcomes.csv")
        self.assertEqual(len(csv), len(self.outcomes))
        self.assertEqual(list(csv.columns), list(self.outcomes.columns))
        self.assertEqual(list(self.root.rglob("*.tmp")), [])

    def test_build_table_error_exits_1_and_writes_nothing(self):
        code, out, err = self.run_main(self.argv("build"), side_effect=RuntimeError("run incomplete"))
        self.assertEqual(code, 1)
        self.assertIn("run incomplete", err)
        self.assertEqual(self.written(), [])

    def test_bad_planners_exit_2(self):
        for value in ["pdm-closed", "idm,bogus", "idm,idm"]:
            code, out, err = self.run_main(self.argv("build", planners=value))
            self.assertEqual(code, 2, value)
            self.assertIn("--planners", err)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.written(), [])

    def test_build_then_check_pass_and_provenance(self):
        code, out, err = self.run_main(self.argv("build"))
        self.assertEqual(code, 0, err)
        for path in (self.out_dir / bpl.LABELS_FILE, self.out_dir / bpl.MODELING_FILE):
            provenance = bpl.read_provenance(path)
            for key in ("tool_version", "built_utc", "planners", "run_roots", "label_columns",
                        "feature_columns", "sha256", "excluded", "split"):
                self.assertIn(key, provenance, (path, key))
            self.assertEqual(provenance["tool_version"], bpl.TOOL_VERSION)
            self.assertEqual(provenance["planners"], self.planners)
            self.assertEqual(provenance["label_columns"], bpl.label_columns(self.planners))
            self.assertEqual(provenance["feature_columns"], bpl.feature_columns(list(self.features.columns)))
            self.assertEqual(set(provenance["sha256"]), {"sample", "features", "schema"})
            self.assertEqual(provenance["sha256"]["sample"], validator.sha256_file(self.sample_path))
            self.assertEqual(provenance["run_roots"]["pdm-closed"],
                             str(experiment_run_root(self.output_root, EXPERIMENT, "pdm-closed")))
            self.assertEqual(provenance["split"], {"folds": 5, "seed": 0, "group": "log_name"})
        modeling = pd.read_parquet(self.out_dir / bpl.MODELING_FILE)
        self.assertEqual(len(modeling), len(self.sample))
        self.assertIn("f1_x", modeling.columns)
        self.assertEqual(list(self.root.rglob("*.tmp")), [])

        code, out, err = self.run_main(self.argv("check"))
        self.assertEqual(code, 0, out + err)
        self.assertEqual(out.splitlines(), [f"PASS {name}" for name in CHECK_NAMES])

    def test_check_fails_on_corrupted_delta_and_missing_file(self):
        self.assertEqual(self.run_main(self.argv("build"))[0], 0)
        path = self.out_dir / bpl.LABELS_FILE
        provenance = bpl.read_provenance(path)
        labels = pd.read_parquet(path)
        labels.loc[0, "delta_pdm_closed_vs_idm"] += 0.25
        bpl.write_parquet_atomic(labels, path, provenance)
        code, out, err = self.run_main(self.argv("check"))
        self.assertEqual(code, 1)
        lines = out.splitlines()
        self.assertEqual(len(lines), len(CHECK_NAMES))
        self.assertTrue(any(line.startswith("FAIL delta_exact:") for line in lines))
        self.assertIn("PASS log_folds", lines)

        (self.out_dir / bpl.MODELING_FILE).unlink()
        code, out, err = self.run_main(self.argv("check"))
        self.assertEqual(code, 2)
        self.assertIn(bpl.MODELING_FILE, err)

    def test_existing_output_requires_overwrite(self):
        self.report_dir.mkdir(parents=True)
        existing = self.report_dir / "planner_labels.md"
        existing.write_text("old\n", encoding="utf-8")
        code, out, err = self.run_main(self.argv("build"))
        self.assertEqual(code, 2)
        self.assertIn(str(existing), err)
        self.assertEqual(self.calls, [])
        self.assertEqual(existing.read_text(encoding="utf-8"), "old\n")
        self.assertFalse((self.out_dir / bpl.LABELS_FILE).exists())

        code, out, err = self.run_main(self.argv("build", "--overwrite"))
        self.assertEqual(code, 0, err)
        self.assertNotEqual(existing.read_text(encoding="utf-8"), "old\n")
        code, out, err = self.run_main(self.argv("build"))
        self.assertEqual(code, 2)
        self.assertIn("--overwrite", err)

    def test_coverage_gap_exits_1_without_modeling_table(self):
        missing = sorted(self.sample["scenario_id"])[:2]
        partial = self.features_path.with_name("partial_features.parquet")
        self.features[~self.features["scenario_id"].isin(missing)].to_parquet(partial, index=False)
        code, out, err = self.run_main(self.argv("build", features=partial))
        self.assertEqual(code, 1)
        self.assertIn("2 label scenario_id absent", err)
        self.assertIn(missing[0], err)
        self.assertEqual(self.written(), [])

        mismatched = self.root / "mismatch_features.parquet"
        bad = self.features.copy()
        bad.loc[0, "log_name"] = "other_log"
        bad.to_parquet(mismatched, index=False)
        code, out, err = self.run_main(self.argv("build", features=mismatched))
        self.assertEqual(code, 1)
        self.assertIn("1 rows with mismatched", err)
        self.assertFalse((self.out_dir / bpl.MODELING_FILE).exists())

    def test_report_contents_with_failed_scenario(self):
        failed_id = sorted(self.sample["scenario_id"])[0]
        mask = (self.outcomes["scenario_id"] == failed_id) & (self.outcomes["planner"] == "pdm-closed")
        self.outcomes.loc[mask, "succeeded"] = False
        self.outcomes.loc[mask, "error_message"] = "planner crashed"
        code, out, err = self.run_main(self.argv("build"))
        self.assertEqual(code, 0, err)
        n = len(self.sample)

        # failed scenario keeps its outcome rows but leaves labels and modeling table
        self.assertEqual(len(pd.read_parquet(self.experiment_dir / "planner_outcomes.parquet")), 2 * n)
        labels = pd.read_parquet(self.out_dir / bpl.LABELS_FILE)
        modeling = pd.read_parquet(self.out_dir / bpl.MODELING_FILE)
        self.assertNotIn(failed_id, set(labels["scenario_id"]))
        self.assertNotIn(failed_id, set(modeling["scenario_id"]))

        report = json.loads((self.report_dir / "planner_labels.json").read_text(encoding="utf-8"))
        self.assertEqual(report["rows"], {"outcomes": 2 * n, "labels": n - 1, "modeling": n - 1, "sample": n})
        self.assertEqual(report["planners"], ["idm", "pdm_closed"])
        self.assertEqual(report["scores"]["idm"]["official"], 0.61)
        self.assertEqual(report["scores"]["pdm_closed"]["official"], 0.83)
        self.assertAlmostEqual(report["scores"]["idm"]["mean"],
                               self.outcomes.loc[self.outcomes["planner"] == "idm", "score"].mean())
        delta = report["deltas"]["delta_pdm_closed_vs_idm"]
        values = labels["delta_pdm_closed_vs_idm"]
        self.assertAlmostEqual(delta["mean"], values.mean())
        self.assertAlmostEqual(delta["median"], values.median())
        self.assertAlmostEqual(delta["std"], values.std())
        self.assertAlmostEqual(delta["share_positive"], float((values > 0).mean()))
        self.assertEqual(sum(report["best_planner_counts"].values()), n - 1)
        self.assertEqual(report["best_planner_counts"],
                         {t: int((labels["best_planner"] == t).sum()) for t in ["idm", "pdm_closed"]})
        self.assertEqual(report["failed"], {failed_id: ["pdm-closed: simulation failed (planner crashed)"]})
        self.assertEqual(sum(f["rows"] for f in report["fold_sizes"].values()), n - 1)
        self.assertEqual(report["coverage"]["label_ids"], n - 1)
        self.assertEqual(report["coverage"]["in_features"], n - 1)
        self.assertEqual(report["coverage"]["absent"], 0)

        markdown = (self.report_dir / "planner_labels.md").read_text(encoding="utf-8")
        for text in ("# Planner labels", "## Scores", "## Deltas against IDM", "## Best planner",
                     "## Excluded scenarios", "## Folds", "## Feature coverage", failed_id,
                     "planner crashed", "`delta_pdm_closed_vs_idm`", "0.8300"):
            self.assertIn(text, markdown)
        self.assertEqual(list(self.root.rglob("*.tmp")), [])

        code, out, err = self.run_main(self.argv("check"))
        self.assertEqual(code, 0, out + err)


if __name__ == "__main__":
    unittest.main()
