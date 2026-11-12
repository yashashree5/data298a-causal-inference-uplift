"""Tests for tools/train_baseline.py on synthetic modeling-table parquet files in temp dirs.

Every table carries a `planner_labels_provenance` entry. No E: or docs/models access: every CLI call
passes --table, --out-dir, --report-dir (train) and --schema under a temp dir, and the real feature
schema is read only. Skipped when scikit-learn is not importable."""
from pathlib import Path
import contextlib
import importlib.util
import io
import json
import math
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tools import train_baseline as tb  # noqa: E402
from tools import build_planner_labels as bpl  # noqa: E402
from tools import validate_scenario_subset as validator  # noqa: E402

HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None
SCHEMA_PATH = Path(__file__).resolve().parents[1] / "configs" / "features" / "first_experiment_schema.yaml"
SCHEMA = validator.load_schema(SCHEMA_PATH)  # read-only: forbidden_feature_patterns
PATTERNS = [str(p) for p in SCHEMA["forbidden_feature_patterns"]]
ITERATIONS = 100
PLANNERS = list(bpl.PLANNER_ORDER)
OTHERS = [p for p in PLANNERS if p != tb.CONTROL]
TYPES = ["changing_lane", "following_lane_with_lead", "starting_left_turn", "stationary"]
MAPS = ["us-pa-pittsburgh-hazelwood", "us-nv-las-vegas-strip"]
FEATURES = ["map_name", "lane_type", "traffic_light_state", "weather_code", "in_intersection",
            "ego_speed", "ego_speed_standardized", "ego_acceleration", "num_vehicles",
            "num_tracked_agents", "nearest_agent_distance_m", "nearest_agent_distance_m_missing",
            "nearest_agent_relative_speed_mps"]
CHECK_NAMES = ["oof_coverage", "unique_predictions", "fold_and_truth", "metrics_recompute",
               "feature_separation", "table_hash"]
EXPERIMENT = "exp_test"
FAST = ["--no-importance", "--bootstrap-resamples", "20"]


# ------------------------------------------------------------------------------------ helpers --

def make_table(rng: np.random.Generator, log_sizes, planners, n_folds, extras=(),
               planted: bool = False) -> pd.DataFrame:
    """Synthetic Modeling_Table: whole logs assigned to `n_folds` folds (needs len(log_sizes) >=
    n_folds), nulls in nearest_agent_*, lane_type, weather_code and in_intersection, and
    score_/delta_/succeeded_/best_planner label columns for `planners`."""
    order = rng.permutation(len(log_sizes))
    log_fold = {int(log): i % n_folds for i, log in enumerate(order)}
    logs, folds = [], []
    for i, size in enumerate(log_sizes):
        logs += [f"2021.06.{i:02d}.12.00.00_veh-{i % 9}"] * size
        folds += [log_fold[i]] * size
    n = len(logs)
    tokens = [f"{i:04x}{int(rng.integers(0, 2**48)):012x}" for i in range(n)]
    data = {
        "scenario_id": [f"{log}:{tok}" for log, tok in zip(logs, tokens)],
        "log_name": logs, "scenario_token": tokens,
        "scenario_type": rng.choice(TYPES, n).tolist(),
        "map_name": rng.choice(MAPS, n).tolist(),
        "lane_type": [None if rng.random() < 0.2 else str(rng.choice(["lane", "lane_connector"]))
                      for _ in range(n)],
        "traffic_light_state": rng.choice(["green", "red", "none", "unknown"], n).tolist(),
        "weather_code": pd.array([None if rng.random() < 0.15 else int(rng.choice([0, 3, 61]))
                                  for _ in range(n)], dtype="Int64"),
        "in_intersection": pd.array([None if rng.random() < 0.15 else bool(rng.random() < 0.4)
                                     for _ in range(n)], dtype="boolean"),
    }
    speed = rng.uniform(0, 20, n)
    data["ego_speed"] = speed
    data["ego_speed_standardized"] = (speed - speed.mean()) / (speed.std() or 1.0)
    data["ego_acceleration"] = rng.normal(0, 1, n)
    data["num_vehicles"] = rng.integers(0, 30, n).astype("int64")
    data["num_tracked_agents"] = data["num_vehicles"] + rng.integers(0, 10, n)
    distance = np.where(rng.random(n) < 0.25, np.nan, rng.uniform(1, 80, n))
    data["nearest_agent_distance_m"] = distance
    data["nearest_agent_distance_m_missing"] = np.isnan(distance)
    data["nearest_agent_relative_speed_mps"] = np.where(np.isnan(distance), np.nan,
                                                        rng.normal(0, 3, n))
    for column in extras:
        data[column] = rng.normal(0, 1, n)
    idm = rng.uniform(0.2, 0.8, n)
    scores = {"idm": idm}
    for planner in planners:
        tok = bpl.token(planner)
        if planner == tb.CONTROL:
            continue
        if planted and tok == "pdm_closed":
            delta = 0.3 * data["ego_speed_standardized"] + rng.normal(0, 0.02, n)
        else:
            delta = rng.normal(0, 0.1, n)
        scores[tok] = idm + delta
    for planner in planners:
        data[f"score_{bpl.token(planner)}"] = scores[bpl.token(planner)]
    for planner in planners:
        if planner != tb.CONTROL:
            tok = bpl.token(planner)
            data[f"delta_{tok}_vs_idm"] = scores[tok] - idm
    for planner in planners:
        data[f"succeeded_{bpl.token(planner)}"] = np.ones(n, dtype=bool)
    toks = [bpl.token(p) for p in planners]
    stacked = np.column_stack([scores[t] for t in toks])
    data["best_planner"] = [toks[i] for i in np.argmax(stacked, axis=1)]
    data["fold"] = np.asarray(folds, dtype="int64")
    return pd.DataFrame(data)


def provenance_for(planners, features=FEATURES, experiment=EXPERIMENT) -> dict:
    labels = []
    for planner in planners:
        tok = bpl.token(planner)
        labels += [f"score_{tok}", f"succeeded_{tok}"]
        if planner != tb.CONTROL:
            labels.append(f"delta_{tok}_vs_idm")
    labels.append("best_planner")
    result = {"planners": list(planners), "feature_columns": list(features), "label_columns": labels}
    if experiment is not None:
        result["experiment"] = experiment
    return result


def write_table(frame: pd.DataFrame, provenance: dict, path: Path) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pandas(frame, preserve_index=False)
    table = table.replace_schema_metadata({**(table.schema.metadata or {}),
                                           bpl.PROVENANCE_KEY: json.dumps(provenance).encode()})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return path


def small_table(seed: int, planners=None, rows_per_log=(1, 4), logs=(4, 8), folds=(2, 4)):
    rng = np.random.default_rng(seed)
    planners = PLANNERS if planners is None else planners
    n_logs = int(rng.integers(logs[0], logs[1] + 1))
    sizes = [int(rng.integers(rows_per_log[0], rows_per_log[1] + 1)) for _ in range(n_logs)]
    k = int(rng.integers(folds[0], min(folds[1], n_logs) + 1))
    return make_table(rng, sizes, planners, k), rng


def model_inputs(frame: pd.DataFrame, features=FEATURES):
    categorical = tb.categorical_columns(frame, list(features))
    numeric = [c for c in features if c not in categorical]
    return tb.to_model_frame(frame, list(features), categorical), numeric, categorical


def run_cli(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = tb.main([str(a) for a in argv])
    return code, out.getvalue(), err.getvalue()


def train_argv(table, out_dir, report_dir, *extra):
    return ["train", "--table", table, "--out-dir", out_dir, "--report-dir", report_dir,
            "--schema", SCHEMA_PATH, *FAST, *extra]


def check_argv(table, out_dir):
    return ["check", "--table", table, "--out-dir", out_dir, "--schema", SCHEMA_PATH]


def files_under(*dirs) -> list[Path]:
    return sorted(p for d in dirs if Path(d).exists() for p in Path(d).rglob("*") if p.is_file())


# --------------------------------------------------------------------------------- properties --

@unittest.skipUnless(HAS_SKLEARN, "scikit-learn is not installed")
class PropertyTests(unittest.TestCase):

    def test_property_7_out_of_fold_isolation(self):
        """Feature: classical-baseline, Property 7: out-of-fold isolation.

        **Validates: Requirements 2.6, 3.3, 4.1, 4.2, 4.3**
        """
        for seed in range(ITERATIONS):
            frame, rng = small_table(seed)
            x, numeric, categorical = model_inputs(frame)
            folds = frame["fold"]
            n = len(frame)
            splits = tb.fold_splits(folds)
            self.assertEqual([f for f, _, _ in splits], sorted(set(folds.tolist())))
            seen = np.zeros(n, dtype=int)
            for fold, train, test in splits:
                self.assertFalse(set(train) & set(test))
                self.assertEqual(len(train) + len(test), n)
                self.assertTrue((folds.to_numpy()[test] == fold).all())
                self.assertTrue((folds.to_numpy()[train] != fold).all())
                seen[test] += 1
            self.assertTrue((seen == 1).all(), "every row is held out exactly once")

            target = str(rng.choice([c for c in frame.columns if c.startswith("delta_")]))
            y = frame[target].to_numpy(dtype="float64")
            fold, _train, test = splits[int(rng.integers(len(splits)))]
            y_changed = y.copy()
            y_changed[test] = rng.normal(0, 5, len(test))
            # hgb dominates runtime; it runs on every fourth table, constant and ridge on all.
            families = tb.MODEL_FAMILIES if seed % 4 == 0 else ("constant", "ridge")
            for family in families:
                pred, imp = tb.cross_fit(x, y, folds, family, numeric, categorical, seed,
                                         importance=False)
                self.assertIsNone(imp)
                self.assertEqual(pred.shape, (n,))
                self.assertTrue(np.isfinite(pred).all(), f"seed {seed} {family}")
                if family == "constant":
                    for f, train, held in splits:
                        np.testing.assert_allclose(pred[held], y[train].mean(), rtol=0, atol=1e-12)
                changed, _ = tb.cross_fit(x, y_changed, folds, family, numeric, categorical, seed,
                                          importance=False)
                np.testing.assert_array_equal(changed[test], pred[test],
                                              err_msg=f"seed {seed} {family} fold {fold}")

    def test_property_8_feature_selection_and_leakage_guard(self):
        """Feature: classical-baseline, Property 8: feature selection and leakage guard.

        **Validates: Requirements 1.2, 2.1, 2.2**
        """
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for seed in range(ITERATIONS):
                rng = np.random.default_rng(seed)
                others = [str(p) for p in rng.permutation(OTHERS)[:int(rng.integers(1, len(OTHERS) + 1))]]
                planners = others[:]
                planners.insert(int(rng.integers(0, len(planners) + 1)), tb.CONTROL)
                extras = [f"meta_{i}" for i in range(int(rng.integers(0, 4)))]
                features = [str(c) for c in rng.permutation(FEATURES)]
                sizes = [int(rng.integers(1, 4)) for _ in range(int(rng.integers(3, 7)))]
                frame = make_table(rng, sizes, planners, 3, extras=extras)
                provenance = provenance_for(planners, features)

                expected = [f"delta_{bpl.token(p)}_vs_idm" for p in planners if p != tb.CONTROL]
                self.assertEqual(list(tb.resolve_targets(planners).values()), expected)
                x, _numeric, _categorical = model_inputs(frame, features)
                self.assertEqual(list(x.columns), features)
                predictions, _metrics, core = tb.train(frame, provenance, SCHEMA, ["constant"], seed,
                                                       importance=False, bootstrap_resamples=5)
                self.assertEqual(core["feature_columns"], features)
                self.assertEqual(core["targets"], expected)
                self.assertEqual(list(dict.fromkeys(predictions["target"])), expected)

                kind = int(rng.integers(3))
                if kind == 0:
                    leak = f"{rng.choice(bpl.LABEL_PREFIXES)}x{seed}"
                elif kind == 1:
                    leak = bpl.BEST_PLANNER
                else:
                    pattern = str(rng.choice(PATTERNS))
                    pattern = pattern.upper() if rng.random() < 0.5 else pattern
                    leak = f"pre_{pattern}_{seed}"
                leaky = frame.copy()
                if leak not in leaky.columns:
                    leaky[leak] = rng.normal(0, 1, len(leaky))
                leaky_features = features[:]
                leaky_features.insert(int(rng.integers(0, len(features) + 1)), leak)
                case = tmp / f"case_{seed}"
                table = write_table(leaky, provenance_for(planners, leaky_features), case / "t.parquet")
                out_dir, report_dir = case / "out", case / "report"
                with mock.patch.object(tb, "cross_fit", side_effect=AssertionError("fitted")):
                    code, _out, err = run_cli(*train_argv(table, out_dir, report_dir))
                self.assertEqual(code, 2, err)
                self.assertIn(leak, err)
                self.assertEqual(files_under(out_dir, report_dir), [])
                with self.assertRaises(tb.InputError):
                    tb.train(leaky, provenance_for(planners, leaky_features), SCHEMA,
                             ["constant"], seed, importance=False, bootstrap_resamples=5)

    def test_property_9_check_recomputes_saved_metrics(self):
        """Feature: classical-baseline, Property 9: check recomputes the saved metrics.

        **Validates: Requirements 9.1, 9.2, 9.3, 9.4, 9.5**
        """
        for seed in range(ITERATIONS):
            rng = np.random.default_rng(seed)
            planners = [tb.CONTROL] + [str(p) for p in rng.permutation(OTHERS)[:int(rng.integers(1, 4))]]
            frame, _ = small_table(seed, planners=planners)
            provenance = provenance_for(planners)
            models = [m for m in ("constant", "ridge") if rng.random() < 0.7] or ["ridge"]
            predictions, metrics, core = tb.train(frame, provenance, SCHEMA, models, seed,
                                                  importance=False, bootstrap_resamples=5)
            config = {**core, "table_sha256": "sha"}

            def failed(preds=predictions, mets=metrics, cfg=config):
                results = tb.check_outputs(frame, provenance, SCHEMA, preds, mets, cfg, "sha")
                self.assertEqual([name for name, _ in results], CHECK_NAMES)
                return {name for name, errors in results if errors}

            self.assertEqual(failed(), set(), f"seed {seed}")
            row = int(rng.integers(len(predictions)))

            self.assertIn("oof_coverage", failed(preds=predictions.drop(index=row)))
            dup = pd.concat([predictions, predictions.iloc[[row]]], ignore_index=True)
            self.assertIn("unique_predictions", failed(preds=dup))
            for column, change in (("fold", 1), ("y_true", 0.5)):
                bad = predictions.copy()
                bad.loc[row, column] = bad.loc[row, column] + change
                self.assertIn("fold_and_truth", failed(preds=bad), column)
            bad = predictions.copy()
            bad.loc[row, "y_pred"] = bad.loc[row, "y_pred"] + float(rng.uniform(0.1, 1.0))
            self.assertIn("metrics_recompute", failed(preds=bad))

            slots = [("per_target", m, t, k) for m, by_t in metrics["per_target"].items()
                     for t, values in by_t.items() for k, v in values.items()
                     if k != "pehe_ci" and math.isfinite(v)]
            slots += [("policy", name, k) for name, values in metrics["policy"].items()
                      for k in ("value", "regret")]
            slot = slots[int(rng.integers(len(slots)))]
            mets = json.loads(json.dumps(tb.json_safe(metrics)))
            parent = mets
            for key in slot[:-1]:
                parent = parent[key]
            parent[slot[-1]] += float(rng.choice([-1, 1]) * rng.uniform(2e-6, 1e-3))
            self.assertEqual(failed(mets=mets), {"metrics_recompute"}, slot)

            leaky = dict(config, feature_columns=config["feature_columns"] + [str(rng.choice(PATTERNS))])
            self.assertIn("feature_separation", failed(cfg=leaky))


# ----------------------------------------------------------------------------------- examples --

@unittest.skipUnless(HAS_SKLEARN, "scikit-learn is not installed")
class ModelTests(unittest.TestCase):

    def test_planted_signal_hgb_beats_constant(self):
        rng = np.random.default_rng(3)
        planners = ["idm", "pdm-closed"]
        frame = make_table(rng, [4] * 20, planners, 5, planted=True)
        _preds, metrics, _core = tb.train(frame, provenance_for(planners), SCHEMA,
                                          ["constant", "hgb"], 0, importance=False,
                                          bootstrap_resamples=20)
        target = "delta_pdm_closed_vs_idm"
        hgb, constant = metrics["per_target"]["hgb"][target], metrics["per_target"]["constant"][target]
        self.assertLess(hgb["pehe"], 0.5 * constant["pehe"])
        self.assertGreater(hgb["spearman"], 0.8)
        self.assertEqual(metrics["best_model"], "hgb")

    def test_to_model_frame_encodings(self):
        frame = pd.DataFrame({
            "flag": pd.array([True, False, None], dtype="boolean"),
            "plain_bool": [True, False, True],
            "map_name": ["a", None, "b"],
            "weather_code": pd.array([61, None, 3], dtype="Int64"),
            "kind": ["x", "y", None],
            "count": pd.array([1, None, 3], dtype="Int64"),
            "speed": [1.5, np.nan, 2.0],
            "unused": [9, 9, 9],
        })
        features = ["speed", "flag", "plain_bool", "map_name", "weather_code", "kind", "count"]
        before = frame.copy()
        categorical = tb.categorical_columns(frame, features)
        self.assertEqual(categorical, ["map_name", "weather_code", "kind"])
        x = tb.to_model_frame(frame, features, categorical)
        pd.testing.assert_frame_equal(frame, before)
        self.assertEqual(list(x.columns), features)
        np.testing.assert_array_equal(x["flag"].to_numpy(), [1.0, 0.0, np.nan])
        np.testing.assert_array_equal(x["plain_bool"].to_numpy(), [1.0, 0.0, 1.0])
        np.testing.assert_array_equal(x["count"].to_numpy(), [1.0, np.nan, 3.0])
        for column in ("flag", "plain_bool", "count", "speed"):
            self.assertEqual(x[column].dtype, np.float64)
        self.assertEqual(x["map_name"].tolist(), ["a", tb.MISSING_CATEGORY, "b"])
        self.assertEqual(x["weather_code"].tolist(), ["61", tb.MISSING_CATEGORY, "3"])
        self.assertEqual(x["kind"].tolist(), ["x", "y", tb.MISSING_CATEGORY])

    def test_ridge_preprocessing_learned_from_training_rows(self):
        rng = np.random.default_rng(5)
        frame = make_table(rng, [3] * 10, PLANNERS, 5)
        held = frame["fold"] == 0
        frame.loc[held, "ego_speed"] = frame.loc[held, "ego_speed"] + 1000.0
        frame.loc[held, "map_name"] = "only_in_fold_0"
        x, numeric, categorical = model_inputs(frame)
        y = frame["delta_pdm_closed_vs_idm"].to_numpy()
        manual = np.full(len(y), np.nan)
        for fold, train, test in tb.fold_splits(frame["fold"]):
            model = tb.make_model("ridge", numeric, categorical, 0).fit(x.iloc[train], y[train])
            prep = model.named_steps["prep"]
            num = prep.named_transformers_["num"]
            x_num = x.iloc[train][numeric].to_numpy(dtype="float64")
            medians = np.nanmedian(x_num, axis=0)
            np.testing.assert_allclose(num.named_steps["impute"].statistics_, medians)
            filled = np.where(np.isnan(x_num), medians, x_num)
            np.testing.assert_allclose(num.named_steps["scale"].mean_, filled.mean(axis=0))
            cats = prep.named_transformers_["cat"].categories_[categorical.index("map_name")]
            self.assertEqual(sorted(cats.tolist()), sorted(set(x.iloc[train]["map_name"])))
            self.assertEqual("only_in_fold_0" in cats.tolist(), fold != 0)
            manual[test] = model.predict(x.iloc[test])
        pred, _ = tb.cross_fit(x, y, frame["fold"], "ridge", numeric, categorical, 0, importance=False)
        np.testing.assert_allclose(pred, manual)

    def test_hgb_fits_with_nan_features_unfilled(self):
        rng = np.random.default_rng(7)
        frame = make_table(rng, [3] * 10, PLANNERS, 5)
        frame.loc[frame["fold"] == 1, "map_name"] = "unseen_in_training"
        x, numeric, categorical = model_inputs(frame)
        self.assertTrue(x[numeric].isna().any().any())
        y = frame["delta_urban_driver_vs_idm"].to_numpy()
        pred, imp = tb.cross_fit(x, y, frame["fold"], "hgb", numeric, categorical, 0, importance=False)
        self.assertIsNone(imp)
        self.assertTrue(np.isfinite(pred).all())
        _fold, train, _test = tb.fold_splits(frame["fold"])[0]
        model = tb.make_model("hgb", numeric, categorical, 0).fit(x.iloc[train], y[train])
        transformed = model.named_steps["prep"].transform(x.iloc[train])
        numeric_part = transformed[:, len(categorical):]
        self.assertEqual(int(np.isnan(numeric_part).sum()), int(x.iloc[train][numeric].isna().sum().sum()))

    def test_hyperparameters_in_estimators_and_config(self):
        numeric, categorical = ["a", "b"], ["map_name"]
        constant = tb.make_model("constant", numeric, categorical, 4)
        self.assertEqual(constant.get_params()["strategy"], "mean")
        ridge = tb.make_model("ridge", numeric, categorical, 4).named_steps["model"]
        self.assertEqual(type(ridge).__name__, "Ridge")
        self.assertEqual(ridge.alpha, 1.0)
        hgb = tb.make_model("hgb", numeric, categorical, 4).named_steps["model"]
        self.assertEqual(type(hgb).__name__, "HistGradientBoostingRegressor")
        params = hgb.get_params()
        expected = {"max_depth": 3, "learning_rate": 0.05, "max_iter": 200, "min_samples_leaf": 5,
                    "l2_regularization": 1.0, "early_stopping": False, "random_state": 4}
        self.assertEqual({k: params[k] for k in expected}, expected)
        self.assertEqual(list(params["categorical_features"]), [0])
        frame, _ = small_table(11, planners=["idm", "gc-pgp"])
        _p, _m, core = tb.train(frame, provenance_for(["idm", "gc-pgp"]), SCHEMA, list(tb.MODEL_FAMILIES),
                                4, importance=False, bootstrap_resamples=5)
        self.assertEqual(core["hyperparameters"], {"constant": {"strategy": "mean"},
                                                   "ridge": {"alpha": 1.0}, "hgb": expected})

    def test_training_is_deterministic(self):
        planners = ["idm", "pdm-closed", "urban-driver"]
        frame = make_table(np.random.default_rng(9), [3] * 12, planners, 5)
        provenance = provenance_for(planners)
        runs = [tb.train(frame, provenance, SCHEMA, list(tb.MODEL_FAMILIES), 2, importance=False,
                         bootstrap_resamples=30) for _ in range(2)]
        pd.testing.assert_frame_equal(runs[0][0], runs[1][0])
        self.assertEqual(json.dumps(tb.json_safe(runs[0][1])), json.dumps(tb.json_safe(runs[1][1])))
        self.assertEqual(len(runs[0][0]), len(frame) * 2 * 3)


@unittest.skipUnless(HAS_SKLEARN, "scikit-learn is not installed")
class CliTests(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.frame = make_table(np.random.default_rng(1), [3] * 20, PLANNERS, 5)
        self.table = write_table(self.frame, provenance_for(PLANNERS), self.tmp / "in" / "table.parquet")
        self.out_dir, self.report_dir = self.tmp / "out", self.tmp / "report"

    def train(self, *extra, table=None):
        return run_cli(*train_argv(table or self.table, self.out_dir, self.report_dir, *extra))

    def assert_nothing_written(self):
        self.assertEqual(files_under(self.out_dir, self.report_dir), [])

    def test_train_check_outputs_and_report(self):
        code, out, err = self.train()
        self.assertEqual(code, 0, err)
        self.assertIn("Best model:", out)
        written = {p.relative_to(self.tmp).as_posix() for p in files_under(self.out_dir, self.report_dir)}
        self.assertEqual(written, {"out/oof_predictions.parquet", "out/metrics.json", "out/config.json",
                                   f"report/baseline_{EXPERIMENT}.md", f"report/baseline_{EXPERIMENT}.json"})
        self.assertEqual(list(self.tmp.rglob("*.tmp")), [])

        predictions = pd.read_parquet(self.out_dir / "oof_predictions.parquet")
        self.assertEqual(list(predictions.columns), list(tb.PREDICTION_COLUMNS))
        self.assertEqual(len(predictions), len(self.frame) * 4 * 3)
        config = json.loads((self.out_dir / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(config["table_sha256"], validator.sha256_file(self.table))
        self.assertEqual(config["schema_sha256"], validator.sha256_file(SCHEMA_PATH))
        self.assertEqual(config["feature_columns"], FEATURES)
        self.assertEqual(config["models"], list(tb.MODEL_FAMILIES))
        self.assertEqual(config["targets"], [f"delta_{bpl.token(p)}_vs_idm" for p in OTHERS])
        self.assertEqual(config["hyperparameters"]["hgb"]["random_state"], 0)
        self.assertEqual(config["hyperparameters"]["ridge"], {"alpha": 1.0})
        self.assertEqual(config["seed"], 0)
        self.assertEqual(config["bootstrap_resamples"], 20)
        self.assertIn("train", config["command"])
        for key in ("python", "numpy", "pandas", "sklearn"):
            self.assertTrue(config["versions"][key], key)
        metrics_text = (self.out_dir / "metrics.json").read_text(encoding="utf-8")
        self.assertNotIn("NaN", metrics_text)
        metrics = json.loads(metrics_text)
        self.assertEqual(metrics["experiment"], EXPERIMENT)
        self.assertEqual(set(metrics["policy"]),
                         {*tb.MODEL_FAMILIES, "best_single_planner", "always_idm", "oracle"})

        markdown = (self.report_dir / f"baseline_{EXPERIMENT}.md").read_text(encoding="utf-8")
        for text in (EXPERIMENT, "`constant`", "`ridge`", "`hgb`", "## Policy value",
                     "`best_single_planner`", "`always_idm`", "`oracle`", "## Protocol"):
            self.assertIn(text, markdown)
        report = json.loads((self.report_dir / f"baseline_{EXPERIMENT}.json").read_text(encoding="utf-8"))
        self.assertEqual(report["experiment"], EXPERIMENT)
        self.assertEqual(report["config"]["table_sha256"], config["table_sha256"])
        self.assertEqual(set(report["per_target"]), set(tb.MODEL_FAMILIES))

        code, out, err = run_cli(*check_argv(self.table, self.out_dir))
        self.assertEqual(code, 0, out + err)
        self.assertEqual(out.splitlines(), [f"PASS {name}" for name in CHECK_NAMES])

        metrics["per_target"]["ridge"][config["targets"][0]]["pehe"] += 1e-3
        (self.out_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        code, out, _err = run_cli(*check_argv(self.table, self.out_dir))
        self.assertEqual(code, 1)
        lines = out.splitlines()
        self.assertEqual(len(lines), len(CHECK_NAMES))
        self.assertTrue(lines[3].startswith("FAIL metrics_recompute: "), lines[3])
        self.assertEqual([l for i, l in enumerate(lines) if i != 3],
                         [f"PASS {n}" for n in CHECK_NAMES if n != "metrics_recompute"])

        (self.out_dir / "config.json").unlink()
        code, _out, err = run_cli(*check_argv(self.table, self.out_dir))
        self.assertEqual(code, 2)
        self.assertIn("config.json", err)

    def test_overwrite_guard(self):
        code, _out, err = self.train("--models", "constant")
        self.assertEqual(code, 0, err)
        before = {p: p.read_bytes() for p in files_under(self.out_dir, self.report_dir)}
        code, _out, err = self.train("--models", "constant")
        self.assertEqual(code, 2)
        self.assertIn("--overwrite", err)
        self.assertIn(str(self.out_dir / "oof_predictions.parquet"), err)
        self.assertEqual({p: p.read_bytes() for p in files_under(self.out_dir, self.report_dir)}, before)
        code, _out, err = self.train("--models", "constant", "--overwrite")
        self.assertEqual(code, 0, err)
        self.assertEqual(list(self.tmp.rglob("*.tmp")), [])

        report_only = self.tmp / "other_out"
        code, _out, err = run_cli(*train_argv(self.table, report_only, self.report_dir, "--models", "constant"))
        self.assertEqual(code, 2)
        self.assertIn(f"baseline_{EXPERIMENT}.md", err)
        self.assertFalse(report_only.exists())

    def test_input_errors_exit_2(self):
        cases = []
        cases.append(("missing file", self.tmp / "nope.parquet", "nope.parquet"))
        prov = provenance_for(PLANNERS)
        del prov["feature_columns"]
        cases.append(("no features", write_table(self.frame, prov, self.tmp / "a.parquet"), "feature_columns"))
        prov = provenance_for(PLANNERS)
        del prov["planners"]
        cases.append(("no planners", write_table(self.frame, prov, self.tmp / "b.parquet"), "planners"))
        for column in ("scenario_type", "delta_gc_pgp_vs_idm", "score_idm", "lane_type"):
            path = write_table(self.frame.drop(columns=[column]), provenance_for(PLANNERS),
                               self.tmp / f"drop_{column}.parquet")
            cases.append((f"no {column}", path, column))
        one_fold = self.frame.assign(fold=np.int64(0))
        cases.append(("one fold", write_table(one_fold, provenance_for(PLANNERS), self.tmp / "c.parquet"),
                      "1 distinct"))
        for name, table, needle in cases:
            code, _out, err = self.train(table=table)
            self.assertEqual(code, 2, name)
            self.assertIn(needle, err, name)
            self.assert_nothing_written()
        for value, needle in (("ridge,lasso", "'lasso'"), ("ridge,ridge", "'ridge'"), ("", "''")):
            code, _out, err = self.train("--models", value)
            self.assertEqual(code, 2, value)
            self.assertIn(needle, err, value)
            self.assert_nothing_written()
        self.assertEqual(tb.parse_models("hgb, constant"), ["constant", "hgb"])

    def test_missing_sklearn_exits_2(self):
        with mock.patch.dict(sys.modules, {"sklearn": None}):
            code, _out, err = self.train()
        self.assertEqual(code, 2)
        self.assertIn(tb.INSTALL_HINT, err)
        self.assert_nothing_written()

    def test_experiment_name_falls_back_to_table_stem(self):
        self.assertEqual(tb.experiment_name({}, Path("x/my_table.parquet")), "my_table")
        self.assertEqual(tb.experiment_name({"experiment": "e1"}, Path("x/t.parquet")), "e1")


if __name__ == "__main__":
    unittest.main()
