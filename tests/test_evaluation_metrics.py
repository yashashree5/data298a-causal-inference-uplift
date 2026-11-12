"""Tests for causal_planner.evaluation.metrics on synthetic vectors (numpy/pandas only; runs
without scikit-learn). Property loops use seeded `random`/`numpy` generators, 100 iterations."""
from pathlib import Path
import math
import os
import random
import subprocess
import sys
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from causal_planner.evaluation import metrics as m  # noqa: E402
ITERATIONS = 100
CONTROL = "idm"
TOKENS = ["pdm_closed", "pdm_open", "pdm_hybrid", "gc_pgp", "urban", "lqr"]
def non_constant(rng: np.random.Generator, low: int, high: int) -> np.ndarray:
    """Random delta vector of length low..high with at least two distinct values; some ties."""
    while True:
        n = int(rng.integers(low, high + 1))
        y = rng.normal(0.0, 1.0, size=n)
        if rng.random() < 0.3:
            y = np.round(y, 1)  # create ties
        if not np.all(y == y[0]):
            return y
def reference_choice(pred: pd.DataFrame) -> list[str]:
    out = []
    for row in pred.itertuples(index=False):
        values = list(row)
        best = max(values)
        out.append(pred.columns[values.index(best)] if best > 0 else CONTROL)
    return out
# ----------------------------------------------------------------------------------- examples --
class MetricExamplesTest(unittest.TestCase):
    Y = np.array([1.0, 2.0, 3.0, 4.0])
    P = np.array([2.0, 2.0, 5.0, 3.0])
    def test_hand_computed_four_rows(self):
        self.assertAlmostEqual(m.pehe(self.Y, self.P), math.sqrt(1.5), places=12)
        self.assertAlmostEqual(m.mae(self.Y, self.P), 1.0, places=12)
        # ranks: y [1,2,3,4], p [1.5,1.5,4,3] -> 3.5 / sqrt(5 * 4.5)
        self.assertAlmostEqual(m.spearman(self.Y, self.P), 3.5 / math.sqrt(22.5), places=12)
        # model order rows 2,3,0,1 -> G = 0,3,7,8,10 (area 23); oracle G = 0,4,7,9,10 (area 25);
        # random area 4 * 10 / 2 = 20; ties in P go to row 0 before row 1.
        self.assertAlmostEqual(m.qini_coefficient(self.Y, self.P), 0.6, places=12)
        np.testing.assert_array_equal(m.gain_curve(self.Y, [3, 2, 1, 0]), [0, 4, 7, 9, 10])
        self.assertEqual(m.top_k_recall(self.Y, self.P, 0.5), 1.0)
        self.assertEqual(m.top_k_recall(self.Y, self.P, 0.25), 0.0)
    def test_nan_cases(self):
        self.assertTrue(math.isnan(m.spearman([1.0, 2.0, 3.0], [5.0, 5.0, 5.0])))
        self.assertTrue(math.isnan(m.spearman([2.0, 2.0, 2.0], [1.0, 2.0, 3.0])))
        self.assertTrue(math.isnan(m.qini_coefficient([0.5, 0.5, 0.5, 0.5], [1.0, 2.0, 3.0, 4.0])))
    def test_top_k_count(self):
        self.assertEqual(m.top_k_count(65, 0.1), 7)
        self.assertEqual(m.top_k_count(65, 0.2), 13)
        self.assertEqual(m.top_k_count(3, 0.1), 1)
        self.assertEqual(m.TOP_K_FRACTIONS, (0.10, 0.20))
    def test_input_validation(self):
        with self.assertRaises(ValueError):
            m.pehe([1.0, 2.0], [1.0])
        with self.assertRaises(ValueError):
            m.mae([], [])
    def test_policy_example(self):
        scores = pd.DataFrame({"idm": [0.5, 0.9, 0.2], "a": [0.7, 0.4, 0.2], "b": [0.6, 0.8, 0.9]})
        pred = pd.DataFrame({"a": [0.3, -0.1, 0.2], "b": [0.3, -0.2, 0.1]})
        choice = m.choose_planner(pred)
        self.assertEqual(choice.tolist(), ["a", "idm", "a"])
        self.assertAlmostEqual(m.policy_value(scores, choice), (0.7 + 0.9 + 0.2) / 3, places=12)
        self.assertAlmostEqual(m.oracle_value(scores), (0.7 + 0.9 + 0.9) / 3, places=12)
    def test_import_loads_no_sklearn_or_tools(self):
        env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
        script = (
            "import sys\n"
            "import causal_planner.evaluation.metrics\n"
            "assert not any(n == 'sklearn' or n.startswith('sklearn.') for n in sys.modules)\n"
            "assert not any(n == 'tools' or n.startswith('tools.') for n in sys.modules)\n"
            "print('ok')\n"
        )
        result = subprocess.run([sys.executable, "-c", script], cwd=ROOT, env=env,
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ok", result.stdout)
# --------------------------------------------------------------------------------- properties --
class MetricPropertiesTest(unittest.TestCase):
    def test_property_1_perfect_predictions(self):
        """Feature: classical-baseline, Property 1: Perfect predictions score perfectly.
        **Validates: Requirements 5.1, 5.2, 5.3**"""
        rng = np.random.default_rng(1)
        for _ in range(ITERATIONS):
            y = non_constant(rng, 2, 300)
            self.assertEqual(m.pehe(y, y.copy()), 0.0)
            for f in m.TOP_K_FRACTIONS:
                self.assertEqual(m.top_k_recall(y, y.copy(), f), 1.0)
            self.assertAlmostEqual(m.qini_coefficient(y, y.copy()), 1.0, places=12)
    def test_property_2_metric_bounds(self):
        """Feature: classical-baseline, Property 2: Metric bounds.
        **Validates: Requirements 5.1, 5.2, 5.3, 5.4**"""
        rng = np.random.default_rng(2)
        for i in range(ITERATIONS):
            n = int(rng.integers(1, 301))
            y = np.full(n, float(rng.normal())) if i % 10 == 0 else rng.normal(size=n)
            p = rng.normal(size=n)
            self.assertGreaterEqual(m.pehe(y, p), 0.0)
            self.assertGreaterEqual(m.mae(y, p), 0.0)
            for f in m.TOP_K_FRACTIONS:
                r = m.top_k_recall(y, p, f)
                self.assertTrue(0.0 <= r <= 1.0, r)
            q = m.qini_coefficient(y, p)
            if np.all(y == y[0]):
                self.assertTrue(math.isnan(q))
            else:
                self.assertFalse(math.isnan(q))
                self.assertLessEqual(q, 1.0 + 1e-12)
    def test_property_3_order_only(self):
        """Feature: classical-baseline, Property 3: Ranking metrics depend only on prediction order.
        **Validates: Requirements 5.2, 5.3, 5.5**"""
        rng = np.random.default_rng(3)
        for _ in range(ITERATIONS):
            n = int(rng.integers(5, 301))
            y = np.round(rng.normal(size=n), 2)
            p = rng.permutation(n).astype(float) / n - 0.5 + rng.uniform(-1e-3, 1e-3)  # distinct
            scale, shift = float(rng.uniform(0.1, 10.0)), float(rng.normal(0.0, 5.0))
            for q in (scale * p + shift, p ** 3):
                self.assertEqual(len(np.unique(q)), n)
                for f in m.TOP_K_FRACTIONS:
                    self.assertEqual(m.top_k_recall(y, q, f), m.top_k_recall(y, p, f))
                a, b = m.qini_coefficient(y, q), m.qini_coefficient(y, p)
                self.assertTrue((math.isnan(a) and math.isnan(b)) or a == b, (a, b))
                a, b = m.spearman(y, q), m.spearman(y, p)
                self.assertTrue((math.isnan(a) and math.isnan(b)) or a == b, (a, b))
    def test_property_4_random_qini_near_zero(self):
        """Feature: classical-baseline, Property 4: Random rankings have zero expected qini.
        **Validates: Requirements 5.3**"""
        rng = np.random.default_rng(4)
        for _ in range(ITERATIONS):
            y = non_constant(rng, 30, 300)
            n = y.shape[0]
            values = [m.qini_coefficient(y, rng.permutation(n).astype(float)) for _ in range(200)]
            self.assertLess(abs(float(np.mean(values))), 0.1)
    def test_property_5_policy_bounded_by_oracle(self):
        """Feature: classical-baseline, Property 5: Policy choice and value are bounded by the oracle.
        **Validates: Requirements 6.1, 6.2, 6.5**"""
        rng = np.random.default_rng(5)
        for _ in range(ITERATIONS):
            k = int(rng.integers(1, 7))
            n = int(rng.integers(1, 80))
            tokens = TOKENS[:k]
            scores = pd.DataFrame(rng.uniform(0.0, 1.0, size=(n, k + 1)), columns=[CONTROL, *tokens])
            pred = pd.DataFrame(np.round(rng.normal(0.0, 0.3, size=(n, k)), 1), columns=tokens)
            choice = m.choose_planner(pred, control=CONTROL)
            self.assertEqual(choice.tolist(), reference_choice(pred))
            all_nonpositive = (pred.to_numpy() <= 0).all(axis=1)
            np.testing.assert_array_equal(choice.to_numpy() == CONTROL, all_nonpositive)
            oracle = m.oracle_value(scores)
            self.assertLessEqual(m.policy_value(scores, choice), oracle + 1e-12)
            observed = scores[tokens].sub(scores[CONTROL], axis=0)
            best = m.choose_planner(observed, control=CONTROL)
            regret = oracle - m.policy_value(scores, best)
            self.assertAlmostEqual(regret, 0.0, places=12)
    def test_property_6_bootstrap(self):
        """Feature: classical-baseline, Property 6: Bootstrap intervals contain the point estimate
        and are reproducible.
        **Validates: Requirements 7.1**"""
        rng = np.random.default_rng(6)
        for i in range(ITERATIONS):
            n = int(rng.integers(10, 201))
            x = rng.normal(0.0, 1.0, size=n)
            y = rng.normal(0.0, 1.0, size=n)
            seed = int(rng.integers(0, 2**31 - 1))
            for stat, arrays in ((lambda a: float(np.mean(a)), (x,)), (m.pehe, (x, y))):
                first = m.bootstrap_ci(stat, *arrays, n_resamples=200, seed=seed)
                second = m.bootstrap_ci(stat, *arrays, n_resamples=200, seed=seed)
                self.assertEqual(first, second)
                point, low, high = first
                self.assertEqual(point, float(stat(*arrays)))
                self.assertTrue(low <= point <= high, (i, first))
if __name__ == "__main__":
    unittest.main()
