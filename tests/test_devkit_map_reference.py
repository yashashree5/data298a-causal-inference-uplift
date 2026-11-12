"""Tests for the devkit map reference script; nuplan is not needed."""
from pathlib import Path
import random
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.devkit_map_reference import (  # noqa: E402
    classify_lane_type, normalize_records, relevant_connector_ids, resolve_traffic_light,
)

STATUSES = ["red", "yellow", "green", "RED", "Green", "unknown", "flashing", None]


class HelpTest(unittest.TestCase):
    def test_help_exits_zero_without_importing_nuplan(self):
        code = ("import sys, runpy; sys.argv = ['devkit_map_reference', '--help']\n"
                "try:\n    runpy.run_module('tools.devkit_map_reference', run_name='__main__')\n"
                "except SystemExit as e:\n    assert e.code in (0, None), e.code\n"
                "assert not any(m.startswith('nuplan') for m in sys.modules)\n")
        res = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertIn("--maps-root", res.stdout)
        self.assertIn("--map-version", res.stdout)


class RulesTest(unittest.TestCase):
    def test_lane_type(self):
        self.assertEqual(classify_lane_type(["1"], ["2"]), "lane")
        self.assertEqual(classify_lane_type([], ["2"]), "lane_connector")
        self.assertEqual(classify_lane_type([], []), "none")

    def test_relevant_prefers_ego_connectors_then_outgoing_edges(self):
        self.assertEqual(relevant_connector_ids([7, "3"], ["9"]), (["3", "7"], "ego_connector"))
        self.assertEqual(relevant_connector_ids([], [9, "9", 4]), (["4", "9"], "ego_lane_outgoing"))
        self.assertEqual(relevant_connector_ids([], []), ([], "none"))

    def test_resolve_examples(self):
        recs = [("5", "green"), ("6", "RED"), ("7", "yellow")]
        self.assertEqual(resolve_traffic_light(["5", "7"], recs), ("yellow", 2))
        self.assertEqual(resolve_traffic_light(["5", "6"], recs), ("red", 2))
        self.assertEqual(resolve_traffic_light(["8"], recs), ("unknown", 0))
        self.assertEqual(resolve_traffic_light(["5"], [("5", "flashing")]), ("unknown", 0))
        self.assertEqual(resolve_traffic_light([], recs), ("none", 0))

    def test_normalize_parquet_records(self):
        self.assertEqual(normalize_records(None), [])
        self.assertEqual(normalize_records(float("nan")), [])
        recs = [{"lane_connector_id": 12, "status": "red"}, {"lane_connector_id": None, "status": "x"}]
        self.assertEqual(normalize_records(recs), [("12", "red")])

    def test_resolve_property_most_restrictive_and_order_invariant(self):
        """Property 3 (traffic-light resolution) for the devkit rule. **Validates: Requirements 4.1, 4.2, 4.3**"""
        rank = {"red": 3, "yellow": 2, "green": 1}
        rng = random.Random(0)
        for _ in range(500):
            relevant = [str(i) for i in rng.sample(range(10), rng.randint(0, 4))]
            records = [(str(rng.randrange(12)), rng.choice(STATUSES)) for _ in range(rng.randint(0, 8))]
            state, matched = resolve_traffic_light(relevant, records)
            valid = [s.lower() for c, s in records if c in relevant and s and s.lower() in rank]
            if not relevant:
                expected = "none"
            elif not valid:
                expected = "unknown"
            else:
                expected = max(valid, key=rank.__getitem__)
            self.assertEqual((state, matched), (expected, len(valid)))
            shuffled = records[:]
            rng.shuffle(shuffled)
            noise = [("99", rng.choice(STATUSES))]
            self.assertEqual(resolve_traffic_light(relevant, shuffled + noise), (state, matched))


if __name__ == "__main__":
    unittest.main()
