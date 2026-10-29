"""Unit tests for the eligibility rules on a synthetic nuPlan-shaped SQLite database."""
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import pandas  # noqa: F401
except ImportError:  # the fast suite must still run without pandas
    pandas = None

if pandas is not None:
    from tools.build_scenario_eligibility import (  # noqa: E402
        BENCHMARK_TYPES, COLUMNS, anchor_frame, benchmark_type, funnel, source_recording,
    )

LOG = "2021.08.17.17.55.18_veh-43_00358_00673"
SECOND = 1_000_000
T0 = 1_629_000_000_000_000
needs_pandas = unittest.skipUnless(pandas, "pandas is required")


def token(i: int) -> bytes:
    return i.to_bytes(8, "big")


def make_db(path: Path, scenes: list[dict], log_seconds: int = 120) -> None:
    """Build the subset of the nuPlan schema the eligibility query reads.

    scenes: one dict per scene (in name order) with optional 'goal' (bool, default True),
    'roadblocks' (str, default '1 2'), and 'anchors': list of (seconds_from_t0, [tags]).
    lidar_pc frames run every second from T0 to T0 + log_seconds, defining the log range.
    """
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE log (token BLOB PRIMARY KEY, map_version TEXT);
        CREATE TABLE scene (token BLOB PRIMARY KEY, name TEXT, goal_ego_pose_token BLOB, roadblock_ids TEXT);
        CREATE TABLE ego_pose (token BLOB PRIMARY KEY);
        CREATE TABLE lidar_pc (token BLOB PRIMARY KEY, scene_token BLOB, timestamp INTEGER);
        CREATE TABLE scenario_tag (token BLOB PRIMARY KEY, lidar_pc_token BLOB, type TEXT);
    """)
    conn.execute("INSERT INTO log VALUES (?, ?)", (token(1), "us-pa-pittsburgh-hazelwood"))
    frame_scene = {}
    for idx, scene in enumerate(scenes):
        scene_tok = token(100 + idx)
        goal_tok = token(200 + idx) if scene.get("goal", True) else None
        if goal_tok is not None:
            conn.execute("INSERT INTO ego_pose VALUES (?)", (goal_tok,))
        conn.execute("INSERT INTO scene VALUES (?, ?, ?, ?)",
                     (scene_tok, f"scene-{idx:04d}", goal_tok, scene.get("roadblocks", "1 2")))
        for sec, _ in scene.get("anchors", []):
            frame_scene[sec] = scene_tok
    tag_id = 10_000
    for sec in range(log_seconds + 1):
        conn.execute("INSERT INTO lidar_pc VALUES (?, ?, ?)",
                     (token(1000 + sec), frame_scene.get(sec, token(100)), T0 + sec * SECOND))
    for scene in scenes:
        for sec, tags in scene.get("anchors", []):
            for tag in tags:
                conn.execute("INSERT INTO scenario_tag VALUES (?, ?, ?)", (token(tag_id), token(1000 + sec), tag))
                tag_id += 1
    conn.commit()
    conn.close()


class PureFunctionTest(unittest.TestCase):
    @needs_pandas
    def test_source_recording_drops_segment_numbers(self):
        self.assertEqual(source_recording(LOG), "2021.08.17.17.55.18_veh-43")
        with self.assertRaises(ValueError):
            source_recording("not_a_log")

    @needs_pandas
    def test_benchmark_type_is_alphabetically_last_matching_tag(self):
        self.assertEqual(len(BENCHMARK_TYPES), 14)
        self.assertEqual(benchmark_type(["behind_long_vehicle", "stationary_in_traffic", "on_intersection"]),
                         "stationary_in_traffic")
        self.assertIsNone(benchmark_type(["on_intersection", "medium_magnitude_speed"]))


@needs_pandas
class AnchorFrameTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / f"{LOG}.db"
        # Seven scenes: indices 1-2 and 6-7 are outside the valid range [3, scene_count - 1).
        make_db(self.db, [
            {"anchors": [(5, ["stationary_in_traffic"])]},                     # scene 1: first two
            {},                                                                 # scene 2
            {"anchors": [(1, ["changing_lane"]),                               # scene 3: window starts before log
                         (30, ["changing_lane", "behind_long_vehicle", "on_intersection"])]},
            {"goal": False, "anchors": [(40, ["changing_lane"])]},             # scene 4: no mission goal
            {"roadblocks": "", "anchors": [(50, ["changing_lane"])]},          # scene 5: no roadblocks
            {"anchors": [(60, ["medium_magnitude_speed"]),                     # scene 6: last two
                         (115, ["changing_lane"])]},
            {},                                                                 # scene 7
        ])

    def tearDown(self):
        self.tmp.cleanup()

    def frame(self, mini_logs=frozenset(), mini_recordings=frozenset()):
        return anchor_frame(self.db, "pittsburgh", set(mini_logs), set(mini_recordings)).set_index("anchor_timestamp_us")

    def row(self, df, sec):
        return df.loc[T0 + sec * SECOND]

    def test_one_row_per_anchor_with_all_columns_and_nothing_dropped(self):
        df = anchor_frame(self.db, "pittsburgh", set(), set())
        self.assertEqual(list(df.columns), COLUMNS)
        self.assertEqual(len(df), 7)
        self.assertTrue(df["scenario_id"].is_unique)
        self.assertEqual(int(df["tag_rows"].sum()), 9)
        self.assertTrue(df["scenario_token"].str.fullmatch(r"[0-9a-f]{16}").all())
        self.assertTrue((df["scenario_id"] == df["log_name"] + ":" + df["scenario_token"]).all())

    def test_each_rule_flags_exactly_the_violating_anchor(self):
        df = self.frame()
        self.assertFalse(self.row(df, 5)["in_valid_scene"])
        self.assertFalse(self.row(df, 1)["complete_window"])
        self.assertFalse(self.row(df, 40)["valid_mission_goal"])
        self.assertFalse(self.row(df, 50)["has_route_roadblocks"])
        self.assertFalse(self.row(df, 60)["in_valid_scene"])
        self.assertFalse(self.row(df, 115)["complete_window"])  # ends after the log
        good = self.row(df, 30)
        self.assertTrue(good[["in_valid_scene", "valid_mission_goal", "has_route_roadblocks", "complete_window"]].all())
        self.assertEqual(int(df["simulation_ready"].sum()), 1)

    def test_window_is_three_seconds_before_to_twelve_after(self):
        good = self.row(self.frame(), 30)
        self.assertEqual(good["window_start_us"], T0 + 27 * SECOND)
        self.assertEqual(good["window_end_us"], T0 + 42 * SECOND)

    def test_tags_and_benchmark_label(self):
        good = self.row(self.frame(), 30)
        self.assertEqual(good["all_scenario_tags"], "behind_long_vehicle|changing_lane|on_intersection")
        self.assertEqual(good["tag_count"], 3)
        self.assertEqual(good["benchmark_scenario_type"], "changing_lane")
        self.assertTrue(pandas.isna(self.row(self.frame(), 60)["benchmark_scenario_type"]))

    def test_mini_recording_excludes_candidates_but_keeps_rows(self):
        clean = self.frame()
        self.assertTrue(self.row(clean, 30)["benchmark_candidate"])
        shared = self.frame(mini_recordings={"2021.08.17.17.55.18_veh-43"})
        self.assertEqual(len(shared), 7)
        self.assertTrue(shared["shares_mini_recording"].all())
        self.assertFalse(shared["benchmark_candidate"].any())
        self.assertFalse(shared["in_mini_split"].any())
        self.assertTrue(self.frame(mini_logs={LOG})["in_mini_split"].all())

    def test_funnel_is_monotonic_and_ends_at_candidates(self):
        df = anchor_frame(self.db, "pittsburgh", set(), set())
        steps = funnel(df)
        counts = [s["anchors"] for s in steps]
        self.assertEqual(counts[0], 7)
        self.assertEqual(counts, sorted(counts, reverse=True))
        self.assertEqual(counts[-1], int(df["benchmark_candidate"].sum()))


if __name__ == "__main__":
    unittest.main()
