"""Unit tests for ego and agent features on a synthetic nuPlan-shaped SQLite database."""
import math
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import numpy as np
    import pandas as pd
except ImportError:  # the fast suite must still run without pandas
    pd = None

if pd is not None:
    from tools.extract_ego_agent_features import (  # noqa: E402
        FEATURES, REAR_AXLE_TO_CENTER_M, extract_log, feature_table, heading_from_quaternion,
        reference_features, start_frames,
    )

needs_pandas = unittest.skipUnless(pd, "pandas is required")
LOG = "2021.08.17.17.55.18_veh-43_00358_00673"
T0 = 1_629_000_000_000_000
STEP = 50_000  # 20 Hz lidar


def tok(i: int) -> bytes:
    return i.to_bytes(8, "big")


def quat(yaw: float):
    return math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)


def make_db(path: Path, ego_yaw: float = 0.0, ego_v_local=(10.0, 0.0), boxes_at=None, frames: int = 400) -> None:
    """One log: lidar_pc every 50 ms, ego at the origin, optional boxes on chosen frames.

    boxes_at: {frame_index: [(category, x, y, vx, vy), ...]}, velocities global.
    """
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE log (token BLOB PRIMARY KEY);
        CREATE TABLE lidar (token BLOB PRIMARY KEY, channel TEXT);
        CREATE TABLE lidar_pc (token BLOB PRIMARY KEY, lidar_token BLOB, ego_pose_token BLOB, timestamp INTEGER);
        CREATE TABLE ego_pose (token BLOB PRIMARY KEY, timestamp INTEGER, x REAL, y REAL, qw REAL, qx REAL,
            qy REAL, qz REAL, vx REAL, vy REAL, acceleration_x REAL, acceleration_y REAL, angular_rate_z REAL);
        CREATE TABLE category (token BLOB PRIMARY KEY, name TEXT);
        CREATE TABLE track (token BLOB PRIMARY KEY, category_token BLOB);
        CREATE TABLE lidar_box (token BLOB PRIMARY KEY, lidar_pc_token BLOB, track_token BLOB,
            x REAL, y REAL, vx REAL, vy REAL);
    """)
    conn.execute("INSERT INTO log VALUES (?)", (tok(1),))
    conn.execute("INSERT INTO lidar VALUES (?, ?)", (tok(2), "MergedPointCloud"))
    names = ["vehicle", "pedestrian", "bicycle", "traffic_cone", "barrier", "czone_sign", "generic_object"]
    for i, n in enumerate(names):
        conn.execute("INSERT INTO category VALUES (?, ?)", (tok(10 + i), n))
    qw, qx, qy, qz = quat(ego_yaw)
    for f in range(frames):
        ts = T0 + f * STEP
        conn.execute("INSERT INTO ego_pose VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                     (tok(100_000 + f), ts + 300, 0.0, 0.0, qw, qx, qy, qz, ego_v_local[0], ego_v_local[1],
                      3.0, 4.0, 0.1))
        conn.execute("INSERT INTO lidar_pc VALUES (?,?,?,?)", (tok(200_000 + f), tok(2), tok(100_000 + f), ts))
    n = 0
    for f, boxes in (boxes_at or {}).items():
        for cat, x, y, vx, vy in boxes:
            conn.execute("INSERT INTO track VALUES (?, ?)", (tok(300_000 + n), tok(10 + names.index(cat))))
            conn.execute("INSERT INTO lidar_box VALUES (?,?,?,?,?,?,?)",
                         (tok(400_000 + n), tok(200_000 + f), tok(300_000 + n), x, y, vx, vy))
            n += 1
    conn.commit()
    conn.close()


def scenarios(*anchor_offsets_us):
    return pd.DataFrame({"scenario_id": [f"{LOG}:{i:016x}" for i in range(len(anchor_offsets_us))],
                         "scenario_token": [f"{i:016x}" for i in range(len(anchor_offsets_us))],
                         "anchor_timestamp_us": [T0 + o for o in anchor_offsets_us]})


@needs_pandas
class PureFunctionTest(unittest.TestCase):
    def test_heading_round_trips_known_angles(self):
        for yaw in (0.0, 0.5, -1.2, math.pi / 2, 3.0):
            self.assertAlmostEqual(float(heading_from_quaternion(*quat(yaw))), yaw, places=12)

    def test_start_frame_is_first_lidar_at_or_after_anchor_minus_3s(self):
        lidar = np.arange(0, 400) * STEP + T0
        anchor = np.array([T0 + 3_000_000, T0 + 3_000_001, T0 + 3_049_999, T0 + 2_000_000])
        self.assertEqual(start_frames(lidar, anchor).tolist(), [0, 1, 1, 0])

    def test_pacifica_rear_axle_to_center(self):
        self.assertAlmostEqual(REAR_AXLE_TO_CENTER_M, 1.461, places=3)


@needs_pandas
class ExtractLogTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / f"{LOG}.db"

    def tearDown(self):
        self.tmp.cleanup()

    def test_ego_features_and_frame_alignment(self):
        make_db(self.db, ego_yaw=0.3)
        # anchor 5.02 s -> start at 2.02 s -> first lidar at or after is frame 41 (2.05 s)
        df = extract_log(self.db, scenarios(5_020_000), "pittsburgh")
        r = df.iloc[0]
        self.assertEqual(r["start_timestamp_us"], T0 + 41 * STEP)
        self.assertEqual(r["start_offset_us"], 41 * STEP - 5_020_000)
        self.assertEqual(r["ego_pose_lag_us"], 300)
        self.assertAlmostEqual(r["ego_heading"], 0.3, places=12)
        self.assertAlmostEqual(r["ego_speed"], 10.0)
        self.assertAlmostEqual(r["ego_acceleration"], 5.0)   # |(3, 4)|
        self.assertAlmostEqual(r["ego_yaw_rate"], 0.1)
        self.assertEqual(r["log_token"], tok(1).hex())

    def test_agent_counts_distance_and_closing_speed(self):
        # Ego faces +x at 10 m/s (local). Ego centre is at (1.461, 0).
        boxes = {60: [("vehicle", 21.461, 0.0, 0.0, 0.0),      # 20 m ahead, stationary: closing +10 m/s
                      ("vehicle", 1.461, 30.0, 0.0, 0.0),      # 30 m to the side: closing 0
                      ("pedestrian", 101.461, 0.0, 0.0, 0.0),  # 100 m: outside 50 m
                      ("traffic_cone", 3.461, 0.0, 0.0, 0.0),  # static: counted as tracked only
                      ("bicycle", 1.461, -45.0, 0.0, 0.0)]}
        make_db(self.db, boxes_at=boxes)
        df = extract_log(self.db, scenarios(6_000_000), "pittsburgh")  # start 3.0 s = frame 60
        r = df.iloc[0]
        self.assertEqual((r["num_tracked_agents"], r["num_vehicles"], r["num_pedestrians"]), (5, 2, 1))
        self.assertEqual(r["agents_within_50m"], 3)
        self.assertAlmostEqual(r["nearest_agent_distance_m"], 20.0, places=9)
        self.assertAlmostEqual(r["nearest_agent_relative_speed_mps"], 10.0, places=9)

    def test_ego_velocity_is_rotated_from_the_ego_frame(self):
        # Ego faces +y (yaw 90 deg) moving forward at 5 m/s: global velocity (0, 5).
        # A stationary car 10 m ahead in +y closes at +5; one 10 m in +x closes at 0.
        boxes = {60: [("vehicle", 0.0, 1.461 + 10.0, 0.0, 0.0)]}
        make_db(self.db, ego_yaw=math.pi / 2, ego_v_local=(5.0, 0.0), boxes_at=boxes)
        r = extract_log(self.db, scenarios(6_000_000), "boston").iloc[0]
        self.assertAlmostEqual(r["nearest_agent_distance_m"], 10.0, places=9)
        self.assertAlmostEqual(r["nearest_agent_relative_speed_mps"], 5.0, places=9)

    def test_no_agents_gives_null_with_indicator(self):
        make_db(self.db, boxes_at={60: [("barrier", 5.0, 5.0, 0.0, 0.0)]})
        df = extract_log(self.db, scenarios(6_000_000), "singapore")
        self.assertEqual(df.iloc[0]["num_tracked_agents"], 1)
        self.assertEqual(df.iloc[0]["agents_within_50m"], 0)
        self.assertTrue(pd.isna(df.iloc[0]["nearest_agent_distance_m"]))
        table = feature_table(df)
        self.assertTrue(table["nearest_agent_distance_m_missing"].all())
        self.assertNotIn("ego_speed_missing", table.columns)

    def test_only_the_start_frame_is_read(self):
        # Boxes exist only after the start frame: none must be counted.
        make_db(self.db, boxes_at={61: [("vehicle", 5.0, 0.0, 0.0, 0.0)], 200: [("vehicle", 5.0, 0.0, 0.0, 0.0)]})
        r = extract_log(self.db, scenarios(6_000_000), "boston").iloc[0]
        self.assertEqual(r["num_tracked_agents"], 0)

    def test_vectorized_equals_devkit_query_reference(self):
        boxes = {f: [("vehicle", 12.0 + f / 10, 3.0, -1.0, 0.5), ("pedestrian", -4.0, 7.0, 1.0, 0.0)]
                 for f in (40, 41, 60, 100)}
        make_db(self.db, ego_yaw=-0.7, ego_v_local=(4.0, 0.3), boxes_at=boxes)
        offsets = (5_000_000, 5_020_000, 6_000_000, 8_000_000)
        df = extract_log(self.db, scenarios(*offsets), "las_vegas")
        for (_, row), off in zip(df.iterrows(), offsets):
            ref = reference_features(self.db, T0 + off)
            self.assertEqual(ref.pop("start_lidar_token"), row["start_lidar_token"])
            for k, v in ref.items():
                self.assertAlmostEqual(float(row[k]), float(v), places=9, msg=k)
        self.assertEqual(set(FEATURES) - set(df.columns), set())


class SchemaTest(unittest.TestCase):
    def test_ego_agent_fields_are_declared(self):
        try:
            import yaml  # noqa: F401
        except ImportError:
            self.skipTest("PyYAML is required")
        from tools.validate_scenario_subset import DEFAULT_SCHEMA, load_schema, schema_fields
        names = {f["name"]: f for f in schema_fields(load_schema(DEFAULT_SCHEMA))}
        expected = ["ego_x", "ego_y", "ego_heading", "ego_speed", "ego_acceleration", "ego_yaw_rate",
                    "num_tracked_agents", "num_vehicles", "num_pedestrians", "nearest_agent_distance_m",
                    "nearest_agent_relative_speed_mps", "agents_within_50m"]
        self.assertTrue(set(expected) <= set(names))
        self.assertFalse(any(names[n]["model_feature"] for n in ("ego_x", "ego_y", "ego_heading")))


if __name__ == "__main__":
    unittest.main()
