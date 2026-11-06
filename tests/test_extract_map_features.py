"""Unit tests for map-context features on synthetic GeoPackage and SQLite fixtures."""
from pathlib import Path
import json
import sqlite3
import random
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import numpy as np
    import pandas as pd
    import pyproj
    import shapely
    from shapely.geometry import Point, box
except ImportError:  # the fast suite must still run without the geo stack
    shapely = None

if shapely is not None:
    from tools.extract_map_features import (  # noqa: E402
        CityMap, RESTRICTIVENESS, classify_lane_type, geometry_features, load_city_map, point_hits,
        projected_crs, read_layer, relevant_connectors, resolve_traffic_light, strip_gpkg_header,
    )
    from shapely import STRtree

needs_geo = unittest.skipUnless(shapely, "pandas, shapely and pyproj are required")
MAP = "us-pa-pittsburgh-hazelwood"
VERSION = "9.17.1937"
UTM = "EPSG:32617"
X0, Y0 = 587_000.0, 4_475_000.0  # a Pittsburgh UTM point


def gpkg_blob(geom, envelope_code: int = 1, srs_id: int = 4326) -> bytes:
    """GeoPackage geometry blob: 'GP', version 0, flags (little-endian + envelope), srs_id, envelope, WKB."""
    flags = (envelope_code << 1) | 0x01
    header = b"GP" + bytes([0, flags]) + struct.pack("<i", srs_id)
    if envelope_code == 1:
        minx, miny, maxx, maxy = geom.bounds
        header += struct.pack("<4d", minx, maxx, miny, maxy)
    return header + shapely.to_wkb(geom)


def to_lonlat(geom):
    t = pyproj.Transformer.from_crs(UTM, "EPSG:4326", always_xy=True)
    return shapely.transform(geom, lambda c: np.column_stack(t.transform(c[:, 0], c[:, 1])))


def make_gpkg(maps_dir: Path, lanes: dict, conn_polys: dict, connectors: dict, inters: dict) -> Path:
    """Minimal map.gpkg. lanes/inters: {fid: utm polygon}; conn_polys: {poly_fid: (connector_fid, polygon)};
    connectors: {fid: exit_lane_fid}. Geometries are stored as lon/lat like the real maps."""
    path = maps_dir / MAP / VERSION / "map.gpkg"
    path.parent.mkdir(parents=True)
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE meta (fid INTEGER PRIMARY KEY, key TEXT, value TEXT)")
    con.executemany("INSERT INTO meta (key, value) VALUES (?, ?)",
                    [("geographicCoordSystem", "epsg:4326"), ("projectedCoordSystem", "epsg:32617")])
    con.execute("CREATE TABLE lanes_polygons (fid INTEGER PRIMARY KEY, geom BLOB, lane_fid INTEGER)")
    con.executemany("INSERT INTO lanes_polygons VALUES (?, ?, ?)",
                    [(f, gpkg_blob(to_lonlat(g)), f) for f, g in lanes.items()])
    con.execute("CREATE TABLE gen_lane_connectors_scaled_width_polygons "
                "(fid INTEGER PRIMARY KEY, geom BLOB, lane_connector_fid INTEGER)")
    con.executemany("INSERT INTO gen_lane_connectors_scaled_width_polygons VALUES (?, ?, ?)",
                    [(f, gpkg_blob(to_lonlat(g)), c) for f, (c, g) in conn_polys.items()])
    con.execute("CREATE TABLE lane_connectors (fid INTEGER PRIMARY KEY, geom BLOB, exit_lane_fid INTEGER, "
                "entry_lane_fid INTEGER)")
    con.executemany("INSERT INTO lane_connectors VALUES (?, NULL, ?, NULL)", list(connectors.items()))
    con.execute("CREATE TABLE intersections (fid INTEGER PRIMARY KEY, geom BLOB)")
    con.executemany("INSERT INTO intersections VALUES (?, ?)",
                    [(f, gpkg_blob(to_lonlat(g))) for f, g in inters.items()])
    con.commit()
    con.close()
    return path


@needs_geo
class GpkgHeaderTests(unittest.TestCase):
    def test_round_trip_envelope_codes_0_and_1(self):
        geom = box(-79.96, 40.42, -79.95, 40.43)
        for code in (0, 1):
            with self.subTest(envelope_code=code):
                blob = gpkg_blob(geom, envelope_code=code)
                self.assertEqual(len(blob) - len(shapely.to_wkb(geom)), 8 + (32 if code else 0))
                back = shapely.from_wkb(strip_gpkg_header(blob))
                self.assertTrue(back.equals(geom))

    def test_rejects_non_gpkg_blob(self):
        with self.assertRaises(ValueError):
            strip_gpkg_header(shapely.to_wkb(Point(0, 0)))


@needs_geo
class LoadCityMapTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.maps = Path(self.tmp.name)
        lane = box(X0 - 10, Y0 - 2, X0 + 10, Y0 + 2)
        conn_poly = box(X0 + 10, Y0 - 2, X0 + 30, Y0 + 2)
        inter = box(X0 + 8, Y0 - 20, X0 + 40, Y0 + 20)
        # connector polygon fid 900 belongs to lane_connectors.fid 55; connectors 55, 56 leave lane 7
        self.path = make_gpkg(self.maps, lanes={7: lane, 8: box(X0 - 10, Y0 + 2, X0 + 10, Y0 + 6)},
                              conn_polys={900: (55, conn_poly)}, connectors={55: 7, 56: 7, 57: 8},
                              inters={3: inter})

    def tearDown(self):
        self.tmp.cleanup()

    def test_read_layer_and_crs(self):
        con = sqlite3.connect(self.path)
        try:
            self.assertEqual(projected_crs(con), "epsg:32617")
            attrs, geoms = read_layer(con, "lanes_polygons", ["fid"])
        finally:
            con.close()
        self.assertEqual(attrs["fid"].tolist(), [7, 8])
        self.assertLess(geoms[0].bounds[0], -79.0)  # stored as lon/lat

    def test_reprojected_utm_point_falls_in_its_lane(self):
        city = load_city_map(self.maps, MAP, VERSION)
        self.assertEqual(city.crs, "epsg:32617")
        hit = city.lane_tree.query(Point(X0, Y0), predicate="intersects")
        self.assertEqual(city.lane_fids[hit].tolist(), [7])
        hit = city.conn_tree.query(Point(X0 + 20, Y0), predicate="intersects")
        self.assertEqual(city.conn_ids[hit].tolist(), [55])  # lane_connector_fid, not polygon fid 900
        hit = city.inter_tree.query(Point(X0 + 20, Y0), predicate="intersects")
        self.assertEqual(city.inter_fids[hit].tolist(), [3])
        self.assertEqual(len(city.lane_tree.query(Point(X0 - 50, Y0), predicate="intersects")), 0)
        self.assertEqual(city.outgoing, {7: [55, 56], 8: [57]})

    def test_missing_gpkg_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_city_map(self.maps, "sg-one-north", VERSION)


def random_boxes(rng: random.Random, n: int, first_id: int) -> dict:
    """n axis-aligned boxes on a 0..20 integer grid, keyed by consecutive ids."""
    out = {}
    for k in range(n):
        x0, y0 = rng.randint(0, 18), rng.randint(0, 18)
        out[first_id + k] = (x0, y0, x0 + rng.randint(1, 6), y0 + rng.randint(1, 6))
    return out


def covers(b, px, py) -> bool:
    return b[0] <= px <= b[2] and b[1] <= py <= b[3]


def make_city(lanes: dict, conns: dict, inters: dict, outgoing: dict | None = None) -> "CityMap":
    """In-memory CityMap from {id: (minx, miny, maxx, maxy)} boxes (no gpkg)."""
    def _tree(d):
        return STRtree([box(*b) for b in d.values()])
    return CityMap(map_name=MAP, version=VERSION, crs="epsg:32617",
                   lane_fids=np.array(list(lanes), dtype=np.int64), lane_tree=_tree(lanes),
                   conn_ids=np.array(list(conns), dtype=np.int64), conn_tree=_tree(conns),
                   inter_fids=np.array(list(inters), dtype=np.int64), inter_tree=_tree(inters),
                   outgoing=outgoing or {})


def random_points(rng: random.Random, n: int) -> tuple[list, list]:
    """Points on a half-unit grid so many land exactly on box edges and corners."""
    return ([rng.randint(-2, 52) / 2 for _ in range(n)], [rng.randint(-2, 52) / 2 for _ in range(n)])


@needs_geo
class GeometryRuleExampleTests(unittest.TestCase):
    def test_point_on_polygon_edge_is_inside(self):
        city = make_city({1: (0, 0, 4, 2)}, {}, {5: (4, 0, 8, 8)})
        out = geometry_features(city, pd.DataFrame({"ego_x": [4.0, 0.0, 9.0], "ego_y": [1.0, 2.0, 1.0]}))
        self.assertEqual(out["lane_type"].tolist(), ["lane", "lane", "none"])  # edge, corner, outside
        self.assertEqual(out["in_intersection"].tolist(), [True, False, False])
        self.assertEqual(out["intersection_fids"].tolist(), [[5], [], []])

    def test_relevant_connectors_use_exit_lane_fid(self):
        # outgoing is keyed by lane_connectors.exit_lane_fid: connectors 55, 56 leave lane 7
        outgoing = {7: [56, 55], 8: [57]}
        self.assertEqual(relevant_connectors([], [7], outgoing), ([55, 56], "ego_lane_outgoing"))
        self.assertEqual(relevant_connectors([57], [7], outgoing), ([57], "ego_connector"))
        self.assertEqual(relevant_connectors([], [9], outgoing), ([], "none"))
        self.assertEqual(relevant_connectors([], [], outgoing), ([], "none"))

    def test_relevant_connectors_from_loaded_map_follow_exit_lane(self):
        with tempfile.TemporaryDirectory() as tmp:
            maps = Path(tmp)
            make_gpkg(maps, lanes={7: box(X0 - 10, Y0 - 2, X0 + 10, Y0 + 2)},
                      conn_polys={900: (55, box(X0 + 10, Y0 - 2, X0 + 30, Y0 + 2))},
                      connectors={55: 7, 56: 7, 57: 8}, inters={})
            city = load_city_map(maps, MAP, VERSION)
        out = geometry_features(city, pd.DataFrame({"ego_x": [X0, X0 + 20], "ego_y": [Y0, Y0]}))
        self.assertEqual(out["relevant_connector_fids"].tolist(), [[55, 56], [55]])
        self.assertEqual(out["relevant_source"].tolist(), ["ego_lane_outgoing", "ego_connector"])

    def test_null_ego_point_gives_null_features(self):
        city = make_city({1: (0, 0, 4, 2)}, {2: (0, 0, 4, 2)}, {3: (0, 0, 4, 2)}, {1: [9]})
        frame = pd.DataFrame({"ego_x": [1.0, None, np.nan], "ego_y": [1.0, 1.0, 1.0]}, index=[10, 11, 12])
        out = geometry_features(city, frame)
        self.assertEqual(out.index.tolist(), [10, 11, 12])
        self.assertTrue(out["lane_type"].iloc[1:].isna().all())
        self.assertTrue(out["in_intersection"].iloc[1:].isna().all())
        self.assertEqual(out["ego_lane_fids"].tolist(), [[1], [], []])
        self.assertEqual(out["relevant_connector_fids"].tolist(), [[2], [], []])

    def test_point_hits_empty_inputs(self):
        city = make_city({1: (0, 0, 1, 1)}, {}, {})
        self.assertEqual(point_hits(city.lane_tree, city.lane_fids, np.array([]), np.array([])), [])
        self.assertEqual(point_hits(city.conn_tree, city.conn_ids, np.array([0.5]), np.array([0.5])), [[]])

    def test_resolve_traffic_light_examples(self):
        self.assertEqual(resolve_traffic_light([1, 2], [(1, "GREEN"), (2, "Red")]), ("red", 2, 0))
        self.assertEqual(resolve_traffic_light([1], [(1, "flashing"), (3, "red")]), ("unknown", 0, 1))
        self.assertEqual(resolve_traffic_light([], [(1, "red")]), ("none", 0, 0))


@needs_geo
class GeometryRulePropertyTests(unittest.TestCase):
    def test_lane_type_decision_table(self):
        """Feature: map-features, Property 1: lane_type decision table"""
        rng = random.Random(1)
        for case in range(100):
            lanes = random_boxes(rng, rng.randint(0, 4), 1)
            conns = random_boxes(rng, rng.randint(0, 4), 100)
            city = make_city(lanes, conns, {})
            xs, ys = random_points(rng, 30)
            out = geometry_features(city, pd.DataFrame({"ego_x": xs, "ego_y": ys}))
            for i, (px, py) in enumerate(zip(xs, ys)):
                in_lane = any(covers(b, px, py) for b in lanes.values())
                in_conn = any(covers(b, px, py) for b in conns.values())
                expected = "lane" if in_lane else "lane_connector" if in_conn else "none"
                with self.subTest(case=case, point=(px, py)):
                    self.assertEqual(out["lane_type"].iloc[i], expected)
                    self.assertEqual(out["ego_lane_fids"].iloc[i],
                                     sorted(f for f, b in lanes.items() if covers(b, px, py)))
                    self.assertEqual(classify_lane_type(out["ego_lane_fids"].iloc[i],
                                                        out["ego_connector_fids"].iloc[i]), expected)

    def test_in_intersection_matches_coverage(self):
        """Feature: map-features, Property 2: in_intersection matches coverage"""
        rng = random.Random(2)
        for case in range(100):
            inters = random_boxes(rng, rng.randint(0, 4), 1)
            city = make_city({}, {}, inters)
            xs, ys = random_points(rng, 30)
            out = geometry_features(city, pd.DataFrame({"ego_x": xs, "ego_y": ys}))
            for i, (px, py) in enumerate(zip(xs, ys)):
                expected = sorted(f for f, b in inters.items() if covers(b, px, py))
                with self.subTest(case=case, point=(px, py)):
                    self.assertEqual(bool(out["in_intersection"].iloc[i]), bool(expected))
                    self.assertEqual(out["intersection_fids"].iloc[i], expected)

    def test_traffic_light_resolution(self):
        """Feature: map-features, Property 3: Traffic-light resolution"""
        rng = random.Random(3)
        statuses = ["red", "yellow", "green", "RED", "Yellow", "GrEeN"]
        junk = ["unknown", "flashing", "", "off"]
        for case in range(100):
            relevant = rng.sample(range(1, 20), rng.randint(0, 4))
            ids = relevant + list(range(20, 25))
            records = [(rng.choice(ids), rng.choice(statuses + junk)) for _ in range(rng.randint(0, 8))]
            rel = set(relevant)
            known = [s.lower() for c, s in records if c in rel and s.lower() in RESTRICTIVENESS]
            if not relevant:
                expected = "none"
            elif not known:
                expected = "unknown"
            else:
                expected = max(known, key=RESTRICTIVENESS.get)
            with self.subTest(case=case, relevant=relevant, records=records):
                state, matched, ignored = resolve_traffic_light(relevant, records)
                self.assertEqual(state, expected)
                if relevant:
                    self.assertEqual(matched, len(known))
                shuffled = records[:]
                rng.shuffle(shuffled)
                noise = [(rng.randint(20, 24), rng.choice(statuses)) for _ in range(3)]
                noise += [(rng.choice(relevant), rng.choice(junk))] if relevant else []
                self.assertEqual(resolve_traffic_light(relevant, shuffled + noise)[0], expected)


try:  # the DB readers need only numpy and pandas
    from tools.extract_map_features import TOKEN_CHUNK, read_start_status, status_worker  # noqa: E402
except ImportError:
    read_start_status = None

needs_db = unittest.skipUnless(read_start_status, "numpy and pandas are required")


def make_status_db(path: Path, records: list[tuple[str, int, str]]) -> Path:
    """Log DB with the nuPlan traffic_light_status columns; records are (token_hex, connector, status)."""
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE traffic_light_status (token BLOB PRIMARY KEY, lidar_pc_token BLOB, "
                "lane_connector_id INTEGER, status VARCHAR(8))")
    con.executemany("INSERT INTO traffic_light_status VALUES (?, ?, ?, ?)",
                    [(struct.pack(">Q", k) * 2, bytes.fromhex(t), c, s)
                     for k, (t, c, s) in enumerate(records)])
    con.commit()
    con.close()
    return path


@needs_db
class StartStatusTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_start_frame_isolation(self):
        """Feature: map-features, Property 4: Start-frame isolation"""
        rng = random.Random(4)
        for case in range(100):
            tokens = sorted({rng.getrandbits(128).to_bytes(16, "big").hex() for _ in range(rng.randint(1, 8))})
            records = [(rng.choice(tokens), rng.randint(1, 60), rng.choice(["red", "green", "yellow", "unknown"]))
                       for _ in range(rng.randint(0, 25))]
            start = rng.sample(tokens, rng.randint(0, len(tokens)))
            start += [rng.getrandbits(128).to_bytes(16, "big").hex()]  # a token with no records
            query = [t.upper() if rng.random() < 0.3 else t for t in start]
            db = make_status_db(self.dir / f"log_{case}.db", records)
            wanted = {t.lower() for t in start}
            with self.subTest(case=case):
                self.assertEqual(read_start_status(db, query), sorted(r for r in records if r[0] in wanted))

    def test_many_tokens_are_chunked(self):
        tokens = [k.to_bytes(16, "big").hex() for k in range(TOKEN_CHUNK * 2 + 7)]
        records = [(t, k, "green") for k, t in enumerate(tokens)]
        db = make_status_db(self.dir / "big.db", records)
        self.assertEqual(read_start_status(db, tokens[1:]), records[1:])
        self.assertEqual(read_start_status(db, []), [])

    def test_worker_returns_rows(self):
        token = "ab" * 16
        db = make_status_db(self.dir / "ok.db", [(token, 5, "red"), ("cd" * 16, 6, "green")])
        self.assertEqual(status_worker(("log_a", db, [token])), {"log_name": "log_a", "rows": [(token, 5, "red")]})

    def test_unreadable_db_yields_error_entry(self):
        bad = self.dir / "bad.db"
        bad.write_bytes(b"this is not a sqlite database" * 100)
        for path in (bad, self.dir / "missing.db"):
            with self.subTest(path=path.name):
                out = status_worker(("log_b", path, ["ab" * 16]))
                self.assertEqual(out["log_name"], "log_b")
                self.assertNotIn("rows", out)
                self.assertTrue(out["error"])

    def test_db_is_opened_read_only(self):
        token = "ef" * 16
        db = make_status_db(self.dir / "ro.db", [(token, 1, "red")])
        before = db.read_bytes()
        read_start_status(db, [token])
        self.assertEqual(db.read_bytes(), before)
        self.assertFalse((self.dir / "ro.db-journal").exists())


if shapely is not None:
    import contextlib
    import io
    import zipfile
    import pyarrow.parquet as pq
    from tools.extract_map_features import (  # noqa: E402
        MAP_COLUMNS, AUDIT_COLUMNS, SUBSET_CANDIDATES, SUBSET_MINI, assemble_features, audit_table, main,
        map_table, reference_input, write_parquet,
    )


def synthetic_frame(rng: random.Random, n: int) -> "pd.DataFrame":
    """Scenario frame with null ego points, unknown/null maps, null tokens, over a few logs per subset."""
    rows = []
    for k in range(n):
        subset = rng.choice([SUBSET_CANDIDATES, SUBSET_MINI])
        null_ego = rng.random() < 0.2
        rows.append({
            "subset": subset, "scenario_id": f"s{k:04d}", "log_name": f"log{rng.randint(0, 4)}",
            "map_name": rng.choice([MAP, MAP, MAP, "unknown-map", None]), "map_version": VERSION,
            "db_file": None,
            "ego_x": None if null_ego else rng.randint(-2, 52) / 2,
            "ego_y": None if null_ego else rng.randint(-2, 52) / 2,
            "start_lidar_token": None if rng.random() < 0.15 else f"{rng.randint(0, 5):016x}",
        })
    df = pd.DataFrame(rows)
    for c in ("subset", "scenario_id", "log_name", "map_name", "map_version", "db_file", "start_lidar_token"):
        df[c] = df[c].astype("string")
    return df


@needs_geo
class AssemblyPropertyTests(unittest.TestCase):
    def test_missing_indicators_and_row_coverage(self):
        """Feature: map-features, Property 5: Missing indicators and row coverage"""
        rng = random.Random(5)
        for case in range(100):
            frame = synthetic_frame(rng, rng.randint(0, 25))
            city = make_city(random_boxes(rng, 3, 1), random_boxes(rng, 3, 100), random_boxes(rng, 2, 1),
                             {1: [7], 2: [8]})
            keys = [(s, f"log{j}") for s in (SUBSET_CANDIDATES, SUBSET_MINI) for j in range(5)]
            rows, errors = {}, {}
            for key in keys:  # each log: readable, unreadable/missing DB, or never read
                r = rng.random()
                if r < 0.5:
                    rows[key] = [(f"{rng.randint(0, 5):016x}", rng.choice([7, 8, 100, 101]),
                                  rng.choice(["red", "green", "yellow", "off"])) for _ in range(rng.randint(0, 6))]
                elif r < 0.8:
                    errors[key] = "FileNotFoundError: missing.db"
            out = assemble_features(frame, {MAP: city}, rows, errors)
            with self.subTest(case=case):
                self.assertEqual(out["scenario_id"].tolist(), frame["scenario_id"].tolist())
                for subset in (SUBSET_CANDIDATES, SUBSET_MINI):
                    t = map_table(out, subset)
                    self.assertEqual(list(t.columns), MAP_COLUMNS)
                    self.assertEqual(sorted(t["scenario_id"]),
                                     sorted(frame.loc[frame["subset"] == subset, "scenario_id"]))
                    for f in ("lane_type", "in_intersection", "traffic_light_state"):
                        self.assertEqual(t[f"{f}_missing"].tolist(), t[f].isna().tolist())
                for i, r in frame.iterrows():
                    geo_missing = pd.isna(r["ego_x"]) or r["map_name"] is pd.NA or r["map_name"] != MAP
                    key = (r["subset"], r["log_name"])
                    tl_missing = (geo_missing or pd.isna(r["start_lidar_token"])
                                  or key in errors or key not in rows)
                    self.assertEqual(bool(out["lane_type_missing"].iat[i]), bool(geo_missing))
                    self.assertEqual(bool(out["in_intersection_missing"].iat[i]), bool(geo_missing))
                    self.assertEqual(bool(out["traffic_light_state_missing"].iat[i]), bool(tl_missing))
                self.assertEqual(len(audit_table(out)), len(frame))


def write_inputs(root: Path, drop_ego_column: str | None = None) -> list[str]:
    """Synthetic build inputs in `root`; returns the CLI flags pointing at them.

    Trainval: logA (DB present, ego on lane 7 / connector 55 / off-map / null point), logB (DB missing).
    Mini: logM in a zip under data/cache/mini/, one scenario on connector 55 and one on an unknown map.
    """
    maps = root / "maps"
    make_gpkg(maps, lanes={7: box(X0 - 10, Y0 - 2, X0 + 10, Y0 + 2)},
              conn_polys={900: (55, box(X0 + 10, Y0 - 2, X0 + 30, Y0 + 2))},
              connectors={55: 7, 56: 7}, inters={3: box(X0 + 8, Y0 - 20, X0 + 40, Y0 + 20)})
    (maps / "nuplan-maps-v1.0.json").write_text(f'{{"{MAP}": {{"version": "{VERSION}"}}}}', encoding="utf-8")
    t = {k: f"{k:016x}" for k in range(1, 9)}
    ego = pd.DataFrame({
        "scenario_id": ["logA:1", "logA:2", "logA:3", "logA:4", "logB:5"],
        "log_name": ["logA"] * 4 + ["logB"],
        "ego_x": [X0, X0 + 20, X0 - 500, None, X0], "ego_y": [Y0, Y0, Y0, None, Y0]})
    mini_ego = pd.DataFrame({"scenario_id": ["logM:6", "logM:7"], "log_name": ["logM", "logM"],
                             "ego_x": [X0 + 20, X0], "ego_y": [Y0, Y0]})
    if drop_ego_column:
        ego = ego.drop(columns=[drop_ego_column])
    ego.to_parquet(root / "ego.parquet", index=False)
    mini_ego.to_parquet(root / "mini_ego.parquet", index=False)
    pd.DataFrame({
        "subset": [SUBSET_CANDIDATES] * 5 + [SUBSET_MINI] * 2,
        "scenario_id": ["logA:1", "logA:2", "logA:3", "logA:4", "logB:5", "logM:6", "logM:7"],
        "start_lidar_token": [t[1], t[2], t[3], t[4], t[5], t[6], t[7]]}).to_parquet(root / "align.parquet")
    pd.DataFrame({
        "scenario_id": ["logA:1", "logA:2", "logA:3", "logA:4", "logB:5", "logX:9"],
        "map_name": [MAP] * 6, "db_file": ["logA.db"] * 4 + ["logB.db", "logX.db"],
        "benchmark_candidate": [True] * 5 + [False]}).to_parquet(root / "elig.parquet")
    pd.DataFrame({"scenario_id": ["logM:6", "logM:7"], "log_name": ["logM", "logM"],
                  "db_file": ["logM.db", "logM.db"], "map_name": [MAP, "xx-unknown"]}).to_csv(
        root / "mini.csv", index=False)
    db_dir = root / "trainval"
    db_dir.mkdir()
    make_status_db(db_dir / "logA.db", [(t[1], 55, "green"), (t[1], 56, "RED"), (t[2], 55, "green"),
                                        (t[8], 55, "red")])
    make_status_db(root / "logM.db", [(t[6], 55, "yellow"), (t[6], 99, "red")])
    with zipfile.ZipFile(root / "mini.zip", "w") as z:
        z.write(root / "logM.db", "data/cache/mini/logM.db")
    (root / "logM.db").unlink()
    return ["--ego", str(root / "ego.parquet"), "--mini-ego", str(root / "mini_ego.parquet"),
            "--alignment", str(root / "align.parquet"), "--eligibility", str(root / "elig.parquet"),
            "--mini-sample", str(root / "mini.csv"), "--db-dir", str(db_dir), "--mini-zip", str(root / "mini.zip"),
            "--maps-dir", str(maps), "--out-dir", str(root / "out"), "--scratch", str(root / "scratch"),
            "--reference-dir", str(root / "ref"), "--report-dir", str(root / "report")]


def run_main(argv: list[str]) -> tuple[int, str]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = main(argv)
    return code, buf.getvalue()


@needs_geo
class BuildCliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_end_to_end_build(self):
        flags = write_inputs(self.root)
        code, log = run_main(["build", *flags, "--workers", "2"])
        self.assertEqual(code, 0, log)
        out = self.root / "out"
        self.assertEqual(sorted(p.name for p in out.iterdir()),
                         ["scenario_map.parquet", "scenario_map_alignment.parquet", "scenario_map_mini_68.parquet"])
        cand = pd.read_parquet(out / "scenario_map.parquet").set_index("scenario_id")
        self.assertEqual(list(cand.reset_index().columns), MAP_COLUMNS)
        self.assertEqual(cand["lane_type"].tolist(), ["lane", "lane_connector", "none", None, "lane"])
        self.assertEqual(cand["in_intersection"].tolist(), [False, True, False, None, False])
        # logA:1 on lane 7 -> outgoing 55 (green) + 56 (RED) -> red; logA:2 on connector 55 -> green;
        # logA:3 off-map -> none; logA:4 null ego; logB DB missing -> null
        self.assertEqual(cand["traffic_light_state"].tolist(), ["red", "green", "none", None, None])
        self.assertEqual(cand["traffic_light_state_missing"].tolist(), [False, False, False, True, True])
        mini = pd.read_parquet(out / "scenario_map_mini_68.parquet")
        self.assertEqual(mini["scenario_id"].tolist(), ["logM:6", "logM:7"])
        self.assertEqual(mini["traffic_light_state"].tolist(), ["yellow", None])  # logM:7 has an unknown map
        self.assertEqual(mini["lane_type_missing"].tolist(), [False, True])
        audit = pd.read_parquet(out / "scenario_map_alignment.parquet")
        self.assertEqual(list(audit.columns), AUDIT_COLUMNS)
        row = audit.set_index("scenario_id").loc["logA:1"]
        self.assertEqual(list(row["ego_lane_fids"]), [7])
        self.assertEqual(list(row["relevant_connector_fids"]), [55, 56])
        self.assertEqual(row["relevant_source"], "ego_lane_outgoing")
        self.assertEqual(row["status_record_count"], 2)
        ref = pd.read_parquet(self.root / "ref" / "mini_68_reference_input.parquet")
        self.assertEqual([dict(d) for d in ref.loc[0, "status_records"]],
                         [{"lane_connector_id": 55, "status": "yellow"}, {"lane_connector_id": 99, "status": "red"}])
        meta = pq.read_schema(out / "scenario_map.parquet").metadata[b"map_features_provenance"]
        prov = json.loads(meta)
        self.assertEqual(prov["maps"][MAP]["crs"], "epsg:32617")
        self.assertIn("size_bytes", prov["inputs"]["ego"])
        self.assertIn("trainval_candidates/logB", prov["status_errors"])
        self.assertFalse(any((self.root / "scratch").rglob("*.db")))  # mini DBs deleted

    def test_limit_logs_writes_smoke_outputs(self):
        flags = write_inputs(self.root)
        code, log = run_main(["build", *flags, "--workers", "1", "--limit-logs", "1"])
        self.assertEqual(code, 0, log)
        cand = pd.read_parquet(self.root / "out" / "scenario_map_smoke.parquet")
        self.assertEqual(cand["scenario_id"].tolist(), ["logA:1", "logA:2", "logA:3", "logA:4"])
        self.assertTrue((self.root / "ref" / "mini_68_reference_input_smoke.parquet").exists())
        self.assertFalse((self.root / "out" / "scenario_map.parquet").exists())

    def test_missing_input_path_exits_2_without_outputs(self):
        flags = write_inputs(self.root)
        flags[flags.index("--mini-zip") + 1] = str(self.root / "nope.zip")
        code, log = run_main(["build", *flags])
        self.assertEqual(code, 2)
        self.assertIn(f"missing input: {self.root / 'nope.zip'}", log)
        self.assertFalse((self.root / "out").exists())
        self.assertFalse((self.root / "ref").exists())

    def test_missing_column_exits_2_and_is_named(self):
        flags = write_inputs(self.root, drop_ego_column="ego_y")
        code, log = run_main(["build", *flags])
        self.assertEqual(code, 2)
        self.assertIn("missing columns ego_y", log)
        self.assertIn("ego.parquet", log)
        self.assertFalse((self.root / "out").exists())

    def test_write_parquet_leaves_no_tmp(self):
        df = pd.DataFrame({"scenario_id": ["a"], "lane_type": pd.array(["lane"], dtype="string")})
        path = self.root / "x" / "t.parquet"
        write_parquet(df, path, {"k": 1})
        self.assertEqual([p.name for p in path.parent.iterdir()], ["t.parquet"])
        self.assertEqual(json.loads(pq.read_schema(path).metadata[b"map_features_provenance"]), {"k": 1})

    def test_check_and_verify_mini_are_registered(self):
        for cmd in ("check", "verify-mini"):
            with self.subTest(cmd=cmd):
                code, log = run_main([cmd])
                self.assertEqual(code, 1)
                self.assertIn("not implemented", log)


if __name__ == "__main__":
    unittest.main()
