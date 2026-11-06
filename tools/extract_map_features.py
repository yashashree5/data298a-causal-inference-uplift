"""Map-context features at the simulation-start ego pose, for nuPlan scenarios.

At the ego rear-axle position of the start lidar frame (see extract_ego_agent_features):
    lane_type            lane / lane_connector / none (a lane wins over a connector)
    in_intersection      ego point covered by an intersection polygon (boundary counts as inside)
    traffic_light_state  most restrictive red > yellow > green status at the start frame among the
                         relevant lane connectors; unknown if none is recorded, none if there are no
                         relevant connectors

Map geometry is read from each city's map.gpkg with sqlite3 (GeoPackage blob = "GP" header + WKB).
Layers are stored in WGS84 lon/lat (EPSG:4326) and are reprojected once per city to the map's
meta.projectedCoordSystem (UTM) with always_xy=True, the transform the devkit map loader applies.

Id conventions (design D5-D7):
    lanes_polygons.fid                                 lane id (equals lane_fid)
    gen_lane_connectors_scaled_width_polygons.lane_connector_fid
                                                       connector id (= lane_connectors.fid)
    lane_connectors.exit_lane_fid                      upstream lane of a connector

Usage (from the repository root):
    python -m tools.extract_map_features build
    python -m tools.extract_map_features check
    python -m tools.extract_map_features verify-mini --reference <devkit parquet>
"""
from __future__ import annotations

if __package__ in (None, ""):
    from _bootstrap import bootstrap
else:
    from ._bootstrap import bootstrap

bootstrap()

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import sqlite3  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import zipfile  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

FEATURE_VERSION = "map-v1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LANE_TYPES = ("lane", "lane_connector", "none")
TL_STATES = ("red", "yellow", "green", "unknown", "none")
RESTRICTIVENESS = {"red": 3, "yellow": 2, "green": 1}
GEOGRAPHIC_CRS = "EPSG:4326"

LANE_LAYER = "lanes_polygons"
CONNECTOR_POLYGON_LAYER = "gen_lane_connectors_scaled_width_polygons"
CONNECTOR_LAYER = "lane_connectors"
INTERSECTION_LAYER = "intersections"

# GeoPackage envelope contents indicator (flags bits 1-3) -> envelope size in bytes
_ENVELOPE_BYTES = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}


# ---------------------------------------------------------------------------- GeoPackage I/O --

def strip_gpkg_header(blob: bytes) -> bytes:
    """WKB payload of a GeoPackage geometry blob (header: magic, version, flags, srs_id, envelope)."""
    if len(blob) < 8 or blob[:2] != b"GP":
        raise ValueError("not a GeoPackage geometry blob")
    flags = blob[3]
    code = (flags >> 1) & 0x07
    if code not in _ENVELOPE_BYTES:
        raise ValueError(f"invalid GeoPackage envelope code {code}")
    return bytes(blob[8 + _ENVELOPE_BYTES[code]:])


def read_layer(conn: sqlite3.Connection, table: str, columns: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    """Attribute columns and shapely geometries (source CRS) of a GeoPackage feature table."""
    import shapely

    cols = ", ".join(f'"{c}"' for c in columns)
    rows = conn.execute(f'SELECT {cols}, "geom" FROM "{table}" ORDER BY "fid"').fetchall()
    attrs = pd.DataFrame([r[:-1] for r in rows], columns=columns)
    wkb = [strip_gpkg_header(r[-1]) if r[-1] is not None else None for r in rows]
    geoms = shapely.from_wkb(np.array(wkb, dtype=object)) if wkb else np.empty(0, dtype=object)
    return attrs, geoms


def projected_crs(conn: sqlite3.Connection) -> str:
    """meta.projectedCoordSystem of a nuPlan map (e.g. 'epsg:32617')."""
    row = conn.execute("SELECT value FROM meta WHERE key = 'projectedCoordSystem'").fetchone()
    if row is None or not row[0]:
        raise ValueError("map.gpkg meta has no projectedCoordSystem")
    return str(row[0])


def reproject(geoms: np.ndarray, source_crs: str, target_crs: str) -> np.ndarray:
    """Transform shapely geometries between CRSs, x = lon/easting first (always_xy)."""
    import shapely
    from pyproj import Transformer

    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)

    def _xy(coords: np.ndarray) -> np.ndarray:
        x, y = transformer.transform(coords[:, 0], coords[:, 1])
        return np.column_stack([x, y])

    return shapely.transform(geoms, _xy)


@dataclass
class CityMap:
    """Map layers of one city, geometries already in projected (UTM) metres."""
    map_name: str
    version: str
    crs: str
    lane_fids: np.ndarray
    lane_tree: object
    conn_ids: np.ndarray
    conn_tree: object
    inter_fids: np.ndarray
    inter_tree: object
    outgoing: dict[int, list[int]] = field(default_factory=dict)


def gpkg_path(maps_dir: Path, map_name: str, version: str) -> Path:
    return Path(maps_dir) / map_name / version / "map.gpkg"


def load_city_map(maps_dir: Path, map_name: str, version: str) -> CityMap:
    """Load lanes, connector polygons, connectors and intersections of one city into STRtrees."""
    from shapely import STRtree

    path = gpkg_path(maps_dir, map_name, version)
    if not path.is_file():
        raise FileNotFoundError(path)
    conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        crs = projected_crs(conn)
        lanes, lane_geoms = read_layer(conn, LANE_LAYER, ["fid"])
        cpoly, cpoly_geoms = read_layer(conn, CONNECTOR_POLYGON_LAYER, ["lane_connector_fid"])
        inter, inter_geoms = read_layer(conn, INTERSECTION_LAYER, ["fid"])
        links = pd.DataFrame(conn.execute(
            f'SELECT fid, exit_lane_fid FROM "{CONNECTOR_LAYER}" ORDER BY fid').fetchall(),
            columns=["fid", "exit_lane_fid"])
    finally:
        conn.close()

    outgoing: dict[int, list[int]] = {}
    for fid, exit_lane in links.itertuples(index=False):
        if exit_lane is None or pd.isna(exit_lane):
            continue
        outgoing.setdefault(int(exit_lane), []).append(int(fid))

    def _tree(geoms: np.ndarray):
        return STRtree(reproject(geoms, GEOGRAPHIC_CRS, crs))

    return CityMap(
        map_name=map_name, version=version, crs=crs,
        lane_fids=lanes["fid"].to_numpy(dtype=np.int64), lane_tree=_tree(lane_geoms),
        conn_ids=cpoly["lane_connector_fid"].to_numpy(dtype=np.int64), conn_tree=_tree(cpoly_geoms),
        inter_fids=inter["fid"].to_numpy(dtype=np.int64), inter_tree=_tree(inter_geoms),
        outgoing=outgoing,
    )


# ------------------------------------------------------------------------------- pure rules --

def point_hits(tree, ids: np.ndarray, x: np.ndarray, y: np.ndarray) -> list[list[int]]:
    """Per point, the sorted unique ids of tree geometries that intersect it (boundary = inside).

    Points with a null/NaN coordinate get an empty list. One vectorised STRtree query for all points.
    """
    import shapely

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    out: list[list[int]] = [[] for _ in range(n)]
    valid = np.flatnonzero(np.isfinite(x) & np.isfinite(y))
    if n == 0 or len(valid) == 0 or len(ids) == 0:
        return out
    points = shapely.points(x[valid], y[valid])
    pairs = tree.query(points, predicate="intersects")
    if pairs.shape[1] == 0:
        return out
    point_idx = valid[pairs[0]]
    hit_ids = np.asarray(ids, dtype=np.int64)[pairs[1]]
    order = np.lexsort((hit_ids, point_idx))
    point_idx, hit_ids = point_idx[order], hit_ids[order]
    starts = np.flatnonzero(np.r_[True, point_idx[1:] != point_idx[:-1]])
    ends = np.r_[starts[1:], len(point_idx)]
    for s, e in zip(starts.tolist(), ends.tolist()):
        out[int(point_idx[s])] = sorted(set(hit_ids[s:e].tolist()))
    return out


def classify_lane_type(lane_hits: list[int], conn_hits: list[int]) -> str:
    """'lane' if any lane covers ego (a lane wins over a connector), else 'lane_connector', else 'none'."""
    if lane_hits:
        return "lane"
    if conn_hits:
        return "lane_connector"
    return "none"


def relevant_connectors(conn_hits: list[int], lane_hits: list[int],
                        outgoing: dict[int, list[int]]) -> tuple[list[int], str]:
    """Relevant_Connectors: the Ego_Connectors, else the connectors leaving the Ego_Lane(s).

    `outgoing` maps lane_connectors.exit_lane_fid (the upstream lane, design D5) to connector fids.
    Returns (sorted unique ids, source) with source in {"ego_connector", "ego_lane_outgoing", "none"}.
    """
    if conn_hits:
        return sorted({int(c) for c in conn_hits}), "ego_connector"
    out = sorted({int(c) for lane in lane_hits for c in outgoing.get(int(lane), ())})
    if out:
        return out, "ego_lane_outgoing"
    return [], "none"


def resolve_traffic_light(relevant: list[int], records: list[tuple[int, str]]) -> tuple[str, int, int]:
    """Most restrictive red > yellow > green status among records of relevant connectors.

    Statuses are compared case-insensitively. A matching record with any other status is ignored
    (counted in the third value) and does not count as a record. Returns
    (state, matched_record_count, ignored_status_count); state is 'unknown' when relevant connectors
    exist but none has a recognised record, and 'none' when there are no relevant connectors.
    """
    rel = {int(r) for r in relevant}
    if not rel:
        return "none", 0, 0
    best, matched, ignored = None, 0, 0
    for conn_id, status in records:
        try:
            cid = int(conn_id)
        except (TypeError, ValueError):
            continue
        if cid not in rel:
            continue
        s = str(status).strip().lower() if status is not None else ""
        if s not in RESTRICTIVENESS:
            ignored += 1
            continue
        matched += 1
        if best is None or RESTRICTIVENESS[s] > RESTRICTIVENESS[best]:
            best = s
    return (best if best is not None else "unknown"), matched, ignored


def geometry_features(city: CityMap, frame: pd.DataFrame) -> pd.DataFrame:
    """Geometry features for each row of `frame` (columns ego_x, ego_y; nullable), same index.

    Columns: lane_type, in_intersection (None for null ego points), ego_lane_fids,
    ego_connector_fids, intersection_fids, relevant_connector_fids (lists), relevant_source.
    """
    x = pd.to_numeric(frame["ego_x"], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    y = pd.to_numeric(frame["ego_y"], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    valid = np.isfinite(x) & np.isfinite(y)
    lanes = point_hits(city.lane_tree, city.lane_fids, x, y)
    conns = point_hits(city.conn_tree, city.conn_ids, x, y)
    inters = point_hits(city.inter_tree, city.inter_fids, x, y)

    lane_type: list[str | None] = []
    in_inter: list[bool | None] = []
    rel_ids: list[list[int]] = []
    rel_src: list[str | None] = []
    for i in range(len(x)):
        if not valid[i]:
            lane_type.append(None)
            in_inter.append(None)
            rel_ids.append([])
            rel_src.append(None)
            continue
        lane_type.append(classify_lane_type(lanes[i], conns[i]))
        in_inter.append(bool(inters[i]))
        ids, src = relevant_connectors(conns[i], lanes[i], city.outgoing)
        rel_ids.append(ids)
        rel_src.append(src)

    return pd.DataFrame({
        "lane_type": pd.array(lane_type, dtype="string"),
        "in_intersection": pd.array(in_inter, dtype="boolean"),
        "ego_lane_fids": lanes,
        "ego_connector_fids": conns,
        "intersection_fids": inters,
        "relevant_connector_fids": rel_ids,
        "relevant_source": pd.array(rel_src, dtype="string"),
    }, index=frame.index)


# ---------------------------------------------------------------------------------------- DB --

STATUS_TABLE = "traffic_light_status"
TOKEN_CHUNK = 500  # stays below SQLite's bound-variable limit (999 on old builds)


def read_start_status(db_path: Path, tokens_hex: list[str]) -> list[tuple[str, int, str]]:
    """traffic_light_status records at the given start lidar tokens only.

    Opens the log DB read-only and runs one `WHERE lidar_pc_token IN (...)` query per chunk of
    TOKEN_CHUNK tokens (one query for a typical log). Returns (token_hex lowercase,
    lane_connector_id, status) tuples sorted for determinism. Raises on unreadable DBs.
    """
    tokens = sorted({str(t).lower() for t in tokens_hex if t is not None and str(t)})
    if not tokens:
        return []
    blobs = [bytes.fromhex(t) for t in tokens]
    path = Path(db_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    rows: list[tuple[str, int, str]] = []
    try:
        for start in range(0, len(blobs), TOKEN_CHUNK):
            chunk = blobs[start:start + TOKEN_CHUNK]
            marks = ", ".join("?" * len(chunk))
            cur = conn.execute(
                f'SELECT lidar_pc_token, lane_connector_id, status FROM "{STATUS_TABLE}" '
                f"WHERE lidar_pc_token IN ({marks})", chunk)
            for token, conn_id, status in cur:
                if conn_id is None:
                    continue
                token_hex = bytes(token).hex() if isinstance(token, (bytes, bytearray, memoryview)) \
                    else str(token).lower()
                rows.append((token_hex, int(conn_id), "" if status is None else str(status)))
    finally:
        conn.close()
    rows.sort()
    return rows


def status_worker(task: tuple[str, Path, list[str]]) -> dict:
    """ProcessPoolExecutor task: start-frame records of one log; never raises.

    Returns {"log_name", "rows"} on success or {"log_name", "error"} with the exception text.
    """
    log_name, db_path, tokens_hex = task
    try:
        return {"log_name": log_name, "rows": read_start_status(Path(db_path), list(tokens_hex))}
    except Exception as exc:  # noqa: BLE001 - reported per log, the run continues
        return {"log_name": log_name, "error": f"{type(exc).__name__}: {exc}"}


# ------------------------------------------------------------------------------ build inputs --

DEFAULT_FEATURES_DIR = Path(r"E:\Project\nuplan\processed\features")
DEFAULT_ELIGIBILITY = Path(r"E:\Project\nuplan\processed\scenario_eligibility.parquet")
DEFAULT_DB_DIR = Path(r"C:\nuplan\dataset\nuplan-v1.1\trainval")
DEFAULT_MINI_ZIP = Path(r"E:\Project\nuplan\downloads\nuplan-v1.1_mini.zip")
DEFAULT_MAPS_DIR = Path(r"C:\nuplan\dataset\maps")
DEFAULT_SCRATCH = Path(r"E:\Project\nuplan\scratch")
DEFAULT_MINI_SAMPLE = PROJECT_ROOT / "configs" / "scenarios" / "idm_mini_sample.csv"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "docs" / "data"
DEFAULT_REFERENCE_DIR = PROJECT_ROOT / "artifacts" / "additional_planners" / "map_reference"
MAP_VERSIONS_FILE = "nuplan-maps-v1.0.json"
PROVENANCE_KEY = b"map_features_provenance"

SUBSET_CANDIDATES = "trainval_candidates"
SUBSET_MINI = "mini_68"

OUTPUT_STEMS = {
    "map": "scenario_map",
    "mini": "scenario_map_mini_68",
    "audit": "scenario_map_alignment",
    "reference": "mini_68_reference_input",
}
FEATURES = ("lane_type", "in_intersection", "traffic_light_state")
MAP_COLUMNS = ["scenario_id", "lane_type", "lane_type_missing", "in_intersection", "in_intersection_missing",
               "traffic_light_state", "traffic_light_state_missing"]
LIST_COLUMNS = ["ego_lane_fids", "ego_connector_fids", "intersection_fids", "relevant_connector_fids"]
COUNT_COLUMNS = ["status_record_count", "start_frame_record_count", "ignored_status_count"]
AUDIT_COLUMNS = (["subset", "scenario_id", "log_name", "map_name", "map_version", "map_crs", "start_lidar_token"]
                 + LIST_COLUMNS + ["relevant_source"] + COUNT_COLUMNS)

EGO_COLUMNS = ["scenario_id", "log_name", "ego_x", "ego_y"]
ALIGNMENT_COLUMNS = ["subset", "scenario_id", "start_lidar_token"]
ELIGIBILITY_COLUMNS = ["scenario_id", "map_name", "db_file"]
ELIGIBILITY_FILTER_COLUMN = "benchmark_candidate"
MINI_SAMPLE_COLUMNS = ["scenario_id", "map_name", "db_file"]
FRAME_COLUMNS = ["subset", "scenario_id", "log_name", "map_name", "map_version", "db_file",
                 "ego_x", "ego_y", "start_lidar_token"]


class InputError(Exception):
    """Invalid build inputs; each message is printed and the tool exits with code 2."""

    def __init__(self, messages: list[str]):
        super().__init__("; ".join(messages))
        self.messages = list(messages)


def input_paths(args) -> dict[str, Path]:
    """Every input `build` reads, by flag name."""
    return {
        "ego": Path(args.ego), "mini_ego": Path(args.mini_ego), "alignment": Path(args.alignment),
        "eligibility": Path(args.eligibility), "mini_sample": Path(args.mini_sample),
        "db_dir": Path(args.db_dir), "mini_zip": Path(args.mini_zip), "maps_dir": Path(args.maps_dir),
        "map_versions": Path(args.maps_dir) / MAP_VERSIONS_FILE,
    }


def _table_columns(path: Path) -> list[str]:
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as stream:
            return next(csv.reader(stream), [])
    import pyarrow.parquet as pq
    return list(pq.read_schema(path).names)


def require_columns(path: Path, needed: list[str]) -> list[str]:
    """Error messages for columns of `needed` that the table at `path` lacks (empty if none)."""
    available = set(_table_columns(path))
    missing = [c for c in needed if c not in available]
    return [f"{path}: missing columns {', '.join(missing)}"] if missing else []


def _unique_or_error(df: pd.DataFrame, name: str) -> list[str]:
    dup = int(df["scenario_id"].duplicated().sum())
    return [f"{name}: {dup} duplicate scenario_id values"] if dup else []


def load_frames(args) -> tuple[pd.DataFrame, dict[str, str]]:
    """Scenario frame (FRAME_COLUMNS, one row per Candidate and per Mini_Scenario) and map versions.

    The scenario universe is the ego tables (Candidate / Mini_Scenario definition); alignment,
    eligibility and the mini sample are left-joined, so a scenario missing there gets null values.
    Raises InputError before anything is written if a path or a column is missing.
    """
    paths = input_paths(args)
    missing = [f"missing input: {p}" for p in paths.values() if not p.exists()]
    if missing:
        raise InputError(missing)
    errors = (require_columns(paths["ego"], EGO_COLUMNS) + require_columns(paths["mini_ego"], EGO_COLUMNS)
              + require_columns(paths["alignment"], ALIGNMENT_COLUMNS)
              + require_columns(paths["eligibility"], ELIGIBILITY_COLUMNS + [ELIGIBILITY_FILTER_COLUMN])
              + require_columns(paths["mini_sample"], MINI_SAMPLE_COLUMNS))
    if errors:
        raise InputError(errors)

    versions_raw = json.loads(paths["map_versions"].read_text(encoding="utf-8"))
    versions = {str(k): str(v["version"]) for k, v in versions_raw.items()
                if isinstance(v, dict) and v.get("version")}

    ego = pd.read_parquet(paths["ego"], columns=EGO_COLUMNS)
    mini_ego = pd.read_parquet(paths["mini_ego"], columns=EGO_COLUMNS)
    align = pd.read_parquet(paths["alignment"], columns=ALIGNMENT_COLUMNS)
    elig = pd.read_parquet(paths["eligibility"], columns=ELIGIBILITY_COLUMNS,
                           filters=[(ELIGIBILITY_FILTER_COLUMN, "==", True)])
    mini = pd.read_csv(paths["mini_sample"], usecols=MINI_SAMPLE_COLUMNS, dtype=str)

    if getattr(args, "limit_logs", None):
        keep = sorted(ego["log_name"].dropna().unique())[: int(args.limit_logs)]
        ego = ego[ego["log_name"].isin(keep)]

    align_c = align.loc[align["subset"] == SUBSET_CANDIDATES, ["scenario_id", "start_lidar_token"]]
    align_m = align.loc[align["subset"] == SUBSET_MINI, ["scenario_id", "start_lidar_token"]]
    errors = (_unique_or_error(ego, "ego table") + _unique_or_error(mini_ego, "mini ego table")
              + _unique_or_error(align_c, f"alignment {SUBSET_CANDIDATES}")
              + _unique_or_error(align_m, f"alignment {SUBSET_MINI}")
              + _unique_or_error(elig, "eligibility candidates") + _unique_or_error(mini, "mini sample"))
    if errors:
        raise InputError(errors)

    cand = ego.merge(align_c, on="scenario_id", how="left").merge(elig, on="scenario_id", how="left")
    cand.insert(0, "subset", SUBSET_CANDIDATES)
    mini_f = mini_ego.merge(align_m, on="scenario_id", how="left").merge(mini, on="scenario_id", how="left")
    mini_f.insert(0, "subset", SUBSET_MINI)
    frame = pd.concat([cand, mini_f], ignore_index=True)
    frame["map_version"] = frame["map_name"].map(versions)
    for col in ("scenario_id", "log_name", "map_name", "map_version", "db_file", "start_lidar_token"):
        frame[col] = frame[col].astype("string")
    frame["start_lidar_token"] = frame["start_lidar_token"].str.lower()
    return frame[FRAME_COLUMNS].reset_index(drop=True), versions


# ----------------------------------------------------------------------------- pure assembly --

def assemble_features(frame: pd.DataFrame, cities: dict[str, CityMap],
                      status_rows: dict[tuple[str, str], list[tuple[str, int, str]]],
                      status_errors: dict[tuple[str, str], str]) -> pd.DataFrame:
    """Features, missing indicators and audit columns for every row of `frame`, same row order.

    `cities` holds the loaded City_Maps by map_name (a map absent here is unavailable).
    `status_rows` maps (subset, log_name) to the start-frame records read from that log;
    a log in `status_errors` or absent from `status_rows` has no usable status data.

    Null-out rules: null ego point or unavailable map -> all three features null;
    otherwise null start token or no usable status data -> traffic_light_state null.
    """
    df = frame.reset_index(drop=True)
    n = len(df)
    lane_type = np.full(n, None, dtype=object)
    in_inter = np.full(n, None, dtype=object)
    lists = {c: [[] for _ in range(n)] for c in LIST_COLUMNS}
    source = np.full(n, None, dtype=object)
    crs = np.full(n, None, dtype=object)

    for map_name, idx in df.groupby("map_name", dropna=True, sort=True).indices.items():
        city = cities.get(str(map_name))
        if city is None:
            continue
        geo = geometry_features(city, df.iloc[idx][["ego_x", "ego_y"]])
        lt = geo["lane_type"].astype(object).where(geo["lane_type"].notna(), None).to_numpy()
        ii = geo["in_intersection"].astype(object).where(geo["in_intersection"].notna(), None).to_numpy()
        rs = geo["relevant_source"].astype(object).where(geo["relevant_source"].notna(), None).to_numpy()
        lane_type[idx], in_inter[idx], source[idx] = lt, ii, rs
        crs[idx] = city.crs
        for c in LIST_COLUMNS:
            values = geo[c].tolist()
            for k, pos in enumerate(idx.tolist()):
                lists[c][pos] = [int(v) for v in values[k]]

    by_token: dict[tuple[str, str], dict[str, list[tuple[int, str]]]] = {}
    for key, rows in status_rows.items():
        grouped: dict[str, list[tuple[int, str]]] = {}
        for token, conn_id, status in rows:
            grouped.setdefault(str(token).lower(), []).append((int(conn_id), status))
        by_token[key] = grouped

    tl_state = np.full(n, None, dtype=object)
    counts = {c: np.full(n, None, dtype=object) for c in COUNT_COLUMNS}
    start_records: list[list[dict] | None] = [None] * n
    subsets = df["subset"].astype(object).to_numpy()
    logs = df["log_name"].astype(object).to_numpy()
    tokens = df["start_lidar_token"].astype(object).to_numpy()
    for i in range(n):
        token = tokens[i]
        if token is None or pd.isna(token):
            continue
        key = (str(subsets[i]), str(logs[i]))
        if key in status_errors or key not in by_token:
            continue
        records = by_token[key].get(str(token).lower(), [])
        start_records[i] = [{"lane_connector_id": c, "status": s} for c, s in records]
        counts["start_frame_record_count"][i] = len(records)
        if lane_type[i] is None:
            continue
        state, matched, ignored = resolve_traffic_light(lists["relevant_connector_fids"][i], records)
        tl_state[i] = state
        counts["status_record_count"][i] = matched
        counts["ignored_status_count"][i] = ignored

    out = df[["subset", "scenario_id", "log_name", "map_name", "map_version", "start_lidar_token",
              "ego_x", "ego_y"]].copy()
    out["map_crs"] = pd.array(crs, dtype="string")
    out["lane_type"] = pd.array(lane_type, dtype="string")
    out["in_intersection"] = pd.array(in_inter, dtype="boolean")
    out["traffic_light_state"] = pd.array(tl_state, dtype="string")
    for f in FEATURES:
        out[f"{f}_missing"] = out[f].isna().to_numpy(dtype=bool)
    for c in LIST_COLUMNS:
        out[c] = lists[c]
    out["relevant_source"] = pd.array(source, dtype="string")
    for c in COUNT_COLUMNS:
        out[c] = pd.array(counts[c], dtype="Int64")
    out["start_frame_records"] = start_records
    return out


def map_table(features: pd.DataFrame, subset: str) -> pd.DataFrame:
    """Model-ready Map_Table of one subset: scenario_id, three features, *_missing; sorted by id."""
    sub = features.loc[features["subset"] == subset, MAP_COLUMNS]
    return sub.sort_values("scenario_id", kind="mergesort").reset_index(drop=True)


def audit_table(features: pd.DataFrame) -> pd.DataFrame:
    return features[AUDIT_COLUMNS].sort_values(["subset", "scenario_id"], kind="mergesort").reset_index(drop=True)


def reference_input(features: pd.DataFrame) -> pd.DataFrame:
    """Devkit reference input for the Mini_Scenarios, with the unfiltered start-frame records."""
    sub = features.loc[features["subset"] == SUBSET_MINI,
                       ["scenario_id", "map_name", "ego_x", "ego_y", "start_lidar_token", "start_frame_records"]]
    sub = sub.rename(columns={"start_frame_records": "status_records"})
    return sub.sort_values("scenario_id", kind="mergesort").reset_index(drop=True)


# ------------------------------------------------------------------------------------ output --

def _arrow_schema(df: pd.DataFrame):
    """Explicit Arrow types for list and nullable columns, so all-empty columns keep their type."""
    import pyarrow as pa

    record = pa.struct([("lane_connector_id", pa.int64()), ("status", pa.string())])
    types = {c: pa.list_(pa.int64()) for c in LIST_COLUMNS}
    types.update({c: pa.int64() for c in COUNT_COLUMNS})
    types.update({"status_records": pa.list_(record), "in_intersection": pa.bool_(),
                  "ego_x": pa.float64(), "ego_y": pa.float64()})
    types.update({f"{f}_missing": pa.bool_() for f in FEATURES})
    return pa.schema([(c, types.get(c, pa.string())) for c in df.columns])


def write_parquet(df: pd.DataFrame, path: Path, meta: dict) -> None:
    """Write `df` with provenance metadata to `<path>.tmp`, then move it into place (os.replace)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, schema=_arrow_schema(df), preserve_index=False)
    table = table.replace_schema_metadata({**(table.schema.metadata or {}),
                                           PROVENANCE_KEY: json.dumps(meta, default=str).encode()})
    tmp = path.with_name(path.name + ".tmp")
    try:
        pq.write_table(table, tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def output_paths(args) -> dict[str, Path]:
    suffix = "_smoke" if getattr(args, "limit_logs", None) else ""
    out, ref = Path(args.out_dir), Path(args.reference_dir)
    return {"map": out / f"{OUTPUT_STEMS['map']}{suffix}.parquet",
            "mini": out / f"{OUTPUT_STEMS['mini']}{suffix}.parquet",
            "audit": out / f"{OUTPUT_STEMS['audit']}{suffix}.parquet",
            "reference": ref / f"{OUTPUT_STEMS['reference']}{suffix}.parquet"}


def _file_info(path: Path) -> dict:
    p = Path(path)
    info: dict = {"path": str(p)}
    if p.is_file():
        st = p.stat()
        info.update(size_bytes=st.st_size,
                    mtime_utc=datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat())
    elif p.is_dir():
        info["directory"] = True
    return info


def provenance(args, cities: dict[str, CityMap], map_errors: dict[str, str],
               status_errors: dict[tuple[str, str], str]) -> dict:
    import pyproj
    import shapely

    maps = {}
    for name, city in sorted(cities.items()):
        gpkg = gpkg_path(Path(args.maps_dir), name, city.version)
        maps[name] = {"version": city.version, "crs": city.crs, "gpkg": _file_info(gpkg)}
    return {
        "feature_version": FEATURE_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "argv": list(getattr(args, "argv", None) or sys.argv[1:]),
        "args": {k: str(v) for k, v in sorted(vars(args).items()) if k not in ("func", "argv")},
        "inputs": {k: _file_info(p) for k, p in input_paths(args).items()},
        "maps": maps,
        "map_errors": map_errors,
        "status_errors": {f"{s}/{log}": e for (s, log), e in sorted(status_errors.items())},
        "libraries": {"shapely": shapely.__version__, "pyproj": pyproj.__version__,
                      "pandas": pd.__version__},
    }


# ------------------------------------------------------------------------------- status reads --

def status_tasks(frame: pd.DataFrame, subset: str) -> list[tuple[str, str, list[str]]]:
    """(log_name, db file name, start tokens) per log of `subset` with at least one start token."""
    sub = frame[(frame["subset"] == subset) & frame["start_lidar_token"].notna() & frame["log_name"].notna()]
    tasks = []
    for log_name, g in sub.groupby("log_name", sort=True):
        db_files = g["db_file"].dropna()
        db_file = str(db_files.iat[0]) if len(db_files) else f"{log_name}.db"
        tasks.append((str(log_name), db_file, sorted(set(g["start_lidar_token"].astype(str)))))
    return tasks


def read_trainval_status(tasks: list[tuple[str, str, list[str]]], db_dir: Path, workers: int):
    """Start-frame records per trainval log, in a process pool (serial for workers <= 1)."""
    rows: dict[tuple[str, str], list] = {}
    errors: dict[tuple[str, str], str] = {}
    jobs = [(log, Path(db_dir) / db_file, tokens) for log, db_file, tokens in tasks]

    def _take(res: dict) -> None:
        key = (SUBSET_CANDIDATES, res["log_name"])
        if "error" in res:
            errors[key] = res["error"]
        else:
            rows[key] = res["rows"]

    t0 = time.time()
    if workers <= 1 or len(jobs) <= 1:
        for job in jobs:
            _take(status_worker(job))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(status_worker, job) for job in jobs]
            for done, fut in enumerate(as_completed(futures), start=1):
                _take(fut.result())
                if done % 500 == 0 or done == len(futures):
                    print(f"  status: {done}/{len(futures)} logs, {len(errors)} errors, "
                          f"{time.time() - t0:.0f} s", flush=True)
    return rows, errors


def read_mini_status(tasks: list[tuple[str, str, list[str]]], mini_zip: Path, scratch: Path):
    """Start-frame records per mini log: copy each needed DB from the zip, query it, delete it."""
    rows: dict[tuple[str, str], list] = {}
    errors: dict[tuple[str, str], str] = {}
    if not tasks:
        return rows, errors
    work = Path(scratch) / "map_mini"
    work.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(mini_zip) as z:
        entries = {Path(i.filename).stem: i for i in z.infolist() if i.filename.lower().endswith(".db")}
        for log_name, db_file, tokens in tasks:
            key = (SUBSET_MINI, log_name)
            entry = entries.get(Path(db_file).stem) or entries.get(log_name)
            if entry is None:
                errors[key] = f"not in {mini_zip}: {db_file}"
                continue
            db = work / f"{log_name}.db"
            try:
                with z.open(entry) as src, open(db, "wb") as dst:
                    shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
                rows[key] = read_start_status(db, tokens)
            except Exception as exc:  # noqa: BLE001 - reported per log, the run continues
                errors[key] = f"{type(exc).__name__}: {exc}"
            finally:
                db.unlink(missing_ok=True)
    try:
        work.rmdir()
    except OSError:
        pass
    return rows, errors


# ------------------------------------------------------------------------------------- build --

def load_cities(frame: pd.DataFrame, maps_dir: Path, versions: dict[str, str]):
    cities: dict[str, CityMap] = {}
    errors: dict[str, str] = {}
    for name in sorted(frame["map_name"].dropna().astype(str).unique()):
        version = versions.get(name)
        if version is None:
            errors[name] = f"map_name not in {MAP_VERSIONS_FILE}"
            continue
        try:
            t0 = time.time()
            cities[name] = load_city_map(maps_dir, name, version)
            print(f"  map {name} {version} ({cities[name].crs}) loaded in {time.time() - t0:.1f} s", flush=True)
        except Exception as exc:  # noqa: BLE001 - that city's features become null
            errors[name] = f"{type(exc).__name__}: {exc}"
    for name, err in errors.items():
        print(f"  map {name} unavailable: {err}", flush=True)
    return cities, errors


def build(args) -> int:
    t0 = time.time()
    try:
        frame, versions = load_frames(args)
    except InputError as exc:
        for msg in exc.messages:
            print(msg)
        return 2
    n_c = int((frame["subset"] == SUBSET_CANDIDATES).sum())
    n_m = int((frame["subset"] == SUBSET_MINI).sum())
    print(f"{n_c:,} candidates and {n_m} mini scenarios loaded in {time.time() - t0:.1f} s", flush=True)

    cities, map_errors = load_cities(frame, Path(args.maps_dir), versions)
    trainval_tasks = status_tasks(frame, SUBSET_CANDIDATES)
    print(f"reading start-frame status of {len(trainval_tasks)} trainval logs with {args.workers} workers",
          flush=True)
    rows, status_errors = read_trainval_status(trainval_tasks, Path(args.db_dir), int(args.workers))
    mini_rows, mini_errors = read_mini_status(status_tasks(frame, SUBSET_MINI), Path(args.mini_zip),
                                              Path(args.scratch))
    rows.update(mini_rows)
    status_errors.update(mini_errors)
    for (subset, log), err in sorted(status_errors.items())[:20]:
        print(f"  status {subset}/{log}: {err}")

    features = assemble_features(frame, cities, rows, status_errors)
    meta = provenance(args, cities, map_errors, status_errors)
    paths = output_paths(args)
    write_parquet(map_table(features, SUBSET_CANDIDATES), paths["map"], meta)
    write_parquet(map_table(features, SUBSET_MINI), paths["mini"], meta)
    write_parquet(audit_table(features), paths["audit"], meta)
    write_parquet(reference_input(features), paths["reference"], meta)
    for f in FEATURES:
        print(f"  {f}: {int(features[f'{f}_missing'].sum()):,} missing")
    print(f"{len(map_errors)} unavailable maps, {len(status_errors)} logs without status data")
    for p in paths.values():
        print(f"wrote {p}")
    print(f"build finished in {time.time() - t0:.1f} s")
    return 0


def check(args) -> int:
    """Placeholder: implemented in task 5.1 of the map-features spec."""
    print("check is not implemented yet (map-features task 5.1)")
    return 1


def verify_mini(args) -> int:
    """Placeholder: implemented in task 5.1 of the map-features spec."""
    print("verify-mini is not implemented yet (map-features task 5.1)")
    return 1


# --------------------------------------------------------------------------------------- CLI --

def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--ego", type=Path, default=DEFAULT_FEATURES_DIR / "scenario_ego_agent.parquet")
    common.add_argument("--mini-ego", type=Path, default=DEFAULT_FEATURES_DIR / "scenario_ego_agent_mini_68.parquet")
    common.add_argument("--alignment", type=Path,
                        default=DEFAULT_FEATURES_DIR / "scenario_ego_agent_alignment.parquet")
    common.add_argument("--eligibility", type=Path, default=DEFAULT_ELIGIBILITY)
    common.add_argument("--mini-sample", type=Path, default=DEFAULT_MINI_SAMPLE)
    common.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR)
    common.add_argument("--mini-zip", type=Path, default=DEFAULT_MINI_ZIP)
    common.add_argument("--maps-dir", type=Path, default=DEFAULT_MAPS_DIR,
                        help=f"city map folders and {MAP_VERSIONS_FILE}")
    common.add_argument("--out-dir", type=Path, default=DEFAULT_FEATURES_DIR)
    common.add_argument("--scratch", type=Path, default=DEFAULT_SCRATCH)
    common.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    common.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    common.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 4))
    common.add_argument("--frame-threshold", type=float, default=0.90)
    common.add_argument("--limit-logs", type=int, default=None,
                        help="smoke test: first N trainval logs; outputs get a _smoke suffix")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build", parents=[common], help="compute the map features").set_defaults(func=build)
    sub.add_parser("check", parents=[common], help="validate outputs and write the report").set_defaults(func=check)
    vm = sub.add_parser("verify-mini", parents=[common], help="compare mini_68 with the devkit reference")
    vm.add_argument("--reference", type=Path, default=None,
                    help="devkit parquet (default <reference-dir>/devkit_map_mini_68.parquet)")
    vm.set_defaults(func=verify_mini)
    return parser


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    args.argv = argv
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
