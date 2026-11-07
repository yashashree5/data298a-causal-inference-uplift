"""nuPlan devkit reference for the map features of the 68 mini scenarios (verify-mini input).

Reads `mini_68_reference_input.parquet` written by `extract_map_features build` (scenario_id, map_name,
ego_x, ego_y, start_lidar_token, status_records = list of {lane_connector_id, status} at the start
frame, unfiltered) and recomputes, with the devkit map API instead of the GeoPackage reader:
    lane_type            "lane" if a LANE contains the ego point, else "lane_connector" if a
                         LANE_CONNECTOR does, else "none"
    in_intersection      an INTERSECTION contains the ego point
    traffic_light_state  most restrictive red > yellow > green status among records of the relevant
                         connectors (the ego connectors, else the outgoing_edges of the ego lanes);
                         "unknown" if none matches, "none" if there are no relevant connectors

nuplan is imported only inside main() after argument parsing, so `--help` works without it.

Run inside the planner image (only when the planner containers are idle):
    docker compose --project-directory . -f infra/docker/compose.planners.yaml \
        run --rm planner python -m tools.devkit_map_reference
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple

RESTRICTIVENESS = {"red": 3, "yellow": 2, "green": 1}
OUTPUT_COLUMNS = (
    "scenario_id", "lane_type", "in_intersection", "traffic_light_state",
    "lane_ids", "lane_connector_ids", "intersection_ids", "relevant_connector_ids",
    "relevant_source", "matched_record_count", "error",
)


# ------------------------------------------------------------------------------- pure rules --

def classify_lane_type(lane_ids: Sequence[str], connector_ids: Sequence[str]) -> str:
    """A lane wins over a connector; neither gives "none"."""
    if len(lane_ids) > 0:
        return "lane"
    if len(connector_ids) > 0:
        return "lane_connector"
    return "none"


def relevant_connector_ids(connector_ids: Iterable[Any], outgoing_ids: Iterable[Any]) -> Tuple[List[str], str]:
    """Ego connectors if any, else the connectors leaving the ego lanes. Ids are compared as strings."""
    conns = sorted({str(c) for c in connector_ids})
    if conns:
        return conns, "ego_connector"
    out = sorted({str(c) for c in outgoing_ids})
    if out:
        return out, "ego_lane_outgoing"
    return [], "none"


def normalize_records(value: Any) -> List[Tuple[str, Optional[str]]]:
    """Status records from parquet (None, list or array of dicts / pairs) as (id string, status)."""
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    records = []
    for rec in value:
        if rec is None:
            continue
        if isinstance(rec, dict):
            cid, status = rec.get("lane_connector_id"), rec.get("status")
        else:
            cid, status = rec[0], rec[1]
        if cid is None:
            continue
        records.append((str(cid), status))
    return records


def resolve_traffic_light(relevant: Sequence[str], records: Iterable[Tuple[str, Optional[str]]]) -> Tuple[str, int]:
    """(state, matched_record_count). Statuses are case-insensitive; other strings are ignored."""
    if len(relevant) == 0:
        return "none", 0
    wanted = {str(r) for r in relevant}
    best, matched = None, 0
    for cid, status in records:
        if str(cid) not in wanted or status is None:
            continue
        s = str(status).strip().lower()
        if s not in RESTRICTIVENESS:
            continue
        matched += 1
        if best is None or RESTRICTIVENESS[s] > RESTRICTIVENESS[best]:
            best = s
    return (best if best is not None else "unknown"), matched


def _finite(v: Any) -> bool:
    try:
        return v is not None and math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


# ------------------------------------------------------------------------------------- devkit --

def reference_row(api: Any, row: dict, point_cls: Any, layers: Any) -> dict:
    """Devkit features for one input row (api is a NuPlanMap for row["map_name"])."""
    p = point_cls(float(row["ego_x"]), float(row["ego_y"]))
    lanes = api.get_all_map_objects(p, layers.LANE)
    conns = api.get_all_map_objects(p, layers.LANE_CONNECTOR)
    inter = api.get_all_map_objects(p, layers.INTERSECTION)
    lane_ids = sorted(str(o.id) for o in lanes)
    conn_ids = sorted(str(o.id) for o in conns)
    outgoing = [e.id for lane in lanes for e in lane.outgoing_edges] if not conn_ids else []
    relevant, source = relevant_connector_ids(conn_ids, outgoing)
    state, matched = resolve_traffic_light(relevant, normalize_records(row.get("status_records")))
    return {
        "scenario_id": row["scenario_id"],
        "lane_type": classify_lane_type(lane_ids, conn_ids),
        "in_intersection": bool(inter),
        "traffic_light_state": state,
        "lane_ids": lane_ids,
        "lane_connector_ids": conn_ids,
        "intersection_ids": sorted(str(o.id) for o in inter),
        "relevant_connector_ids": relevant,
        "relevant_source": source,
        "matched_record_count": matched,
        "error": None,
    }


def empty_row(scenario_id: Any, error: str) -> dict:
    row = {c: None for c in OUTPUT_COLUMNS}
    row.update(scenario_id=scenario_id, error=error)
    for c in ("lane_ids", "lane_connector_ids", "intersection_ids", "relevant_connector_ids"):
        row[c] = []
    return row


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Devkit map-feature reference for the mini scenarios (run in the planner image).")
    parser.add_argument("--input", type=Path,
                        default=Path("/artifacts/map_reference/mini_68_reference_input.parquet"))
    parser.add_argument("--output", type=Path,
                        default=Path("/artifacts/map_reference/devkit_map_mini_68.parquet"))
    parser.add_argument("--maps-root", default="/data/maps")
    parser.add_argument("--map-version", default="nuplan-maps-v1.0")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if not args.input.is_file():
        print(f"missing input: {args.input}", file=sys.stderr)
        return 2

    import pandas as pd
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    from nuplan.common.maps.nuplan_map.map_factory import get_maps_api

    frame = pd.read_parquet(args.input)
    required = ["scenario_id", "map_name", "ego_x", "ego_y", "status_records"]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        print(f"missing columns in {args.input}: {', '.join(missing)}", file=sys.stderr)
        return 2

    apis: dict = {}
    rows = []
    for row in frame.to_dict("records"):
        sid, map_name = row["scenario_id"], row["map_name"]
        if not map_name or not (_finite(row["ego_x"]) and _finite(row["ego_y"])):
            rows.append(empty_row(sid, "missing map_name or ego point"))
            continue
        try:
            if map_name not in apis:
                apis[map_name] = get_maps_api(args.maps_root, args.map_version, map_name)
            rows.append(reference_row(apis[map_name], row, Point2D, SemanticMapLayer))
        except Exception as exc:  # recorded per scenario; verify-mini reports the disagreement
            rows.append(empty_row(sid, f"{type(exc).__name__}: {exc}"))

    out = pd.DataFrame(rows, columns=list(OUTPUT_COLUMNS))
    out = out.sort_values("scenario_id", kind="mergesort").reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_name(args.output.name + ".tmp")
    out.to_parquet(tmp, index=False)
    os.replace(tmp, args.output)

    errors = int(out["error"].notna().sum())
    print(f"wrote {args.output} ({len(out)} rows, {errors} errors)")
    for f in ("lane_type", "in_intersection", "traffic_light_state"):
        counts = out[f].value_counts(dropna=False).to_dict()
        print(f"  {f}: {counts}")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
