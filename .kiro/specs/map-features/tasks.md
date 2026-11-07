# Implementation Plan: map-features

## Overview

Build `tools/extract_map_features.py` bottom-up: GeoPackage I/O and reprojection, the pure geometry and traffic-light rules, the start-frame status reader, then `build`, `check` and `verify-mini`, plus `tools/devkit_map_reference.py`. Each code task carries its own `unittest` tests in `tests/test_extract_map_features.py` (property tests are 100-case `random.Random(seed)` loops tagged `Feature: map-features, Property N: <title>`). The plan ends with the real-data acceptance run and an optional Docker devkit comparison.

## Tasks

- [x] 1. Correct the frame assumption and add map loading
  - [x] 1.1 Fix the coordinate-frame assumption in requirements.md
    - Replace the Assumptions bullet saying GPKG coordinates are the local UTM frame: layers are WGS84 lon/lat (EPSG:4326) and are reprojected per city to `meta.projectedCoordSystem` (sg-one-north 32648, us-ma-boston 32619, us-nv-las-vegas-strip 32611, us-pa-pittsburgh-hazelwood 32617) with `always_xy=True`, as in design D2; Requirement 7 remains the guard
    - _Requirements: 7.1_
  - [x] 1.2 Implement GeoPackage I/O and `load_city_map` with tests
    - Create `tools/extract_map_features.py` (`from __future__ import annotations`, `_bootstrap` as sibling tools) with constants, `strip_gpkg_header`, `read_layer`, `projected_crs`, `CityMap` and `load_city_map` (lanes by `fid`, connector ids from `lane_connector_fid`, `outgoing` keyed by `exit_lane_fid`, STRtrees on reprojected geometry)
    - Tests: `strip_gpkg_header` round trip for envelope codes 0 and 1; tiny sqlite-built GeoPackage fixture reprojected 4326 → 32617 puts a known UTM point in its lane
    - _Requirements: 2.1, 3.1, 7.1_

- [x] 2. Implement geometry and traffic-light rules
  - [x] 2.1 Implement `point_hits`, `classify_lane_type`, `relevant_connectors`, `resolve_traffic_light`, `geometry_features` with tests
    - `intersects` predicate so boundary points count as inside; "lane wins" over connector; Relevant_Connectors from Ego_Connectors else `outgoing[ego lane]`; most restrictive red > yellow > green, case-insensitive, unknown statuses counted as ignored
    - Property tests on synthetic axis-aligned polygons: **Property 1: lane_type decision table**, **Property 2: in_intersection matches coverage**, **Property 3: Traffic-light resolution**
    - Example tests: point on polygon edge is inside; `relevant_connectors` uses `exit_lane_fid`
    - _Requirements: 2.1, 2.2, 2.3, 3.1, 3.2, 4.1, 4.2, 4.3, 9.1_

- [x] 3. Implement start-frame status reading
  - [x] 3.1 Implement `read_start_status` and `status_worker` with tests
    - One `WHERE lidar_pc_token IN (...)` query per log using `bytes.fromhex`; worker returns `{"log_name", "rows" | "error"}` instead of raising
    - Property test on a temporary SQLite `traffic_light_status` table with several tokens: **Property 4: Start-frame isolation**; test that an unreadable DB yields an `error` entry
    - _Requirements: 4.4, 4.5, 9.1_

- [x] 4. Implement `build` and the CLI
  - [x] 4.1 Implement `load_frames`, `write_parquet`, `build` and `main` with tests
    - Argparse subcommands and all flags/defaults from the design CLI table, including `--limit-logs` (outputs get `_smoke` suffix) and `--workers`
    - Input validation (missing paths and columns exit 2 before any output); per-city geometry in the main process; trainval status reads in `ProcessPoolExecutor`; mini DBs copied from the zip to `--scratch/map_mini/` and deleted in `finally`
    - Null-out rules and `*_missing` columns; write `scenario_map.parquet`, `scenario_map_mini_68.parquet`, `scenario_map_alignment.parquet` and `<reference-dir>/mini_68_reference_input.parquet` via `.tmp` + `os.replace` with `map_features_provenance` metadata
    - Tests: **Property 5: Missing indicators and row coverage** on synthetic frames with null ego points, unknown maps and missing DBs; missing input path exits 2 with no outputs; missing column exits 2 and is named; no `.tmp` left after write
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 2.4, 3.3, 4.5, 5.1, 5.2, 5.3, 5.4, 5.5_

- [ ] 5. Implement `check`, `verify-mini` and the devkit reference script
  - [~] 5.1 Implement `check` and `verify_mini` with tests
    - The seven checks from the design, `--frame-threshold` (default 0.90), PASS/FAIL lines, `docs/data/map_features.md` and `map_features.json` (per-city value counts, missing counts, check results, shapely/pyproj listed as added dependencies)
    - `verify-mini` prints `agree n/68` per feature, lists disagreeing `scenario_id`s with both values, exits 0 only on full agreement
    - Tests: a city with 0.5 share yields a FAIL line naming the city and share; `verify-mini` on hand-built tables with one disagreement exits 1 and lists it
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 7.1, 7.2, 7.3, 8.1, 8.2, 8.3_
  - [~] 5.2 Implement `tools/devkit_map_reference.py` with a `--help` test
    - Lazy nuplan imports; per-row devkit LANE / LANE_CONNECTOR / INTERSECTION lookup, `outgoing_edges` fallback and the restrictiveness rule as in the design pseudocode; writes `devkit_map_mini_68.parquet`
    - Test: `--help` exits 0 without nuplan installed
    - _Requirements: 8.1_

- [~] 6. Checkpoint - Ensure all tests pass
  - `.venv\Scripts\python.exe -m pytest -q` passes (baseline 2379 passed, 6 skipped, plus the new tests). Ensure all tests pass, ask the user if questions arise.

- [ ] 7. Real-data acceptance run
  - [~] 7.1 Run smoke and full `build`, then `check`
    - `build --limit-logs 20` first and inspect the `_smoke` outputs; then the full `build` (expected under 5 minutes)
    - `check` must pass all checks, with per-city frame alignment ≥ 0.90 (expected about 99.9–100%); confirm `docs/data/map_features.md` and `.json` are written
    - Report per-city counts for `lane_type`, `in_intersection` and `traffic_light_state`, plus missing counts; fix code and re-run tests if a check fails
    - Do not run Docker in this task
    - _Requirements: 5.1, 6.1, 6.4, 7.1, 7.2_

- [ ] 8. Devkit verification on mini
  - [ ]* 8.1 Run the Docker devkit reference and `verify-mini`
    - Only when `docker ps` shows no planner containers running
    - `docker compose --project-directory . -f infra/docker/compose.planners.yaml run --rm planner python -m tools.devkit_map_reference`, then `python tools/extract_map_features.py verify-mini --reference artifacts/additional_planners/map_reference/devkit_map_mini_68.parquet`; expect 68/68 on all three features and document any rule change if the devkit disagrees
    - _Requirements: 8.1, 8.2, 8.3_

## Notes

- Tasks marked with `*` are optional; 8.1 depends on idle planner containers.
- Tests live with the code they test, so each task leaves the suite green.
- Property numbers refer to the Correctness Properties in design.md.
- Git and Linear steps are handled outside this plan.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2"] },
    { "id": 1, "tasks": ["2.1"] },
    { "id": 2, "tasks": ["3.1"] },
    { "id": 3, "tasks": ["4.1"] },
    { "id": 4, "tasks": ["5.1"] },
    { "id": 5, "tasks": ["5.2"] },
    { "id": 6, "tasks": ["7.1"] },
    { "id": 7, "tasks": ["8.1"] }
  ]
}
```
