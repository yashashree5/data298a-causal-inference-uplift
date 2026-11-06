# Requirements Document

## Introduction

Linear issue 298-39 ("Extract map features: lane_type, in_intersection, traffic_light_state"), branch `298-39-map-features`. A new command-line tool, `tools/extract_map_features.py`, computes three pre-simulation map features for every nuPlan benchmark candidate (1,406,469 scenarios) and for the 68 mini scenarios: `lane_type`, `in_intersection` and `traffic_light_state`. The features are looked up at the ego rear-axle pose at simulation start, using the city map GeoPackages and the `traffic_light_status` table of each log database at the start lidar frame.

The tool follows the conventions of `tools/extract_ego_agent_features.py` and `tools/extract_weather_features.py`: argparse subcommands, `from __future__ import annotations`, pandas/pyarrow, a process pool over cities or logs where it helps, atomic writes through a temporary file and `os.replace`, provenance metadata in the parquet schema, PASS/FAIL check lines, integer exit codes and defaults overridable by flags. Subcommands are `build`, `check` and the optional `verify-mini`.

### Assumptions

- Ego position is (`ego_x`, `ego_y`) from `scenario_ego_agent.parquet` / `scenario_ego_agent_mini_68.parquet`: rear axle, global map frame in UTM metres. The start lidar frame is `start_lidar_token` from `scenario_ego_agent_alignment.parquet`.
- The GeoPackage `gpkg_geometry_columns` declares SRS 4326, but coordinates are treated as the same local UTM frame as `ego_x`/`ego_y`. This is verified by the frame-alignment check (Requirement 7), not assumed silently.
- A point on a polygon boundary counts as inside the polygon.
- When the ego point lies inside both a lane polygon and a lane connector polygon, `lane_type` is `lane`. The `verify-mini` comparison confirms this rule against the devkit; design may change it if the devkit disagrees, and the change is documented.
- "Lane connectors leaving the ego lane" means connectors whose upstream lane fid equals the ego lane fid. Design confirms which of `exit_lane_fid` / `entry_lane_fid` holds the upstream lane by inspecting the map data.
- Status strings are compared case-insensitively. Strings other than green, red and yellow are treated as no status record and counted in the report.
- The default frame-alignment threshold is 90% of candidates per city with `lane_type` not equal to `none`; the threshold is a flag.
- geopandas/shapely (or an equivalent geometry library) may be added to `.venv`; any added dependency is listed in `docs/data/map_features.md`.
- Optional extras (lane speed limit, crosswalk proximity, distance to next intersection) are out of scope for this issue unless design adds them at no extra input cost; the three required fields come first.
- `configs/features/first_experiment_schema.yaml` is not modified; schema integration is issue 298-40.

## Glossary

- **Map_Tool**: The command-line program `tools/extract_map_features.py` with subcommands `build`, `check` and `verify-mini`.
- **Candidate**: A row of `scenario_ego_agent.parquet`, keyed by `scenario_id`.
- **Mini_Scenario**: A row of `scenario_ego_agent_mini_68.parquet`, keyed by `scenario_id`.
- **Ego_Point**: The point (`ego_x`, `ego_y`) of a Candidate or Mini_Scenario.
- **Start_Frame**: The lidar frame identified by `start_lidar_token` for the scenario in `scenario_ego_agent_alignment.parquet`.
- **City_Map**: The `map.gpkg` file for a scenario's `map_name`, at the version given by `maps/nuplan-maps-v1.0.json`.
- **Lane_Polygon**: A feature of the City_Map layer `lanes_polygons`.
- **Connector_Polygon**: A feature of the City_Map layer `gen_lane_connectors_scaled_width_polygons`, linked to a row of the `lane_connectors` layer.
- **Intersection_Polygon**: A feature of the City_Map layer `intersections`.
- **Ego_Lane**: The Lane_Polygon containing the Ego_Point.
- **Ego_Connector**: A Connector_Polygon containing the Ego_Point.
- **Relevant_Connectors**: The Ego_Connectors when at least one exists; otherwise the lane connectors whose upstream lane is the Ego_Lane.
- **Status_Record**: A row of the log database table `traffic_light_status` whose `lidar_pc_token` equals the Start_Frame token.
- **Restrictiveness_Order**: red is more restrictive than yellow, and yellow is more restrictive than green.
- **Map_Table**: The output file `scenario_map.parquet` (Candidates) or `scenario_map_mini_68.parquet` (Mini_Scenarios).
- **Audit_Table**: The output file `scenario_map_alignment.parquet`.
- **Devkit_Reference**: A parquet file of the three features for the Mini_Scenarios computed with the nuPlan devkit map API inside Docker image `data298a/pdm-closed-mini:1.0`.

## Requirements

### Requirement 1: Inputs and paths

**User Story:** As a researcher, I want every input and output path configurable, so that the tool runs from the E: drive or from the C: copy when the USB drive is disconnected.

#### Acceptance Criteria

1. THE Map_Tool SHALL accept a flag for each input path (ego table, mini ego table, alignment table, eligibility table, mini sample CSV, trainval database directory, mini dataset zip, maps directory) and for the output directory.
2. THE Map_Tool SHALL default the output directory to `E:\Project\nuplan\processed\features\`.
3. IF a required input file or directory does not exist, THEN THE Map_Tool SHALL print the missing path and exit with a nonzero exit code before writing any output.
4. IF an input table lacks a column the Map_Tool reads, THEN THE Map_Tool SHALL print the missing column names and exit with a nonzero exit code.
5. WHEN `build` processes Mini_Scenarios, THE Map_Tool SHALL extract only the needed log databases from the mini dataset zip into a scratch directory.

### Requirement 2: lane_type

**User Story:** As a researcher, I want to know whether ego starts on a lane, a lane connector or neither, so that road context can enter the uplift model.

#### Acceptance Criteria

1. WHEN the Ego_Point lies inside a Lane_Polygon, THE Map_Tool SHALL set `lane_type` to `lane`.
2. WHEN the Ego_Point lies inside a Connector_Polygon and inside no Lane_Polygon, THE Map_Tool SHALL set `lane_type` to `lane_connector`.
3. WHEN the Ego_Point lies inside no Lane_Polygon and no Connector_Polygon, THE Map_Tool SHALL set `lane_type` to `none`.
4. IF the Ego_Point or the City_Map is unavailable for a scenario, THEN THE Map_Tool SHALL set `lane_type` to null and `lane_type_missing` to True.

### Requirement 3: in_intersection

**User Story:** As a researcher, I want a flag for ego starting inside an intersection, so that intersection scenarios can be separated in analysis.

#### Acceptance Criteria

1. WHEN the Ego_Point lies inside an Intersection_Polygon, THE Map_Tool SHALL set `in_intersection` to True.
2. WHEN the Ego_Point lies inside no Intersection_Polygon, THE Map_Tool SHALL set `in_intersection` to False.
3. IF the Ego_Point or the City_Map is unavailable for a scenario, THEN THE Map_Tool SHALL set `in_intersection` to null and `in_intersection_missing` to True.

### Requirement 4: traffic_light_state

**User Story:** As a researcher, I want the traffic-light status relevant to ego at simulation start, so that signal context is available without using any post-start information.

#### Acceptance Criteria

1. WHEN at least one Relevant_Connector has a Status_Record, THE Map_Tool SHALL set `traffic_light_state` to the most restrictive status under the Restrictiveness_Order among the Status_Records of the Relevant_Connectors.
2. WHEN Relevant_Connectors exist and none has a Status_Record, THE Map_Tool SHALL set `traffic_light_state` to `unknown`.
3. WHEN no Relevant_Connector exists, THE Map_Tool SHALL set `traffic_light_state` to `none`.
4. THE Map_Tool SHALL read Status_Records only for the Start_Frame of each scenario.
5. IF the Start_Frame token, the log database or the City_Map is unavailable for a scenario, THEN THE Map_Tool SHALL set `traffic_light_state` to null and `traffic_light_state_missing` to True.

### Requirement 5: Output tables

**User Story:** As a researcher, I want one row per scenario in model-ready tables and the map ids in a separate audit table, so that high-cardinality ids never become model inputs.

#### Acceptance Criteria

1. WHEN `build` completes, THE Map_Tool SHALL write `scenario_map.parquet` with exactly one row per Candidate and `scenario_map_mini_68.parquet` with exactly one row per Mini_Scenario, keyed on `scenario_id`.
2. THE Map_Tool SHALL write to each Map_Table only `scenario_id`, the three feature columns and their `*_missing` indicator columns.
3. THE Map_Tool SHALL write to the Audit_Table, per scenario, the `scenario_id`, `map_name`, map version, Ego_Lane fid, Ego_Connector fids, Intersection_Polygon fid, Relevant_Connector fids and the Status_Record count.
4. THE Map_Tool SHALL write each output file to a temporary path and move the file into place with `os.replace`.
5. THE Map_Tool SHALL store provenance in each output parquet schema metadata, including input paths, input file sizes, map versions, tool arguments and the creation timestamp.

### Requirement 6: Check subcommand

**User Story:** As a researcher, I want a check that validates the outputs and writes a report, so that the feature tables can be trusted before integration.

#### Acceptance Criteria

1. WHEN `check` runs, THE Map_Tool SHALL print one PASS or FAIL line per check and exit with code 0 only if every check passes.
2. THE Map_Tool SHALL check that each Map_Table has one row per input scenario, unique `scenario_id` values and only the allowed values for each feature column.
3. THE Map_Tool SHALL check that each `*_missing` column is True exactly where the corresponding feature value is null.
4. WHEN `check` completes, THE Map_Tool SHALL write `docs/data/map_features.md` and a JSON report with per-city value counts, missing counts and the check results.

### Requirement 7: Map frame alignment

**User Story:** As a researcher, I want evidence that ego coordinates and map coordinates share one frame, so that a coordinate mismatch cannot silently produce all-`none` features.

#### Acceptance Criteria

1. WHEN `check` runs, THE Map_Tool SHALL compute, per city, the share of Candidates with `lane_type` equal to `lane` or `lane_connector`.
2. IF the share for any city is below the frame-alignment threshold, THEN THE Map_Tool SHALL report a FAIL line naming the city and the share.
3. THE Map_Tool SHALL accept a flag that sets the frame-alignment threshold, with a default of 0.90.

### Requirement 8: Devkit verification on mini

**User Story:** As a researcher, I want the extractor compared with the nuPlan devkit on the mini scenarios, so that the geometry and status rules match the simulator.

#### Acceptance Criteria

1. WHERE `verify-mini` is used, THE Map_Tool SHALL compare `scenario_map_mini_68.parquet` with the Devkit_Reference per scenario and per feature.
2. WHERE `verify-mini` is used, THE Map_Tool SHALL print the number of agreeing scenarios per feature and list each disagreeing `scenario_id` with both values.
3. WHERE `verify-mini` is used, THE Map_Tool SHALL exit with code 0 only if all 68 Mini_Scenarios agree on all three features.

### Requirement 9: Tests

**User Story:** As a developer, I want unit tests on synthetic data, so that the rules are verified without real nuPlan files.

#### Acceptance Criteria

1. THE test module `tests/test_extract_map_features.py` SHALL cover Requirements 2, 3 and 4 using synthetic polygons and a synthetic SQLite `traffic_light_status` table.
2. WHEN `.venv\Scripts\python.exe -m pytest -q` runs, THE test suite SHALL pass with no failures.
