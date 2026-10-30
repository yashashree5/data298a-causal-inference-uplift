# nuPlan Scenario Subset and Required Fields

Owner: Ameya Khond (data and nuPlan scenario pipeline)

Machine-readable contract: [`configs/features/first_experiment_schema.yaml`](../../configs/features/first_experiment_schema.yaml)
Validator: [`tools/validate_scenario_subset.py`](../../tools/validate_scenario_subset.py)
Tests: [`tests/test_scenario_subset.py`](../../tests/test_scenario_subset.py)

```bash
python3 tools/validate_scenario_subset.py
```

## Objective

Freeze the scenario subset and the field contract for the first causal/uplift
experiment, before any model is trained. The question the experiment asks is:

> For a given driving scenario, how does the planner (IDM vs PDM-Closed) change
> the nuPlan outcome, and which pre-simulation scenario characteristics explain
> where the change is large or small?

This document defines which scenarios are used, what one analysis unit is, what
the treatment and outcomes are, which inputs a model may see, which fields are
forbidden because they leak the outcome, and how data is split and stored. It
does not train a model or report causal effects.

## Existing development subset

The first experiment reuses the completed **`mini_68`** experiment unchanged. No
new selection system is introduced. The selection rules, category table and
coverage notes are documented in
[`configs/scenarios/README.md`](../../configs/scenarios/README.md) and implemented
in the [EDA notebook](../../notebooks/nuplan_mini_eda.ipynb); they are not
repeated here.

| Property | Value | Source |
| --- | --- | --- |
| Scenarios | 68 | manifest |
| Challenge categories | 14 (12 × 5 + 2 × 4) | manifest |
| Logs | 38 (of 52 logs with eligible anchors) | manifest |
| Maps | 4: Las Vegas 52, Boston 7, Pittsburgh 7, Singapore 2 | manifest |
| Dataset | nuPlan v1.1 mini split (64 SQLite logs) | `experiment.json` |
| Devkit | nuPlan 1.2.2, commit `a581fbca…` | `experiment.json` |
| PDM-Closed | tuPlan Garage commit `b51d5d04…` | `experiment.json` |
| Simulation | `closed_loop_reactive_agents`, seed 0, `nuplan_mini` builder | `experiment.json` |
| Planners | IDM and PDM-Closed on the same 68 scenarios | outcome table |
| Outcome rows | 136 = 68 scenarios × 2 planners, all simulations succeeded | outcome table |
| Manifest SHA-256 | `9162d8ae9d81a1068ca40a1d925a363b16a26e8df9026be84c7726c89e5bfcb5` | manifest |

Committed artifacts:

- [`configs/scenarios/idm_mini_sample.csv`](../../configs/scenarios/idm_mini_sample.csv): the frozen manifest (one row per scenario).
- [`artifacts/mini_68/planner_outcomes.csv`](../../artifacts/mini_68/planner_outcomes.csv) and [`.parquet`](../../artifacts/mini_68/planner_outcomes.parquet): long-format outcomes (one row per scenario and planner).
- [`artifacts/mini_68/experiment.json`](../../artifacts/mini_68/experiment.json): shared settings, provenance and official aggregates (IDM 0.7601, PDM-Closed 0.9004).

## Why the subset is manageable

- **Already simulated.** Both planners have been run and validated on all 68
  scenarios (68/68 scored, 0 failures each), so the outcome side of the
  experiment exists today.
- **Small inputs.** The manifest, the outcome CSV and Parquet, and
  `experiment.json` total 117,236 bytes (measured, see
  [Storage requirements](#storage-requirements)).
  Pre-simulation feature extraction needs only the 38 selected mini log
  databases plus maps; no sensor blobs.
- **Covers the scenario space.** All 14 challenge categories and all four maps
  appear, so heterogeneity in planner effect can be explored without the full
  dataset.
- **Fixed before scoring.** The sample was frozen before its first simulation,
  so the subset is not tuned to the observed scores.

## Scenario-selection criteria

The criteria are those already applied by the EDA notebook (details and counts
in [`configs/scenarios/README.md`](../../configs/scenarios/README.md#selection-rules)):

1. nuPlan v1.1 mini logs; devkit valid-scene rule (first and last two scenes excluded).
2. Anchor carries one of the 14 `nuplan_challenge_scenarios` tags; category = lexically last matching tag (`MAX(type)`).
3. Mission goal and non-empty route roadblocks present; 15 s window (anchor − 3 s to anchor + 12 s) inside the log.
4. Up to five scenarios per category, rarest category first, ordered by SHA-256 of `"7:<scenario_id>"` (sampling seed 7).
5. No two selected windows overlap within the same log.

For this experiment one more criterion applies: **a scenario is analysable only
if both planners produced a successful, scored row.** All 68 currently qualify.

The validator checks the manifest hash, unique `scenario_id` and
`scenario_token`, `scenario_id == log_name:scenario_token`, the window offset and
duration, and the absence of overlapping windows within a log.

## Unit of analysis

**One scenario** = one `(log_name, scenario_token)` pair, keyed by
`scenario_id = "<log_name>:<scenario_token>"`.

- The outcome table is long format: one row per `(scenario_id, planner)`.
- A **matched pair** is the same scenario simulated under both planners. Each
  scenario contributes exactly one IDM row and one PDM-Closed row.
- Because both planners see the identical initial state, map, route, agents and
  seed, the pair design removes scenario-level confounding by construction: the
  within-pair outcome difference is attributable to the planner (up to
  simulation nondeterminism, see [Limitations](#limitations)).
- Scenarios from the same log are not independent; the log is the grouping
  unit for splits.

## Treatment definition

| Field | Values | Column |
| --- | --- | --- |
| `planner_name` | `idm` (control) / `pdm-closed` (treatment) | `planner` |
| `planner_configuration` | `idm_planner` / `pdm_closed_planner` Hydra configs | derived from `tools/pdm_mini_reproducer.py` `PLANNERS` |
| `simulation_seed` | 0 for both | `experiment.json` |

Treatment is assigned by design (every scenario receives both), not chosen by
the data. The planner is never a model covariate; it defines which potential
outcome a row represents.

## Outcome definition

- **Primary outcome:** `scenario_score` (column `score`), the nuPlan
  closed-loop reactive weighted score in [0, 1]. A zero is a valid outcome, not
  a failure. `closed_loop_reactive_score` is an alias of the same value.
- **Pairwise uplift target (derived):** `score_delta = score(pdm-closed) − score(idm)`
  per scenario. It is computed by pairing rows and is never stored as a feature.
- **Secondary outcomes:** the nuPlan metric columns listed in
  [Required post-simulation outcome fields](#required-post-simulation-outcome-fields).
- **Experiment-level:** `overall_score` per planner = mean of `scenario_score`;
  the validator confirms it equals `official_score` in `experiment.json`.

## Required pre-simulation input fields

**Feature time.** Every pre-simulation feature is measured at **simulation
iteration 0**: the first `lidar_pc` at or after `window_start_us`
(= anchor − 3 s), using only log data at or before that time (the devkit's 2 s
history buffer is allowed). Nothing after iteration 0 may enter a feature.

Status legend: **available** = in a committed artifact; **derived** = computable
from committed data; **unavailable** = must be extracted from the nuPlan
databases (not done yet).

Scenario metadata:

| Field | Status | Model feature | Note |
| --- | --- | --- | --- |
| `scenario_id`, `scenario_token`, `log_name`, `db_file` | available | no | Identifiers and join keys |
| `log_token` | **unavailable** | no | Not stored; `log_name` is the log key. Read `log.token` from the 38 DBs at extraction |
| `scenario_type`, `all_scenario_tags` | available | no | Stratification and subgroup reporting only (look-ahead, see leakage) |
| `map_name` | available | **yes** | 4-level categorical |
| `city` | derived | no | 1:1 with `map_name` via `city_by_map_name`; confirm with `log.location` at extraction |
| `scenario_timestamp` (`anchor_timestamp_us`), `window_start_us`, `window_end_us`, `time_block_id` | available | no | Time and selection bookkeeping |
| `dataset_version` | derived | no | Constant |

Ego and scene state at iteration 0 (all **unavailable** today; devkit sources are
listed per field in the schema):

| Field | Required | Model feature | Source (devkit / map API) |
| --- | --- | --- | --- |
| `ego_x`, `ego_y`, `ego_heading` | yes | no | `initial_ego_state.rear_axle` (used to compute relative features; raw global pose identifies location) |
| `ego_speed`, `ego_acceleration`, `ego_yaw_rate` | yes | yes | `initial_ego_state.dynamic_car_state` |
| `route_length_m` | no | yes | route roadblock baseline-path lengths |
| `lane_id` | no | no | lane / lane connector at ego position (high-cardinality identifier) |
| `lane_type`, `in_intersection` | yes | yes | map layers at ego position |
| `traffic_light_state` | yes | yes | `get_traffic_light_status_at_iteration(0)` |
| `num_tracked_agents`, `num_vehicles`, `num_pedestrians`, `agents_within_50m` | yes | yes | `initial_tracked_objects` |
| `nearest_agent_distance_m`, `nearest_agent_relative_speed_mps` | yes | yes | `initial_tracked_objects` relative to ego |
| `lane_curvature` | no | yes | ego lane baseline path, next 50 m |

The schema currently declares **15 model features**: `map_name` plus the 14
ego/scene covariates above. Only `map_name` is available now.

**How the unavailable fields will be obtained:** a read-only extraction step
will load each manifest row through the devkit `NuPlanScenarioBuilder` (same
commit, `nuplan_mini` builder) from the 38 selected log databases and maps,
evaluate the fields at iteration 0, and write a feature table keyed by
`scenario_id`. That table must pass
`python3 tools/validate_scenario_subset.py --features <table.csv>`, which rejects
any column that is not a join key, a declared model feature or a `<feature>_missing`
indicator. This requires the local nuPlan mini data (see
[Limitations](#limitations)).

## Required post-simulation outcome fields

All are `available_before_simulation: false` and `model_feature: false`.

| Field | Column | Status |
| --- | --- | --- |
| `scenario_score` (primary) | `score` | available |
| `closed_loop_reactive_score` | alias of `score` | derived |
| `overall_score` | `experiment.json` `official_score` | derived |
| `progress_score` | `ego_progress_along_expert_route` | available |
| `making_progress` | `ego_is_making_progress` | available |
| `comfort_score` | `ego_is_comfortable` | available |
| `no_at_fault_collision_score` | `no_ego_at_fault_collisions` | available |
| `at_fault_collision` | `no_ego_at_fault_collisions < 1` | derived |
| `time_to_collision_within_bound` | same | available |
| `drivable_area_compliance` | same | available |
| `driving_direction_compliance` | same | available |
| `speed_limit_compliance` | same | available |
| `simulation_success` | `succeeded` | available |
| `failure_reason` | `error_message` (empty when successful) | available |
| `score_delta` | pdm-closed − idm per pair | derived |

Provenance columns (`run_id`, `sample_sha256`, `nuplan_commit`,
`tuplan_garage_commit`) are carried for traceability and are never features.
Every column in the committed manifest and outcome table is declared in the
schema; an undeclared or missing column fails validation.

## Fields excluded because of leakage

A field is a valid model input only if it is known before simulation starts and
cannot be affected by the planner. Excluded:

- **All outcomes and anything derived from them:** scores, progress, comfort,
  collision and TTC metrics, compliance metrics, `succeeded`, `error_message`,
  `score_delta`, per-planner aggregates, ranks or residuals computed from scores.
- **The planner's own behavior:** simulated ego trajectory, controller commands,
  any state after iteration 0.
- **Future log data:** anything measured after `window_start_us`, including the
  logged expert's future trajectory and agent futures. `window_end_us` marks
  the end of the evaluated window.
- **Treatment and run identity:** `planner`, `planner_configuration`, `run_id`.
- **Look-ahead labels:** `scenario_type` and `all_scenario_tags` are unaffected
  by the planner, but they are assigned at the anchor (3 s after simulation
  start) and summarize the expert's upcoming behavior. They are used for
  stratification and subgroup reporting, not as default model inputs.
- **Identifiers that memorize location or log:** `scenario_id`,
  `scenario_token`, `log_name`, `log_token`, `db_file`, `time_block_id`,
  timestamps, raw global `ego_x`/`ego_y`/`ego_heading`, `lane_id`.

Enforcement in [`tools/validate_scenario_subset.py`](../../tools/validate_scenario_subset.py):

1. Outcome-role fields must have `available_before_simulation: false` and `model_feature: false`.
2. Every model feature must be available before simulation and belong to the metadata or covariate role.
3. No model feature name or column may contain a forbidden pattern (`score`, `progress`, `comfort`, `collision`, `compliance`, `succeeded`, `success`, `failure`, `error`, `trajectory`, `planner`, `run_id`, `window_end`, `delta`, `uplift`).
4. With `--features`, the feature table may contain only join keys, declared model features and `_missing` indicators.

## Missing-data policy

Each field in the schema carries one policy:

| Policy | Meaning | Used for |
| --- | --- | --- |
| `fail_validation` | Any missing value fails validation | Identifiers, design fields, treatment, `simulation_success`, provenance |
| `exclude_pair` | Drop the scenario for **both** planners and report it; never impute | Core ego state, agent counts, all score/metric outcomes |
| `null_with_indicator` | Keep the scenario, store null, add `<name>_missing` | Map/context features that can legitimately be absent (e.g. no nearby agent), `log_token` |
| `not_applicable_when_succeeded` | Null expected when the run succeeded | `failure_reason` |
| `constant_from_metadata` | Experiment-level constant | `dataset_version`, `simulation_seed` |

Rules: no imputation from outcomes or from the other planner's row; a failed
simulation for either planner removes the whole pair; exclusions are reported
with counts. Current state: 0 missing values in required committed fields and
136/136 successful simulations, so no pair is excluded.

## Train/test split strategy

- **Grouping:** by `log_name`. All scenarios of a log, and both planner rows
  of each scenario, stay in the same fold. Scenarios from one log share driver,
  route, weather and time, so a scenario-level split would leak.
- **Scheme:** 5-fold log-grouped cross-validation (`split` in the schema:
  `folds: 5`, `seed: 0`). With 68 scenarios a single hold-out set would be too
  small to evaluate anything.
- **Deterministic assignment** (`assign_log_folds`): logs sorted by scenario
  count (largest first), then by SHA-256 of `"<seed>:<log_name>"`; each log is
  placed in the fold with the fewest scenarios. The result does not depend on
  row order. Current folds: 8/8/8/7/7 logs and 14/14/14/13/13 scenarios.
- **No outcome-based stratification**, so the split cannot be tuned to scores.
- **Known imbalance:** Singapore is a single log (2 scenarios), so its fold
  tests on an unseen `map_name` level; a test fold covers 7–11 of the 14
  categories. Report results per fold and treat map/category subgroup results as
  descriptive.

The validator recomputes the assignment and checks that no log or scenario
appears in more than one fold. The assignment is reproducible from code and is
not stored as a separate file.

## Storage requirements

Committed files were measured with `stat -f %z` (bytes) on the current branch.

| Category | Status | Size |
| --- | --- | --- |
| Committed subset metadata: manifest | measured | 24,857 B |
| Committed outcomes: `planner_outcomes.csv` / `.parquet` | measured | 62,005 B / 29,093 B |
| Committed metadata: `experiment.json`, `idm/result.json`, `pdm_closed/result.json` | measured | 1,281 B, 492 B, 512 B |
| All tracked `artifacts/mini_68` files (5) | measured | 93,383 B |
| Feature contract `first_experiment_schema.yaml` | measured | 25,127 B |
| Whole repository, all tracked files | measured | 277,617 B (31 files, before this change) |
| Processed pre-simulation feature table | **does not exist yet** | — (68 rows × ~20 columns; to be measured after extraction) |
| Raw nuPlan mini DBs | **not present locally, not measured** | Published size of the mini split: 13 GB for 64 logs (nuPlan devkit tutorial). Only the 38 selected DBs are needed |
| Maps | **not present locally, not measured** | — |
| Sensor blobs (camera/lidar) | **not used** | 0; not needed for any field in this contract |
| Full simulation outputs | **not committed, not present locally** | Only the two `result.json` summaries are committed |
| Docker images | **not measured** | Docker daemon not running on this machine |

To measure the raw inputs once the data is available (`NUPLAN_HOST_DATA_ROOT`
contains `maps/` and `nuplan-v1.1/`, as in [`.env.example`](../../.env.example)):

```bash
python3 tools/validate_scenario_subset.py --data-root "$NUPLAN_HOST_DATA_ROOT"   # bytes of the 38 selected DBs
du -sh "$NUPLAN_HOST_DATA_ROOT"/nuplan-v1.1/splits/mini "$NUPLAN_HOST_DATA_ROOT"/maps
docker image ls
```

Git policy: `.gitignore` excludes `*.db`, `*.db-shm`, `*.db-wal`, `*.gpkg`,
`*.pcd`, `sensor_blobs/`, `nuplan-v1.1/`, `nuplan-maps-v*/`, `/external/` and all
of `artifacts/` except the reviewed `mini_68` snapshot. Raw nuPlan data, sensor
blobs, Docker images and full simulation outputs stay out of Git. A processed
feature table for 68 scenarios is expected to be small enough to commit, but it
should be measured and reviewed first.

## Limitations

- **Pre-simulation features are not extracted yet.** 15 required fields
  (ego state, map context, traffic lights and agents) are marked
  `unavailable`; only `map_name` is usable as a model feature today. Extraction
  needs the local nuPlan mini DBs and maps, which are not on this machine.
- **`log_token` is not stored**; `log_name` serves as the log key until extraction.
- **Small sample:** 68 pairs in 38 logs. Category- and map-level estimates
  (5 or fewer scenarios per category, 2 in Singapore) are descriptive only.
- **Not representative:** category-balanced sample, not mini's natural
  frequencies; 52 of 68 scenarios are in Las Vegas.
- **Single-label categories:** 49 of 68 anchors have several tags; the assigned
  category is a deterministic convention.
- **Look-ahead in labels:** scenario tags describe the expert's behavior in the
  window, so they are not used as default features.
- **One seed, one run per planner:** simulation nondeterminism is not
  quantified; the pair difference assumes reruns would reproduce the scores.
- **Upstream planners only:** IDM and PDM-Closed validate the pipeline; they are
  not the models the team will build.

## Acceptance-criteria checklist

- [x] Existing `mini_68` subset reused; no new selection system.
- [x] Subset definition documented with counts (68 scenarios, 14 categories, 38 logs, 4 maps, 136 rows).
- [x] Unit of analysis, matched pair, treatment and outcome defined.
- [x] Required pre-simulation fields defined with source, dtype, required flag, availability before simulation, model-feature flag and missing-data policy.
- [x] Required post-simulation outcome fields defined.
- [x] Leakage exclusions defined and enforced by the validator and tests.
- [x] Missing-data policy defined per field.
- [x] Log-level split strategy defined and checked.
- [x] Storage values measured for committed artifacts; unmeasured items marked as such.
- [x] Validator checks manifest, unique tokens, log and type presence, planner pairing, both-planner outcome rows, duplicate scenario-planner rows, missing required values and outcome-as-feature leakage; reports counts by type, city, map and log; exits nonzero on failure.
- [x] Unit tests for manifest loading, unique tokens, matched pairs, required fields, duplicate detection, leakage and deterministic ordering.
- [x] Raw nuPlan data excluded from Git.
- [ ] Pre-simulation feature table extracted and validated (blocked on local nuPlan mini data).
- [ ] `log_token` read from the log databases (blocked on local nuPlan mini data).

## Next steps

1. Mount the nuPlan v1.1 mini split and maps (`NUPLAN_HOST_DATA_ROOT`, see the
   [project README](../../README.md)) and measure the 38 selected DBs with
   `--data-root`.
2. Write the read-only extraction step for the iteration-0 fields in the
   schema, in the pinned devkit environment (`infra/docker`), keyed by
   `scenario_id`, including `log_token` and a `city` cross-check against `log.location`.
3. Validate the feature table with `--features`, flip the extracted fields to
   `available` in the schema, and commit the table only after measuring its size.
4. Build the paired analysis table (one row per scenario: features, IDM score,
   PDM-Closed score, `score_delta`) and the log-grouped folds; only then start
   modeling.
