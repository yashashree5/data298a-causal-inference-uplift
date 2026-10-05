# IDM mini scenario sample

[`idm_mini_sample.csv`](idm_mini_sample.csv) identifies the 68 scenarios used
for our first IDM evaluation on nuPlan mini. It is a **category-balanced pilot
sample**, intended to cover different driving situations at a manageable local
runtime. It does not preserve the natural scenario frequencies in mini and is
not the paper's original evaluation cohort.

The CSV contains scenario identifiers and metadata, not driving trajectories.
The [Python runner](../../tools/idm_mini_reproducer.py) reads the log names and
scenario tokens and passes them to nuPlan, which loads the driving data from the
local databases. `make idm-mini` requests all 68 rows. Always check the runner
report and scored scenario IDs afterward: a requested count alone does not
prove that every scenario was simulated and scored.

## Selection rules

Categories originate in the databases' `scenario_tag.type` field and are attached
to anchors through `scenario_tag.lidar_pc_token`. The EDA's **Original categories
stored in the databases** section prints every observed tag, its raw row and
anchor counts, the number of logs containing it, and whether it belongs to the
14-category benchmark filter. These raw counts precede filtering and single-label
assignment; the eligible-anchor counts below come afterward.

The [EDA notebook](../../notebooks/nuplan_mini_eda.ipynb) implements these rules:

1. Scan the 64 nuPlan v1.1 mini SQLite databases. One database represents one
   driving log. Apply the devkit discovery rule that excludes anchors in the
   first two and last two scenes of each log.
2. Build one catalog row per tagged LiDAR anchor token. Combine multiple tags
   into `all_scenario_tags`, so multiple labels do not duplicate an anchor.
3. Keep anchors with at least one of the 14 benchmark tags. When an anchor has
   several matching tags, assign the alphabetically last matching benchmark tag
   as its category, matching the discovery query's `MAX(type)` convention.
   This is a deterministic label choice, not a ranking of driving importance.
4. Require a mission-goal ego-pose record and nonempty route-roadblock IDs.
   Require the extraction window to fit inside the log's timestamp range:
   **15 seconds, starting 3 seconds before the anchor and ending 12 seconds
   after it**. This boundary check does not verify every intermediate timestamp
   or guarantee that IDM will find a complete route.
5. Target **five scenarios per category**, processing categories with fewer
   eligible anchors first.
6. Within each category, order candidates by a reproducible hash of
   `"7:<scenario_id>"`. Specifically, take the first 16 hexadecimal characters
   of SHA-256 and interpret them as an integer. The sampling seed is **7**;
   the simulation seed is separately set to **0**.
7. Accept candidates in that order unless their window overlaps a window
   already selected from the same log. This exclusion applies across all
   categories. Windows that only touch at an endpoint are allowed. Stop at
   five accepted candidates or when the category has no candidates left.

The catalog contains 334,753 distinct tagged anchors after the scene rule.
There are 57,992 benchmark candidates with valid mission-goal records; all of
these also pass the window-boundary and route-record checks in this snapshot.
These are candidate anchors, not 57,992 independent driving events.

## Why 68 instead of 70?

Twelve categories supplied five scenarios each. Two supplied four after the
overlap exclusion: `following_lane_with_lead` and
`waiting_for_pedestrian_to_cross`. Thus **12 × 5 + 4 + 4 = 68**.

| Assigned category | Eligible anchors | Selected |
| --- | ---: | ---: |
| behind_long_vehicle | 663 | 5 |
| changing_lane | 5 | 5 |
| following_lane_with_lead | 26 | 4 |
| high_lateral_acceleration | 160 | 5 |
| high_magnitude_speed | 20,024 | 5 |
| low_magnitude_speed | 2,032 | 5 |
| near_multiple_vehicles | 835 | 5 |
| starting_left_turn | 268 | 5 |
| starting_right_turn | 130 | 5 |
| starting_straight_traffic_light_intersection_traversal | 173 | 5 |
| stationary_in_traffic | 20,190 | 5 |
| stopping_with_lead | 64 | 5 |
| traversing_pickup_dropoff | 13,312 | 5 |
| waiting_for_pedestrian_to_cross | 110 | 4 |
| Total | 57,992 | 68 |

The algorithm is greedy: earlier selections affect what remains available.
Four accepted scenarios does not mean only four candidate anchors exist, nor
that four is the maximum achievable with another ordering. The category counts
above use the single assigned label; they are not counts of every matching tag.

## Coverage and interpretation

The sample covers all 14 categories and 38 of the 52 logs containing eligible
benchmark anchors. It includes all four map locations: Las Vegas (52 scenarios),
Boston (7), Pittsburgh (7), and Singapore (2). There are no location or per-log
quotas; this coverage is an outcome of the selection.

This design gives rare categories a place in the pilot and reduces repeated
evaluation of nearly identical time windows. It has important limits:

- Five scenarios per category are too few for precise category estimates.
- Balanced category counts differ substantially from mini's natural frequency
  distribution, as the table shows. The aggregate describes this selected cohort.
- Non-overlapping windows from the same log can still be correlated.
- Single-label assignment affects category membership; 49 of the selected
  anchors have multiple original tags.
- A mini-sample score close to 0.76 is numerical agreement on this cohort, not
  proof of the same performance across all mini data or the original benchmark.

The sample was fixed before its first 68-scenario simulation. Do not change the
seed, selection rules, or sample after observing scores just to approach 0.76.
For broader evaluation, define and save the new cohort before running it and
report its result separately.

## Recreating the manifest

Set the local dataset root as described in the [project README](../../README.md),
then run the EDA notebook from top to bottom. It writes the local catalog to
`artifacts/eda/scenario_catalog.csv` and the generated sample to
`artifacts/eda/idm_mini_sample.csv`. It checks unique scenario IDs, absence of
overlapping windows, and the five-per-category cap.

Compare that generated CSV with this directory's saved manifest before replacing
it. Keep the saved manifest fixed for repeat runs of this experiment. Changes
to the data, labels, or selection algorithm can change the selected scenarios.

The current 68-row CSV has SHA-256:

```text
9162d8ae9d81a1068ca40a1d925a363b16a26e8df9026be84c7726c89e5bfcb5
```
