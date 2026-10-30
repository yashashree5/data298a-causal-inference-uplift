# Tools

These files are stable command-line entry points. The reusable implementation
lives under `src/causal_planner/`; existing commands and public function imports
remain available. `tools/_bootstrap.py` makes the source package importable from
a checkout without installation. Docker already mounts the checkout, so this
reorganization does not require an image rebuild. See the
[module mapping and migration notes](../docs/repository_structure.md).

The `extract_scenario_features.py` and `build_model_dataset.py` commands remain
package-migration placeholders: execution exits with status 2 and writes no data.
The implemented feature extractors are `extract_weather_features.py` and
`extract_ego_agent_features.py`, documented below. Use `build_paired_dataset.py`
for the existing IDM/PDM-Closed cleaning, pairing, and grouped-fold workflow;
joining the extracted feature tables into the modeling dataset remains separate
work. `configs/features/first_experiment_schema.yaml` is its field contract;
`feature_schema.json` is an unused placeholder, not a replacement contract.

## Data layer: train-city archives, catalog and extracted logs

These tools prepare the four nuPlan v1.1 train-city archives (Boston, Pittsburgh,
Singapore, Las Vegas part 1). Machine layout, timings and setup are in
[`docs/local_gpu_setup.md`](../docs/local_gpu_setup.md).

| Tool | Purpose | Output |
|---|---|---|
| [`download_nuplan_drive.ps1`](download_nuplan_drive.ps1) | Resumable rclone copy of the archives from the team Drive (folder ID from `.env`) | `D:\nuplan\downloads\` |
| [`build_scenario_catalog.py`](build_scenario_catalog.py) | One Parquet shard per log of every tagged anchor, then a combined, checked catalog | `D:\nuplan\processed\scenario_catalog.parquet` |
| [`extract_nuplan_logs.py`](extract_nuplan_logs.py) | Extracts every log DB into the devkit's `nuplan` builder path and records per-log provenance | `D:\nuplan\dataset\nuplan-v1.1\trainval\` |
| [`build_scenario_eligibility.py`](build_scenario_eligibility.py) | Flags every tagged anchor with the `mini_68` simulation rules, its drive (`source_recording`) and mini overlap | `D:\nuplan\processed\scenario_eligibility.parquet`, [`docs/data/trainval_eligibility.md`](../docs/data/trainval_eligibility.md) |

```powershell
.\.venv\Scripts\python -m tools.build_scenario_catalog build; .\.venv\Scripts\python -m tools.build_scenario_catalog check
.\.venv\Scripts\python -m tools.extract_nuplan_logs extract;  .\.venv\Scripts\python -m tools.extract_nuplan_logs check
.\.venv\Scripts\python -m tools.build_scenario_eligibility verify-mini   # rules reproduce the mini_68 counts
.\.venv\Scripts\python -m tools.build_scenario_eligibility build; .\.venv\Scripts\python -m tools.build_scenario_eligibility check
```

All tools are resumable and never modify the archives. Verified state (2026-10-29):
7,353 DBs in 4 archives, all extracted (441 GB) with matching sizes; the catalog has
43,943,311 tag rows from 7,114 logs (239 logs have no scenario tags) and 73 scenario types.

The eligibility table has one row per distinct tagged anchor (20,650,110) and drops nothing;
rules are flags. `verify-mini` first runs the same code on the 64 mini DBs and must reproduce
the documented 334,753 anchors, 57,992 benchmark candidates and all 68 manifest rows. On the
train cities, 1,406,469 anchors are benchmark candidates, from 2,639 logs and 828 drives;
see the [report](../docs/data/trainval_eligibility.md) for the funnel and category-by-city counts.
Candidates overlap in time, so the manifest step must still select non-overlapping windows
per `source_recording` and split train/val/test by recording.

## Weather features

[`extract_weather_features.py`](extract_weather_features.py) adds historical weather and sun
elevation at simulation start for every benchmark candidate and the 68 mini scenarios. Source:
Open-Meteo Historical Weather API, model `ecmwf_ifs` (~9 km), CC BY 4.0, non-commercial use.

```powershell
.\.venv\Scripts\python -m tools.extract_weather_features fetch   # 4 API calls (one per city), cached
.\.venv\Scripts\python -m tools.extract_weather_features build   # offline from the cache
.\.venv\Scripts\python -m tools.extract_weather_features check   # validation + docs/data reports
```

Each scenario uses the UTC hour at or before its simulation start; precipitation, rain and gusts
cover the preceding hour, so no value describes time after the start. Outputs in
`D:\nuplan\processed\weather\`: `scenario_weather.parquet` (1,406,469 candidates),
`scenario_weather_mini_68.parquet`, the hourly table, an alignment audit table and the raw API
responses. The 10 columns are declared in `configs/features/first_experiment_schema.yaml`
(`weather_pre_simulation`) and pass `validate_scenario_subset.py --features`. Coverage, limits
and checks: [`docs/data/weather_features.md`](../docs/data/weather_features.md).

## Ego and agent features

[`extract_ego_agent_features.py`](extract_ego_agent_features.py) extracts ego vehicle state
(position, speed, acceleration, yaw rate) and surrounding agent features (counts, nearest distance,
closing speed) at simulation iteration 0 for every benchmark candidate and the 68 mini scenarios.

```powershell
.\.venv\Scripts\python -m tools.extract_ego_agent_features build-mini   # 68 mini scenarios
.\.venv\Scripts\python -m tools.extract_ego_agent_features verify-mini  # compare with devkit SQL
.\.venv\Scripts\python -m tools.extract_ego_agent_features build        # 1.4M train candidates
.\.venv\Scripts\python -m tools.extract_ego_agent_features check        # validation + reports
```

Iteration 0 is defined by the devkit: the first lidar_pc at or after `anchor - 3 s`, reading
only data at or before that timestamp. Ego velocities are rotated from the ego frame to the
global frame; agent velocities are already global. Outputs in `D:\nuplan\processed\features\ego_agent\`:
one Parquet per log with candidates, `scenario_ego_agent_mini_68.parquet`, and audit reports.
The 12 columns are declared in `configs/features/first_experiment_schema.yaml`
(`ego_scene_pre_simulation`) and pass all unit tests and validation checks.

## IDM mini reproducer

[`idm_mini_reproducer.py`](idm_mini_reproducer.py) starts our IDM mini experiment.
It gives everyone on the team the same entry point and simulation settings.

The script:

1. Reads and validates the saved [68-scenario CSV](../configs/scenarios/idm_mini_sample.csv).
2. Passes its log names and scenario tokens to the nuPlan devkit.
3. Runs the official IDM planner in closed-loop reactive simulation.
4. Records the launch command and saves outputs under `artifacts/idm_mini/`.

In Docker, the runner uses `/artifacts/idm_mini/`. The `/artifacts` mount points
to `NUPLAN_HOST_ARTIFACT_ROOT` from `.env`, which defaults to the repository's
`./artifacts`. A custom host output directory is honored too. Outside Docker,
the default is the repository's `artifacts/idm_mini/`; `--output-root` can
override either default.

The IDM algorithm, simulator, and scoring code are supplied by nuPlan inside
Docker. This script connects our selected sample to those components.

After completing the [local setup](../README.md#local-setup), run these commands
from the repository root:

```bash
make idm-mini-dry-run  # Validate the sample and print the command; no simulation.
make idm-mini          # Start the simulation for all 68 selected scenarios.
```

The runner prints the output directory when finished. Check nuPlan's
`runner_report.parquet` for completed/failed scenarios and the `aggregator_metric/`
folder for scores. The script does not yet print a score comparison with 0.76.

See the [sample README](../configs/scenarios/README.md) for how the 68 scenarios
were selected.

## PDM-Closed mini comparison

[`pdm_mini_reproducer.py`](pdm_mini_reproducer.py) runs upstream PDM-Closed or
IDM in the new PDM image. It reuses the original manifest loader and command
settings: the **same 68 IDs**, 14 categories, mini builder, closed-loop reactive
agents, sequential worker, simulation seed 0, and official nuPlan scoring.
It adds the upstream PDM configuration search path; it does not use Val14 or
resample scenarios. The historical CSV name `idm_mini_sample.csv` is retained
because its contents are shared and must not change between planners.

After the [Docker setup](../infra/docker/README.md):

```bash
make pdm-mini-smoke                      # First saved scenario
make idm-mini-matched RUN_ARGS=--limit=1  # Matched baseline on that scenario
make pdm-mini RUN_ARGS="--experiment mini_68_local"  # All 68
make idm-mini-matched RUN_ARGS="--experiment mini_68_local"  # Same environment
```

For host-side command validation without Docker or data:

```bash
python3 -m tools.pdm_mini_reproducer --dry-run
```

Outputs are organized by experiment setup, not by planner or timestamp:

```text
artifacts/mini_68/
├── idm/
├── pdm_closed/
├── planner_outcomes.parquet
├── planner_outcomes.csv
└── experiment.json
```

The committed `mini_68` snapshot contains only the combined tables, experiment
metadata, and two result summaries, not the full audit files listed below. The
commands above use `mini_68_local` to preserve that snapshot. Use another explicit
experiment name if this local output already exists.

The default experiment is `mini_<selected count>`: full runs use `mini_68`,
one-scenario smoke checks use `mini_1`. Each planner directory contains:

- `run_manifest.json`: exact selected IDs, full-sample hash, upstream commits,
  command, and dataset paths;
- `environment.txt`: installed Python packages;
- `console.log` and nested nuPlan logs, resolved configuration and metrics;
- `result.json`: success/failure counts and the official aggregate score;
- `scenario_scores.csv`: individual scenario scores and metric values.

Run IDs remain inside metadata and nuPlan's internal generated layout. An
existing planner directory is never overwritten. Use
`RUN_ARGS="--experiment mini_68_alternative"` for a genuinely different setup, consistently for both
planners, or deliberately move an old experiment aside before rerunning it.
For different samples with the same row count, explicitly give different
experiment names. No automatic replacement or deletion is performed.
`PLANNER_OUTPUT_ROOT` is the artifact root inside Docker (`/artifacts`), mounted
from `NUPLAN_HOST_ARTIFACT_ROOT`. The original v1.2 IDM runner retains its legacy
`artifacts/idm_mini/` layout and is unchanged.

The runner checks exact log/token membership, duplicate or missing results,
category and planner identity, simulation failures, finite scores, and the
aggregate. A zero score is a valid scored outcome, not a simulation failure.
Validation errors produce a nonzero exit status; an aggregate over a silently
reduced subset must not be treated as a successful comparison. If simulation
fails before reports are available, inspect `console.log`; no validated result
is claimed.

This image uses nuPlan **1.2.2**, whereas the recorded original IDM result used
**1.2.0**. Use `idm-mini-matched` for the comparison, not just the historical
0.760088 score. Join paired results on **both `log_name` and `scenario`** and
check matching sample hashes and simulation settings first. Do not compare a
one-scenario smoke score with a 68-scenario average. This is a mini pilot, not
the published benchmark's original evaluation cohort.

## Long-format planner outcome table

`build_planner_outcomes.py` combines **completed full-sample** IDM and PDM runs.
It rereads the original reports and aggregate Parquet files, verifies matching
samples, code versions, packages and evaluation settings, and requires exactly
one valid outcome for each planner on each scenario. Smoke runs and incomplete
runs are rejected. It never silently drops scenarios to make the join succeed.

Run inside the PDM image after completing both planners in `mini_68_local` as
above. This command is for a new output and refuses to overwrite existing files.
The committed `mini_68` summaries alone cannot be used as source runs:

```bash
docker compose --project-directory . -f infra/docker/compose.pdm.yaml run --rm pdm \
  python -m tools.build_planner_outcomes \
  --idm-run /artifacts/mini_68_local/idm \
  --pdm-run /artifacts/mini_68_local/pdm_closed \
  --output /artifacts/mini_68_local/planner_outcomes.parquet \
  --csv /artifacts/mini_68_local/planner_outcomes.csv
```

The output is a **long-format dataset**, with 136 rows for 68 scenarios and two
planners. Its unique key is `(scenario_id, planner)` within this selected pair
of runs. Keep the experiment name when combining different experiment tables;
the recorded `run_id` remains available for source tracing.
It retains identifiers, map/category/window metadata, planner, run provenance,
success status, score, and the eight component metrics used in nuPlan scoring.
Component metrics are scores, not raw collision counts or physical units.
There is no stored difference column. Effects can be derived by pairing rows.

This is the **outcome table**, not yet a complete causal-model feature table.
Weather and ego/agent extractors now produce separate pre-simulation tables;
those tables still need to be joined to the outcomes by scenario ID. Scenario
category tags may describe future events and
must not be blindly used as model inputs. All rows from a driving log should
stay together in any train/test split; 136 rows are only 68 paired scenarios.

Parquet preserves types and embeds source run IDs and artifact hashes. The
builder rereads the saved file to verify it and refuses to overwrite an existing
output. The adjacent `experiment.json` records the sample, revisions, row count,
source run IDs, scores, artifact hashes and relative source directories. Original
logs/configs retain the paths used when the simulation ran; those historical
paths are not rewritten during relocation. Read the table in a notebook with
`pandas.read_parquet(...)`.

For a local CSV inspection copy, add `--csv /artifacts/mini_68/planner_outcomes.csv`
when building a new table. To export an **existing** Parquet without rebuilding
or rerunning simulations, use the same script's function inside the PDM Python
environment (replace the path with the desired table):

```python
from pathlib import Path
import pandas as pd
from tools.build_planner_outcomes import save_csv

source = Path("/artifacts/mini_68/planner_outcomes.parquet")
save_csv(pd.read_parquet(source), source.with_suffix(".csv"))
```

CSV contains the same rows and columns, without a pandas index, and refuses to
overwrite an existing file. Parquet remains the modeling source: CSV does not
preserve embedded provenance metadata or column types. When opening CSV in
Excel, import scenario tokens/IDs and microsecond timestamps as text to prevent
automatic conversion or precision loss.

If the two simulations are still running, add `--wait-seconds 10800` to queue
assembly for up to three hours. This does not start or restart simulations.
Failed validation stops assembly; a timeout leaves existing runs untouched and
does not create a partial table. Inspect the command's exit status afterward.
