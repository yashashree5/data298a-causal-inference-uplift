# Tools

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
make pdm-mini                           # All 68, not just the smoke scenario
make idm-mini-matched                    # All 68 in the same environment
```

For host-side command validation without Docker or data:

```bash
python3 -m tools.pdm_mini_reproducer --dry-run
```

Each execution creates a unique directory under `artifacts/pdm_mini/`, or
`pdm_mini/` under the configured host artifact root. It contains:

- `run_manifest.json`: exact selected IDs, full-sample hash, upstream commits,
  command, and dataset paths;
- `environment.txt`: installed Python packages;
- `console.log` and nested nuPlan logs, resolved configuration and metrics;
- `result.json`: success/failure counts and the official aggregate score;
- `scenario_scores.csv`: individual scenario scores and metric values.

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

Run inside the PDM image, replacing the two run-directory placeholders:

```bash
docker compose --project-directory . -f infra/docker/compose.pdm.yaml run --rm pdm \
  python -m tools.build_planner_outcomes \
  --idm-run /artifacts/pdm_mini/<full-idm-run> \
  --pdm-run /artifacts/pdm_mini/<full-pdm-run> \
  --output /artifacts/pdm_mini/planner_outcomes.parquet
```

The output is a **long-format dataset**, with 136 rows for 68 scenarios and two
planners. Its unique key is `(scenario_id, planner)` within this selected pair
of runs. Adding another experiment later also requires `run_id` in the key.
It retains identifiers, map/category/window metadata, planner, run provenance,
success status, score, and the eight component metrics used in nuPlan scoring.
Component metrics are scores, not raw collision counts or physical units.
There is no stored difference column. Effects can be derived by pairing rows.

This is the **outcome table**, not yet a complete causal-model feature table.
Initial speed, surrounding-agent counts and other pre-simulation conditions
still need extraction. Scenario category tags may describe future events and
must not be blindly used as model inputs. All rows from a driving log should
stay together in any train/test split; 136 rows are only 68 paired scenarios.

Parquet preserves types and embeds source run IDs and artifact hashes. The
builder rereads the saved file to verify it and refuses to overwrite an existing
output. Read it in a notebook with `pandas.read_parquet(...)`.

For a local CSV inspection copy, add `--csv /artifacts/pdm_mini/outcomes.csv`
when building a new table. To export an **existing** Parquet without rebuilding
or rerunning simulations, use the same script's function inside the PDM Python
environment (replace the path with the desired table):

```python
from pathlib import Path
import pandas as pd
from tools.build_planner_outcomes import save_csv

source = Path("/artifacts/pdm_mini/comparison_20261001T0218/planner_outcomes.parquet")
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
