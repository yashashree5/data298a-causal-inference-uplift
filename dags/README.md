# Airflow DAG: `nuplan_pipeline`

One manually triggered DAG that runs the data pipeline and the five-planner
comparison in order. Every task runs an existing tool's `build` and then its
`check`. A failed build or check exits non-zero, fails the task and blocks
everything downstream: the checks are the quality gate.

- [`pipeline_spec.py`](pipeline_spec.py): the task graph, shell commands, retries
  and pools. Plain Python, unit-tested without Airflow
  ([`tests/test_nuplan_pipeline_dag.py`](../tests/test_nuplan_pipeline_dag.py)).
- [`nuplan_pipeline.py`](nuplan_pipeline.py): turns each step into a `BashOperator`.
- [`infra/docker/compose.airflow.yaml`](../infra/docker/compose.airflow.yaml) and
  [`Dockerfile.airflow`](../infra/docker/Dockerfile.airflow): local Airflow 2.10.

## Task graph

```text
extract_logs ─────┐
                  ├─► eligibility ─┬─► ego_agent_features ─┬─► map_features* ─┐
scenario_catalog ─┘                │                       └──────────────────┼─► integrate_features* ─┐
                                   ├─► weather_features ──────────────────────┘                        │
                                   └─► freeze_manifest ─► planner_<name> x5 ─► build_labels* ──────────┴─► model_dataset* ─► publish_drive*
```

| Task | Command (from the repository root) | Retries | Notes |
|---|---|---:|---|
| `extract_logs` | `tools.extract_nuplan_logs extract && check` | 0 | Skips DBs that are already extracted |
| `scenario_catalog` | `tools.build_scenario_catalog build && check` | 0 | Reads the ZIPs; eligibility's check compares against it |
| `eligibility` | `tools.build_scenario_eligibility build && check` | 0 | Skips logs that already have a shard |
| `ego_agent_features` | `build-mini` (once) `&& build && check` | 0 | `check` needs the mini_68 table |
| `weather_features` | `tools.extract_weather_features fetch && build && check` | 3 | `fetch` is a no-op when `open_meteo_raw.json` is cached |
| `freeze_manifest` | `tools.freeze_manifest check --manifest <MANIFEST>` | 0 | Never rebuilds the frozen manifest |
| `planner_<name>` | `docker compose ... compose.planners.yaml run --rm planner python -m tools.pdm_mini_reproducer` | 1 | Pool `simulation` (5 slots); IDM, PDM-Closed, PDM-Hybrid, UrbanDriver, GC-PGP |

\* Deferred: these tasks exit with code 99 and show as **skipped**, never as a
fake success, until their tools are on `main`:

- `map_features`: `tools.extract_map_features` is unfinished (branch `298-39-map-features`).
- `integrate_features`: `tools/build_scenario_features.py` and its leakage check
  (branch `298-40-feature-integration`).
- `build_labels`: score difference vs IDM (Linear 298-51).
- `model_dataset`: join of features, labels and split.
- `publish_drive`: `rclone copy --immutable` to the shared Drive.

Every task uses the trigger rule `none_failed`: a skipped upstream task does not
block, a failed one does. Enabling a deferred step means replacing its
`deferred(...)` entry in `pipeline_spec.py` with a real `Step`.

## Run it locally

1. Build the planner image once (see [`infra/docker`](../infra/docker/README.md)):
   the planner tasks use `data298a/pdm-closed-mini:1.0`.
2. Add the Airflow paths to `.env` (template: [`.env.example`](../.env.example)):

   | Variable | Meaning |
   |---|---|
   | `NUPLAN_HOST_DATA_ROOT` | Dataset root with `maps/` and `nuplan-v1.1/`; mounted at `/data` |
   | `NUPLAN_HOST_PROCESSED_ROOT` | Processed outputs; mounted at `/processed` |
   | `NUPLAN_HOST_ZIPS_DIR` | nuPlan ZIPs, including `nuplan-v1.1_mini.zip`; mounted read-only at `/zips` |
   | `NUPLAN_HOST_SCRATCH` | Temporary DB extraction folder; mounted at `/scratch` |
   | `NUPLAN_CITIES` | Optional, e.g. `pittsburgh`: limits extraction and catalog |
   | `AIRFLOW_DOCKER_REPO_DIR` | The repository as the Docker daemon sees it |
   | `AIRFLOW_DOCKER_DATA_ROOT` | The dataset root as the Docker daemon sees it |

   The planner container is started from inside Airflow through the host's
   Docker daemon, so `AIRFLOW_DOCKER_*` must be paths the daemon understands. On
   Linux they equal the host paths. On Windows with Docker Desktop, write
   `C:\Users\me\repo` as `/run/desktop/mnt/host/c/Users/me/repo`.
3. Start Airflow:

   ```text
   docker compose --project-directory . -f infra/docker/compose.airflow.yaml up -d --build
   ```

   Open http://localhost:8080 (user `airflow`, password `airflow`). The DAG is
   created paused; unpause it before triggering.
4. Stop it with `docker compose --project-directory . -f infra/docker/compose.airflow.yaml down`.

## Run options

Trigger with a JSON configuration ("Trigger DAG w/ config", or
`airflow dags trigger nuplan_pipeline --conf '{...}'`):

| Key | Default | Meaning |
|---|---|---|
| `cities` | Variable `CITIES` (all cities) | List or comma string of `boston`, `las_vegas`, `pittsburgh`, `singapore` |
| `planner_limit` | whole sample | Run only the first N scenarios per planner |
| `experiment` | `airflow_<logical date>` | Planner experiment folder; the runner refuses to overwrite one |
| `sample` | `configs/scenarios/pittsburgh_65_sample.csv` | Scenario CSV the planners run |
| `dataset_split` | `train_pittsburgh` | Split folder under `nuplan-v1.1/splits/` |
| `manifest` | `configs/scenarios/city4_ranked_top5.csv` | Frozen manifest re-validated before simulation |

Smoke run (Pittsburgh, one scenario per planner):

```json
{"cities": "pittsburgh", "planner_limit": 1, "experiment": "airflow_smoke"}
```

Without Docker Compose, set the Airflow Variables `DATA_ROOT`, `PROC_ROOT`,
`ZIPS_DIR`, `SCRATCH`, `HOST_REPO_DIR` and `HOST_DATA_ROOT` yourself; `MINI_ZIP`,
`REPORT_ROOT`, `HOST_ARTIFACT_ROOT`, `HOST_CHECKPOINT_ROOT`, `MANIFEST`,
`PLANNER_SAMPLE` and `DATASET_SPLIT` are optional. Check reports go to
`<PROC_ROOT>/reports/` because the repository is mounted read-only.

Weather stays historical (2021 Open-Meteo archive); the DAG never fetches live
weather for model inputs.
