# Additional matched planners

The matched mini experiment supports five logical planners on the unchanged
68-scenario manifest:

| CLI name | Upstream planner | Checkpoint |
| --- | --- | --- |
| `idm` | IDMPlanner | none |
| `pdm-closed` | PDMClosedPlanner | none |
| `pdm-hybrid` | PDMHybridPlanner | `pdm_offset_checkpoint.ckpt` |
| `urban-driver` | MLPlanner with UrbanDriver model | `urbandriver_checkpoint.ckpt` |
| `gc-pgp` | MLPlanner with GC-PGP model | `gc_pgp_checkpoint.ckpt` |

The learned planners use official tuPlan Garage checkpoints. Keep checkpoint
files out of Git and mount them read-only. Before simulation, the runner checks
the filename, size, and SHA-256 against `checkpoints/provenance.json`. It records
the verified checkpoint metadata in the run manifest.

## Isolation from other experiments

Use `compose.planners.yaml`, which has a distinct Compose project name and a
separate default artifact mount. It does not reuse the existing `mini_68`
directories. The dataset and checkpoints are read-only, and the source checkout
is also mounted read-only.

Set the dataset root in `.env`, then prepare distinct host directories:

```bash
mkdir -p artifacts/additional_planners checkpoints
```

Optional external locations:

```bash
export ADDITIONAL_PLANNER_HOST_ARTIFACT_ROOT=/absolute/path/to/new/artifacts
export PLANNER_HOST_CHECKPOINT_ROOT=/absolute/path/to/checkpoints
```

## Dry runs

Dry runs validate the saved manifest and print commands without requiring a
checkpoint or starting a containerized simulation:

```bash
python3 -m tools.pdm_mini_reproducer --planner pdm-hybrid --dry-run
python3 -m tools.pdm_mini_reproducer --planner urban-driver --dry-run
python3 -m tools.pdm_mini_reproducer --planner gc-pgp --dry-run
```

## One-scenario smoke tests

Use a new experiment name. Run one planner at a time so a failure cannot affect
another planner's artifacts:

```bash
docker compose --project-directory . -f infra/docker/compose.planners.yaml run --rm planner \
  python -m tools.pdm_mini_reproducer --planner pdm-hybrid --limit 1 \
  --experiment extra_planners_smoke

docker compose --project-directory . -f infra/docker/compose.planners.yaml run --rm planner \
  python -m tools.pdm_mini_reproducer --planner urban-driver --limit 1 \
  --experiment extra_planners_smoke

docker compose --project-directory . -f infra/docker/compose.planners.yaml run --rm planner \
  python -m tools.pdm_mini_reproducer --planner gc-pgp --limit 1 \
  --experiment extra_planners_smoke
```

Only after all three smoke tests validate should the same commands be run
without `--limit 1`, using a new full experiment such as
`mini_68_five_planners_20261007`.

## Assemble an arbitrary matched planner set

The outcome builder accepts repeatable `--run PLANNER=DIR` arguments. A complete
five-planner table contains 340 rows (68 scenarios times five planners):

```bash
python -m tools.build_planner_outcomes \
  --run idm=/artifacts/mini_68_five_planners_20261007/idm \
  --run pdm-closed=/artifacts/mini_68_five_planners_20261007/pdm_closed \
  --run pdm-hybrid=/artifacts/mini_68_five_planners_20261007/pdm_hybrid \
  --run urban-driver=/artifacts/mini_68_five_planners_20261007/urban_driver \
  --run gc-pgp=/artifacts/mini_68_five_planners_20261007/gc_pgp \
  --output /artifacts/mini_68_five_planners_20261007/planner_outcomes.parquet \
  --csv /artifacts/mini_68_five_planners_20261007/planner_outcomes.csv
```

The legacy `--idm-run` and `--pdm-run` pair remains supported for the existing
two-planner workflow.

## Pittsburgh 65-scenario batch

Pittsburgh uses the fixed `configs/scenarios/pittsburgh_65_sample.csv` manifest:
five scenarios from each of the 13 categories available in the Pittsburgh
split. Pittsburgh has no `traversing_pickup_dropoff` examples, so this is a
custom 13-category cohort rather than an official Val14 reproduction.

Point the artifact mount at a Pittsburgh-specific directory and run the batch:

```bash
export ADDITIONAL_PLANNER_HOST_ARTIFACT_ROOT=./artifacts/pittsburgh_planners

docker compose --project-directory . -f infra/docker/compose.planners.yaml run --rm planner \
  python -m tools.run_five_planners \
  --sample configs/scenarios/pittsburgh_65_sample.csv \
  --dataset-split train_pittsburgh \
  --smoke-experiment pittsburgh_smoke_YYYYMMDD \
  --experiment pittsburgh_65_five_planners_YYYYMMDD
```

The batch runs a one-scenario smoke test immediately before each planner's full
run. It stops on the first failure and writes `batch_status.json`. A successful
five-planner batch produces 325 matched rows: 65 scenarios times five planners.
Use new smoke and full experiment names for every attempt; existing outputs are
never overwritten.

## Queued batch after the active regression

`tools.run_additional_planners` waits for valid IDM and PDM-Closed
`result.json` files under a read-only `/reference` mount (up to 24 hours).
It runs PDM-Hybrid, UrbanDriver, and GC-PGP sequentially, each with a
one-scenario smoke test immediately before its 68-scenario run. Any failing
stage stops the batch. The final assembly rechecks matched configurations,
scenario coverage, planner/model identity, and checkpoint provenance before
writing a new 340-row table; it never updates the reference tables.

The October 7 batch uses the already-installed image by immutable image ID,
a four-CPU/eight-GiB cap, no network, and separate output directories. Its
status is saved atomically in
`artifacts/additional_planners/mini_68_five_planners_20261007/batch_status.json`.
Check progress with `docker logs data298-five-planner-batch-20261007`.
The earlier `extra_smoke_20261007` test was interrupted to free resources
for the active PDM-Closed run; its outputs are retained but are not used.
The queued batch uses the fresh smoke experiment `extra_smoke_queued_20261007`.

Downloaded checkpoint sizes and SHA-256 hashes are recorded in
`checkpoints/provenance.json`. Unit tests passing and downloaded checkpoints
do not establish simulation success: only a completed batch with validated
artifacts does. Keep Docker and the computer running for this local batch;
inspect a failed stage before retrying with a new experiment name.
