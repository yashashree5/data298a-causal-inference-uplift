# Repository structure and migration

## What changed

Reusable implementation now lives in `src/causal_planner/`. The existing
`tools/` filenames remain command-line entry points and re-export the public
functions used by existing tests and notebooks. This is a structural refactor:
mini scenario selection, simulation settings, output paths, and validation
rules are unchanged.

| Responsibility | Implementation | Existing entry point |
| --- | --- | --- |
| Manifest loading | `data/scenarios.py` | Imported through the existing runner tools |
| Planner commands and execution | `simulation/runners.py` | `tools/idm_mini_reproducer.py`, `tools/pdm_mini_reproducer.py` |
| Report and matched-run validation | `simulation/validation.py` | Re-exported through the existing tools |
| Outcome assembly and exports | `data/outcomes.py` | `tools/build_planner_outcomes.py` |
| Ego/agent and weather extraction | Existing tools retained on main | `tools/extract_ego_agent_features.py`, `tools/extract_weather_features.py` |
| Mini outcome cleaning, pairing, and grouped folds | Existing tool retained on main | `tools/build_paired_dataset.py` |
| Feature extraction | `data/features.py` (scaffold) | `tools/extract_scenario_features.py` (scaffold) |
| Modeling table assembly and log splits | `data/dataset.py` (scaffold) | `tools/build_model_dataset.py` (scaffold) |

Implementation paths in the table are relative to `src/causal_planner/`.
Tests continue to exercise the original imports; additional compatibility tests
cover direct script execution, module execution, and package imports.

## Teammates with an existing checkout

Use the same commands after updating the checkout:

```bash
make check
python3 -m tools.pdm_mini_reproducer --dry-run
python3 -m tools.pdm_mini_reproducer --planner idm --dry-run
python3 tools/idm_mini_reproducer.py --dry-run
```

The tools add this checkout's `src/` directory to the import path through
`tools/_bootstrap.py`. No package installation or Docker image rebuild is
required for existing commands: Compose already mounts the whole checkout.

For development or notebooks that import `causal_planner` directly, install
the package in editable mode in that interpreter's environment:

```bash
python -m pip install --no-deps -e .
```

Editable installation is optional and uses `pyproject.toml`; defaults resolve
relative to this source checkout, not the caller's working directory. Run the
project from a checkout or editable install. Standalone wheel distribution with
bundled manifests/configuration is not part of this migration.

Heavy dependencies and upstream revisions remain defined by the existing Docker
files and requirement files under `infra/docker/`. Installing this package alone
does not install nuPlan, pandas, or the simulation environment.

## Feature pipeline and remaining assembly work

The package feature and dataset files reserve responsibilities only. Their
`--help` commands work; attempting execution exits with status 2 and an explicit
not-implemented message. They cannot generate feature data. The merged
`tools/extract_ego_agent_features.py` and `tools/extract_weather_features.py`
already implement extraction, with reports in `docs/data/`; the original tools
remain the supported entry points. Generated tables live outside Git and must
be supplied or regenerated with paths appropriate to the current machine.

`tools/build_paired_dataset.py` implements cleaning and pairing of the committed
IDM/PDM-Closed outcomes and grouped folds. Its field contract is
`configs/features/first_experiment_schema.yaml`. It does not yet join the separate
weather and ego/agent feature tables into a complete modeling table.

`configs/features/feature_schema.json` is marked `scaffold` and contains no
approved feature definitions. It is not the active field contract and must not
replace the YAML contract. Before extending dataset assembly:

1. Verify the actual simulation start from the resolved scenario configuration;
   do not assume it equals the anchor timestamp.
2. Define feature names, units, history window, and missing-value behavior.
3. Reuse the existing extractors and include only information available at or
   before simulation start.
4. Validate one feature row per scenario and matching manifest provenance.
5. Keep the five-planner outcome table in long format. For modeling, compare
   each candidate against IDM with `candidate_score - idm_score`; the existing
   paired builder still supports IDM/PDM-Closed only.
6. Assign whole source recordings to splits before fitting any preprocessing;
   multiple log segments from the same recording must stay together. The current
   mini paired builder groups by log; changing that policy is dataset work.
7. Verify on `mini_68` before generating a larger experiment.

Outcome/component scores and retrospective tags must not become predictor inputs
by default. The planned modeling table has one row per scenario, not one row per
planner. Preserve the existing long-format outcome files as source evidence.

## Pittsburgh

`artifacts/pittsburgh_v1/` contains a planning README for a larger modeling cohort.
The existing custom pilot is frozen in `configs/scenarios/pittsburgh_65_sample.csv`
with provenance in the adjacent JSON: 65 scenarios across 13 available categories.
Its recorded CRLF bytes are preserved by a file-specific `.gitattributes` rule;
normalizing the file would invalidate hashes in existing experiment artifacts.

`tools/run_five_planners.py` runs IDM, PDM-Closed, PDM-Hybrid, UrbanDriver, and
GC-PGP on that manifest using `--dataset-split train_pittsburgh`. Generated
experiments remain ignored under `artifacts/pittsburgh_planners/`. The mini
workflow remains available. See [additional planners](additional_planners.md)
for checkpoint verification, mounts, smoke checks, and full batch commands.

The merged data and feature tools remain available alongside these runners.
Generated weather and ego/agent tables must be supplied or regenerated; joining
them and extending the two-planner paired builder are separate dataset tasks.
