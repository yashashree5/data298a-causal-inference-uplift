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

## Future feature pipeline

The new feature and dataset files reserve responsibilities only. Their
`--help` commands work; attempting execution exits with status 2 and an explicit
not-implemented message. They cannot generate feature data yet.

`configs/features/feature_schema.json` is marked `scaffold` and contains no
approved feature definitions. Before implementation:

1. Verify the actual simulation start from the resolved scenario configuration;
   do not assume it equals the anchor timestamp.
2. Define feature names, units, history window, and missing-value behavior.
3. Extract only information available at or before simulation start.
4. Validate one feature row per scenario and matching manifest provenance.
5. Pair the two planner outcomes and calculate `score_pdm - score_idm`.
6. Assign whole driving logs to splits before fitting any preprocessing.
7. Verify on `mini_68` before generating a larger experiment.

Outcome/component scores and retrospective tags must not become predictor inputs
by default. The planned modeling table has one row per scenario, not one row per
planner. Preserve the existing long-format outcome files as source evidence.

## Pittsburgh

`artifacts/pittsburgh_v1/` contains only a planning README. Generated output
remains ignored by Git. `configs/scenarios/pittsburgh_sample.csv` does not yet
exist: it must be created from the inspected catalog, not an empty or fabricated
cohort. Record equal feasible category counts and log coverage before simulations.

The EDA notebook and current runners still use the mini split. Adding configurable
Pittsburgh data paths, selecting the cohort, extracting features, and running new
simulations are subsequent work, not completed by this structural refactor.
