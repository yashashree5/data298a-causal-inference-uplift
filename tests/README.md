# Tests

Implementation lives in `src/causal_planner/`. The original tests intentionally
retain their `tools.*` imports to check compatibility for existing checkouts and
notebooks. `test_repository_layout.py` additionally checks package imports,
direct/module CLI invocation, manifest resolution from another directory, and
that unimplemented feature commands cannot produce successful-looking outputs.

[`test_idm_mini_reproducer.py`](test_idm_mini_reproducer.py) checks our sample
loader and simulation command before spending time on an actual IDM run.

It checks that:

- The saved sample contains 68 unique scenario IDs across 14 categories.
- A scenario limit selects the requested number of rows.
- The command selects IDM, reactive simulation, and the requested scenario tokens.
- A CSV missing required columns is rejected.

## Install the host test environment

Follow the root [host development setup](../README.md#host-development-setup)
to create a Python 3.11 virtual environment, then install from the repository root:

```bash
python -m pip install -r requirements.txt
```

Use the virtual environment's interpreter for every command below. On Windows,
this can be `.\.venv\Scripts\python` without activating the environment.
These host checks need no Docker, GPU, or downloaded nuPlan databases. They do
not run simulations or verify the recorded planner scores. The base package's
declared Python 3.9 minimum does not describe this combined Python 3.11 test
environment.

## Test commands and known failures

`make check` runs `unittest` discovery and checks that the EDA notebook and draft
feature schema are valid JSON. It requires `make` and a `python3` executable in
the selected environment. On Windows, or without `make`, run its commands using
the virtual environment's interpreter:

```bash
python -m unittest discover -s tests -v
python -m json.tool notebooks/nuplan_mini_eda.ipynb
python -m json.tool configs/features/feature_schema.json
```

Unittest discovery does not collect standalone pytest functions or their
parametrized cases. Its reported count is therefore different from a pytest
run; do not present `make check` as the full pytest suite.

On macOS, the temporary directory can resolve from `/var/...` to
`/private/var/...`, causing the path assertion in
`test_build_planner_labels.CliTests.test_build_passes_run_roots_and_writes_outcomes_unchanged`
to fail. Use a canonical temporary path for the current process:

```bash
TMPDIR=/private/tmp make check
```

The same prefix can be applied to `python -m unittest` or `python -m pytest`.
This works around the existing path assertion; it does not fix that test.

Pull requests and pushes to `main` run pytest on Python 3.11, Linux and Windows
([`.github/workflows/ci.yml`](../.github/workflows/ci.yml)). To match its current
test selection, run:

```bash
python -m pytest -q \
  --deselect tests/test_extract_map_features.py::AssemblyPropertyTests::test_missing_indicators_and_row_coverage \
  --deselect tests/test_extract_map_features.py::BuildCliTests::test_end_to_end_build
python -m tools.validate_scenario_subset
```

The multiline command above uses bash/zsh syntax. In PowerShell, put the pytest
command and both `--deselect` arguments on one line, using
`.\.venv\Scripts\python` when the virtual environment is not activated.

CI also performs the two JSON checks above. Its two map-feature exclusions were
introduced for Linear 298-39 / PR #30; `make check` and an unfiltered
`python -m pytest -q` include those tests. Installing the shared requirements
enables their optional geospatial dependencies, so they run where an earlier
minimal environment skipped them. Record their actual results rather than
assuming an excluded test still fails. Keep any remaining test defects separate
from dependency installation, and report the command, platform, revision,
passed/failed counts, skips, and deselections for each verification.
See [`.github/README.md`](../.github/README.md) for the exclusion and
branch-protection notes.

## PDM-Closed checks

`test_pdm_mini_reproducer.py` checks that PDM and matched IDM use identical
evaluation settings except for the planner. It also tests result validation:
missing/extra/duplicate scenarios, failed simulations, wrong categories or
planners, invalid scores, and incorrect aggregates. These are small synthetic
tests, not simulations and not evidence of driving performance.

Run the host checks described above. Run the real one-scenario simulation with
`make pdm-mini-smoke`; see [tools](../tools/README.md) for the matched IDM run.

`test_build_planner_outcomes.py` checks that the long-format table builder rejects
different samples, revisions, packages, evaluation settings and metric weights.
Its artifact discovery requires one unambiguous source file per run.
