# Implementation Plan: Planner Outcome Labels

## Overview

Add `tools/build_planner_labels.py` (Python) with `build` and `check` subcommands, test it on synthetic data, then build and check the real tables on IDM + PDM-Closed now and on all five planners once the remaining simulations finish. Do not modify `build_table_for_runs`, `tools/build_paired_dataset.py`, `tools/extract_scenario_features.py` or `src/causal_planner/data/features.py`.

## Tasks

- [x] 1. Implement label and modeling-table core
  - [x] 1.1 Create `tools/build_planner_labels.py` with constants, `token`, `parse_planners`, `label_columns`, `find_failures`, `best_planner`, `build_labels`, `InputError`, `CoverageError`
    - Module docstring with usage, `from __future__ import annotations`, `TOOL_VERSION`, `PLANNER_ORDER`, `TIE_TOLERANCE = 1e-9`, default paths (output root `artifacts/additional_planners`, experiment `pittsburgh_65_five`, Sample, Feature_Table and Out_Dir under `E:\Project\nuplan\processed\`)
    - Exclude Failed_Scenarios with one reason per failing planner; delta = `score_p - score_idm`; ties go to the earliest planner in `PLANNER_ORDER`
    - _Requirements: 1.1, 1.2, 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2_
  - [x] 1.2 Add `feature_columns` and `join_features`
    - Inner join on `scenario_id`; `CoverageError` with absent count and first ids; identity mismatch check on `log_name`, `scenario_token`, `map_name`; `fold` from `assign_log_folds` (imported from `tools.validate_scenario_subset`) with the Schema `split` folds and seed
    - _Requirements: 4.1, 4.2, 4.3, 4.4_
  - [x] 1.3 Add `check_tables`
    - Checks `unique_ids`, `row_counts`, `outcome_completeness`, `delta_exact`, `score_range`, `feature_coverage`, `column_separation` (provenance lists vs actual columns, disjointness, label prefixes, `best_planner`, Schema `forbidden_feature_patterns`), `log_folds` (reuse `check_log_split`)
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6_

- [x] 2. Add CLI, I/O, provenance and report
  - [x] 2.1 Implement `build` and `check` subcommands with every path as a flag, `--planners`, and `--overwrite`
    - Resolve Run_Roots with `experiment_run_root`; call `build_table_for_runs` unchanged; check output paths before any work; write `planner_outcomes.parquet` + `.csv` under the Experiment_Root and labels + modeling table in `--out-dir`
    - Local `write_parquet_atomic` (tmp + `os.replace`) reusing `arrow_schema` from `tools.build_scenario_features`, storing `planner_labels_provenance` (label/feature columns, planners, run roots, sha256 of Sample, Feature_Table and Schema, timestamp, tool version, exclusions, split)
    - Read the Feature_Table with a `scenario_id` filter; `build_report` + `markdown_report` writing `planner_labels.md` and `.json` to `--report-dir`
    - Exit 2 on input errors or existing outputs, 1 on run, coverage or check failures; `PASS`/`FAIL` lines
    - _Requirements: 1.3, 1.4, 1.5, 4.5, 5.7, 6.1, 6.2, 6.3, 6.4, 7.1_

- [x] 3. Write tests and run the suite
  - [x] 3.1 Create `tests/test_build_planner_labels.py` (unittest, synthetic long outcome frames and feature parquet in temp dirs, no E: or `docs/data` access)
    - Seeded 100-iteration loops for Properties 1-6 (label faithfulness and delta exactness, best planner consistency, failed-scenario exclusion, coverage gaps, fold disjointness by log, label/feature column separation), tagged `Feature: planner-outcome-labels, Property N`
    - Examples: `parse_planners` errors, `cmd_build` with `build_table_for_runs` patched (receives `experiment_run_root` paths; raising exits 1 with no output), identity mismatch, provenance keys, duplicate ids and row counts, CLI `build` + `check` exit codes 0/1/2 and `PASS`/`FAIL` output, existing output without `--overwrite`, no leftover `.tmp`, report JSON and Markdown contents
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2, 4.2, 4.3, 4.4, 4.5, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 6.2, 6.3, 6.4, 7.2_
  - [ ]* 3.2 Add a test that runs `build` then `check` on a five-planner synthetic experiment laid out with `experiment_run_root` directories and asserts five `score_`, four `delta_` and five `succeeded_` columns
    - _Requirements: 1.3, 2.1_
  - [x] 3.3 Checkpoint: run `.venv\Scripts\python.exe -m pytest -q -p no:subtests` (baseline 8,776 passed, 6 skipped, plus the new tests); ensure all tests pass, ask the user if questions arise
    - _Requirements: 7.1, 7.3_

- [ ] 4. Build and check real data
  - [x] 4.1 Run `.venv\Scripts\python.exe -m tools.build_planner_labels build --planners idm,pdm-closed`, then `check` (planners read from provenance); report row counts (expect 130 outcome rows and 65 label rows minus exclusions), excluded scenarios with reasons, mean scores vs official scores, delta summary, `best_planner` counts, fold sizes and feature coverage
    - _Requirements: 1.3, 1.4, 2.5, 3.2, 4.1, 5.7, 6.4_
  - [~] 4.2 After the planner containers finish, confirm each of the five logs `C:\nuplan\logs\pgh65_<planner>.log` ends with a JSON summary, then run `build --overwrite` and `check` with all five planners (default `--planners`); report the same figures (expect 325 outcome rows) and the regenerated `docs/data/planner_labels.md` / `.json`
    - _Requirements: 1.1, 1.3, 2.1, 5.2, 5.7, 6.3, 6.4_

## Notes

- Task marked `*` is optional.
- E: is a USB drive and Run_Roots are long Windows paths; keep all paths as flags and build them with `pathlib` and `experiment_run_root`.
- Label and modeling tables stay on E: and are never committed; only the Report under `docs/data` is committed.
- No Git, Linear or Drive steps.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2"] },
    { "id": 2, "tasks": ["1.3"] },
    { "id": 3, "tasks": ["2.1"] },
    { "id": 4, "tasks": ["3.1"] },
    { "id": 5, "tasks": ["3.2"] },
    { "id": 6, "tasks": ["3.3"] },
    { "id": 7, "tasks": ["4.1"] },
    { "id": 8, "tasks": ["4.2"] }
  ]
}
```
