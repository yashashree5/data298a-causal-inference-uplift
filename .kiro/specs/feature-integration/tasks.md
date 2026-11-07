# Implementation Plan: Feature Integration

## Overview

Add `tools/build_scenario_features.py` (Python), update the Schema and the validator's `--features` reader, test on synthetic parquet, then build and check the real tables. Do not modify `tools/extract_scenario_features.py` or `causal_planner/data/features.py`.

## Tasks

- [ ] 1. Implement integration core
  - [~] 1.1 Create `tools/build_scenario_features.py` with `select_feature_columns`, `integrate`, `CoverageError`, `InputError`
    - Module docstring with usage, `from __future__ import annotations`, constants `KEYS`, `LOCATION_COLUMNS`, default paths under `E:\Project\nuplan\processed\`
    - Select fields by `column` or `name`, exclude `map_name` and non-model fields; derive `_missing` for `null_with_indicator` when absent; inner-join coverage check with per-table absent and duplicate counts; sort by `scenario_id`
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 3.1_
  - [~] 1.2 Add `check_table` reusing `check_leakage` and `check_feature_table_columns` imported from `tools.validate_scenario_subset`
    - Forbidden pattern and location-column scan, unique ids, row count, indicator == isna, no nulls in `exclude_pair`/`fail_validation` features, `artifacts: [features]` columns present
    - _Requirements: 3.2, 3.3, 4.1, 4.2, 4.3, 4.4_

- [ ] 2. Add CLI, I/O, provenance and report
  - [~] 2.1 Implement `build` and `check` subcommands with every path as a flag and `--only {full,mini,both}`
    - Read only needed columns via pyarrow (eligibility filtered on `benchmark_candidate`; mini `map_name` from `configs/scenarios/idm_mini_sample.csv`); exit 2 on missing file/column, 1 on coverage or check failures; PASS/FAIL lines
    - `write_parquet_atomic` (tmp + `os.replace`) with `scenario_features_provenance` metadata
    - `build_report` writing `docs/data/scenario_features.md` and `.json` (columns, rows, null counts, rows by city, not-integrated features)
    - _Requirements: 2.3, 4.5, 5.1, 5.2, 5.3, 5.4_

- [ ] 3. Update Schema and validator
  - [~] 3.1 Edit `configs/features/first_experiment_schema.yaml` by hand: the 22 integrated fields get `status: available`, `column`, `artifacts: [features]`; add `features` to `map_name` artifacts; leave `route_length_m`, `lane_curvature`, `ego_x/y/heading`, `lane_id` unchanged
    - _Requirements: 6.1_
  - [~] 3.2 In `tools/validate_scenario_subset.py`, read `--features` column names from the parquet schema when the path ends in `.parquet`; update the `--features` help text
    - _Requirements: 6.2_

- [ ] 4. Write tests and run the suite
  - [~] 4.1 Create `tests/test_build_scenario_features.py` (unittest, synthetic parquet in a temp dir)
    - Seeded 100-iteration loops for Properties 1-5 (column contract, join faithfulness under shuffle, indicator consistency, coverage-gap counts, forbidden columns fail check), tagged `Feature: feature-integration, Property N`
    - Examples: duplicates, missing file/column exit 2, exclude_pair nulls, absent declared column, CLI build+check exit codes and PASS/FAIL output, provenance metadata, report files, validator parquet `--features`
    - _Requirements: 1.1, 1.3, 1.4, 2.1, 2.2, 2.3, 3.3, 4.2, 4.3, 4.4, 4.5, 5.1, 5.3, 6.2_
  - [ ]* 4.2 Add a test that the real Schema's model features minus not-integrated ones match the 22 expected columns
    - _Requirements: 6.1_
  - [~] 4.3 Checkpoint: run `.venv\Scripts\python.exe -m pytest -q -p no:subtests` and `.venv\Scripts\python.exe -m tools.validate_scenario_subset`; ensure all tests pass, ask the user if questions arise
    - _Requirements: 6.3_

- [ ] 5. Build and check real data
  - [~] 5.1 Run `.venv\Scripts\python.exe -m tools.build_scenario_features build`, then `check`, then `.venv\Scripts\python.exe -m tools.validate_scenario_subset --features E:\Project\nuplan\processed\scenario_features.parquet`; report row counts (expect 1,406,469 and 68), column count, null counts and per-city counts
    - _Requirements: 2.1, 4.1, 5.3, 6.2_

## Notes

- Task marked `*` is optional.
- E: is a USB drive; keep all paths as flags.
- No Git or Linear steps.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "3.1"] },
    { "id": 1, "tasks": ["1.2", "3.2"] },
    { "id": 2, "tasks": ["2.1"] },
    { "id": 3, "tasks": ["4.1", "4.2"] },
    { "id": 4, "tasks": ["4.3"] },
    { "id": 5, "tasks": ["5.1"] }
  ]
}
```
