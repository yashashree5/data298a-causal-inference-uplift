# Implementation Plan: Ranked 4-City Manifest

## Overview

Build `tools/freeze_manifest.py` in two layers. First the pure core over pandas DataFrames (keys, candidate filtering, ranking with overlap filter, splits, manifest and cuts, reporting, checks), each followed by its tests in `tests/test_freeze_manifest.py`. Then the I/O shell (reading, hashing, atomic writes, `build`, `check`, `main`), end-to-end tests in `tmp_path`, docs, and an acceptance run on the real eligibility table. Tests use plain pytest with a seeded synthetic generator (`random.Random(seed)`, `@pytest.mark.parametrize("seed", range(100))`); Hypothesis is not added. Run tests with `.venv\Scripts\python.exe -m pytest -q`.

## Tasks

- [x] 1. Module skeleton, keys and candidate filtering
  - [x] 1.1 Create `tools/freeze_manifest.py` with constants and first pure functions
    - `from __future__ import annotations`; constants `TOOL_VERSION`, `CITIES`, `BENCHMARK_TYPES` (imported from `tools.build_scenario_eligibility`), `DEFAULT_ELIGIBILITY`, `DEFAULT_RANKING`, `DEFAULT_OUTPUT_DIR`, `DEFAULT_PREFIX`, `DEFAULT_SEED`, `DEFAULT_RATIOS`, `READ_COLUMNS`, `MANIFEST_COLUMNS`
    - Implement `rank_key`, `split_key`, `missing_columns` (sorted), `drop_mini` (returns frame and excluded count)
    - `READ_COLUMNS` holds only identifier, timing, map, tag and eligibility-flag columns
    - _Requirements: 1.2, 1.3, 1.4, 2.7_

  - [x] 1.2 Create `tests/test_freeze_manifest.py` with the seeded synthetic generator
    - Helper that builds a random candidate frame with all `READ_COLUMNS`: 1-4 cities, 1-5 types, 1-8 drives per city, 0-30 candidates per drive with frequently overlapping windows, random mini and `benchmark_candidate` flags, 16-hex tokens
    - Smoke test that the generator is deterministic per seed
    - _Requirements: 9.1_

  - [x]* 1.3 Write property test for candidate filtering
    - **Property 1: Candidate filtering**
    - **Validates: Requirements 1.1, 1.2**

  - [x]* 1.4 Write property test for missing column detection
    - **Property 2: Missing column detection**
    - **Validates: Requirements 1.3**

  - [x]* 1.5 Write property test for seed sensitivity
    - **Property 7: Seed sensitivity**
    - **Validates: Requirements 2.7**

- [ ] 2. Ranking with overlap filter
  - [x] 2.1 Implement `rank_and_filter(candidates, seed)`
    - Per city, groups in `(count, type)` order; within a group sort by `(rank_key, scenario_id)`
    - Shared per-drive sorted interval list with `bisect_left`, half-open windows, neighbour-only overlap check
    - Assign dense `group_rank` starting at 1
    - _Requirements: 2.1, 2.2, 2.3, 2.4_

  - [x] 2.2 Write property test for ranking order and dense group ranks
    - **Property 3: Ranking order and dense group ranks**
    - **Validates: Requirements 2.1, 2.4, 8.5**

  - [x] 2.3 Write property test comparing against a naive reference model
    - **Property 4: Overlap filter equals the reference model**
    - Reference uses a pairwise overlap check; also assert no accepted windows overlap within a drive
    - **Validates: Requirements 2.2, 2.3, 8.4**

  - [~] 2.4 Write example tests for the overlap filter
    - Touching windows `[0,10)` and `[10,20)` both accepted; nested window rejected; a rarer type claims a window before a more common type in the same drive
    - _Requirements: 2.2, 2.3_

- [ ] 3. Drive-grouped splits
  - [x] 3.1 Implement `validate_ratios`, `split_counts` and `assign_splits`
    - `validate_ratios` raises `ValueError` for non-finite values, negatives, wrong length, or sum off by more than 1e-9
    - `split_counts` per the design algorithm (round half to even, minimum one drive per positive-ratio split when n >= 3)
    - `assign_splits` per city, drives ordered by `(split_key, drive)`
    - _Requirements: 3.1, 3.2, 3.3, 3.5_

  - [~] 3.2 Write property test for split partition and coverage
    - **Property 8: Split partition and coverage**
    - **Validates: Requirements 3.1, 3.2, 3.3**

  - [~]* 3.3 Write property test for ratio validation
    - **Property 10: Ratio validation**
    - **Validates: Requirements 3.5**

  - [~] 3.4 Write example tests for `split_counts`
    - n = 0, 1, 2, 3, 4, 7, 208 with 0.70/0.15/0.15 and a triple with a zero ratio; check 208 gives 146/31/31
    - _Requirements: 3.2, 3.3_

- [ ] 4. Manifest rows and cuts
  - [x] 4.1 Implement `to_manifest(ranked, splits)` and `make_cut(manifest, n)`
    - Rename to `scenario_type`, lowercase tokens, derive and verify `db_file`, map split per drive, select `MANIFEST_COLUMNS`, sort by `(source_city, scenario_type, group_rank)`, raise `ValueError` on duplicate `scenario_id` or `db_file` mismatch
    - `make_cut` raises `ValueError` for n < 1 and keeps rows with `group_rank <= n` in order
    - _Requirements: 2.6, 3.4, 4.1, 4.2, 4.3, 4.4, 4.5, 5.1, 5.4_

  - [~] 4.2 Write property test for the manifest format contract
    - **Property 11: Manifest format contract**
    - Serialize with `to_csv(..., index=False, lineterminator="\n")` to bytes, assert no `\r`, and load the file with `load_scenarios` from a `tmp_path` copy
    - **Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 8.2**

  - [~] 4.3 Write property test for cut nesting and exact prefix
    - **Property 6: Cut nesting and exact prefix**
    - **Validates: Requirements 2.6, 5.1**

  - [~] 4.4 Write property test that splits are independent of the cut
    - **Property 9: Splits are independent of the cut**
    - **Validates: Requirements 3.4, 8.6**

  - [~] 4.5 Write property test for determinism
    - **Property 5: Determinism**
    - Build twice, once from a row-shuffled input, and compare CSV bytes
    - **Validates: Requirements 2.5**

- [x] 5. Reporting and provenance
  - [x] 5.1 Implement `group_table`, `split_coverage`, `manifest_counts` and `build_provenance`
    - `group_table` over all 56 `CITIES × BENCHMARK_TYPES` pairs with candidates, ranked, and per-N selected/shortfall
    - `split_coverage` per split: `groups_with_rows` and sorted `groups_without_rows` as `"city/type"`
    - `manifest_counts`: rows by split and city, rows by split and type, drives by split and city
    - `build_provenance` assembles the dict shown in the design (eligibility, ranking, manifests, groups, drive counts, git commit or null)
    - _Requirements: 3.6, 5.2, 5.3, 6.1, 6.2, 6.3, 6.4_

  - [~]* 5.2 Write property test for group table and shortfall accounting
    - **Property 12: Group table and shortfall accounting**
    - **Validates: Requirements 5.2, 5.3**

  - [~]* 5.3 Write property test for provenance count consistency
    - **Property 13: Provenance counts are consistent**
    - **Validates: Requirements 3.6, 6.2, 6.3**

  - [~]* 5.4 Write example test for a single-drive group
    - A Las-Vegas-like group with one drive appears in `groups_without_rows` for two splits
    - _Requirements: 3.6_

- [ ] 6. Check logic
  - [x] 6.1 Implement `run_checks(manifest, candidates, mini_recordings, recorded_sha, actual_sha, loader_error)`
    - Evaluate each of the seven checks from the design table independently and return `dict[str, bool]` plus a reason for each failure
    - A missing provenance entry (`recorded_sha` None) fails the SHA-256 check
    - _Requirements: 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8_

  - [~] 6.2 Write property test for corruption detection
    - **Property 14: Check detects each corruption**
    - Parametrize over the seven corruption kinds; assert exactly the matching check is False, and all True on the clean manifest
    - **Validates: Requirements 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 8.10**

- [~] 7. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 8. I/O shell and CLI
  - [x] 8.1 Implement file helpers
    - `read_candidates` (pyarrow schema check, then filtered read with `benchmark_candidate == True`, single `eligibility_version`), `mini_recordings`, `file_sha256` (16 MiB chunks), `git_commit` (None on any failure)
    - `write_csv_atomic` (UTF-8, `newline=""`, LF, returns SHA-256 of written bytes) and `write_json_atomic` (sorted keys, indent 2, trailing LF); temp file in target dir, `os.replace`, temp removed on exception
    - _Requirements: 1.1, 1.3, 4.6, 6.4, 6.5, 7.2_

  - [~]* 8.2 Write unit tests for the file helpers
    - LF-only CSV and JSON bytes, sorted JSON keys, temp file removed when a write raises, `git_commit` returns None when the subprocess fails
    - _Requirements: 4.6, 6.4, 6.5, 7.2_

  - [x] 8.3 Implement `build` and the argparse `main`
    - Flags from the design CLI section; `--per-group` repeatable, default `[5]`, de-duplicated and sorted; validate ratios and N before reading (exit 2)
    - Resolve all target paths first; refuse existing or protected targets (`idm_mini_sample.csv`, `pittsburgh_65_sample.csv`, `pittsburgh_65_sample.json`) with exit 3 before any read or write
    - Run the pure core, write ranking, cuts, then provenance; call `load_scenarios` on every written manifest (`[FAIL]` and exit 1 on error); print a summary JSON
    - Map missing columns, multiple eligibility versions, duplicate ids, `db_file` mismatch and missing input to `error: ...` with exit 2
    - _Requirements: 1.1, 1.2, 1.3, 2.5, 2.7, 3.3, 3.5, 4.7, 5.4, 6.1, 7.1, 7.3, 7.4_

  - [~] 8.4 Implement the `check` subcommand
    - Load manifest, provenance (default: manifest path with `.json` suffix), candidates and mini recordings; capture `load_scenarios` errors; call `run_checks`; print one `[PASS]`/`[FAIL]` line per check; return 0 or 1
    - _Requirements: 8.1, 8.2, 8.7, 8.9, 8.10_

- [ ] 9. End-to-end tests in `tmp_path`
  - [~] 9.1 Write end-to-end test for `build`
    - Write a small synthetic parquet; run `main(["build", ...])` with all outputs in `tmp_path`; assert ranking, cut and provenance exist, provenance has sorted keys and LF endings, and `git_commit` is null when monkeypatched to fail
    - Second run returns 3 and leaves files byte-identical; protected file names are refused with 3
    - _Requirements: 2.5, 5.2, 5.3, 6.1, 6.4, 6.5, 7.1, 7.3, 9.1_

  - [~] 9.2 Write end-to-end test for `check`
    - Returns 0 on fresh output; returns 1 with the expected `[FAIL]` line after each corruption from Property 14 applied to the written CSV or provenance
    - _Requirements: 8.1, 8.9, 8.10, 9.1_

  - [~] 9.3 Write tests for input errors
    - Missing column, invalid ratios and N = 0 return 2 and write no files
    - _Requirements: 1.3, 3.5, 5.4_

- [~] 10. Checkpoint - Ensure all tests pass
  - Run `.venv\Scripts\python.exe -m pytest -q`; the existing 124 tests and all new tests pass. Ask the user if questions arise.

- [ ] 11. Usage documentation
  - [~] 11.1 Add a `freeze_manifest.py` section to `tools/README.md`
    - `build` and `check` commands with defaults, output locations (full Ranking outside Git, cuts under `configs/scenarios/`), exit codes
    - _Requirements: 7.4_

  - [~] 11.2 Add the `city4_ranked_top5` files to `configs/scenarios/README.md`
    - What the cut is, how it nests in larger cuts, where the full Ranking lives, how to re-run `check`
    - _Requirements: 2.6, 7.4_

- [ ] 12. Acceptance run on the real eligibility table
  - [~] 12.1 Run `build` and `check` on `E:\Project\nuplan\processed\scenario_eligibility.parquet`
    - `.venv\Scripts\python.exe -m tools.freeze_manifest build` with defaults; writes `E:\Project\nuplan\processed\manifests\city4_ranked_seed7_full.csv`, `configs/scenarios/city4_ranked_top5.csv` and `configs/scenarios/city4_ranked_top5.json`
    - `.venv\Scripts\python.exe -m tools.freeze_manifest check --manifest configs/scenarios/city4_ranked_top5.csv` returns 0; also run `check` on the full Ranking with `--provenance configs/scenarios/city4_ranked_top5.json`
    - Compare with the design Measurements: 15,838 ranked rows, 828 drives (boston 208, las_vegas 98, pittsburgh 193, singapore 329), 53 non-empty groups, two groups under 5 (las_vegas/high_lateral_acceleration, singapore/following_lane_with_lead); report any difference
    - _Requirements: 1.1, 2.5, 3.2, 4.7, 5.2, 6.1, 8.9_

## Notes

- Tasks marked with `*` are optional. The core property tests (ranking order, overlap reference model, cut nesting, split partition, cut-independent splits, determinism, manifest format, corruption detection) are required.
- All implementation is in `tools/freeze_manifest.py`; all tests are in `tests/test_freeze_manifest.py`. Property tests carry the docstring tag `Feature: ranked-city-manifest, Property <n>: <title>`.
- Property tests use a seeded generator over `range(100)` seeds; Hypothesis is not installed and is not added.
- The real 1.4 M-row table is used only in task 12.1.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2", "2.1"] },
    { "id": 2, "tasks": ["1.3", "3.1"] },
    { "id": 3, "tasks": ["1.4", "4.1"] },
    { "id": 4, "tasks": ["1.5", "5.1"] },
    { "id": 5, "tasks": ["2.2", "6.1"] },
    { "id": 6, "tasks": ["2.3", "8.1"] },
    { "id": 7, "tasks": ["2.4", "8.3"] },
    { "id": 8, "tasks": ["3.2", "8.4"] },
    { "id": 9, "tasks": ["3.3", "11.1", "11.2"] },
    { "id": 10, "tasks": ["3.4"] },
    { "id": 11, "tasks": ["4.2"] },
    { "id": 12, "tasks": ["4.3"] },
    { "id": 13, "tasks": ["4.4"] },
    { "id": 14, "tasks": ["4.5"] },
    { "id": 15, "tasks": ["5.2"] },
    { "id": 16, "tasks": ["5.3"] },
    { "id": 17, "tasks": ["5.4"] },
    { "id": 18, "tasks": ["6.2"] },
    { "id": 19, "tasks": ["8.2"] },
    { "id": 20, "tasks": ["9.1"] },
    { "id": 21, "tasks": ["9.2"] },
    { "id": 22, "tasks": ["9.3"] },
    { "id": 23, "tasks": ["12.1"] }
  ]
}
```
