# Requirements Document

## Introduction

Linear issue 298-49 ("Freeze the ranked 4-city manifest with drive-grouped splits"). A new command-line tool, `tools/freeze_manifest.py`, freezes one deterministic ranking of nuPlan benchmark scenarios across all four cities (boston, las_vegas, pittsburgh, singapore) from the scenario eligibility table at `E:\Project\nuplan\processed\scenario_eligibility.parquet`. The five-planner demo set (top 5 per city/type group) and the later training set (top N per group) are both cuts of this single ranking, so the demo set is always a prefix of the training set. Every drive (`source_recording`) is assigned to exactly one of train, val or test by a seeded hash, independent of the cut size. The tool has a `build` subcommand that writes the manifest and provenance, and a `check` subcommand that re-validates a frozen manifest.

The tool follows the conventions of `tools/build_scenario_eligibility.py` and `tools/freeze_pittsburgh_sample.py`: argparse subcommands, `from __future__ import annotations`, pandas/pyarrow, atomic writes through a temporary file and `os.replace`, defaults overridable by flags, provenance JSON, PASS/FAIL check lines and integer exit codes.

### Assumptions

- The ranking key reuses the Pittsburgh convention: `sha256(seed + ":" + scenario_id)` truncated to 16 hex characters, with default seed 7, ties broken by `scenario_id`.
- The time-overlap filter is shared across all scenario types within one `source_recording`, as in `tools/freeze_pittsburgh_sample.py`. Groups are processed from rarest to most common candidate count within each city, so rare groups claim windows first.
- The overlap filter is run to exhaustion (no quota cap), so `group_rank` is defined for every non-overlapping candidate and any cut size N yields a nested set. Whether the full ranking is committed or kept under `E:\Project\nuplan\processed\manifests\` is decided in design based on its measured size.
- Split assignment is stratified by city: drives are ordered by a seeded hash within each city and the first ~70% go to train, the next ~15% to val and the rest to test. Groups with fewer drives than splits (for example las_vegas/high_lateral_acceleration with 1 drive) land entirely in one or two splits; this is reported, not corrected.
- Held-out-city evaluation is done at analysis time and is out of scope.
- The 3 missing (city, type) groups out of 56 are reported as zero-availability groups, not errors.

## Glossary

- **Freeze_Tool**: The command-line program `tools/freeze_manifest.py` with subcommands `build` and `check`.
- **Eligibility_Table**: The parquet file of scenario candidates produced by `tools/build_scenario_eligibility.py`, default `E:\Project\nuplan\processed\scenario_eligibility.parquet`.
- **Candidate**: A row of the Eligibility_Table with `benchmark_candidate` equal to True.
- **Drive**: A distinct value of `source_recording`; the unit of split assignment and of overlap checking.
- **Group**: The set of Candidates sharing one (`source_city`, `benchmark_scenario_type`) pair.
- **Window**: The interval [`window_start_us`, `window_end_us`) of a Candidate.
- **Overlap**: Two Windows in the same Drive where each starts before the other ends.
- **Rank_Key**: The first 16 hex characters of SHA-256 of the string `"<seed>:<scenario_id>"`.
- **Group_Rank**: The 1-based position of a Candidate within its Group after the overlap filter, in Rank_Key order.
- **Ranking**: The full set of Candidates that survive the overlap filter, each with a Group_Rank and a split.
- **Cut**: The subset of the Ranking with Group_Rank less than or equal to a per-group count N.
- **Manifest**: A CSV file written by the Freeze_Tool containing a Ranking or a Cut.
- **Split**: One of the labels `train`, `val` or `test` assigned to a Drive.
- **Provenance**: The JSON file written alongside a Manifest that records inputs, parameters, counts and hashes.
- **Shortfall**: The difference between a requested per-group count N and the number of Candidates the Group provides in a Cut, when positive.
- **Mini_Recording**: A `source_recording` that also appears in the nuPlan mini split (`shares_mini_recording` True or `in_mini_split` True).
- **Scenario_Loader**: The function `load_scenarios` in `src/causal_planner/data/scenarios.py`.

## Requirements

### Requirement 1: Candidate selection

**User Story:** As a researcher, I want the manifest drawn only from validated benchmark candidates, so that every scenario can be simulated and none leaks from the mini split.

#### Acceptance Criteria

1. WHEN `build` runs, THE Freeze_Tool SHALL read only Eligibility_Table rows whose `benchmark_candidate` value is True.
2. IF a Candidate row has `shares_mini_recording` True or `in_mini_split` True, THEN THE Freeze_Tool SHALL exclude the row and report the excluded count in the Provenance.
3. IF the Eligibility_Table lacks any column the Freeze_Tool reads, THEN THE Freeze_Tool SHALL print the missing column names and exit with a nonzero exit code.
4. THE Freeze_Tool SHALL read only pre-simulation identifier, timing, map and tag columns from the Eligibility_Table.

### Requirement 2: Deterministic ranking with overlap filter

**User Story:** As a researcher, I want one deterministic ranking per city/type group, so that the demo set and training set come from the same frozen order.

#### Acceptance Criteria

1. THE Freeze_Tool SHALL order Candidates within each Group by Rank_Key ascending, breaking ties by `scenario_id` ascending.
2. THE Freeze_Tool SHALL process Groups within each city in ascending order of Candidate count, breaking ties by scenario type name.
3. WHEN a Candidate's Window has an Overlap with a Window already accepted in the same Drive, THE Freeze_Tool SHALL skip the Candidate.
4. WHEN a Candidate is accepted, THE Freeze_Tool SHALL assign the Candidate a Group_Rank equal to one plus the number of Candidates previously accepted in the same Group.
5. WHEN `build` runs twice with the same Eligibility_Table and seed, THE Freeze_Tool SHALL produce byte-identical Manifest files.
6. WHEN Cuts with per-group counts N1 less than N2 are produced from the same Ranking, THE Freeze_Tool SHALL produce a Cut for N1 that is a subset of the Cut for N2.
7. WHERE a `--seed` flag is given, THE Freeze_Tool SHALL use the given integer in the Rank_Key and split hash instead of the default seed 7.

### Requirement 3: Drive-grouped splits

**User Story:** As a researcher, I want each drive assigned to exactly one split, so that train, val and test never share a recording.

#### Acceptance Criteria

1. THE Freeze_Tool SHALL assign each Drive in the Ranking exactly one Split from a seeded hash of the Drive identifier.
2. THE Freeze_Tool SHALL assign Splits within each city separately, so that each city with at least 3 Drives has at least one Drive in each of train, val and test.
3. THE Freeze_Tool SHALL target split ratios of 0.70 train, 0.15 val and 0.15 test, overridable by a `--split-ratios` flag.
4. THE Freeze_Tool SHALL compute Split assignments from the Drive set of the full Ranking, so that Split assignments are identical for every Cut size N.
5. IF the given split ratios are negative or do not sum to 1 within 1e-9, THEN THE Freeze_Tool SHALL print an error and exit with a nonzero exit code.
6. THE Freeze_Tool SHALL record in the Provenance, for each Split, the number of Groups with at least one Manifest row and the list of Groups with zero Manifest rows.

### Requirement 4: Manifest output contract

**User Story:** As a pipeline developer, I want the manifest to load with the existing scenario loader, so that the simulation runners accept it unchanged.

#### Acceptance Criteria

1. THE Freeze_Tool SHALL write Manifest columns in the order `scenario_id`, `log_name`, `db_file`, `scenario_token`, `anchor_timestamp_us`, `scenario_type`, `all_scenario_tags`, `map_name`, `source_recording`, `window_start_us`, `window_end_us`, `source_city`, `group_rank`, `split`.
2. THE Freeze_Tool SHALL set `scenario_type` to the Candidate's `benchmark_scenario_type`.
3. THE Freeze_Tool SHALL write `scenario_token` as 16 lowercase hexadecimal characters and `db_file` as `log_name` followed by `.db`.
4. THE Freeze_Tool SHALL write each `scenario_id` at most once per Manifest.
5. THE Freeze_Tool SHALL write Manifest rows sorted by `source_city`, `scenario_type`, `group_rank`.
6. THE Freeze_Tool SHALL write Manifest CSV files as UTF-8 with LF line endings.
7. WHEN a Manifest is written, THE Scenario_Loader SHALL load the Manifest without raising an error.

### Requirement 5: Cuts and shortfalls

**User Story:** As a researcher, I want to emit a demo cut and a training cut from the same ranking and see where groups fall short, so that I know the real coverage of each set.

#### Acceptance Criteria

1. WHERE a per-group count N is requested, THE Freeze_Tool SHALL write a Manifest containing exactly the Ranking rows with Group_Rank less than or equal to N.
2. WHEN a Group provides fewer than N rows in a Cut, THE Freeze_Tool SHALL record the Group's available, selected and Shortfall counts in the Provenance and continue.
3. THE Freeze_Tool SHALL record in the Provenance, for every one of the 56 (city, type) pairs, the Candidate count, the Ranking count, and the selected count and Shortfall for each requested N.
4. IF a requested N is less than 1, THEN THE Freeze_Tool SHALL print an error and exit with a nonzero exit code.

### Requirement 6: Provenance

**User Story:** As a reviewer, I want a provenance record next to each manifest, so that I can reproduce and audit the freeze.

#### Acceptance Criteria

1. WHEN `build` completes, THE Freeze_Tool SHALL write a Provenance JSON file containing the tool version, seed, split ratios, requested per-group counts, Eligibility_Table path and fingerprint, eligibility version, Git commit, and the SHA-256 of each written Manifest.
2. THE Freeze_Tool SHALL record in the Provenance the row counts per Split by city and by scenario type for each written Manifest.
3. THE Freeze_Tool SHALL record in the Provenance the Drive counts per Split by city.
4. IF the Git commit cannot be determined, THEN THE Freeze_Tool SHALL record the Git commit as null and continue.
5. THE Freeze_Tool SHALL write Provenance JSON with sorted keys, 2-space indentation and LF line endings.

### Requirement 7: Safe writes

**User Story:** As a researcher, I want frozen outputs protected, so that a rerun cannot silently replace a committed manifest.

#### Acceptance Criteria

1. IF any target Manifest or Provenance file already exists, THEN THE Freeze_Tool SHALL print a message stating that nothing was overwritten and exit with a nonzero exit code before writing any file.
2. THE Freeze_Tool SHALL write each output to a temporary file in the target directory and move the file into place with `os.replace`.
3. THE Freeze_Tool SHALL leave `configs/scenarios/idm_mini_sample.csv`, `configs/scenarios/pittsburgh_65_sample.csv` and `configs/scenarios/pittsburgh_65_sample.json` unmodified.
4. THE Freeze_Tool SHALL write committed Cuts and their Provenance under `configs/scenarios/` by default, with every output path overridable by a flag.

### Requirement 8: Check subcommand

**User Story:** As a reviewer, I want to re-validate a frozen manifest against its source, so that I can trust it before running planners.

#### Acceptance Criteria

1. WHEN `check` runs, THE Freeze_Tool SHALL print one PASS or FAIL line per check.
2. WHEN `check` runs, THE Freeze_Tool SHALL verify that the Scenario_Loader loads the Manifest.
3. WHEN `check` runs, THE Freeze_Tool SHALL verify that every Manifest row matches a Candidate in the Eligibility_Table by `scenario_id` with equal `source_recording`, `window_start_us` and `window_end_us`.
4. WHEN `check` runs, THE Freeze_Tool SHALL verify that no two Manifest rows in the same Drive have an Overlap.
5. WHEN `check` runs, THE Freeze_Tool SHALL verify that the Group_Rank values of each Group form the sequence 1 to k with no gaps.
6. WHEN `check` runs, THE Freeze_Tool SHALL verify that all Manifest rows of each Drive carry the same Split.
7. WHEN `check` runs, THE Freeze_Tool SHALL verify that the Manifest SHA-256 equals the value recorded in the Provenance.
8. WHEN `check` runs, THE Freeze_Tool SHALL verify that no Manifest row belongs to a Mini_Recording.
9. WHEN every check passes, THE Freeze_Tool SHALL exit with code 0.
10. IF any check fails, THEN THE Freeze_Tool SHALL exit with code 1.

### Requirement 9: Tests

**User Story:** As a developer, I want unit tests on synthetic data, so that the tool is verified without the 20-million-row table.

#### Acceptance Criteria

1. THE test module `tests/test_freeze_manifest.py` SHALL exercise ranking, overlap filtering, nesting of Cuts, split assignment, shortfall reporting, overwrite refusal and each `check` failure using synthetic DataFrames.
2. WHEN `.venv\Scripts\python.exe -m pytest -q` runs, THE test suite SHALL pass all existing tests and all new tests.
