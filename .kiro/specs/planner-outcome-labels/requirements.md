# Requirements Document

## Introduction

Turn the Pittsburgh 65-scenario x 5-planner simulation runs into model labels and a modeling table. A new tool `tools/build_planner_labels.py` assembles the long outcome table with `causal_planner.data.outcomes.build_table_for_runs`, derives one label row per scenario (per-planner scores, deltas against IDM, best planner), joins the labels with `scenario_features.parquet`, assigns log-grouped folds, checks the result, and writes a data report. The tool works with any finished subset of planners that includes IDM, so it runs now on IDM and PDM-Closed and later on all five. Linear 298-51.

## Glossary

- **Builder**: the `build` subcommand of `tools/build_planner_labels.py`.
- **Checker**: the `check` subcommand of `tools/build_planner_labels.py`.
- **Schema**: `configs/features/first_experiment_schema.yaml`.
- **Sample**: `configs/scenarios/pittsburgh_65_sample.csv` (65 scenarios).
- **Planner**: one of `idm`, `pdm-closed`, `pdm-hybrid`, `urban-driver`, `gc-pgp`, in this fixed order (the Planner_Order).
- **Planner_Token**: the snake_case form of a Planner name (`pdm-closed` becomes `pdm_closed`).
- **Included_Planners**: the Planners selected with `--planners`, default all five; `idm` is the control.
- **Run_Root**: the directory returned by `causal_planner.simulation.runners.experiment_run_root(output_root, experiment, planner)`.
- **Experiment_Root**: the parent directory of the Run_Roots, default `artifacts/additional_planners/pittsburgh_65_five/`.
- **Outcome_Table**: `planner_outcomes.parquet` (and `planner_outcomes.csv`) under the Experiment_Root, one row per (`scenario_id`, Planner), as returned by `build_table_for_runs`.
- **Failed_Scenario**: a Sample scenario where any Included_Planner row has `succeeded` false, a null `score`, or no row.
- **Label_Table**: `planner_labels.parquet`, one row per non-failed Sample scenario.
- **Label_Columns**: the Label_Table columns `score_<token>`, `delta_<token>_vs_idm`, `succeeded_<token>` and `best_planner`.
- **Feature_Table**: `E:\Project\nuplan\processed\scenario_features.parquet`.
- **Feature_Columns**: every Feature_Table column other than `scenario_id`, `log_name` and `scenario_token`.
- **Modeling_Table**: `planner_modeling_table.parquet`, the Label_Table inner-joined with the Feature_Table on `scenario_id`, plus `fold`.
- **Out_Dir**: the Label_Table and Modeling_Table directory, default `E:\Project\nuplan\processed\labels\pittsburgh_65_five\`.
- **Report**: `docs/data/planner_labels.md` and `docs/data/planner_labels.json`.
- **Tie_Tolerance**: 1e-9, the score difference below which two scores are treated as equal (the `pair_outcome` convention of `tools/build_paired_dataset.py`).

## Requirements

### Requirement 1: Planner Selection and Outcome Table

**User Story:** As a modeler, I want the long outcome table built from whichever planner runs have finished, so that I can start before every simulation is done.

#### Acceptance Criteria

1. THE Builder SHALL accept `--planners` as a comma-separated list of Planner names with a default of all five Planners.
2. IF `--planners` omits `idm`, names an unknown Planner, or repeats a Planner, THEN THE Builder SHALL exit with status 2 and print the offending value.
3. WHEN the Builder runs, THE Builder SHALL resolve each Included_Planner's Run_Root with `experiment_run_root` and pass the Run_Roots and the Sample to `build_table_for_runs`.
4. WHEN `build_table_for_runs` returns, THE Builder SHALL write the returned table unchanged as the Outcome_Table in Parquet and CSV form under the Experiment_Root.
5. IF `build_table_for_runs` raises an error, THEN THE Builder SHALL exit with status 1, print the error, and write no output.

### Requirement 2: Label Table

**User Story:** As a modeler, I want one label row per scenario, so that I can train models that predict which planner does best.

#### Acceptance Criteria

1. WHEN the Builder runs, THE Builder SHALL write a Label_Table containing `scenario_id`, `log_name`, `scenario_token`, `scenario_type`, `map_name`, and the Label_Columns for the Included_Planners.
2. THE Builder SHALL set `score_<token>` to the Outcome_Table `score` of that Planner and `succeeded_<token>` to the Outcome_Table `succeeded` of that Planner for the same `scenario_id`.
3. THE Builder SHALL set `delta_<token>_vs_idm` to `score_<token>` minus `score_idm` for every Included_Planner other than `idm`.
4. THE Builder SHALL set `best_planner` to the Planner_Token of the Included_Planner with the highest score, where scores within the Tie_Tolerance of the highest score are tied and the tie goes to the earliest Planner in the Planner_Order.
5. THE Builder SHALL write exactly one Label_Table row per Sample scenario that is not a Failed_Scenario, sorted by `scenario_id`.

### Requirement 3: Failed Scenario Policy

**User Story:** As a reviewer, I want failed simulations excluded from labels but kept in the outcomes, so that labels compare planners on equal footing.

#### Acceptance Criteria

1. WHEN a Sample scenario is a Failed_Scenario, THE Builder SHALL keep the scenario's rows in the Outcome_Table and exclude the scenario from the Label_Table and the Modeling_Table.
2. WHEN the Builder excludes a Failed_Scenario, THE Builder SHALL record the `scenario_id` and one reason per failing Planner (simulation failed with its `error_message`, null score, or no row) in the Report.

### Requirement 4: Modeling Table and Folds

**User Story:** As a modeler, I want labels joined with pre-simulation features and leak-free folds, so that cross-validation is honest.

#### Acceptance Criteria

1. WHEN the Builder runs, THE Builder SHALL write a Modeling_Table containing every Label_Table column, every Feature_Columns column, and `fold`, with one row per Label_Table row.
2. IF any Label_Table `scenario_id` is absent from the Feature_Table, THEN THE Builder SHALL exit with status 1, write no Modeling_Table, and print the count and first absent `scenario_id` values.
3. IF the Feature_Table `log_name`, `scenario_token` or `map_name` value differs from the Label_Table value for the same `scenario_id`, THEN THE Builder SHALL exit with status 1 and print the count of mismatched rows.
4. THE Builder SHALL assign `fold` with `assign_log_folds` from `tools/validate_scenario_subset.py`, using the Schema `split` folds (5), seed (0) and group column `log_name`.
5. THE Builder SHALL store a `planner_labels_provenance` JSON entry in the Label_Table and Modeling_Table parquet metadata listing `label_columns`, `feature_columns`, `planners`, the Run_Roots, the SHA-256 of the Sample, the Feature_Table and the Schema, the build timestamp, and the tool version.

### Requirement 5: Integrity Check

**User Story:** As a reviewer, I want the Checker to verify the labels and modeling table, so that a corrupted build is caught before modeling.

#### Acceptance Criteria

1. WHEN the Checker runs, THE Checker SHALL fail if `scenario_id` values in the Label_Table or the Modeling_Table are not unique.
2. WHEN the Checker runs, THE Checker SHALL fail if the Outcome_Table row count differs from the Sample row count times the Included_Planners count, if any Sample scenario lacks an Outcome_Table row for any Included_Planner, or if the Label_Table and Modeling_Table row counts differ from the Sample row count minus the Failed_Scenario count.
3. WHEN the Checker runs, THE Checker SHALL fail if any `delta_<token>_vs_idm` value differs from `score_<token>` minus `score_idm` under exact float equality, or if any `score_<token>` value lies outside [0, 1].
4. WHEN the Checker runs, THE Checker SHALL fail with a coverage error if any Modeling_Table `scenario_id` is absent from the Feature_Table.
5. WHEN the Checker runs, THE Checker SHALL fail if the provenance `feature_columns` differ from the Feature_Columns, if the provenance `label_columns` differ from the Label_Columns, if the two lists share a column, or if any Feature_Columns entry starts with `score_`, `delta_` or `succeeded_`, equals `best_planner`, or contains a Schema `forbidden_feature_patterns` entry.
6. WHEN the Checker runs, THE Checker SHALL fail if any `log_name` appears in more than one `fold` of the Modeling_Table.
7. THE Checker SHALL print one `PASS` or `FAIL` line per check and exit with status 0 when all checks pass, 1 when any check fails, and 2 on a missing input file or column.

### Requirement 6: Outputs, Paths and Report

**User Story:** As a project member, I want reproducible outputs and a committed report, so that the labels are documented without committing data.

#### Acceptance Criteria

1. THE Builder SHALL accept the output root, experiment name (default `pittsburgh_65_five`), Sample, Feature_Table, Out_Dir, report directory and Schema paths as command-line flags.
2. THE Builder SHALL write each Parquet output through a temporary file followed by `os.replace`.
3. IF an output file already exists and `--overwrite` is absent, THEN THE Builder SHALL exit with status 2 and print the existing path.
4. WHEN the Builder completes, THE Builder SHALL write the Report with the row counts, Included_Planners, per-Planner mean score and official score, per-delta mean, median, standard deviation and share above zero, `best_planner` counts, Failed_Scenarios with reasons, fold sizes, and Feature_Table coverage.

### Requirement 7: Compatibility and Tests

**User Story:** As a maintainer, I want the change isolated and tested, so that existing tools keep working.

#### Acceptance Criteria

1. THE Builder SHALL leave `build_table_for_runs`, `tools/build_paired_dataset.py`, `tools/extract_scenario_features.py` and `src/causal_planner/data/features.py` unchanged.
2. WHEN `tests/test_build_planner_labels.py` runs, THE test module SHALL use synthetic data in temporary directories and read or write no file on the E: drive or under `docs/data`.
3. WHEN the full test suite runs with `.venv\Scripts\python.exe -m pytest -q -p no:subtests`, THE test suite SHALL pass.
