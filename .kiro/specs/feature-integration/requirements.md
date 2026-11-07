# Requirements Document

## Introduction

Integrate the per-scenario ego/agent, weather and map feature tables into one model-input table, `scenario_features.parquet` (benchmark candidates) and `scenario_features_mini_68.parquet` (mini sample), joined on `scenario_id`. A new tool `tools/build_scenario_features.py` builds and checks the tables against the schema contract `configs/features/first_experiment_schema.yaml`, and writes a data report. Linear 298-40.

## Glossary

- **Builder**: the `build` subcommand of `tools/build_scenario_features.py`.
- **Checker**: the `check` subcommand of `tools/build_scenario_features.py`.
- **Schema**: `configs/features/first_experiment_schema.yaml`.
- **Source_Tables**: `scenario_ego_agent*.parquet`, `scenario_weather*.parquet` and `scenario_map*.parquet` (full and `_mini_68` variants).
- **Map_Name_Source**: `scenario_eligibility.parquet` rows with `benchmark_candidate` true (full variant) or `configs/scenarios/idm_mini_sample.csv` (mini variant).
- **Feature_Table**: an output parquet file `scenario_features.parquet` or `scenario_features_mini_68.parquet`.
- **Integrated_Feature**: a Schema field with `model_feature: true` whose column is present in the Source_Tables or the Map_Name_Source.
- **Key_Columns**: `scenario_id`, `log_name`, `scenario_token`.
- **Missing_Indicator**: a boolean column `<column>_missing` that is true exactly when `<column>` is null.
- **Validator**: `tools/validate_scenario_subset.py`.

## Requirements

### Requirement 1: Integrated Feature Table

**User Story:** As a modeler, I want one row per scenario with every available pre-simulation model feature, so that I can train models from a single table.

#### Acceptance Criteria

1. WHEN the Builder runs, THE Builder SHALL write a Feature_Table containing the Key_Columns, `map_name`, every Integrated_Feature, and a Missing_Indicator for every Integrated_Feature whose Schema `missing_policy` is `null_with_indicator`.
2. THE Builder SHALL write exactly one Feature_Table row per `scenario_id` in the Map_Name_Source.
3. THE Builder SHALL copy each Integrated_Feature value unchanged from the Source_Table row with the same `scenario_id`.
4. WHEN a Source_Table lacks a Missing_Indicator for an Integrated_Feature with `missing_policy` `null_with_indicator`, THE Builder SHALL derive the Missing_Indicator from the nullness of the feature column.

### Requirement 2: Complete Join Coverage

**User Story:** As a modeler, I want complete coverage, so that no scenario silently drops out of the join.

#### Acceptance Criteria

1. IF any `scenario_id` in the Map_Name_Source is absent from any Source_Table, THEN THE Builder SHALL exit with status 1, write no Feature_Table, and print the per-table count of absent `scenario_id` values.
2. IF any Source_Table or the Map_Name_Source contains a duplicate `scenario_id`, THEN THE Builder SHALL exit with status 1 and print the duplicate count.
3. IF an input file is missing or lacks a required column, THEN THE Builder SHALL exit with status 2 and print the offending path or column.

### Requirement 3: Leakage Guard

**User Story:** As a reviewer, I want leakage guarded by the Schema, so that outcomes and location columns never become model inputs.

#### Acceptance Criteria

1. THE Builder SHALL exclude every Source_Table column that is neither a Key_Column, `map_name`, an Integrated_Feature, nor a Missing_Indicator of an Integrated_Feature.
2. WHEN the Checker runs, THE Checker SHALL fail if the Validator `check_leakage` or `check_feature_table_columns` functions return any error for the Schema and the Feature_Table columns.
3. WHEN the Checker runs, THE Checker SHALL fail if any non-key Feature_Table column contains a Schema `forbidden_feature_patterns` entry or equals `ego_x`, `ego_y` or `ego_heading`.

### Requirement 4: Table Integrity Check

**User Story:** As a reviewer, I want the Checker to verify table integrity, so that a corrupted build is caught before modeling.

#### Acceptance Criteria

1. WHEN the Checker runs, THE Checker SHALL fail if `scenario_id` values in the Feature_Table are not unique or the row count differs from the Map_Name_Source row count.
2. WHEN the Checker runs, THE Checker SHALL fail if any Missing_Indicator disagrees with the nullness of its feature column.
3. WHEN the Checker runs, THE Checker SHALL fail if any Integrated_Feature with `missing_policy` `exclude_pair` or `fail_validation` contains a null value.
4. WHEN the Checker runs, THE Checker SHALL fail if any Schema field with `artifacts` containing `features` has its column absent from the Feature_Table.
5. THE Checker SHALL print one `PASS` or `FAIL` line per check and exit with status 0 when all checks pass, 1 when any check fails, and 2 on input errors.

### Requirement 5: Provenance and Report

**User Story:** As a project member, I want provenance and a report, so that the table is reproducible and documented.

#### Acceptance Criteria

1. THE Builder SHALL store input paths, input row counts, input file sizes, the Schema SHA-256, and the build timestamp in the Feature_Table parquet schema metadata.
2. THE Builder SHALL write each Feature_Table through a temporary file followed by an atomic replace.
3. WHEN the Builder completes, THE Builder SHALL write `docs/data/scenario_features.md` and `docs/data/scenario_features.json` listing the column names, row counts, per-column null counts, per-city row counts, and Schema model features that are not integrated.
4. THE Builder SHALL accept every input path, output path and report path as a command-line flag with defaults under `E:\Project\nuplan\processed\`.

### Requirement 6: Schema and Validator Update

**User Story:** As a maintainer, I want the Schema and Validator updated, so that the contract reflects the new artifact.

#### Acceptance Criteria

1. THE Schema SHALL mark each Integrated_Feature with `status: available`, its `column`, and `features` in `artifacts`.
2. WHEN the Validator runs with `--features` pointing to a parquet file, THE Validator SHALL read the column names from the parquet schema and apply `check_feature_table_columns`.
3. WHEN the full test suite runs, THE test suite SHALL pass, including `tests/test_repository_layout.py` with the `tools/extract_scenario_features.py` and `causal_planner/data/features.py` placeholders unchanged.
