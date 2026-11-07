# Requirements Document

## Introduction

Fit classical, interpretable baseline models that predict each scenario's planner effect (score change against IDM) from pre-simulation features, and evaluate them with the project's metrics. Every planner was simulated on every scenario, so the true individual effect `delta_<token>_vs_idm` is observed and each effect gets its own direct regression model. A new tool `tools/train_baseline.py` trains the `constant`, `ridge` and `hgb` model families with out-of-fold cross-fitting over the Modeling_Table `fold` column, writes predictions, metrics and a configuration record, verifies them, and writes a committed report. Reusable metric functions go into a new package `causal_planner.evaluation` so the later Transformer (Linear 298-27) uses the same definitions. The tool runs on the Pittsburgh 65-scenario table now and on the Pittsburgh top-25 table (about 298 rows) later through a path flag. Linear 298-26.

## Glossary

- **Trainer**: the `train` subcommand of `tools/train_baseline.py`.
- **Checker**: the `check` subcommand of `tools/train_baseline.py`.
- **Metrics_Module**: `src/causal_planner/evaluation/metrics.py`, pure numpy/pandas metric functions without scikit-learn.
- **Modeling_Table**: a `planner_modeling_table.parquet` written by `tools/build_planner_labels.py`, default `E:\Project\nuplan\processed\labels\pittsburgh_65_five\planner_modeling_table.parquet`.
- **Provenance**: the `planner_labels_provenance` metadata entry of the Modeling_Table, read with `tools.build_planner_labels.read_provenance`; it lists `experiment`, `planners`, `label_columns` and `feature_columns`.
- **Experiment_Name**: the Provenance `experiment` value, or the Modeling_Table file stem when the Provenance has no `experiment`.
- **Feature_Columns**: exactly the Provenance `feature_columns` list (38 columns for the current tables).
- **Categorical_Columns**: `map_name`, `lane_type`, `traffic_light_state` and `weather_code`, plus any other Feature_Columns column with a non-numeric, non-boolean dtype.
- **Planner_Token**: the snake_case form of a Provenance planner name (`pdm-closed` becomes `pdm_closed`), as computed by `tools.build_planner_labels.token`.
- **Control**: the planner `idm`.
- **Target**: the column `delta_<token>_vs_idm` for each Provenance planner other than the Control (four Targets for five planners).
- **Model_Family**: one of `constant`, `ridge`, `hgb`.
- **Fold**: the Modeling_Table `fold` value (0-4, log-grouped).
- **OOF_Prediction**: a prediction for a Modeling_Table row made by a model fitted only on rows of the other Folds.
- **Schema**: `configs/features/first_experiment_schema.yaml`.
- **Leakage_Column**: a Feature_Columns entry that starts with `score_`, `delta_` or `succeeded_`, equals `best_planner`, or contains a Schema `forbidden_feature_patterns` entry (case-insensitive).
- **Seed**: the `--seed` value, default 0.
- **Top_K**: `ceil(f * n)` rows for f in {0.10, 0.20}, where n is the number of evaluated rows.
- **Policy**: the per-scenario planner choice that picks the Target planner with the highest predicted delta, or the Control when no predicted delta is above 0.
- **Oracle**: the per-scenario choice of the planner with the highest observed score.
- **Best_Single_Planner**: for each held-out Fold, the planner (Control included) with the highest mean observed score on the training Folds.
- **Out_Dir**: the data output directory, default `E:\Project\nuplan\processed\models\baseline\<Experiment_Name>`.
- **Report**: `baseline_<Experiment_Name>.md` and `baseline_<Experiment_Name>.json` in the report directory, default `docs/models`.

## Requirements

### Requirement 1: Input Table and Targets

**User Story:** As a modeler, I want the baseline to read targets and features from the modeling table's provenance, so that the same tool runs on the 65-row and top-25 tables without code changes.

#### Acceptance Criteria

1. THE Trainer SHALL accept the Modeling_Table path with `--table`.
2. WHEN the Trainer runs, THE Trainer SHALL read the Provenance and define one Target per Provenance planner other than the Control, in Provenance order.
3. IF the Modeling_Table file is missing, the Provenance lacks `planners` or `feature_columns`, or the Modeling_Table lacks `scenario_id`, `scenario_type`, `fold`, a Target column, a `score_<token>` column or a Feature_Columns column, THEN THE Trainer SHALL exit with status 2 and print the missing item.
4. IF the Modeling_Table `fold` column has fewer than 2 distinct values, THEN THE Trainer SHALL exit with status 2 and print the distinct Fold count.

### Requirement 2: Feature Handling and Leakage Guard

**User Story:** As a reviewer, I want features restricted to pre-simulation columns and preprocessed inside each training fold, so that the evaluation is free of leakage.

#### Acceptance Criteria

1. THE Trainer SHALL use exactly the Feature_Columns as model inputs.
2. IF any Feature_Columns entry is a Leakage_Column, THEN THE Trainer SHALL exit with status 2 before fitting any model and print each Leakage_Column with the matched rule.
3. THE Trainer SHALL encode boolean Feature_Columns as 0/1 and treat the Categorical_Columns as categorical inputs.
4. WHERE the Model_Family is `ridge`, THE Trainer SHALL fill null numeric feature values with the training-Fold median, standardize numeric features with training-Fold statistics, and one-hot encode the Categorical_Columns with categories learned from the training Folds.
5. WHERE the Model_Family is `hgb`, THE Trainer SHALL pass null numeric feature values to the model unfilled and encode the Categorical_Columns as native categorical inputs with categories learned from the training Folds.
6. THE Trainer SHALL fit every learned preprocessing step on the training Folds of each split only.

### Requirement 3: Model Families

**User Story:** As a modeler, I want three interpretable model families with fixed, recorded settings, so that later models have an honest comparison point.

#### Acceptance Criteria

1. THE Trainer SHALL accept `--models` as a comma-separated list of Model_Families with a default of `constant,ridge,hgb`.
2. IF `--models` names an unknown Model_Family or repeats a Model_Family, THEN THE Trainer SHALL exit with status 2 and print the offending value.
3. WHERE the Model_Family is `constant`, THE Trainer SHALL predict the training-Fold mean of the Target for every held-out row.
4. WHERE the Model_Family is `ridge`, THE Trainer SHALL fit a scikit-learn `Ridge` regressor with `alpha` 1.0.
5. WHERE the Model_Family is `hgb`, THE Trainer SHALL fit a scikit-learn `HistGradientBoostingRegressor` with `max_depth` 3, `learning_rate` 0.05, `max_iter` 200, `min_samples_leaf` 5, `l2_regularization` 1.0, early stopping disabled, and `random_state` equal to the Seed.
6. THE Trainer SHALL fit one separate model per Model_Family, Target and training split.
7. THE Trainer SHALL use the fixed hyperparameters of 3.4 and 3.5 without hyperparameter search and SHALL record the hyperparameters in `config.json`.

### Requirement 4: Out-of-Fold Protocol

**User Story:** As a reviewer, I want every scenario predicted by a model that never saw it, so that the metrics estimate out-of-sample performance.

#### Acceptance Criteria

1. WHEN the Trainer runs, THE Trainer SHALL, for each Fold value, fit on the rows of all other Folds and predict the rows of that Fold.
2. THE Trainer SHALL produce exactly one OOF_Prediction per Modeling_Table row, Target and selected Model_Family.
3. WHEN the targets of the rows in one Fold change, THE Trainer SHALL produce unchanged OOF_Predictions for the rows of that Fold.
4. WHEN the Trainer runs twice with the same Modeling_Table, Seed and `--models`, THE Trainer SHALL produce identical OOF_Predictions and metrics.

### Requirement 5: Per-Target Metrics

**User Story:** As a modeler, I want standard effect-prediction metrics in a shared module, so that the baseline and the Transformer are scored the same way.

#### Acceptance Criteria

1. THE Metrics_Module SHALL provide `pehe` as the root mean squared difference between predicted and observed deltas.
2. THE Metrics_Module SHALL provide `top_k_recall` as the fraction of the Top_K rows by observed delta that are also among the Top_K rows by predicted delta, with ties broken by row position.
3. THE Metrics_Module SHALL provide `qini_coefficient` as the trapezoid area between the cumulative observed-delta curve ordered by predicted delta (descending) and the random line, divided by the same area for the curve ordered by observed delta.
4. IF all observed deltas passed to `qini_coefficient` are equal, THEN THE Metrics_Module SHALL return NaN.
5. THE Metrics_Module SHALL provide `mae` and `spearman`, where `spearman` is the Pearson correlation of pandas average ranks and is NaN when either input is constant.
6. THE Metrics_Module SHALL import only the Python standard library, numpy and pandas.
7. WHEN the Trainer finishes cross-fitting, THE Trainer SHALL report for each Model_Family and Target the `pehe`, `mae`, `spearman`, `top_k_recall` at 10% and 20%, and `qini_coefficient` computed on the pooled OOF_Predictions.

### Requirement 6: Policy Metrics

**User Story:** As a planner-selection researcher, I want the value of acting on the predictions, so that I can see whether the baseline picks better planners than fixed choices.

#### Acceptance Criteria

1. THE Metrics_Module SHALL provide `choose_planner`, which returns the Policy choice per row, with ties among the highest predicted deltas going to the earliest Target in Provenance order.
2. THE Metrics_Module SHALL provide `policy_value` as the mean observed score of the chosen planner per row.
3. WHEN the Trainer finishes cross-fitting, THE Trainer SHALL report the Policy value of each Model_Family computed from its OOF_Predictions for all Targets.
4. WHEN the Trainer finishes cross-fitting, THE Trainer SHALL report the value of the Best_Single_Planner evaluated on each held-out Fold, of always choosing the Control, and of the Oracle.
5. THE Trainer SHALL report the regret of every reported policy as the Oracle value minus the policy value.

### Requirement 7: Uncertainty and Diagnostics

**User Story:** As a reviewer, I want confidence intervals, subgroup errors, failure cases and feature importances, so that I can judge results from 65 rows.

#### Acceptance Criteria

1. THE Metrics_Module SHALL provide `bootstrap_ci`, which resamples rows with replacement a given number of times with a given seed and returns the point estimate and the 2.5th and 97.5th percentiles of the resampled statistic.
2. WHEN the Trainer finishes cross-fitting, THE Trainer SHALL report 95% bootstrap intervals from 1,000 resamples seeded with the Seed for each `pehe` value and each Policy value.
3. THE Trainer SHALL select as best model the Model_Family with the lowest mean `pehe` across Targets, with ties going to the earlier of `constant`, `ridge`, `hgb`.
4. WHEN the Trainer finishes cross-fitting, THE Trainer SHALL report the best model's `pehe` and row count per Target and `scenario_type`.
5. WHEN the Trainer finishes cross-fitting, THE Trainer SHALL report the best model's 5 OOF_Predictions with the largest absolute error, each with `scenario_id`, `scenario_type`, Target, observed delta and predicted delta.
6. WHERE the Model_Family is `ridge` or `hgb`, THE Trainer SHALL compute scikit-learn permutation importance on each held-out Fold with `n_repeats` 10 and `random_state` equal to the Seed, average the importances across Folds, and report the 10 Feature_Columns with the highest mean importance per Target.

### Requirement 8: Outputs and Reproducibility

**User Story:** As a project member, I want reproducible outputs and a committed report, so that results are documented without committing data.

#### Acceptance Criteria

1. THE Trainer SHALL accept `--table`, `--out-dir`, `--report-dir`, `--schema`, `--seed`, `--models` and `--overwrite`.
2. WHEN the Trainer completes, THE Trainer SHALL write `oof_predictions.parquet` with columns `scenario_id`, `fold`, `model`, `target`, `y_true` and `y_pred` to the Out_Dir.
3. WHEN the Trainer completes, THE Trainer SHALL write `metrics.json` with every metric of Requirements 5, 6 and 7 to the Out_Dir.
4. WHEN the Trainer completes, THE Trainer SHALL write `config.json` with the Python, numpy, pandas and scikit-learn versions, the Seed, the hyperparameters, the Modeling_Table SHA-256, the Feature_Columns, the Targets, the Model_Families and the command line to the Out_Dir.
5. WHEN the Trainer completes, THE Trainer SHALL write the Report with the metrics, the configuration summary and the Experiment_Name.
6. THE Trainer SHALL write every output file through a temporary file followed by `os.replace`.
7. IF an output file already exists and `--overwrite` is absent, THEN THE Trainer SHALL exit with status 2, print the existing path, and write no output.
8. IF scikit-learn is not importable, THEN THE Trainer SHALL exit with status 2 and print the install command `pip install -e .[modeling]`.

### Requirement 9: Output Check

**User Story:** As a reviewer, I want the Checker to verify the saved outputs, so that a corrupted or inconsistent run is caught before results are quoted.

#### Acceptance Criteria

1. WHEN the Checker runs, THE Checker SHALL fail if any Modeling_Table `scenario_id` lacks an OOF_Prediction for any Target and Model_Family listed in `config.json`.
2. WHEN the Checker runs, THE Checker SHALL fail if any (`scenario_id`, `target`, `model`) combination has more than one OOF_Prediction.
3. WHEN the Checker runs, THE Checker SHALL fail if any saved `fold` or `y_true` differs from the Modeling_Table `fold` or Target value of the same `scenario_id`.
4. WHEN the Checker runs, THE Checker SHALL fail if any `pehe`, `mae`, `spearman`, `top_k_recall`, `qini_coefficient`, policy value or regret recomputed from `oof_predictions.parquet` and the Modeling_Table differs from `metrics.json` by more than 1e-9.
5. WHEN the Checker runs, THE Checker SHALL fail if the `config.json` feature list differs from the Feature_Columns, contains a Leakage_Column, or shares a column with the Provenance `label_columns`, or if the `config.json` table SHA-256 differs from the Modeling_Table SHA-256.
6. THE Checker SHALL print one `PASS` or `FAIL` line per check and exit with status 0 when all checks pass, 1 when any check fails, and 2 on a missing input file or column.

### Requirement 10: Compatibility, Packaging and Tests

**User Story:** As a maintainer, I want the change isolated, declared and tested, so that existing tools keep working.

#### Acceptance Criteria

1. THE Trainer SHALL leave `tools/extract_scenario_features.py`, `tools/build_model_dataset.py`, `src/causal_planner/data/features.py`, `src/causal_planner/__init__.py`, the behavior of `tools/build_planner_labels.py`, and `build_table_for_runs` unchanged.
2. THE project SHALL declare `[project.optional-dependencies] modeling = ["scikit-learn>=1.5"]` in `pyproject.toml` and keep `dependencies = []`.
3. WHERE scikit-learn is not importable, THE test module `tests/test_train_baseline.py` SHALL skip all of its tests, and THE test module `tests/test_evaluation_metrics.py` SHALL run without scikit-learn.
4. WHEN the test modules run, THE test modules SHALL use synthetic data in temporary directories and read or write no file on the E: drive or under `docs/models`.
5. WHEN the full test suite runs with `.venv\Scripts\python.exe -m pytest -q -p no:subtests`, THE test suite SHALL pass.
