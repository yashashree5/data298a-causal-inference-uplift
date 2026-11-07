# Implementation Plan: Classical Baseline

## Overview

Add the metric package `causal_planner.evaluation` and `tools/train_baseline.py` (Python) with `train` and `check` subcommands, declare scikit-learn as the `modeling` extra, test everything on synthetic data, then train and check on the Pittsburgh 65 table now and on the Pittsburgh top-25 table once its labels exist. Do not modify `tools/extract_scenario_features.py`, `tools/build_model_dataset.py`, `src/causal_planner/data/features.py`, `src/causal_planner/__init__.py`, the behavior of `tools/build_planner_labels.py`, or `build_table_for_runs`.

## Tasks

- [ ] 1. Implement the shared metrics module
  - [~] 1.1 Create `src/causal_planner/evaluation/__init__.py` (docstring only) and `src/causal_planner/evaluation/metrics.py`
    - numpy/pandas only, no scikit-learn or scipy: `pehe`, `mae`, `spearman` (pandas average ranks, NaN on constant input), `top_k_count`, `top_k_recall` (stable argsort, ties to the earlier row), `gain_curve`, `qini_coefficient` (trapezoid formula from the design, NaN on constant targets), `choose_planner`, `realized_scores`, `policy_value`, `oracle_value`, `bootstrap_ci` (joint row resampling, `default_rng(seed)`, percentile interval)
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 6.1, 6.2, 7.1_
  - [~] 1.2 Create `tests/test_evaluation_metrics.py` (unittest, runs without scikit-learn)
    - Seeded 100-iteration loops for Properties 1-6 (perfect predictions; metric bounds; order-only dependence under positive affine and cube transforms; random-ranking qini near 0; policy choice and value bounded by the oracle; bootstrap interval contains the point estimate and is reproducible), tagged `Feature: classical-baseline, Property N`
    - Examples: hand-computed values on a 4-row vector, NaN cases, `top_k_count(65, 0.1) == 7` and `top_k_count(65, 0.2) == 13`, subprocess import smoke test (no `sklearn`, no `tools` in `sys.modules`)
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 6.1, 6.2, 6.5, 7.1, 10.3, 10.4_

- [ ] 2. Implement the baseline tool core
  - [~] 2.1 Create `tools/train_baseline.py` with constants, `InputError`, `parse_models`, `load_table`, `experiment_name`, `resolve_targets`, `leakage_columns`, `categorical_columns`, `to_model_frame`, `make_model`, `fold_splits`, `cross_fit`
    - Module docstring with usage, `from __future__ import annotations`, `tools._bootstrap` shim; import `read_provenance`, `token`, `LABEL_PREFIXES`, `BEST_PLANNER` from `tools.build_planner_labels` and `load_schema` from `tools.validate_scenario_subset`
    - Fixed `HYPERPARAMETERS` (ridge `alpha` 1.0; hgb `max_depth` 3, `learning_rate` 0.05, `max_iter` 200, `min_samples_leaf` 5, `l2_regularization` 1.0, early stopping off, `random_state` = seed); all learned preprocessing inside the sklearn estimator; scikit-learn imported lazily
    - Permutation importance per held-out fold (`n_repeats` 10, `random_state` = seed) averaged across folds for ridge and hgb
    - _Requirements: 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 4.1, 4.2, 4.3, 7.6_
  - [~] 2.2 Add `train`, `point_metrics`, `best_single_planner` and `evaluate` to `tools/train_baseline.py`
    - Per-target pooled OOF metrics (`pehe`, `mae`, `spearman`, top-10%/20% recall, qini) from `causal_planner.evaluation.metrics`; Policy value per model, Best_Single_Planner per held-out fold, always-IDM, Oracle, regret for each
    - 1,000-resample bootstrap CIs (seed) for every PEHE and policy value; best model by mean PEHE; scenario_type PEHE breakdown and the 5 largest absolute errors for the best model; top 10 features per Target for ridge and hgb
    - _Requirements: 4.4, 5.7, 6.3, 6.4, 6.5, 7.2, 7.3, 7.4, 7.5, 7.6_

- [ ] 3. Add CLI, I/O, check and packaging
  - [~] 3.1 Implement `cmd_train`, `cmd_check`, `check_outputs`, `build_parser` and `main` in `tools/train_baseline.py`
    - Flags `--table`, `--out-dir` (default `E:\Project\nuplan\processed\models\baseline\<Experiment_Name>`), `--report-dir` (default `docs/models`), `--schema`, `--seed`, `--models`, `--overwrite`; check every output path before fitting; exit 2 with `pip install -e .[modeling]` when scikit-learn is missing
    - Atomic writers for `oof_predictions.parquet`, `metrics.json` (NaN as `null`) and `config.json` (versions, seed, hyperparameters, table and schema SHA-256 via `sha256_file`, feature list, targets, models, command line); reuse `write_text_atomic` from `tools.build_planner_labels`
    - `build_report` + `markdown_report` writing `baseline_<Experiment_Name>.md` and `.json`, including the fixed-hyperparameter rationale and the fold-protocol note
    - Checks `oof_coverage`, `unique_predictions`, `fold_and_truth`, `metrics_recompute` (tolerance 1e-9, `null` equals NaN), `feature_separation`, `table_hash`; `PASS`/`FAIL` lines; exit 0/1/2
    - _Requirements: 1.1, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 10.1_
  - [ ] 3.2 Add `[project.optional-dependencies] modeling = ["scikit-learn>=1.5"]` to `pyproject.toml`, keeping `dependencies = []`
    - _Requirements: 10.2_

- [ ] 4. Write tool tests and run the suite
  - [ ] 4.1 Create `tests/test_train_baseline.py` (unittest, `skipUnless` scikit-learn importable, synthetic modeling-table parquet with provenance in temp dirs, no E: or `docs/models` access)
    - Seeded 100-iteration loops for Properties 7-9 (out-of-fold isolation, feature selection and leakage guard, check recomputes the saved metrics), tagged `Feature: classical-baseline, Property N`
    - Examples: planted signal recovered by hgb better than constant; `to_model_frame` encodings; ridge medians from training rows; hgb with NaN; hyperparameters in estimators and `config.json`; determinism; input and `--models` errors exit 2; missing scikit-learn exits 2; `--overwrite` guard; no leftover `.tmp`; CLI `train` + `check` exit codes and `PASS`/`FAIL` lines; Report files
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 3.1, 3.2, 3.3, 3.4, 3.5, 3.7, 4.1, 4.2, 4.3, 4.4, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 10.3, 10.4_
  - [ ]* 4.2 Add a test that runs `train --models constant,hgb` then `check` on a synthetic table with three planners (two Targets) and asserts two Targets, two models and no ridge entries in the outputs
    - _Requirements: 1.2, 3.1, 9.1_
  - [ ] 4.3 Checkpoint: run `.venv\Scripts\python.exe -m pytest -q -p no:subtests` (baseline 8,794 passed, 6 skipped, plus the new tests); ensure all tests pass, ask the user if questions arise
    - _Requirements: 10.1, 10.5_

- [ ] 5. Train and check on real data
  - [ ] 5.1 Run `.venv\Scripts\python.exe -m tools.train_baseline train` on `E:\Project\nuplan\processed\labels\pittsburgh_65_five\planner_modeling_table.parquet`, then `check`; report headline metrics (PEHE with CI, Spearman, top-k recall and qini per model and Target; policy values and regret for each model, Best_Single_Planner, always-IDM and Oracle), the best model, top features, and confirm `docs/models/baseline_pittsburgh_65_five.md` / `.json` were written
    - _Requirements: 5.7, 6.3, 6.4, 6.5, 7.2, 7.3, 7.4, 7.5, 7.6, 8.2, 8.3, 8.4, 8.5, 9.6_
  - [ ] 5.2 After all five `C:\nuplan\logs\pgh25_<planner>.log` files end with a JSON summary, run `python -m tools.build_planner_labels build --experiment pittsburgh_top25_five --sample configs/scenarios/pittsburgh_top25_sample.csv --out-dir E:\Project\nuplan\processed\labels\pittsburgh_top25_five --report-dir docs/data/pittsburgh_top25` and its `check`; then run `train` and `check` with `--table E:\Project\nuplan\processed\labels\pittsburgh_top25_five\planner_modeling_table.parquet` and report the same figures as 5.1 alongside the 65-row results
    - _Requirements: 1.1, 1.2, 8.1, 8.5, 9.6_

## Notes

- Task marked `*` is optional.
- E: is a USB drive; keep every path a flag. Data outputs stay on E: and are never committed; only the Report under `docs/models` is committed.
- Hyperparameters are fixed on purpose (no tuning on 13-row folds); the `fold` column is the evaluation protocol for both tables, and the top-25 manifest train/val/test split is out of scope.
- No Git, Linear or Drive steps.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "3.2"] },
    { "id": 1, "tasks": ["1.2", "2.1"] },
    { "id": 2, "tasks": ["2.2"] },
    { "id": 3, "tasks": ["3.1"] },
    { "id": 4, "tasks": ["4.1"] },
    { "id": 5, "tasks": ["4.2"] },
    { "id": 6, "tasks": ["4.3"] },
    { "id": 7, "tasks": ["5.1"] },
    { "id": 8, "tasks": ["5.2"] }
  ]
}
```
