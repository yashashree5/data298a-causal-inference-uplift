"""Train and check classical baselines that predict per-scenario planner effects.

Every planner is simulated on every scenario, so the effect of planner <t> against the IDM control,
delta_<t>_vs_idm = score_<t> - score_idm, is observed per row. Each such Target gets its own direct
regression model fitted on the pre-simulation Feature_Columns listed in the modeling table's
provenance. Three Model_Families are fitted with fixed hyperparameters:

    constant  training-fold mean of the Target (DummyRegressor)
    ridge     median imputation + standardization + one-hot categoricals, Ridge(alpha=1.0)
    hgb       native categoricals, HistGradientBoostingRegressor (shallow, slow, regularized)

Evaluation is out of fold over the modeling table's log-grouped `fold` column: for each fold, fit
on the other folds and predict the held-out rows. Every learned preprocessing step sits inside the
scikit-learn estimator, so it is fitted on the training folds of each split only.

Outputs:
    <out-dir>/oof_predictions.parquet     one OOF prediction per row x Target x model
    <out-dir>/metrics.json                per-target, policy and diagnostic metrics
    <out-dir>/config.json                 versions, seed, hyperparameters, hashes, features
    <report-dir>/baseline_<experiment>.md and .json

Usage (from the repository root):
    python -m tools.train_baseline train
    python -m tools.train_baseline train --models constant,hgb --seed 1 --overwrite
    python -m tools.train_baseline check
    python -m tools.train_baseline train --table E:/Project/nuplan/processed/labels/pittsburgh_top25_five/planner_modeling_table.parquet

Every input, output and report path is a flag (see --help). scikit-learn is needed for `train`
only (pip install -e .[modeling]). Exit status: 0 on success, 1 on a failed fit or check, 2 on a
bad --models value, missing input file or column, leakage column, missing scikit-learn, or an
existing output without --overwrite.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools._bootstrap import bootstrap  # noqa: E402

bootstrap()

from tools.build_planner_labels import (  # noqa: E402
    BEST_PLANNER,
    LABEL_PREFIXES,
    read_provenance,
    token,
    write_text_atomic,
)
from tools import validate_scenario_subset as validator  # noqa: E402
from tools.validate_scenario_subset import load_schema, sha256_file  # noqa: E402
from causal_planner.evaluation import metrics as M  # noqa: E402

TOOL_VERSION = "classical-baseline-v1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONTROL = "idm"
MODEL_FAMILIES = ("constant", "ridge", "hgb")
IMPORTANCE_FAMILIES = ("ridge", "hgb")
CATEGORICAL = ("map_name", "lane_type", "traffic_light_state", "weather_code")
MISSING_CATEGORY = "__missing__"
REQUIRED_COLUMNS = ("scenario_id", "scenario_type", "fold")
HYPERPARAMETERS = {
    "constant": {"strategy": "mean"},
    "ridge": {"alpha": 1.0},
    "hgb": {"max_depth": 3, "learning_rate": 0.05, "max_iter": 200, "min_samples_leaf": 5,
            "l2_regularization": 1.0, "early_stopping": False},  # random_state = seed
}
PERMUTATION_REPEATS = 10
PERMUTATION_SCORING = "neg_root_mean_squared_error"
BOOTSTRAP_RESAMPLES = 1000
TOP_FEATURES = 10
FAILURE_CASES = 5
CHECK_TOLERANCE = 1e-9

PREDICTIONS_FILE = "oof_predictions.parquet"
METRICS_FILE = "metrics.json"
CONFIG_FILE = "config.json"
REPORT_STEM = "baseline"

DEFAULT_TABLE = Path("E:/Project/nuplan/processed/labels/pittsburgh_65_five/"
                     "planner_modeling_table.parquet")
DEFAULT_MODELS_ROOT = Path("E:/Project/nuplan/processed/models/baseline")
DEFAULT_REPORT_DIR = PROJECT_ROOT / "docs" / "models"
DEFAULT_SCHEMA = PROJECT_ROOT / "configs" / "features" / "first_experiment_schema.yaml"
INSTALL_HINT = "pip install -e .[modeling]"


class InputError(Exception):
    """Bad --models value, missing input file, provenance key or column, too few folds, leakage
    column, missing scikit-learn or existing output (exit status 2)."""


class FitError(Exception):
    """A model fit or prediction failed; the message names the family, Target and Fold
    (exit status 1)."""


# ------------------------------------------------------------------------------------- inputs --

def parse_models(value: str) -> list[str]:
    """Comma-separated Model_Families -> families in MODEL_FAMILIES order.

    Raises InputError naming the offending value for an unknown or repeated family.
    """
    names = [name.strip() for name in str(value).split(",")]
    seen: set[str] = set()
    for name in names:
        if name not in MODEL_FAMILIES:
            raise InputError(f"--models: unknown model {name!r}; "
                             f"choose from {', '.join(MODEL_FAMILIES)}")
        if name in seen:
            raise InputError(f"--models: model {name!r} is repeated")
        seen.add(name)
    return [family for family in MODEL_FAMILIES if family in seen]


def resolve_targets(planners: list[str]) -> dict[str, str]:
    """{planner_token: 'delta_<token>_vs_idm'} for each non-control planner, provenance order."""
    control = token(CONTROL)
    return {token(p): f"delta_{token(p)}_vs_{control}" for p in planners if p != CONTROL}


def load_table(path) -> tuple[pd.DataFrame, dict]:
    """Read the Modeling_Table parquet and its Provenance.

    Raises InputError when the file is missing, the Provenance lacks `planners` or
    `feature_columns`, the planners lack the control, the table lacks scenario_id, scenario_type,
    fold, a Target, a score_<token> or a Feature_Columns column, or fold has fewer than 2 values.
    """
    path = Path(path)
    if not path.is_file():
        raise InputError(f"modeling table not found: {path}")
    provenance = read_provenance(path)
    for key in ("planners", "feature_columns"):
        if not provenance.get(key):
            raise InputError(f"modeling table provenance lacks {key!r}: {path}")
    planners = [str(p) for p in provenance["planners"]]
    if CONTROL not in planners:
        raise InputError(f"provenance planners {planners} lack the control planner {CONTROL!r}")
    frame = pd.read_parquet(path)
    targets = resolve_targets(planners)
    required = (list(REQUIRED_COLUMNS) + list(targets.values())
                + [f"score_{token(p)}" for p in planners] + list(provenance["feature_columns"]))
    missing = [column for column in dict.fromkeys(required) if column not in frame.columns]
    if missing:
        raise InputError(f"modeling table lacks required column(s): {', '.join(missing)}")
    distinct = int(frame["fold"].nunique(dropna=True))
    if distinct < 2:
        raise InputError(f"modeling table fold column has {distinct} distinct value(s); "
                         f"at least 2 are needed for out-of-fold evaluation")
    return frame, provenance


def experiment_name(provenance: dict, table_path: Path) -> str:
    """Provenance `experiment`, else the table file stem."""
    name = (provenance or {}).get("experiment")
    return str(name) if name else Path(table_path).stem


def leakage_columns(features: list[str], schema: dict) -> dict[str, str]:
    """{column: rule} for each Leakage_Column: a label prefix (score_, delta_, succeeded_), the
    best_planner column, or a Schema forbidden_feature_patterns entry (case-insensitive)."""
    patterns = [str(p) for p in (schema or {}).get("forbidden_feature_patterns") or []]
    found: dict[str, str] = {}
    for column in features:
        name = str(column)
        prefix = next((p for p in LABEL_PREFIXES if name.startswith(p)), None)
        if prefix is not None:
            found[name] = f"label prefix {prefix!r}"
        elif name == BEST_PLANNER:
            found[name] = f"equals {BEST_PLANNER!r}"
        else:
            hit = next((p for p in patterns if p.lower() in name.lower()), None)
            if hit is not None:
                found[name] = f"forbidden pattern {hit!r}"
    return found


# ------------------------------------------------------------------------------------ features --

def _is_bool(series: pd.Series) -> bool:
    return pd.api.types.is_bool_dtype(series.dtype)


def categorical_columns(frame: pd.DataFrame, features: list[str]) -> list[str]:
    """Feature_Columns in CATEGORICAL plus any other non-numeric, non-boolean feature, in
    feature order."""
    result = []
    for column in features:
        series = frame[column]
        if column in CATEGORICAL or (not _is_bool(series)
                                     and not pd.api.types.is_numeric_dtype(series.dtype)):
            result.append(column)
    return result


def to_model_frame(frame: pd.DataFrame, features: list[str], categorical: list[str]) -> pd.DataFrame:
    """Stateless conversion to the model input frame: exactly `features`, in order.

    Booleans (bool or nullable boolean) -> float64 0.0/1.0 with NA -> NaN; categoricals -> object
    strings with nulls -> MISSING_CATEGORY; other columns -> float64 with NA -> NaN. Nothing is
    learned from the data. The input is not modified.
    """
    categorical = set(categorical)
    columns: dict[str, pd.Series] = {}
    for column in features:
        series = frame[column]
        if column in categorical:
            values = [MISSING_CATEGORY if pd.isna(v) else str(v) for v in series.tolist()]
            columns[column] = pd.Series(values, index=frame.index, dtype=object)
        elif _is_bool(series):
            columns[column] = series.astype("Float64").astype("float64")
        else:
            columns[column] = pd.to_numeric(series, errors="raise").astype("Float64").astype("float64")
    return pd.DataFrame(columns, index=frame.index)[list(features)]


# -------------------------------------------------------------------------------------- models --

def model_hyperparameters(family: str, seed: int) -> dict:
    """HYPERPARAMETERS[family], with random_state filled for hgb."""
    params = dict(HYPERPARAMETERS[family])
    if family == "hgb":
        params["random_state"] = int(seed)
    return params


def make_model(family: str, numeric: list[str], categorical: list[str], seed: int):
    """Unfitted scikit-learn estimator taking the model frame (see the design document).

    constant: DummyRegressor(strategy='mean').
    ridge:    median imputation + StandardScaler on numeric, one-hot (unknown ignored) on
              categorical, then Ridge(alpha=1.0).
    hgb:      ordinal encoding (unknown -> NaN) of categorical as native categorical features,
              numeric passed through with NaN, then HistGradientBoostingRegressor.
    Every learned step sits inside the returned estimator.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.dummy import DummyRegressor
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

    numeric, categorical = list(numeric), list(categorical)
    params = model_hyperparameters(family, seed)
    if family == "constant":
        return DummyRegressor(**params)
    if family == "ridge":
        transformers = []
        if numeric:
            transformers.append(("num", Pipeline([
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scale", StandardScaler()),
            ]), numeric))
        if categorical:
            transformers.append(("cat", OneHotEncoder(handle_unknown="ignore",
                                                      sparse_output=False), categorical))
        prep = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)
        return Pipeline([("prep", prep), ("model", Ridge(**params))])
    if family == "hgb":
        transformers = []
        if categorical:
            transformers.append(("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                                                       unknown_value=np.nan), categorical))
        if numeric:
            transformers.append(("num", "passthrough", numeric))
        prep = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)
        native = list(range(len(categorical))) if categorical else None
        model = HistGradientBoostingRegressor(categorical_features=native, **params)
        return Pipeline([("prep", prep), ("model", model)])
    raise InputError(f"unknown model {family!r}; choose from {', '.join(MODEL_FAMILIES)}")


def fold_splits(folds: pd.Series) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """(fold, train positions, test positions) per sorted fold value; train = rows of the other
    folds. Positions index the rows of `folds` in order."""
    values = pd.Series(folds).to_numpy()
    splits = []
    for fold in sorted({int(v) for v in values}):
        test = np.flatnonzero(values == fold)
        train = np.flatnonzero(values != fold)
        splits.append((fold, train, test))
    return splits


def cross_fit(model_frame: pd.DataFrame, y, folds: pd.Series, family: str, numeric: list[str],
              categorical: list[str], seed: int,
              importance: bool) -> tuple[np.ndarray, dict[str, float] | None]:
    """OOF predictions for one family and Target.

    For each fold split: a fresh make_model, fit on the training rows, predict the held-out rows.
    When `importance`, run sklearn.inspection.permutation_importance on the held-out rows
    (scoring neg_root_mean_squared_error, n_repeats PERMUTATION_REPEATS, random_state seed) and
    return {feature: importances_mean averaged across folds} in model-frame column order;
    otherwise None.
    """
    from sklearn.inspection import permutation_importance

    y = np.asarray(y, dtype="float64")
    if len(y) != len(model_frame):
        raise ValueError(f"target length {len(y)} differs from model frame length {len(model_frame)}")
    predictions = np.full(len(y), np.nan, dtype="float64")
    totals = np.zeros(model_frame.shape[1], dtype="float64") if importance else None
    splits = fold_splits(folds)
    for fold, train, test in splits:
        model = make_model(family, numeric, categorical, seed)
        x_train, x_test = model_frame.iloc[train], model_frame.iloc[test]
        try:
            model.fit(x_train, y[train])
            predictions[test] = np.asarray(model.predict(x_test), dtype="float64")
            if importance:
                result = permutation_importance(model, x_test, y[test],
                                                scoring=PERMUTATION_SCORING,
                                                n_repeats=PERMUTATION_REPEATS,
                                                random_state=seed)
                totals += np.asarray(result.importances_mean, dtype="float64")
        except Exception as exc:  # noqa: BLE001 - reported with family, Target and Fold
            raise FitError(f"fold {fold}: {type(exc).__name__}: {exc}") from exc
    if not importance:
        return predictions, None
    mean = totals / len(splits)
    return predictions, {str(c): float(v) for c, v in zip(model_frame.columns, mean)}


# ---------------------------------------------------------------------------------- evaluation --

def _recall_key(fraction: float) -> str:
    return f"top{int(round(fraction * 100))}_recall"


def _planner_tokens(targets: dict[str, str], planners=None) -> list[str]:
    """Planner tokens (control included) in Provenance order; defaults to control + Targets."""
    if planners is None:
        return [token(CONTROL)] + list(targets)
    return [token(p) for p in planners]


def _score_frame(frame: pd.DataFrame, tokens: list[str]) -> pd.DataFrame:
    """Observed scores with one column per planner token, positional index."""
    return pd.DataFrame({t: frame[f"score_{t}"].to_numpy(dtype="float64") for t in tokens})


def _aligned_predictions(frame: pd.DataFrame, predictions: pd.DataFrame, model: str,
                         target: str) -> np.ndarray:
    """y_pred for one (model, Target) in Modeling_Table row order, matched on scenario_id.

    Raises ValueError when a scenario_id lacks a prediction or has more than one.
    """
    part = predictions[(predictions["model"] == model) & (predictions["target"] == target)]
    ids = part["scenario_id"].astype(str)
    if ids.duplicated().any():
        raise ValueError(f"duplicate predictions for model {model!r}, target {target!r}")
    lookup = pd.Series(part["y_pred"].to_numpy(dtype="float64"), index=ids.to_numpy())
    aligned = lookup.reindex(frame["scenario_id"].astype(str).to_numpy())
    if aligned.isna().any():
        raise ValueError(f"missing predictions for model {model!r}, target {target!r}")
    return aligned.to_numpy(dtype="float64")


def _target_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    result = {"pehe": M.pehe(y_true, y_pred), "mae": M.mae(y_true, y_pred),
              "spearman": M.spearman(y_true, y_pred)}
    for fraction in M.TOP_K_FRACTIONS:
        result[_recall_key(fraction)] = float(M.top_k_recall(y_true, y_pred, fraction))
    result["qini"] = M.qini_coefficient(y_true, y_pred)
    return result


def best_single_planner(frame: pd.DataFrame, tokens: list[str]) -> tuple[pd.Series, dict[int, str]]:
    """Per held-out Fold, the planner token with the highest mean observed score on the other
    Folds (ties -> earlier in `tokens`, which is Provenance order). Returns the per-row choice
    (positional index) and {fold: token}."""
    scores = _score_frame(frame, list(tokens))
    choice = pd.Series([None] * len(frame), dtype=object)
    per_fold: dict[int, str] = {}
    for fold, train, test in fold_splits(frame["fold"]):
        means = scores.iloc[train].mean(axis=0).to_numpy(dtype="float64")
        winner = str(scores.columns[int(np.argmax(means))])  # first maximum
        per_fold[int(fold)] = winner
        choice.iloc[test] = winner
    return choice, per_fold


def point_metrics(frame: pd.DataFrame, predictions: pd.DataFrame, targets: dict[str, str],
                  models: list[str], planners=None) -> dict:
    """Deterministic metrics recomputable from the OOF predictions and the Modeling_Table.

    `targets` is {planner_token: Target column} in Provenance order; `planners` the Provenance
    planner list (control included), used for score columns and Best_Single_Planner ties.
    Returns {"per_target": {model: {target: {...}}}, "policy": {...}}; every policy entry has
    `value` and `regret` (Oracle value minus the policy value).
    """
    tokens = _planner_tokens(targets, planners)
    control = token(CONTROL)
    scores = _score_frame(frame, tokens)
    oracle = M.oracle_value(scores)

    per_target: dict = {}
    policy: dict = {}
    for model in models:
        per_target[model] = {}
        deltas = {}
        for tok, target in targets.items():
            y_true = frame[target].to_numpy(dtype="float64")
            y_pred = _aligned_predictions(frame, predictions, model, target)
            per_target[model][target] = _target_metrics(y_true, y_pred)
            deltas[tok] = y_pred
        choice = M.choose_planner(pd.DataFrame(deltas, columns=list(targets)), control=control)
        value = M.policy_value(scores, choice)
        counts = choice.value_counts()
        policy[model] = {"value": value, "regret": oracle - value,
                         "choices": {t: int(counts.get(t, 0)) for t in tokens}}

    single_choice, per_fold = best_single_planner(frame, tokens)
    value = M.policy_value(scores, single_choice)
    policy["best_single_planner"] = {"value": value, "regret": oracle - value,
                                     "per_fold": {str(k): v for k, v in sorted(per_fold.items())}}
    idm_choice = pd.Series([control] * len(frame), dtype=object)
    value = M.policy_value(scores, idm_choice)
    policy["always_idm"] = {"value": value, "regret": oracle - value}
    policy["oracle"] = {"value": oracle, "regret": 0.0}
    return {"per_target": per_target, "policy": policy}


def _ci(point_low_high: tuple[float, float, float]) -> list[float]:
    return [float(point_low_high[1]), float(point_low_high[2])]


def evaluate(frame: pd.DataFrame, predictions: pd.DataFrame, targets: dict[str, str],
             models: list[str], seed: int, importances: dict | None, *,
             bootstrap_resamples: int = BOOTSTRAP_RESAMPLES, planners=None) -> dict:
    """All metrics of Requirements 5-7 (see the design's metrics.json layout, minus
    `experiment`): point metrics, bootstrap CIs (`bootstrap_resamples`, Seed) for every PEHE and
    policy value, best model, best-model scenario_type PEHE and failure cases, and the top
    permutation-importance features. `importances` is {model: {target: {feature: importance}}}
    for the families that computed it (others are left out)."""
    seed = int(seed)
    tokens = _planner_tokens(targets, planners)
    control = token(CONTROL)
    n_rows = int(len(frame))
    point = point_metrics(frame, predictions, targets, models, planners=planners)
    scores = _score_frame(frame, tokens)

    def boot(statistic, *arrays):
        return M.bootstrap_ci(statistic, *arrays, n_resamples=bootstrap_resamples, seed=seed)

    aligned = {(m, t): _aligned_predictions(frame, predictions, m, t)
               for m in models for t in targets.values()}
    truth = {t: frame[t].to_numpy(dtype="float64") for t in targets.values()}

    per_target: dict = {}
    for model in models:
        per_target[model] = {}
        for target in targets.values():
            entry = dict(point["per_target"][model][target])
            ci = _ci(boot(M.pehe, truth[target], aligned[(model, target)]))
            per_target[model][target] = {"pehe": entry.pop("pehe"), "pehe_ci": ci, **entry}

    choices: dict[str, pd.Series] = {}
    for model in models:
        deltas = pd.DataFrame({tok: aligned[(model, t)] for tok, t in targets.items()},
                              columns=list(targets))
        choices[model] = M.choose_planner(deltas, control=control)
    choices["best_single_planner"] = best_single_planner(frame, tokens)[0]
    choices["always_idm"] = pd.Series([control] * n_rows, dtype=object)

    policy: dict = {}
    for name, entry in point["policy"].items():
        if name == "oracle":
            ci = _ci(boot(M.oracle_value, scores))
        else:
            ci = _ci(boot(M.policy_value, scores, choices[name]))
        rest = {k: v for k, v in entry.items() if k not in ("value",)}
        policy[name] = {"value": entry["value"], "value_ci": ci, **rest}

    mean_pehe = {m: float(np.mean([point["per_target"][m][t]["pehe"] for t in targets.values()]))
                 for m in models}
    best = min(models, key=lambda m: (mean_pehe[m], MODEL_FAMILIES.index(m)))

    scenario_types = frame["scenario_type"].astype(str).to_numpy()
    breakdown: dict = {}
    for target in targets.values():
        breakdown[target] = {}
        for stype in sorted(set(scenario_types.tolist())):
            mask = scenario_types == stype
            breakdown[target][stype] = {"n": int(mask.sum()),
                                        "pehe": M.pehe(truth[target][mask],
                                                       aligned[(best, target)][mask])}

    ids = frame["scenario_id"].astype(str).to_numpy()
    rows = []
    for t_index, target in enumerate(targets.values()):
        errors = np.abs(aligned[(best, target)] - truth[target])
        for pos in range(n_rows):
            rows.append((-float(errors[pos]), t_index, pos, target))
    rows.sort()  # largest error first; ties by Target order then table row
    failures = [{"scenario_id": ids[pos], "scenario_type": scenario_types[pos], "target": target,
                 "y_true": float(truth[target][pos]), "y_pred": float(aligned[(best, target)][pos]),
                 "abs_error": -neg}
                for neg, _t, pos, target in rows[:FAILURE_CASES]]

    top_features: dict = {}
    for model in models:
        if model not in IMPORTANCE_FAMILIES or not importances or model not in importances:
            continue
        top_features[model] = {}
        for target in targets.values():
            values = importances[model].get(target) or {}
            ranked = sorted(enumerate(values.items()), key=lambda item: (-item[1][1], item[0]))
            top_features[model][target] = [{"feature": f, "importance": float(v)}
                                           for _i, (f, v) in ranked[:TOP_FEATURES]]

    folds = frame["fold"].astype("int64").value_counts().sort_index()
    return {
        "n_rows": n_rows,
        "folds": {str(int(k)): int(v) for k, v in folds.items()},
        "targets": list(targets.values()),
        "models": list(models),
        "top_k": {str(f): M.top_k_count(n_rows, f) for f in M.TOP_K_FRACTIONS},
        "per_target": per_target,
        "policy": policy,
        "best_model": best,
        "scenario_type_pehe": breakdown,
        "failure_cases": failures,
        "permutation_importance": top_features,
    }


def train(frame: pd.DataFrame, provenance: dict, schema: dict, models: list[str], seed: int, *,
          importance: bool = True,
          bootstrap_resamples: int = BOOTSTRAP_RESAMPLES) -> tuple[pd.DataFrame, dict, dict]:
    """Guard leakage, cross-fit every family x Target, evaluate, and return
    (oof_predictions, metrics, config_core).

    Raises InputError naming each Leakage_Column and its rule before any fit, and FitError naming
    family, Target and Fold when a fit fails. `importance` (default on) toggles permutation
    importance for ridge and hgb; `bootstrap_resamples` (default 1,000) sets the CI resamples.
    Both defaults follow the spec; tests may lower them for speed.
    """
    features = [str(c) for c in provenance["feature_columns"]]
    leaks = leakage_columns(features, schema)
    if leaks:
        detail = "; ".join(f"{column} ({rule})" for column, rule in leaks.items())
        raise InputError(f"leakage column(s) in feature_columns: {detail}")
    planners = [str(p) for p in provenance["planners"]]
    targets = resolve_targets(planners)
    models = list(models)
    seed = int(seed)

    frame = frame.reset_index(drop=True)
    categorical = categorical_columns(frame, features)
    numeric = [c for c in features if c not in set(categorical)]
    model_frame = to_model_frame(frame, features, categorical)
    folds = frame["fold"].astype("int64")
    ids = frame["scenario_id"].astype(str)

    parts = []
    importances: dict = {}
    for family in models:
        wants = importance and family in IMPORTANCE_FAMILIES
        for target in targets.values():
            y = frame[target].to_numpy(dtype="float64")
            try:
                y_pred, imp = cross_fit(model_frame, y, folds, family, numeric, categorical,
                                        seed, importance=wants)
            except FitError as exc:
                raise FitError(f"model {family!r}, target {target!r}, {exc}") from exc
            if imp is not None:
                importances.setdefault(family, {})[target] = imp
            parts.append(pd.DataFrame({"scenario_id": ids.to_numpy(), "fold": folds.to_numpy(),
                                       "model": family, "target": target,
                                       "y_true": y, "y_pred": y_pred}))

    predictions = pd.concat(parts, ignore_index=True)
    predictions["_m"] = predictions["model"].map({m: i for i, m in enumerate(models)})
    predictions["_t"] = predictions["target"].map({t: i for i, t in enumerate(targets.values())})
    predictions = (predictions.sort_values(["_m", "_t", "scenario_id"], kind="stable")
                   .drop(columns=["_m", "_t"]).reset_index(drop=True))
    predictions = predictions.astype({"scenario_id": str, "fold": "int64", "model": str,
                                      "target": str, "y_true": "float64", "y_pred": "float64"})

    metrics = evaluate(frame, predictions, targets, models, seed, importances,
                       bootstrap_resamples=bootstrap_resamples, planners=planners)
    config_core = {
        "seed": seed,
        "models": models,
        "hyperparameters": {m: model_hyperparameters(m, seed) for m in models},
        "targets": list(targets.values()),
        "feature_columns": features,
        "categorical_columns": categorical,
        "permutation_repeats": PERMUTATION_REPEATS if importance else 0,
        "bootstrap_resamples": int(bootstrap_resamples),
    }
    return predictions, metrics, config_core


# ------------------------------------------------------------------------------------------ I/O --

PREDICTION_COLUMNS = ("scenario_id", "fold", "model", "target", "y_true", "y_pred")
CONFIG_KEYS = ("models", "targets", "feature_columns", "table_sha256")


def default_out_dir(provenance: dict, table_path: Path) -> Path:
    """DEFAULT_MODELS_ROOT / Experiment_Name."""
    return DEFAULT_MODELS_ROOT / experiment_name(provenance, table_path)


def output_paths(out_dir: Path, report_dir: Path, experiment: str) -> dict[str, Path]:
    """Every file `train` writes."""
    out_dir, report_dir = Path(out_dir), Path(report_dir)
    stem = f"{REPORT_STEM}_{experiment}"
    return {
        "predictions": out_dir / PREDICTIONS_FILE,
        "metrics": out_dir / METRICS_FILE,
        "config": out_dir / CONFIG_FILE,
        "report_md": report_dir / f"{stem}.md",
        "report_json": report_dir / f"{stem}.json",
    }


def json_safe(value):
    """Nested copy with NaN/inf -> None and numpy scalars/arrays -> Python types."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [json_safe(v) for v in value.tolist()]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json_atomic(path: Path, data) -> None:
    """JSON with NaN written as null (allow_nan=False after json_safe), via write_text_atomic."""
    write_text_atomic(path, json.dumps(json_safe(data), indent=2, allow_nan=False) + "\n")


def write_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
    """OOF predictions to `<path>.tmp` with a fixed Arrow schema, then os.replace; the tmp is
    removed on failure."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema([("scenario_id", pa.string()), ("fold", pa.int64()), ("model", pa.string()),
                        ("target", pa.string()), ("y_true", pa.float64()), ("y_pred", pa.float64())])
    frame = pd.DataFrame({
        "scenario_id": df["scenario_id"].astype(str).tolist(),
        "fold": df["fold"].to_numpy(dtype="int64"),
        "model": df["model"].astype(str).tolist(),
        "target": df["target"].astype(str).tolist(),
        "y_true": df["y_true"].to_numpy(dtype="float64"),
        "y_pred": df["y_pred"].to_numpy(dtype="float64"),
    })
    table = pa.Table.from_pandas(frame, schema=schema, preserve_index=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        pq.write_table(table, tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def library_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": platform.python_version(),
                                       "numpy": np.__version__, "pandas": pd.__version__}
    for name, module in (("sklearn", "sklearn"), ("pyarrow", "pyarrow")):
        try:
            versions[name] = __import__(module).__version__
        except ImportError:
            versions[name] = None
    return versions


# --------------------------------------------------------------------------------------- report --

HYPERPARAMETER_RATIONALE = (
    "Hyperparameters are fixed, not tuned. With about 13 rows per fold on the 65-row table, "
    "nested cross-validation would tune on about 10 rows per inner fold, so the chosen settings "
    "would follow fold noise, and tuning on the outer folds would leak evaluation rows into model "
    "selection. Ridge uses `alpha` 1.0; boosting is shallow, slow and regularized "
    "(`max_depth` 3, `learning_rate` 0.05, `max_iter` 200, `min_samples_leaf` 5, "
    "`l2_regularization` 1.0, no early stopping). The same settings are used on every table so "
    "runs stay comparable."
)
PROTOCOL_NOTE = (
    "Evaluation is out of fold over the modeling table's log-grouped `fold` column: for each fold, "
    "every model is fitted on the other folds and predicts the held-out rows, and all metrics are "
    "computed on the pooled out-of-fold predictions. Every learned preprocessing step (median "
    "imputation, scaling, category encoding) is fitted on the training folds only. There is no "
    "separate test split; the top-25 manifest's train/val/test split is not used here."
)


def build_report(metrics: dict, config: dict) -> dict:
    """Report JSON: tool version, time, experiment, configuration summary, then every metric."""
    summary = {
        "seed": config.get("seed"),
        "models": config.get("models"),
        "hyperparameters": config.get("hyperparameters"),
        "versions": config.get("versions"),
        "table": config.get("table"),
        "table_sha256": config.get("table_sha256"),
        "schema_sha256": config.get("schema_sha256"),
        "n_features": len(config.get("feature_columns") or []),
        "categorical_columns": config.get("categorical_columns"),
        "permutation_repeats": config.get("permutation_repeats"),
        "bootstrap_resamples": config.get("bootstrap_resamples"),
    }
    rest = {k: v for k, v in metrics.items() if k != "experiment"}
    return json_safe({"tool_version": config.get("tool_version", TOOL_VERSION),
                      "generated_utc": config.get("created_utc"),
                      "experiment": config.get("experiment", metrics.get("experiment")),
                      "config": summary, **rest})


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    value = float(value)
    return "n/a" if not math.isfinite(value) else f"{value:.{digits}f}"


def _fmt_ci(ci) -> str:
    if not ci:
        return "n/a"
    return f"[{_fmt(ci[0])}, {_fmt(ci[1])}]"


def _cell(value) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def markdown_report(report: dict) -> str:
    cfg = report.get("config") or {}
    experiment = report.get("experiment", "")
    recall_keys = [_recall_key(f) for f in M.TOP_K_FRACTIONS]
    top_k = report.get("top_k") or {}
    lines = [
        f"# Classical baseline: {experiment}",
        "",
        f"Generated by `python -m tools.train_baseline train` "
        f"({report.get('tool_version', TOOL_VERSION)}, {report.get('generated_utc', '')}) for the "
        f"`{experiment}` experiment. Predictions, metrics and configuration stay outside Git; only "
        "this report is committed. Each Target `delta_<planner>_vs_idm` is the observed score "
        "change against IDM and gets its own direct regression model.",
        "",
        "## Configuration",
        "",
        f"- Rows: {report.get('n_rows', 0):,}; folds: "
        + ", ".join(f"{k} ({v})" for k, v in (report.get("folds") or {}).items()),
        f"- Models: " + ", ".join(f"`{m}`" for m in report.get("models") or []),
        f"- Targets: " + ", ".join(f"`{t}`" for t in report.get("targets") or []),
        f"- Seed: {cfg.get('seed')}; bootstrap resamples: {cfg.get('bootstrap_resamples')}; "
        f"permutation repeats: {cfg.get('permutation_repeats')}",
        f"- Features: {cfg.get('n_features')} (categorical: "
        + ", ".join(f"`{c}`" for c in cfg.get("categorical_columns") or []) + ")",
        f"- Top-k rows: " + ", ".join(f"{float(k):.0%} = {v}" for k, v in top_k.items()),
        f"- Table SHA-256: `{cfg.get('table_sha256')}`",
        "- Versions: " + ", ".join(f"{k} {v}" for k, v in (cfg.get("versions") or {}).items()),
        "",
        "| Model | Hyperparameters |",
        "|---|---|",
    ]
    for model, params in (cfg.get("hyperparameters") or {}).items():
        text = ", ".join(f"`{k}`={v}" for k, v in params.items())
        lines.append(f"| `{model}` | {_cell(text)} |")
    lines += ["", "## Protocol", "", PROTOCOL_NOTE, "", HYPERPARAMETER_RATIONALE, "",
              "## Per-target metrics", "",
              "PEHE is the RMSE of the predicted delta, with a 95% bootstrap interval over rows. "
              "Top-k recall is the share of the top-k rows by observed delta that are also in the "
              "top-k by predicted delta. Qini is 1 for a perfect ranking and 0 in expectation for "
              "a random one."]
    per_target = report.get("per_target") or {}
    for model in report.get("models") or []:
        lines += ["", f"### `{model}`", "",
                  "| Target | PEHE | PEHE 95% CI | MAE | Spearman | Top-10% recall | "
                  "Top-20% recall | Qini |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for target, m in (per_target.get(model) or {}).items():
            lines.append(f"| `{target}` | {_fmt(m.get('pehe'))} | {_fmt_ci(m.get('pehe_ci'))} | "
                         f"{_fmt(m.get('mae'))} | {_fmt(m.get('spearman'))} | "
                         f"{_fmt(m.get(recall_keys[0]), 3)} | {_fmt(m.get(recall_keys[1]), 3)} | "
                         f"{_fmt(m.get('qini'))} |")
    lines += ["", "## Policy value", "",
              "Each model's policy picks the planner with the highest predicted delta, or IDM when "
              "no predicted delta is above 0. The best single planner is chosen per held-out fold "
              "from the training folds' mean scores. Regret is the oracle value minus the policy "
              "value.", "",
              "| Policy | Value | 95% CI | Regret | Choices |", "|---|---:|---:|---:|---|"]
    for name, p in (report.get("policy") or {}).items():
        if "choices" in p:
            extra = ", ".join(f"{k} {v}" for k, v in p["choices"].items())
        elif "per_fold" in p:
            extra = "per fold: " + ", ".join(f"{k} {v}" for k, v in p["per_fold"].items())
        else:
            extra = ""
        lines.append(f"| `{name}` | {_fmt(p.get('value'))} | {_fmt_ci(p.get('value_ci'))} | "
                     f"{_fmt(p.get('regret'))} | {_cell(extra)} |")
    best = report.get("best_model")
    lines += ["", "## Best model", "",
              f"`{best}` has the lowest mean PEHE across Targets (ties go to the earlier of "
              "constant, ridge, hgb).", "",
              "### PEHE by scenario type", "", "| Target | Scenario type | Rows | PEHE |",
              "|---|---|---:|---:|"]
    for target, groups in (report.get("scenario_type_pehe") or {}).items():
        for stype, g in groups.items():
            lines.append(f"| `{target}` | {_cell(stype)} | {g.get('n', 0):,} | {_fmt(g.get('pehe'))} |")
    lines += ["", "### Largest errors", "",
              "| scenario_id | Scenario type | Target | Observed | Predicted | Abs. error |",
              "|---|---|---|---:|---:|---:|"]
    for f in report.get("failure_cases") or []:
        lines.append(f"| `{f['scenario_id']}` | {_cell(f['scenario_type'])} | `{f['target']}` | "
                     f"{_fmt(f.get('y_true'))} | {_fmt(f.get('y_pred'))} | {_fmt(f.get('abs_error'))} |")
    lines += ["", "## Top features", ""]
    importance = report.get("permutation_importance") or {}
    if importance:
        lines += [f"Permutation importance (increase in RMSE) on each held-out fold, averaged across "
                  f"folds; top {TOP_FEATURES} per Target.", ""]
        for model, by_target in importance.items():
            for target, items in by_target.items():
                text = ", ".join(f"`{i['feature']}` ({_fmt(i.get('importance'))})" for i in items)
                lines.append(f"- `{model}`, `{target}`: {text or 'none'}")
    else:
        lines.append("Not computed.")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------------------- checks --

CHECK_EXAMPLES = 5


def _first(values) -> str:
    values = [str(v) for v in values]
    return ", ".join(values[:CHECK_EXAMPLES]) + (f" (+{len(values) - CHECK_EXAMPLES} more)"
                                                  if len(values) > CHECK_EXAMPLES else "")


def _as_float(value) -> float:
    if value is None:
        return math.nan
    return float(value)


def _same(saved, recomputed, tolerance: float = CHECK_TOLERANCE) -> bool:
    """True when both are NaN/None, or both finite numbers within `tolerance`."""
    try:
        a, b = _as_float(saved), _as_float(recomputed)
    except (TypeError, ValueError):
        return False
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return abs(a - b) <= tolerance


def _check_coverage(frame, predictions, models, targets) -> list[str]:
    ids = set(frame["scenario_id"].astype(str))
    errors = []
    for model in models:
        for target in targets:
            part = predictions[(predictions["model"] == model) & (predictions["target"] == target)]
            missing = sorted(ids - set(part["scenario_id"].astype(str)))
            if missing:
                errors.append(f"{model}/{target}: {len(missing)} scenario(s) lack a prediction: "
                              f"{_first(missing)}")
    return errors


def _check_unique(predictions) -> list[str]:
    keys = ["scenario_id", "target", "model"]
    dup = predictions[predictions.duplicated(keys, keep=False)]
    if dup.empty:
        return []
    combos = sorted({f"{r.model}/{r.target}/{r.scenario_id}" for r in dup.itertuples()})
    return [f"{len(combos)} (scenario_id, target, model) combination(s) have more than one "
            f"prediction: {_first(combos)}"]


def _check_fold_truth(frame, predictions, targets) -> list[str]:
    errors = []
    table = frame.assign(_id=frame["scenario_id"].astype(str)).drop_duplicates("_id").set_index("_id")
    ids = predictions["scenario_id"].astype(str)
    unknown = sorted(set(ids) - set(table.index))
    if unknown:
        errors.append(f"{len(unknown)} predicted scenario_id(s) not in the modeling table: "
                      f"{_first(unknown)}")
    known = ids.isin(table.index).to_numpy()
    pred = predictions[known]
    pid = ids[known].to_numpy()
    saved_fold = pred["fold"].to_numpy(dtype="int64")
    true_fold = table["fold"].astype("int64").reindex(pid).to_numpy()
    bad = sorted({pid[i] for i in np.flatnonzero(saved_fold != true_fold)})
    if bad:
        errors.append(f"fold differs from the modeling table for {len(bad)} scenario(s): {_first(bad)}")
    for target in targets:
        mask = (pred["target"] == target).to_numpy()
        if not mask.any():
            continue
        if target not in table.columns:
            errors.append(f"target {target!r} is not a modeling table column")
            continue
        saved = pred.loc[mask, "y_true"].to_numpy(dtype="float64")
        truth = table[target].astype("float64").reindex(pid[mask]).to_numpy()
        differs = ~((saved == truth) | (np.isnan(saved) & np.isnan(truth)))
        bad = sorted({pid[mask][i] for i in np.flatnonzero(differs)})
        if bad:
            errors.append(f"y_true differs from {target} for {len(bad)} scenario(s): {_first(bad)}")
    other = sorted(set(pred["target"].astype(str)) - set(targets))
    if other:
        errors.append(f"predictions for target(s) not in config.json: {_first(other)}")
    return errors


def _check_metrics(frame, provenance, predictions, metrics, models, targets) -> list[str]:
    planners = [str(p) for p in provenance["planners"]]
    resolved = resolve_targets(planners)
    if list(resolved.values()) != list(targets):
        return [f"config.json targets {list(targets)} differ from the provenance Targets "
                f"{list(resolved.values())}"]
    try:
        point = point_metrics(frame, predictions, resolved, list(models), planners=planners)
    except (ValueError, KeyError) as error:
        return [f"cannot recompute metrics: {error}"]
    errors = []
    saved_targets = metrics.get("per_target") or {}
    for model, by_target in point["per_target"].items():
        for target, values in by_target.items():
            saved = (saved_targets.get(model) or {}).get(target)
            if saved is None:
                errors.append(f"metrics.json lacks per_target {model}/{target}")
                continue
            for key, value in values.items():
                if key not in saved:
                    errors.append(f"metrics.json lacks {model}/{target}/{key}")
                elif not _same(saved[key], value):
                    errors.append(f"{model}/{target}/{key}: saved {saved[key]!r}, "
                                  f"recomputed {value!r}")
    saved_policy = metrics.get("policy") or {}
    for name, values in point["policy"].items():
        saved = saved_policy.get(name)
        if saved is None:
            errors.append(f"metrics.json lacks policy {name}")
            continue
        for key in ("value", "regret"):
            if key not in saved:
                errors.append(f"metrics.json lacks policy {name}/{key}")
            elif not _same(saved[key], values[key]):
                errors.append(f"policy {name}/{key}: saved {saved[key]!r}, "
                              f"recomputed {values[key]!r}")
    return errors


def _check_features(provenance, schema, config) -> list[str]:
    features = [str(c) for c in config.get("feature_columns") or []]
    expected = [str(c) for c in provenance.get("feature_columns") or []]
    errors = []
    if features != expected:
        extra = [c for c in features if c not in expected]
        missing = [c for c in expected if c not in features]
        detail = []
        if extra:
            detail.append(f"extra {_first(extra)}")
        if missing:
            detail.append(f"missing {_first(missing)}")
        errors.append("config.json feature list differs from the provenance feature_columns"
                      + (f" ({'; '.join(detail)})" if detail else " (order)"))
    leaks = leakage_columns(features, schema)
    if leaks:
        errors.append("leakage column(s) in config.json features: "
                      + _first(f"{c} ({r})" for c, r in leaks.items()))
    labels = {str(c) for c in provenance.get("label_columns") or []}
    shared = [c for c in features if c in labels]
    if shared:
        errors.append(f"config.json features share label column(s): {_first(shared)}")
    return errors


def check_outputs(frame: pd.DataFrame, provenance: dict, schema: dict, predictions: pd.DataFrame,
                  metrics: dict, config: dict, table_sha256: str) -> list[tuple[str, list[str]]]:
    """(check name, errors) for oof_coverage, unique_predictions, fold_and_truth,
    metrics_recompute, feature_separation and table_hash."""
    models = [str(m) for m in config.get("models") or []]
    targets = [str(t) for t in config.get("targets") or []]
    hash_errors = []
    if config.get("table_sha256") != table_sha256:
        hash_errors.append(f"config.json table_sha256 {config.get('table_sha256')} differs from the "
                           f"modeling table SHA-256 {table_sha256}")
    return [
        ("oof_coverage", _check_coverage(frame, predictions, models, targets)),
        ("unique_predictions", _check_unique(predictions)),
        ("fold_and_truth", _check_fold_truth(frame, predictions, targets)),
        ("metrics_recompute", _check_metrics(frame, provenance, predictions, metrics, models, targets)),
        ("feature_separation", _check_features(provenance, schema, config)),
        ("table_hash", hash_errors),
    ]


# ----------------------------------------------------------------------------------- commands --

def _error(message: str, status: int) -> int:
    print(f"error: {message}", file=sys.stderr)
    return status


def _require_file(path: Path) -> Path:
    path = Path(path)
    if not path.is_file():
        raise InputError(f"input file not found: {path}")
    return path


def _require_sklearn() -> None:
    try:
        import sklearn  # noqa: F401
    except ImportError as exc:
        raise InputError(f"scikit-learn is required for train: {INSTALL_HINT}") from exc


def cmd_train(args) -> int:
    """Validate inputs and output paths, fit and evaluate in memory, then write every output."""
    try:
        models = parse_models(args.models)
        table_path = Path(args.table)
        frame, provenance = load_table(table_path)
        schema_path = _require_file(args.schema)
        schema = load_schema(schema_path)
        leaks = leakage_columns([str(c) for c in provenance["feature_columns"]], schema)
        if leaks:
            detail = "; ".join(f"{column} ({rule})" for column, rule in leaks.items())
            raise InputError(f"leakage column(s) in feature_columns: {detail}")
        experiment = experiment_name(provenance, table_path)
        out_dir = Path(args.out_dir) if args.out_dir else default_out_dir(provenance, table_path)
        paths = output_paths(out_dir, args.report_dir, experiment)
        if not args.overwrite:
            existing = [str(p) for p in paths.values() if p.exists()]
            if existing:
                raise InputError(f"output exists (use --overwrite): {', '.join(existing)}")
        _require_sklearn()
        table_sha = sha256_file(table_path)
        schema_sha = sha256_file(schema_path)
    except (InputError, validator.InputError) as error:
        return _error(str(error), 2)

    importance = not getattr(args, "no_importance", False)
    resamples = int(getattr(args, "bootstrap_resamples", BOOTSTRAP_RESAMPLES))
    try:
        predictions, metrics, core = train(frame, provenance, schema, models, int(args.seed),
                                           importance=importance, bootstrap_resamples=resamples)
    except InputError as error:
        return _error(str(error), 2)
    except FitError as error:
        print(f"FAIL fit: {error}; no output written", file=sys.stderr)
        return 1

    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    metrics = {"experiment": experiment, **metrics}
    config = {
        "tool_version": TOOL_VERSION,
        "created_utc": created,
        "experiment": experiment,
        "command": list(getattr(args, "command_line", None) or sys.argv),
        "versions": library_versions(),
        **core,
        "table": str(table_path),
        "table_sha256": table_sha,
        "schema": str(schema_path),
        "schema_sha256": schema_sha,
    }
    report = build_report(metrics, config)
    markdown = markdown_report(report)

    write_parquet_atomic(predictions, paths["predictions"])
    write_json_atomic(paths["metrics"], metrics)
    write_json_atomic(paths["config"], config)
    print(f"Wrote {paths['predictions']} ({len(predictions):,} rows), {METRICS_FILE} and "
          f"{CONFIG_FILE}")
    write_json_atomic(paths["report_json"], report)
    write_text_atomic(paths["report_md"], markdown)
    print(f"Wrote {paths['report_md']} and {paths['report_json'].name}")
    best = metrics["best_model"]
    mean_pehe = np.mean([metrics["per_target"][best][t]["pehe"] for t in metrics["targets"]])
    print(f"Best model: {best} (mean PEHE {mean_pehe:.4f})")
    return 0


def cmd_check(args) -> int:
    """Run check_outputs on the saved outputs; one PASS/FAIL line per check."""
    try:
        table_path = Path(args.table)
        frame, provenance = load_table(table_path)
        schema = load_schema(_require_file(args.schema))
        out_dir = Path(args.out_dir) if args.out_dir else default_out_dir(provenance, table_path)
        paths = {key: _require_file(out_dir / name) for key, name in
                 (("predictions", PREDICTIONS_FILE), ("metrics", METRICS_FILE),
                  ("config", CONFIG_FILE))}
        predictions = pd.read_parquet(paths["predictions"])
        missing = [c for c in PREDICTION_COLUMNS if c not in predictions.columns]
        if missing:
            raise InputError(f"{paths['predictions']} lacks column(s): {', '.join(missing)}")
        try:
            metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
            config = json.loads(paths["config"].read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise InputError(f"invalid JSON output: {error}") from error
        missing = [k for k in CONFIG_KEYS if k not in config]
        if missing:
            raise InputError(f"{paths['config']} lacks key(s): {', '.join(missing)}")
        results = check_outputs(frame, provenance, schema, predictions, metrics, config,
                                sha256_file(table_path))
    except (InputError, validator.InputError) as error:
        return _error(str(error), 2)

    ok = True
    for name, errors in results:
        if errors:
            ok = False
            shown = "; ".join(errors[:3]) + (f" (+{len(errors) - 3} more)" if len(errors) > 3 else "")
            print(f"FAIL {name}: {shown}")
        else:
            print(f"PASS {name}")
    return 0 if ok else 1


# ------------------------------------------------------------------------------------------ CLI --

def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--table", type=Path, default=DEFAULT_TABLE,
                        help="planner_modeling_table.parquet with planner_labels_provenance")
    common.add_argument("--out-dir", type=Path, default=None,
                        help=f"directory of {PREDICTIONS_FILE}, {METRICS_FILE} and {CONFIG_FILE} "
                             f"(default: {DEFAULT_MODELS_ROOT}/<experiment>)")
    common.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA, help="feature schema YAML")

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    train_cmd = sub.add_parser("train", parents=[common],
                               help="fit the baselines out of fold and write outputs and report")
    train_cmd.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR,
                           help=f"directory of {REPORT_STEM}_<experiment>.md and .json")
    train_cmd.add_argument("--seed", type=int, default=0, help="random seed (default 0)")
    train_cmd.add_argument("--models", default=",".join(MODEL_FAMILIES),
                           help=f"comma-separated model families (default {','.join(MODEL_FAMILIES)})")
    train_cmd.add_argument("--overwrite", action="store_true", help="replace existing outputs")
    # Test-only speed knobs; the actual values are recorded in config.json.
    train_cmd.add_argument("--no-importance", action="store_true", help=argparse.SUPPRESS)
    train_cmd.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES,
                           help=argparse.SUPPRESS)
    train_cmd.set_defaults(func=cmd_train)
    sub.add_parser("check", parents=[common],
                   help="verify the saved outputs against the modeling table").set_defaults(func=cmd_check)
    return parser


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    args.command_line = ["python", "-m", "tools.train_baseline", *argv]
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
