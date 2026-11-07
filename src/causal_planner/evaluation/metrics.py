"""Metrics for per-scenario planner-effect predictions and planner-selection policies.

Only the standard library, numpy and pandas are used, so the classical baseline
(`tools/train_baseline.py`) and later models score predictions with the same code
without needing scikit-learn or scipy.

Effect metrics take 1-D arrays of observed deltas ``y_true`` and predicted deltas
``y_pred``. Ranking metrics break ties by row position (stable sort), so they depend
only on the order of the predictions.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
import pandas as pd

TOP_K_FRACTIONS = (0.10, 0.20)


def _pair(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("y_true and y_pred must be 1-D")
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"length mismatch: {a.shape[0]} != {b.shape[0]}")
    if a.shape[0] == 0:
        raise ValueError("y_true and y_pred must be non-empty")
    return a, b


def pehe(y_true, y_pred) -> float:
    """Root mean squared difference between predicted and observed deltas."""
    a, b = _pair(y_true, y_pred)
    return float(np.sqrt(np.mean((b - a) ** 2)))


def mae(y_true, y_pred) -> float:
    """Mean absolute difference between predicted and observed deltas."""
    a, b = _pair(y_true, y_pred)
    return float(np.mean(np.abs(b - a)))


def spearman(y_true, y_pred) -> float:
    """Pearson correlation of pandas average ranks; NaN when either input is constant."""
    a, b = _pair(y_true, y_pred)
    ra = pd.Series(a).rank(method="average").to_numpy(dtype=float)
    rb = pd.Series(b).rank(method="average").to_numpy(dtype=float)
    da = ra - ra.mean()
    db = rb - rb.mean()
    denom = math.sqrt(float(np.sum(da * da)) * float(np.sum(db * db)))
    if denom == 0.0 or not math.isfinite(denom):
        return float("nan")
    return float(np.sum(da * db) / denom)


def top_k_count(n: int, fraction: float) -> int:
    """ceil(fraction * n), at least 1 and at most n (for n >= 1)."""
    if n < 1:
        raise ValueError("n must be at least 1")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    # Rounding guards against float noise such as 0.2 * 65 = 13.000000000000002.
    k = math.ceil(round(fraction * n, 9))
    return int(min(max(k, 1), n))


def _top_order(x: np.ndarray) -> np.ndarray:
    """Row positions by descending value; ties go to the earlier row."""
    return np.argsort(-x, kind="stable")


def top_k_recall(y_true, y_pred, fraction: float) -> float:
    """Share of the top-k rows by observed delta that are also top-k by predicted delta."""
    a, b = _pair(y_true, y_pred)
    k = top_k_count(a.shape[0], fraction)
    true_top = set(_top_order(a)[:k].tolist())
    pred_top = set(_top_order(b)[:k].tolist())
    return len(true_top & pred_top) / k


def gain_curve(y_true, order) -> np.ndarray:
    """Cumulative observed delta along `order`: G[0] = 0, G[j] = sum of the first j rows."""
    y = np.asarray(y_true, dtype=float)
    idx = np.asarray(order, dtype=np.int64)
    return np.concatenate(([0.0], np.cumsum(y[idx])))


def _trapezoid_area(curve: np.ndarray) -> float:
    return float(np.sum((curve[:-1] + curve[1:]) / 2.0))


def qini_coefficient(y_true, y_pred) -> float:
    """(A(G_model) - A(random)) / (A(G_oracle) - A(random)); NaN when y_true is constant.

    G_model orders rows by descending prediction, G_oracle by descending observed delta
    (both stable), the random line is R[j] = j * G[n] / n with area n * G[n] / 2, and A is
    the trapezoid area sum_{j=1..n} (G[j-1] + G[j]) / 2.
    """
    a, b = _pair(y_true, y_pred)
    if np.all(a == a[0]):
        return float("nan")
    n = a.shape[0]
    model = gain_curve(a, _top_order(b))
    oracle = gain_curve(a, _top_order(a))
    random_area = n * float(model[-1]) / 2.0
    denom = _trapezoid_area(oracle) - random_area
    if denom <= 0.0:
        return float("nan")
    return (_trapezoid_area(model) - random_area) / denom


def choose_planner(pred_deltas: pd.DataFrame, control: str = "idm") -> pd.Series:
    """Per row, the first column holding the row maximum if that maximum is above 0, else `control`.

    Columns are the Target planner tokens in Provenance order. NaN predictions never win.
    """
    columns = np.asarray(pred_deltas.columns, dtype=object)
    values = pred_deltas.to_numpy(dtype=float, na_value=np.nan)
    if values.shape[1] == 0:
        return pd.Series([control] * len(pred_deltas), index=pred_deltas.index, dtype=object)
    values = np.where(np.isnan(values), -np.inf, values)
    best = np.argmax(values, axis=1)  # first maximum, so ties go to the earliest column
    best_value = values[np.arange(values.shape[0]), best]
    choice = np.where(best_value > 0.0, columns[best], control)
    return pd.Series(choice, index=pred_deltas.index, dtype=object)


def realized_scores(scores: pd.DataFrame, choice: pd.Series) -> np.ndarray:
    """Row-wise observed score of the chosen planner (matched by position)."""
    chosen = np.asarray(choice, dtype=object)
    if chosen.shape[0] != len(scores):
        raise ValueError(f"length mismatch: {len(scores)} score rows, {chosen.shape[0]} choices")
    positions = scores.columns.get_indexer(chosen)
    if np.any(positions < 0):
        missing = sorted({str(c) for c, p in zip(chosen, positions) if p < 0})
        raise KeyError(f"chosen planners missing from scores: {missing}")
    values = scores.to_numpy(dtype=float, na_value=np.nan)
    return values[np.arange(values.shape[0]), positions]


def policy_value(scores: pd.DataFrame, choice: pd.Series) -> float:
    """Mean observed score of the chosen planner per row."""
    return float(np.mean(realized_scores(scores, choice)))


def oracle_value(scores: pd.DataFrame) -> float:
    """Mean of the row-wise maximum observed score."""
    return float(np.mean(scores.to_numpy(dtype=float, na_value=np.nan).max(axis=1)))


def _take(array, idx: np.ndarray):
    if isinstance(array, (pd.DataFrame, pd.Series)):
        return array.iloc[idx]
    return np.asarray(array)[idx]


def bootstrap_ci(
    statistic: Callable[..., float],
    *arrays,
    n_resamples: int = 1000,
    seed: int = 0,
    level: float = 0.95,
) -> tuple[float, float, float]:
    """(point, low, high) for `statistic(*arrays)` by joint row resampling with replacement.

    Each resample draws n row indices from ``np.random.default_rng(seed)`` and applies them
    to every array. low/high are linear percentiles at (1 - level)/2 and (1 + level)/2 of
    the resampled statistics after dropping NaN values (NaN when none remain).
    """
    if not arrays:
        raise ValueError("bootstrap_ci needs at least one array")
    if n_resamples < 1:
        raise ValueError("n_resamples must be at least 1")
    if not 0.0 < level < 1.0:
        raise ValueError("level must be in (0, 1)")
    lengths = {len(a) for a in arrays}
    if len(lengths) != 1:
        raise ValueError(f"arrays have different lengths: {sorted(lengths)}")
    n = lengths.pop()
    if n == 0:
        raise ValueError("arrays must be non-empty")

    point = float(statistic(*arrays))
    rng = np.random.default_rng(seed)
    stats = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        stats[i] = float(statistic(*(_take(a, idx) for a in arrays)))
    stats = stats[~np.isnan(stats)]
    if stats.size == 0:
        return point, float("nan"), float("nan")
    low, high = np.percentile(stats, [100.0 * (1.0 - level) / 2.0, 100.0 * (1.0 + level) / 2.0], method="linear")
    return point, float(low), float(high)
