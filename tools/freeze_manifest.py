"""Freeze one deterministic ranking of benchmark scenarios across the four nuPlan cities.

Reads the benchmark candidates of the scenario eligibility table (tools/build_scenario_eligibility.py)
for boston, las_vegas, pittsburgh and singapore and ranks every (city, scenario type) group:

    rank key        sha256("<seed>:<scenario_id>")[:16], ties broken by scenario_id
    group order     rarest group first within each city, ties by type name
    overlap filter  a window [start, end) is skipped if it overlaps one already accepted in the
                    same drive (source_recording), shared across all types; run to exhaustion
    group_rank      1-based position of an accepted candidate within its group
    split           every drive goes to exactly one of train/val/test by a seeded hash, per city,
                    computed from the full ranking so every cut carries the same splits

A cut keeps the rows with group_rank <= N, so the top-5 demo set is a prefix of any larger
training cut. The full ranking is written outside Git; cuts and provenance go to configs/scenarios/.

Usage (from the repository root):
    python -m tools.freeze_manifest build                     # ranking + top-5 cut + provenance
    python -m tools.freeze_manifest build --per-group 5 --per-group 50 --prefix city4_ranked_v2
    python -m tools.freeze_manifest check --manifest configs/scenarios/city4_ranked_top5.csv
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import numbers
import os
import subprocess
import sys
import tempfile
from bisect import bisect_left
from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from tools.build_scenario_eligibility import BENCHMARK_TYPES

TOOL_VERSION = "freeze-manifest-v1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CITIES = ("boston", "las_vegas", "pittsburgh", "singapore")
DEFAULT_ELIGIBILITY = Path("E:/Project/nuplan/processed/scenario_eligibility.parquet")
DEFAULT_RANKING = Path("E:/Project/nuplan/processed/manifests/city4_ranked_seed{seed}_full.csv")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "configs" / "scenarios"
DEFAULT_PREFIX = "city4_ranked"
DEFAULT_SEED = 7
DEFAULT_RATIOS = (0.70, 0.15, 0.15)

# Pre-simulation identifier, timing, map, tag and eligibility-flag columns only (Req 1.4).
READ_COLUMNS = [
    "scenario_id", "log_name", "db_file", "scenario_token", "anchor_timestamp_us",
    "benchmark_scenario_type", "all_scenario_tags", "map_name", "source_recording",
    "window_start_us", "window_end_us", "source_city", "benchmark_candidate",
    "shares_mini_recording", "in_mini_split", "eligibility_version",
]
MANIFEST_COLUMNS = [
    "scenario_id", "log_name", "db_file", "scenario_token", "anchor_timestamp_us",
    "scenario_type", "all_scenario_tags", "map_name", "source_recording",
    "window_start_us", "window_end_us", "source_city", "group_rank", "split",
]
MINI_FLAGS = ("shares_mini_recording", "in_mini_split")


def rank_key(seed: int, scenario_id: str) -> str:
    """First 16 hex characters of sha256("<seed>:<scenario_id>") (Req 2.1, 2.7)."""
    return hashlib.sha256(f"{seed}:{scenario_id}".encode()).hexdigest()[:16]


def split_key(seed: int, drive: str) -> str:
    """Seeded hash that orders drives for split assignment (Req 3.1, 2.7)."""
    return hashlib.sha256(f"{seed}:split:{drive}".encode()).hexdigest()


def missing_columns(columns: Iterable[str]) -> list[str]:
    """READ_COLUMNS absent from ``columns``, sorted (Req 1.3)."""
    present = set(columns)
    return sorted(column for column in READ_COLUMNS if column not in present)


def drop_mini(candidates: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove rows from mini recordings; return the kept rows and the excluded count (Req 1.2)."""
    mini = pd.Series(False, index=candidates.index)
    for flag in MINI_FLAGS:
        mini |= candidates[flag].fillna(False).astype(bool)
    return candidates.loc[~mini].copy(), int(mini.sum())


def rank_and_filter(candidates: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Rank candidates per (city, type) group and drop windows overlapping accepted ones.

    Cities are processed in sorted order; within a city, groups go rarest first (ties by type
    name) and rows within a group by ``(rank_key, scenario_id)`` (Req 2.1, 2.2). A row is
    accepted only if its half-open window ``[start, end)`` overlaps no window already accepted
    in the same drive, across all types of the city (Req 2.3). Accepted rows get a dense
    1-based ``group_rank`` (Req 2.4). Returns the accepted rows with their original columns plus
    ``group_rank`` (int64), in processing order, with a fresh RangeIndex.
    """
    city_col, type_col = "source_city", "benchmark_scenario_type"
    if candidates.empty:
        out = candidates.copy()
        out["group_rank"] = pd.Series(dtype="int64")
        return out.reset_index(drop=True)

    frame = candidates.reset_index(drop=True)
    order = pd.DataFrame({
        "city": frame[city_col].astype(str),
        "count": frame.groupby([city_col, type_col])[type_col].transform("size").astype("int64"),
        "type": frame[type_col].astype(str),
        "key": [rank_key(seed, str(sid)) for sid in frame["scenario_id"]],
        "sid": frame["scenario_id"].astype(str),
    })
    order = order.sort_values(["city", "count", "type", "key", "sid"], kind="mergesort")
    positions = order.index.to_numpy()

    cities = order["city"].to_numpy()
    types = order["type"].to_numpy()
    drives = frame["source_recording"].astype(str).to_numpy()[positions]
    starts = frame["window_start_us"].astype("int64").to_numpy()[positions]
    ends = frame["window_end_us"].astype("int64").to_numpy()[positions]

    accepted: list[int] = []
    ranks: list[int] = []
    occupied: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    current_group: tuple[str, str] | None = None
    rank = 0
    for i in range(len(positions)):
        group = (cities[i], types[i])
        if group != current_group:
            current_group, rank = group, 0
        start, end = int(starts[i]), int(ends[i])
        intervals = occupied[(cities[i], drives[i])]
        j = bisect_left(intervals, (start, end))
        if j > 0 and intervals[j - 1][1] > start:
            continue
        if j < len(intervals) and intervals[j][0] < end:
            continue
        intervals.insert(j, (start, end))
        rank += 1
        accepted.append(int(positions[i]))
        ranks.append(rank)

    out = frame.iloc[accepted].reset_index(drop=True)
    out["group_rank"] = pd.Series(ranks, dtype="int64")
    return out


SPLITS = ("train", "val", "test")


def validate_ratios(ratios) -> None:
    """Raise ValueError unless ``ratios`` is three finite, non-negative numbers summing to 1 (Req 3.5)."""
    try:
        values = list(ratios)
    except TypeError:
        raise ValueError(f"split ratios must be three numbers, got {ratios!r}") from None
    if len(values) != 3:
        raise ValueError(f"split ratios must be three numbers, got {len(values)}")
    for value in values:
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise ValueError(f"split ratio {value!r} is not a number")
        if not math.isfinite(value):
            raise ValueError(f"split ratio {value!r} is not finite")
        if value < 0:
            raise ValueError(f"split ratio {value!r} is negative")
    if abs(sum(values) - 1.0) > 1e-9:
        raise ValueError(f"split ratios must sum to 1, got {sum(values)!r}")


def split_counts(n: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    """Number of drives for (train, val, test) out of ``n`` (Req 3.2, 3.3).

    val and test are ``round(n * ratio)`` (half to even), train takes the rest. For n >= 3 each
    split with a positive ratio gets at least one drive, taken from the larger of val/test (ties:
    test). A split with ratio 0 gets no drives. For n < 3 there is no minimum; train is clamped to
    [0, n] and the remainder goes to val, then test.
    """
    if n < 0:
        raise ValueError(f"drive count must be non-negative, got {n}")
    rt, rv, rs = ratios
    n_val, n_test = round(n * rv), round(n * rs)
    n_train = n - n_val - n_test

    if n < 3:
        n_train = min(max(n_train, 0), n)
        rest = n - n_train
        n_val = min(n_val, rest)
        return n_train, n_val, rest - n_val

    if rv > 0 and n_val == 0:
        n_val = 1
    if rs > 0 and n_test == 0:
        n_test = 1
    floor_val, floor_test = (1 if rv > 0 else 0), (1 if rs > 0 else 0)
    train_floor = 1 if rt > 0 else 0
    n_train = n - n_val - n_test
    while n_train < train_floor:
        can_val, can_test = n_val > floor_val, n_test > floor_test
        if not (can_val or can_test):
            break
        if can_test and (n_test >= n_val or not can_val):
            n_test -= 1
        else:
            n_val -= 1
        n_train += 1
    if rt == 0 and n_train > 0:
        # Rounding left drives for a zero-ratio train split; give them to val, else test.
        if rv > 0:
            n_val += n_train
        else:
            n_test += n_train
        n_train = 0
    return n_train, n_val, n_test


def assign_splits(
    drives_by_city: Mapping[str, Iterable[str]],
    seed: int,
    ratios: tuple[float, float, float],
) -> dict[str, str]:
    """Map every drive to "train", "val" or "test", per city (Req 3.1-3.3).

    Within a city, unique drives are ordered by ``(split_key(seed, drive), drive)``; the first
    ``n_train`` go to train, the next ``n_val`` to val, the rest to test, with counts from
    ``split_counts``.
    """
    validate_ratios(ratios)
    assignment: dict[str, str] = {}
    for city in sorted(drives_by_city):
        drives = sorted({str(d) for d in drives_by_city[city]}, key=lambda d: (split_key(seed, d), d))
        n_train, n_val, _ = split_counts(len(drives), ratios)
        for i, drive in enumerate(drives):
            if drive in assignment:
                raise ValueError(f"drive {drive!r} appears in more than one city")
            if i < n_train:
                assignment[drive] = "train"
            elif i < n_train + n_val:
                assignment[drive] = "val"
            else:
                assignment[drive] = "test"
    return assignment


INT_COLUMNS = ("anchor_timestamp_us", "window_start_us", "window_end_us", "group_rank")
TOKEN_PATTERN = r"[0-9a-f]{16}"


def to_manifest(ranked: pd.DataFrame, splits: Mapping[str, str]) -> pd.DataFrame:
    """Turn ``rank_and_filter`` output into the manifest format (Req 4.1-4.5, 3.4).

    Renames ``benchmark_scenario_type`` to ``scenario_type``, lowercases ``scenario_token``,
    sets ``db_file`` to ``log_name + ".db"`` after checking the table value agrees, maps each
    drive to its split, selects ``MANIFEST_COLUMNS`` and sorts stably by
    ``(source_city, scenario_type, group_rank)``. Raises ValueError on a duplicate
    ``scenario_id``, a ``db_file`` mismatch, a malformed token or a drive without a split.
    """
    frame = ranked.rename(columns={"benchmark_scenario_type": "scenario_type"}).copy()

    duplicated = frame["scenario_id"].astype(str).duplicated(keep=False)
    if duplicated.any():
        ids = sorted(frame.loc[duplicated, "scenario_id"].astype(str).unique())
        raise ValueError(f"duplicate scenario_id in ranking: {', '.join(ids[:5])}")

    log_name = frame["log_name"].astype(str)
    expected_db = log_name + ".db"
    mismatch = frame["db_file"].astype(str) != expected_db
    if mismatch.any():
        bad = frame.loc[mismatch, "scenario_id"].astype(str).tolist()
        raise ValueError(f"db_file does not equal log_name + '.db' for: {', '.join(bad[:5])}")
    frame["log_name"] = log_name
    frame["db_file"] = expected_db

    tokens = frame["scenario_token"].astype(str).str.lower()
    malformed = ~tokens.str.fullmatch(TOKEN_PATTERN)
    if malformed.any():
        bad = tokens[malformed].tolist()
        raise ValueError(f"scenario_token is not 16 hex characters: {', '.join(bad[:5])}")
    frame["scenario_token"] = tokens

    drives = frame["source_recording"].astype(str)
    missing = sorted(set(drives) - set(splits))
    if missing:
        raise ValueError(f"no split assigned for drive(s): {', '.join(missing[:5])}")
    frame["source_recording"] = drives
    frame["split"] = drives.map(splits)

    for column in INT_COLUMNS:
        frame[column] = frame[column].astype("int64")

    out = frame[MANIFEST_COLUMNS].sort_values(
        ["source_city", "scenario_type", "group_rank"], kind="mergesort"
    )
    return out.reset_index(drop=True)


def make_cut(manifest: pd.DataFrame, n: int) -> pd.DataFrame:
    """Rows of ``manifest`` with ``group_rank <= n``, order preserved (Req 5.1, 2.6, 5.4)."""
    if isinstance(n, bool) or not isinstance(n, numbers.Integral) or n < 1:
        raise ValueError(f"per-group count must be an integer >= 1, got {n!r}")
    return manifest.loc[manifest["group_rank"] <= n].reset_index(drop=True)


def _type_column(frame: pd.DataFrame) -> str:
    """Scenario type column of a candidate/ranked frame (raw) or a manifest (renamed)."""
    return "benchmark_scenario_type" if "benchmark_scenario_type" in frame.columns else "scenario_type"


def _group_sizes(frame: pd.DataFrame) -> dict[tuple[str, str], int]:
    """Row count per (source_city, scenario type) pair, as plain ints."""
    if frame.empty:
        return {}
    sizes = frame.groupby(
        [frame["source_city"].astype(str), frame[_type_column(frame)].astype(str)]
    ).size()
    return {(str(city), str(kind)): int(count) for (city, kind), count in sizes.items()}


def _all_pairs() -> list[tuple[str, str]]:
    """The 56 (city, type) pairs of CITIES x BENCHMARK_TYPES, sorted."""
    return [(city, kind) for city in sorted(CITIES) for kind in sorted(BENCHMARK_TYPES)]


def group_table(
    candidates: pd.DataFrame,
    manifest: pd.DataFrame,
    cuts: Mapping[int, pd.DataFrame],
) -> list[dict]:
    """One entry per (city, type) pair of CITIES x BENCHMARK_TYPES, sorted (Req 5.2, 5.3).

    Each entry has the candidate count, the count in the full ranking ``manifest``, and for each
    requested N (key ``str(N)``) the rows ``selected`` in that cut and the ``shortfall``
    ``max(0, N - selected)``. Pairs with no candidates are listed with zeros.
    """
    candidate_sizes = _group_sizes(candidates)
    ranked_sizes = _group_sizes(manifest)
    cut_sizes = {int(n): _group_sizes(cut) for n, cut in cuts.items()}
    table = []
    for city, kind in _all_pairs():
        entry_cuts = {}
        for n in sorted(cut_sizes):
            selected = cut_sizes[n].get((city, kind), 0)
            entry_cuts[str(n)] = {"selected": selected, "shortfall": max(0, n - selected)}
        table.append({
            "source_city": city,
            "scenario_type": kind,
            "candidates": candidate_sizes.get((city, kind), 0),
            "ranked": ranked_sizes.get((city, kind), 0),
            "cuts": entry_cuts,
        })
    return table


def split_coverage(manifest: pd.DataFrame) -> dict:
    """Per split, the number of the 56 groups with rows and the sorted "city/type" list without (Req 3.6)."""
    pairs = _all_pairs()
    coverage = {}
    for split in SPLITS:
        rows = manifest.loc[manifest["split"].astype(str) == split] if not manifest.empty else manifest
        present = set(_group_sizes(rows))
        coverage[split] = {
            "groups_with_rows": sum(1 for pair in pairs if pair in present),
            "groups_without_rows": [f"{city}/{kind}" for city, kind in pairs if (city, kind) not in present],
        }
    return coverage


def _count_by(frame: pd.DataFrame, column: str, keys: Iterable[str], unique: str | None = None) -> dict:
    """{split: {key: count}} with every split and every key present (zeros included)."""
    values = frame[column].astype(str) if not frame.empty else pd.Series(dtype=str)
    all_keys = sorted(set(keys) | set(values))
    result = {}
    for split in SPLITS:
        mask = frame["split"].astype(str) == split if not frame.empty else pd.Series(dtype=bool)
        if unique is None:
            counts = values[mask].value_counts() if not frame.empty else {}
        else:
            subset = frame.loc[mask] if not frame.empty else frame
            counts = (
                subset.groupby(subset[column].astype(str))[unique].nunique() if not subset.empty else {}
            )
        result[split] = {key: int(counts.get(key, 0)) for key in all_keys}
    return result


def manifest_counts(manifest: pd.DataFrame) -> dict:
    """Rows by split and city, rows by split and type, drives by split and city (Req 6.2, 6.3)."""
    return {
        "rows_by_split_city": _count_by(manifest, "source_city", CITIES),
        "rows_by_split_type": _count_by(manifest, _type_column(manifest), BENCHMARK_TYPES),
        "drives_by_split_city": _count_by(manifest, "source_city", CITIES, unique="source_recording"),
    }


def _manifest_block(manifest: pd.DataFrame, sha256: str) -> dict:
    """Hash, row count, per-split counts and split coverage of one written manifest (Req 6.1, 6.2)."""
    counts = manifest_counts(manifest)
    return {
        "sha256": str(sha256),
        "rows": int(len(manifest)),
        "rows_by_split_city": counts["rows_by_split_city"],
        "rows_by_split_type": counts["rows_by_split_type"],
        "split_coverage": split_coverage(manifest),
    }


def build_provenance(
    *,
    generated_utc: str,
    git_commit: str | None,
    seed: int,
    split_ratios: Iterable[float],
    eligibility_path: str,
    eligibility_sha256: str,
    eligibility_bytes: int,
    eligibility_version: str,
    candidates: pd.DataFrame,
    mini_excluded_rows: int,
    ranking: pd.DataFrame,
    ranking_path: str,
    ranking_sha256: str,
    ranking_committed: bool,
    cuts: Mapping[int, pd.DataFrame],
    cut_paths: Mapping[int, str],
    cut_sha256: Mapping[int, str],
) -> dict:
    """Assemble the provenance dict of the design (Req 3.6, 5.2, 5.3, 6.1-6.4).

    Pure: every hash, path, timestamp and the Git commit (None when unknown, Req 6.4) are passed
    in. ``candidates`` are the rows after the mini guard (``candidate_rows`` records the rows read,
    i.e. kept plus ``mini_excluded_rows``); ``ranking`` is the full ranking manifest and ``cuts``
    maps each N to its cut. All values are plain Python types.
    """
    per_group = sorted(int(n) for n in cuts)
    manifests = {}
    for n in per_group:
        block = _manifest_block(cuts[n], cut_sha256[n])
        block["per_group"] = n
        manifests[str(cut_paths[n])] = block
    ranking_block = _manifest_block(ranking, ranking_sha256)
    ranking_block["path"] = str(ranking_path)
    ranking_block["committed"] = bool(ranking_committed)
    return {
        "tool_version": TOOL_VERSION,
        "generated_utc": str(generated_utc),
        "git_commit": None if git_commit is None else str(git_commit),
        "seed": int(seed),
        "split_ratios": [float(r) for r in split_ratios],
        "per_group": per_group,
        "eligibility": {
            "path": str(eligibility_path),
            "sha256": str(eligibility_sha256),
            "bytes": int(eligibility_bytes),
            "version": str(eligibility_version),
            "candidate_rows": int(len(candidates)) + int(mini_excluded_rows),
            "mini_excluded_rows": int(mini_excluded_rows),
        },
        "ranking": ranking_block,
        "manifests": manifests,
        "drives_by_split_city": manifest_counts(ranking)["drives_by_split_city"],
        "groups": group_table(candidates, ranking, {n: cuts[n] for n in per_group}),
    }


CHECK_NAMES = (
    "scenario loader accepts manifest",
    "rows match eligibility candidates",
    "no overlapping windows within a drive",
    "group_rank is 1..k per group",
    "one split per drive",
    "sha256 matches provenance",
    "no mini recordings",
)
WINDOW_KEY = ("scenario_id", "source_recording", "window_start_us", "window_end_us")


def _require(frame: pd.DataFrame, columns: Iterable[str], what: str) -> None:
    """Raise ValueError naming the columns of ``columns`` absent from ``frame``."""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{what} is missing column(s): {', '.join(missing)}")


def _int_column(frame: pd.DataFrame, column: str) -> pd.Series:
    """``frame[column]`` as int64; raise ValueError if any value is missing or not an integer."""
    values = pd.to_numeric(frame[column], errors="coerce")
    bad = values.isna() | (values != values.round())
    if bad.any():
        raise ValueError(f"{column} has non-integer value(s): {frame.loc[bad, column].astype(str).tolist()[:5]}")
    return values.astype("int64")


def _window_keys(frame: pd.DataFrame, what: str) -> pd.DataFrame:
    """The WINDOW_KEY columns of ``frame`` with string ids and int64 window bounds."""
    _require(frame, WINDOW_KEY, what)
    return pd.DataFrame({
        "scenario_id": frame["scenario_id"].astype(str),
        "source_recording": frame["source_recording"].astype(str),
        "window_start_us": _int_column(frame, "window_start_us"),
        "window_end_us": _int_column(frame, "window_end_us"),
    })


def _check_loader(loader_error: str | None) -> str | None:
    return None if loader_error is None else f"load_scenarios failed: {loader_error}"


def _check_candidates(manifest: pd.DataFrame, candidates: pd.DataFrame) -> str | None:
    rows = _window_keys(manifest, "manifest")
    known = set(_window_keys(candidates, "candidates").itertuples(index=False, name=None))
    bad = [key[0] for key in rows.itertuples(index=False, name=None) if key not in known]
    if bad:
        return f"{len(bad)} row(s) not matching a candidate: {', '.join(bad[:5])}"
    return None


def _check_overlap(manifest: pd.DataFrame) -> str | None:
    rows = _window_keys(manifest, "manifest").sort_values(
        ["source_recording", "window_start_us", "window_end_us"], kind="mergesort"
    )
    previous_end = rows.groupby("source_recording")["window_end_us"].transform(
        lambda ends: ends.cummax().shift()
    )
    overlap = previous_end.notna() & (rows["window_start_us"] < previous_end)
    if overlap.any():
        drives = sorted(rows.loc[overlap, "source_recording"].unique())
        return f"overlapping windows in drive(s): {', '.join(drives[:5])}"
    return None


def _check_ranks(manifest: pd.DataFrame) -> str | None:
    _require(manifest, ("source_city", "scenario_type", "group_rank"), "manifest")
    ranks = _int_column(manifest, "group_rank")
    groups = [manifest["source_city"].astype(str), manifest["scenario_type"].astype(str)]
    bad = [
        f"{city}/{kind}"
        for (city, kind), values in ranks.groupby(groups)
        if sorted(values.tolist()) != list(range(1, len(values) + 1))
    ]
    if bad:
        return f"group_rank is not 1..k in group(s): {', '.join(sorted(bad)[:5])}"
    return None


def _check_splits(manifest: pd.DataFrame) -> str | None:
    _require(manifest, ("source_recording", "split"), "manifest")
    counts = manifest.groupby(manifest["source_recording"].astype(str))["split"].nunique(dropna=False)
    bad = sorted(counts.index[counts != 1])
    if bad:
        return f"drive(s) with more than one split: {', '.join(bad[:5])}"
    return None


def _check_sha(recorded_sha: str | None, actual_sha: str | None) -> str | None:
    if recorded_sha is None:
        return "no provenance entry for this manifest"
    if actual_sha is None:
        return "manifest SHA-256 could not be computed"
    if str(recorded_sha).strip().lower() != str(actual_sha).strip().lower():
        return f"file SHA-256 {actual_sha} differs from provenance {recorded_sha}"
    return None


def _check_mini(manifest: pd.DataFrame, mini: Iterable[str]) -> str | None:
    _require(manifest, ("source_recording",), "manifest")
    mini_set = {str(drive) for drive in mini}
    found = sorted(set(manifest["source_recording"].astype(str)) & mini_set)
    if found:
        return f"mini recording(s) in manifest: {', '.join(found[:5])}"
    return None


def check_details(
    manifest: pd.DataFrame,
    candidates: pd.DataFrame,
    mini_recordings: Iterable[str],
    recorded_sha: str | None,
    actual_sha: str | None,
    loader_error: str | None,
) -> dict[str, tuple[bool, str]]:
    """Evaluate the seven check lines independently, in CHECK_NAMES order (Req 8.2-8.8).

    Returns ``{name: (passed, reason)}``; ``reason`` is "" on a pass. A check that cannot be
    evaluated (missing column, malformed value, missing provenance entry) fails with the reason
    instead of raising, so one failure never hides another.
    """
    mini = list(mini_recordings) if mini_recordings is not None else []
    evaluators = (
        lambda: _check_loader(loader_error),
        lambda: _check_candidates(manifest, candidates),
        lambda: _check_overlap(manifest),
        lambda: _check_ranks(manifest),
        lambda: _check_splits(manifest),
        lambda: _check_sha(recorded_sha, actual_sha),
        lambda: _check_mini(manifest, mini),
    )
    results: dict[str, tuple[bool, str]] = {}
    for name, evaluate in zip(CHECK_NAMES, evaluators):
        try:
            reason = evaluate()
        except Exception as error:  # report, never crash (design: Check Subcommand)
            reason = f"could not evaluate: {error}"
        results[name] = (reason is None, reason or "")
    return results


def run_checks(
    manifest: pd.DataFrame,
    candidates: pd.DataFrame,
    mini_recordings: Iterable[str],
    recorded_sha: str | None,
    actual_sha: str | None,
    loader_error: str | None,
) -> dict[str, bool]:
    """``{check name: passed}`` for the seven checks, in CHECK_NAMES order (Req 8.2-8.8)."""
    details = check_details(manifest, candidates, mini_recordings, recorded_sha, actual_sha, loader_error)
    return {name: passed for name, (passed, _) in details.items()}


# ---------------------------------------------------------------------------------------------
# File helpers (I/O shell)
# ---------------------------------------------------------------------------------------------

HASH_CHUNK_BYTES = 16 * 1024 * 1024
GIT_TIMEOUT_S = 10


class InputError(ValueError):
    """Invalid or unreadable input table; ``build``/``check`` map it to exit code 2."""

    def __init__(self, message: str, missing: Iterable[str] | None = None) -> None:
        super().__init__(message)
        self.missing = list(missing) if missing is not None else []


def _schema_names(path: Path) -> list[str]:
    """Column names of the parquet file at ``path``; InputError if it is absent or unreadable."""
    import pyarrow.parquet as pq

    path = Path(path)
    if not path.is_file():
        raise InputError(f"eligibility table not found: {path}")
    try:
        return list(pq.read_schema(path).names)
    except Exception as error:
        raise InputError(f"cannot read parquet schema of {path}: {error}") from error


def read_candidates(path: Path) -> tuple[pd.DataFrame, str]:
    """Read the benchmark candidates of the eligibility table (Req 1.1, 1.3, 1.4).

    Checks the schema first and raises InputError (``missing`` set, message
    ``"Missing columns: a, b"``) if any of READ_COLUMNS is absent. Then reads only READ_COLUMNS
    for rows with ``benchmark_candidate == True``. Returns ``(candidates, eligibility_version)``;
    raises InputError unless the candidates carry exactly one eligibility version.
    """
    missing = missing_columns(_schema_names(path))
    if missing:
        raise InputError(f"Missing columns: {', '.join(missing)}", missing=missing)
    try:
        frame = pd.read_parquet(
            path, columns=READ_COLUMNS, filters=[("benchmark_candidate", "==", True)], engine="pyarrow"
        )
    except Exception as error:
        raise InputError(f"cannot read eligibility table {path}: {error}") from error
    frame = frame.loc[frame["benchmark_candidate"].fillna(False).astype(bool)].reset_index(drop=True)
    versions = sorted({str(v) for v in frame["eligibility_version"].dropna().unique()})
    if len(versions) != 1:
        found = ", ".join(versions) if versions else "none"
        raise InputError(f"expected exactly one eligibility_version among candidates, found: {found}")
    return frame, versions[0]


def mini_recordings(path: Path) -> set[str]:
    """``source_recording`` of every table row (candidate or not) with a mini flag set (Req 8.8)."""
    columns = ["source_recording", *MINI_FLAGS]
    missing = sorted(set(columns) - set(_schema_names(path)))
    if missing:
        raise InputError(f"Missing columns: {', '.join(missing)}", missing=missing)
    frame = pd.read_parquet(path, columns=columns, engine="pyarrow")
    mini = pd.Series(False, index=frame.index)
    for flag in MINI_FLAGS:
        mini |= frame[flag].fillna(False).astype(bool)
    return {str(drive) for drive in frame.loc[mini, "source_recording"].dropna().unique()}


def file_sha256(path: Path) -> str:
    """Streaming SHA-256 hex digest of ``path`` in 16 MiB chunks (Req 6.1)."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while chunk := stream.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    """``git rev-parse HEAD`` of the repository, or None on any failure (Req 6.4)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_S,
            check=False,
        )
        commit = result.stdout.strip()
        if result.returncode != 0 or not commit:
            return None
        return commit
    except Exception:
        return None


def _write_atomic(path: Path, write) -> None:
    """Write via a temp file in ``path``'s directory, then ``os.replace`` (Req 7.2).

    ``write`` receives a text stream opened with UTF-8 and ``newline=""``. The temp file is
    removed if anything fails before the replace.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, prefix=f".{path.name}.",
        suffix=".tmp", delete=False,
    )
    try:
        with tmp:
            write(tmp)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp.name, path)
    except BaseException:
        try:
            os.remove(tmp.name)
        except OSError:
            pass
        raise


def write_csv_atomic(df: pd.DataFrame, path: Path) -> str:
    """Write ``df`` as UTF-8 CSV with LF endings and no index; return the SHA-256 of the file (Req 4.6, 7.2).

    ``df`` is not modified. The hash is computed from the bytes written to the temp file, before
    the replace.
    """
    digest: dict[str, str] = {}

    def write(stream) -> None:
        df.to_csv(stream, index=False, lineterminator="\n")
        stream.flush()
        digest["sha256"] = file_sha256(Path(stream.name))

    _write_atomic(path, write)
    return digest["sha256"]


def write_json_atomic(obj: dict, path: Path) -> None:
    """Write ``obj`` as JSON with sorted keys, indent 2 and a trailing LF (Req 6.5, 7.2)."""
    text = json.dumps(obj, indent=2, sort_keys=True) + "\n"
    _write_atomic(path, lambda stream: stream.write(text))


# ---------------------------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------------------------

PROTECTED_FILES = ("idm_mini_sample.csv", "pittsburgh_65_sample.csv", "pittsburgh_65_sample.json")
EXIT_OK, EXIT_FAIL, EXIT_ERROR, EXIT_REFUSED = 0, 1, 2, 3


def _error(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return EXIT_ERROR


def _path_key(path: Path) -> str:
    """Comparable form of an absolute path (case-insensitive on Windows)."""
    return os.path.normcase(str(Path(path).resolve()))


def _display_path(path: Path) -> str:
    """POSIX path relative to PROJECT_ROOT when inside it, else absolute POSIX."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _inside_project(path: Path) -> bool:
    try:
        Path(path).resolve().relative_to(PROJECT_ROOT.resolve())
        return True
    except ValueError:
        return False


def _protected(path: Path) -> bool:
    """True if ``path`` is one of the frozen samples in configs/scenarios/ (Req 7.3)."""
    protected = {_path_key(PROJECT_ROOT / "configs" / "scenarios" / name) for name in PROTECTED_FILES}
    return _path_key(path) in protected


def _per_group_values(values: Iterable[int] | None) -> list[int]:
    """De-duplicated, sorted per-group counts; default [5]; ValueError on N < 1 (Req 5.4)."""
    result = sorted(set(values)) if values else [5]
    bad = [n for n in result if n < 1]
    if bad:
        raise ValueError(f"--per-group must be at least 1, got {', '.join(map(str, bad))}")
    return result


def build_targets(args, per_group: list[int]) -> dict:
    """Resolve every output path of a ``build`` invocation before any work (Req 7.1, 7.4)."""
    output_dir = Path(args.output_dir)
    ranking = Path(args.ranking_output) if args.ranking_output else Path(str(DEFAULT_RANKING).format(seed=args.seed))
    cuts = {n: output_dir / f"{args.prefix}_top{n}.csv" for n in per_group}
    if args.provenance:
        provenance = Path(args.provenance)
    else:
        provenance = output_dir / f"{args.prefix}_{'_'.join(f'top{n}' for n in per_group)}.json"
    return {"ranking": ranking, "cuts": cuts, "provenance": provenance}


def _load_error(path: Path) -> str | None:
    """None if ``load_scenarios`` accepts ``path``, else the error text (Req 4.7)."""
    from tools._bootstrap import bootstrap

    bootstrap()
    from causal_planner.data.scenarios import load_scenarios

    try:
        load_scenarios(Path(path))
    except Exception as error:
        return f"{type(error).__name__}: {error}"
    return None


def build(args) -> int:
    """Rank, split, cut and write the ranking, cuts and provenance; return an exit code."""
    # 1. Validate parameters before touching any file (Req 3.5, 5.4).
    try:
        ratios = tuple(float(r) for r in args.split_ratios)
        validate_ratios(ratios)
        per_group = _per_group_values(args.per_group)
    except ValueError as error:
        return _error(str(error))

    # 2. Resolve and vet every target before any read or write (Req 7.1, 7.3).
    targets = build_targets(args, per_group)
    all_paths = [targets["ranking"], *targets["cuts"].values(), targets["provenance"]]
    keys = [_path_key(p) for p in all_paths]
    if len(set(keys)) != len(keys):
        return _error("output paths must be distinct: " + ", ".join(p.as_posix() for p in all_paths))
    protected = [p for p in all_paths if _protected(p)]
    existing = [p for p in all_paths if p.exists()]
    refused = protected + [p for p in existing if p not in protected]
    if refused:
        print(
            "Refusing to overwrite: " + ", ".join(_display_path(p) for p in refused)
            + "; nothing was overwritten",
            file=sys.stderr,
        )
        return EXIT_REFUSED

    # 3. Pure core (Req 1.1-1.3, 2.1-2.7, 3.1-3.4, 4.1-4.5, 5.1).
    eligibility = Path(args.eligibility)
    try:
        candidates, version = read_candidates(eligibility)
        kept, mini_excluded = drop_mini(candidates)
        ranked = rank_and_filter(kept, args.seed)
        drives_by_city = {
            str(city): rows["source_recording"].astype(str).unique().tolist()
            for city, rows in ranked.groupby(ranked["source_city"].astype(str))
        } if not ranked.empty else {}
        splits = assign_splits(drives_by_city, args.seed, ratios)
        manifest = to_manifest(ranked, splits)
        cuts = {n: make_cut(manifest, n) for n in per_group}
        eligibility_sha = file_sha256(eligibility)
        eligibility_bytes = eligibility.stat().st_size
    except (InputError, ValueError, OSError) as error:
        return _error(str(error))

    # 4. Write ranking, cuts, then provenance last (Req 7.2).
    ranking_sha = write_csv_atomic(manifest, targets["ranking"])
    cut_sha = {n: write_csv_atomic(cuts[n], targets["cuts"][n]) for n in per_group}
    provenance = build_provenance(
        generated_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        git_commit=git_commit(),
        seed=args.seed,
        split_ratios=ratios,
        eligibility_path=_display_path(eligibility),
        eligibility_sha256=eligibility_sha,
        eligibility_bytes=eligibility_bytes,
        eligibility_version=version,
        candidates=kept,
        mini_excluded_rows=mini_excluded,
        ranking=manifest,
        ranking_path=_display_path(targets["ranking"]),
        ranking_sha256=ranking_sha,
        ranking_committed=_inside_project(targets["ranking"]),
        cuts=cuts,
        cut_paths={n: _display_path(p) for n, p in targets["cuts"].items()},
        cut_sha256=cut_sha,
    )
    write_json_atomic(provenance, targets["provenance"])

    # 5. Every written manifest must load with the scenario loader (Req 4.7).
    failed = False
    for path in [targets["ranking"], *targets["cuts"].values()]:
        reason = _load_error(path)
        if reason is not None:
            print(f"[FAIL] scenario loader accepts {_display_path(path)}: {reason}")
            failed = True
    if failed:
        return EXIT_FAIL

    groups = provenance["groups"]
    summary = {
        "rows": int(len(manifest)),
        "drives": int(manifest["source_recording"].nunique()),
        "groups_non_empty": sum(1 for g in groups if g["ranked"] > 0),
        "mini_excluded_rows": int(mini_excluded),
        "cuts": {
            str(n): {
                "path": _display_path(targets["cuts"][n]),
                "rows": int(len(cuts[n])),
                "shortfall_groups": sum(1 for g in groups if g["cuts"][str(n)]["shortfall"] > 0),
            }
            for n in per_group
        },
        "ranking": _display_path(targets["ranking"]),
        "provenance": _display_path(targets["provenance"]),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return EXIT_OK


def check(args) -> int:
    """PLACEHOLDER until task 8.4 implements the check subcommand."""
    return _error("check is not implemented yet")


COMMANDS = {"build": build, "check": check}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="freeze the ranking, cuts and provenance")
    b.add_argument("--eligibility", type=Path, default=DEFAULT_ELIGIBILITY)
    b.add_argument("--seed", type=int, default=DEFAULT_SEED)
    b.add_argument("--split-ratios", type=float, nargs=3, default=list(DEFAULT_RATIOS),
                   metavar=("TRAIN", "VAL", "TEST"))
    b.add_argument("--per-group", type=int, action="append", default=None, metavar="N",
                   help="per-group count of a cut; repeatable (default: 5)")
    b.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    b.add_argument("--prefix", default=DEFAULT_PREFIX)
    b.add_argument("--ranking-output", type=Path, default=None,
                   help=f"full ranking CSV (default: {DEFAULT_RANKING.as_posix()})")
    b.add_argument("--provenance", type=Path, default=None)

    c = sub.add_parser("check", help="re-validate a frozen manifest")
    c.add_argument("--manifest", type=Path, required=True)
    c.add_argument("--provenance", type=Path, default=None)
    c.add_argument("--eligibility", type=Path, default=DEFAULT_ELIGIBILITY)

    args = parser.parse_args(argv)
    return COMMANDS[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
