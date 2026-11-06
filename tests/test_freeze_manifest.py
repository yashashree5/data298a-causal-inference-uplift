"""Tests for tools/freeze_manifest.py on seeded synthetic candidate frames (no file I/O unless noted).

Hypothesis is not installed, so property tests use ``make_candidates(seed)`` over
``@pytest.mark.parametrize("seed", SEEDS)``. Each property test carries the docstring tag
``Feature: ranked-city-manifest, Property <n>: <title>``.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.freeze_manifest import (  # noqa: E402
    BENCHMARK_TYPES,
    CITIES,
    READ_COLUMNS,
    drop_mini,
    rank_and_filter,
)

SEEDS = range(100)
SECOND = 1_000_000
WINDOW_US = 15 * SECOND
ANCHOR_OFFSET_US = 3 * SECOND
T0 = 1_629_000_000_000_000
VERSION = "eligibility-v1"
MAP_NAMES = {
    "boston": "us-ma-boston",
    "las_vegas": "us-nv-las-vegas-strip",
    "pittsburgh": "us-pa-pittsburgh-hazelwood",
    "singapore": "sg-one-north",
}
INT_COLUMNS = ("anchor_timestamp_us", "window_start_us", "window_end_us")
BOOL_COLUMNS = ("benchmark_candidate", "shares_mini_recording", "in_mini_split")


def make_candidates(
    seed: int,
    *,
    mini_rate: float = 0.1,
    candidate_rate: float = 0.85,
    max_cities: int = 4,
    max_types: int = 5,
    max_drives: int = 8,
    max_per_drive: int = 30,
    span_seconds: int = 120,
) -> pd.DataFrame:
    """Random eligibility-table projection with exactly ``READ_COLUMNS``, deterministic per seed.

    1..max_cities cities from CITIES and 1..max_types types from BENCHMARK_TYPES; per city
    1..max_drives drives (unique ``source_recording``), each with 1-3 logs and
    0..max_per_drive rows. Window starts fall on whole seconds in ``[0, span_seconds)`` of the
    drive, so 15 s windows overlap often and sometimes touch exactly. ``shares_mini_recording``
    is set per drive and ``in_mini_split`` per log, each with probability ``mini_rate``;
    ``benchmark_candidate`` is True per row with probability ``candidate_rate``.
    """
    rng = random.Random(seed)
    cities = sorted(rng.sample(CITIES, rng.randint(1, max_cities)))
    types = sorted(rng.sample(BENCHMARK_TYPES, rng.randint(1, max_types)))
    tokens: set[str] = set()
    rows = []
    for city_index, city in enumerate(cities):
        for drive_index in range(rng.randint(1, max_drives)):
            # Month/day encode (city, drive), so drive ids are unique within and across cities.
            drive = (
                f"2021.{city_index + 5:02d}.{drive_index + 10:02d}."
                f"{rng.randrange(24):02d}.{rng.randrange(60):02d}.{rng.randrange(60):02d}"
                f"_veh-{rng.randrange(10, 60)}"
            )
            drive_t0 = T0 + rng.randrange(10_000) * SECOND
            shares_mini = rng.random() < mini_rate
            logs = []
            for log_index in range(rng.randint(1, 3)):
                first = log_index * 1_000 + rng.randrange(1_000)
                logs.append((f"{drive}_{first:05d}_{first + rng.randrange(1, 500):05d}",
                             rng.random() < mini_rate))
            for _ in range(rng.randint(0, max_per_drive)):
                log_name, in_mini = rng.choice(logs)
                token = f"{rng.getrandbits(64):016x}"
                while token in tokens:
                    token = f"{rng.getrandbits(64):016x}"
                tokens.add(token)
                scenario_type = rng.choice(types)
                extra = rng.sample(types, rng.randint(0, len(types) - 1))
                start = drive_t0 + rng.randrange(span_seconds) * SECOND
                rows.append({
                    "scenario_id": f"{log_name}:{token}",
                    "log_name": log_name,
                    "db_file": f"{log_name}.db",
                    "scenario_token": token,
                    "anchor_timestamp_us": start + ANCHOR_OFFSET_US,
                    "benchmark_scenario_type": scenario_type,
                    "all_scenario_tags": "|".join(sorted({scenario_type, *extra})),
                    "map_name": MAP_NAMES[city],
                    "source_recording": drive,
                    "window_start_us": start,
                    "window_end_us": start + WINDOW_US,
                    "source_city": city,
                    "benchmark_candidate": rng.random() < candidate_rate,
                    "shares_mini_recording": shares_mini,
                    "in_mini_split": in_mini,
                    "eligibility_version": VERSION,
                })
    frame = pd.DataFrame(rows, columns=READ_COLUMNS)
    for column in INT_COLUMNS:
        frame[column] = frame[column].astype("int64")
    for column in BOOL_COLUMNS:
        frame[column] = frame[column].astype(bool)
    return frame


@pytest.mark.parametrize("seed", SEEDS)
def test_generator_is_deterministic_and_well_formed(seed):
    frame = make_candidates(seed)
    pd.testing.assert_frame_equal(frame, make_candidates(seed))
    assert list(frame.columns) == READ_COLUMNS
    if frame.empty:
        return
    assert frame["scenario_id"].is_unique
    assert set(frame["source_city"]) <= set(CITIES)
    assert set(frame["benchmark_scenario_type"]) <= set(BENCHMARK_TYPES)
    assert frame["source_city"].nunique() <= 4 and frame["benchmark_scenario_type"].nunique() <= 5
    assert frame["scenario_token"].str.fullmatch(r"[0-9a-f]{16}").all()
    assert (frame["scenario_id"] == frame["log_name"] + ":" + frame["scenario_token"]).all()
    assert (frame["db_file"] == frame["log_name"] + ".db").all()
    assert (frame["window_end_us"] - frame["window_start_us"] == WINDOW_US).all()
    assert (frame["anchor_timestamp_us"] - frame["window_start_us"] == ANCHOR_OFFSET_US).all()
    assert (frame.groupby("source_recording")["source_city"].nunique() == 1).all()
    assert frame.groupby("source_city")["source_recording"].nunique().between(1, 8).all()
    assert frame.groupby("source_recording").size().max() <= 30
    assert frame["eligibility_version"].nunique() == 1


def test_generator_differs_across_seeds_and_covers_flags():
    frames = [make_candidates(seed) for seed in SEEDS]
    assert len({tuple(frame["scenario_id"]) for frame in frames}) == len(frames)
    combined = pd.concat(frames, ignore_index=True)
    for column in BOOL_COLUMNS:
        assert combined[column].any() and not combined[column].all()
    # Windows overlap often within a drive: some consecutive starts are closer than 15 s.
    starts = combined.sort_values(["source_recording", "window_start_us"])
    gaps = starts.groupby("source_recording")["window_start_us"].diff().dropna()
    assert (gaps < WINDOW_US).mean() > 0.5
    assert (gaps == WINDOW_US).any()


def build_ranking(frame: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, int]:
    """Candidate step of ``build``: candidate filter (Req 1.1), mini guard (Req 1.2), ranking."""
    candidates = frame[frame["benchmark_candidate"]]
    kept, excluded = drop_mini(candidates)
    return rank_and_filter(kept, seed), excluded


@pytest.mark.parametrize("seed", SEEDS)
def test_property_1_candidate_filtering(seed):
    """Feature: ranked-city-manifest, Property 1: Candidate filtering

    **Validates: Requirements 1.1, 1.2**
    """
    frame = make_candidates(seed, mini_rate=0.3)
    ranking, excluded = build_ranking(frame, seed)

    source = frame.set_index("scenario_id")
    rows = source.loc[ranking["scenario_id"]]
    assert rows["benchmark_candidate"].all()
    assert not rows["shares_mini_recording"].any()
    assert not rows["in_mini_split"].any()

    candidates = frame[frame["benchmark_candidate"]]
    expected = int((candidates["shares_mini_recording"] | candidates["in_mini_split"]).sum())
    assert excluded == expected

    # The overlap filter can only remove rows: every clean candidate either is ranked or was
    # dropped for overlap, never for a mini or candidate reason.
    clean = candidates[~(candidates["shares_mini_recording"] | candidates["in_mini_split"])]
    assert set(ranking["scenario_id"]) <= set(clean["scenario_id"])
    assert ranking.empty == clean.empty


EXTRA_COLUMNS = (
    "scenario_type", "ego_speed_mps", "weather_condition", "lidar_pc_token", "notes",
)


@pytest.mark.parametrize("seed", SEEDS)
def test_property_2_missing_column_detection(seed):
    """Feature: ranked-city-manifest, Property 2: Missing column detection

    **Validates: Requirements 1.3**
    """
    from tools.freeze_manifest import missing_columns

    rng = random.Random(seed)
    removed = rng.sample(READ_COLUMNS, rng.randint(1, len(READ_COLUMNS)))
    remaining = [column for column in READ_COLUMNS if column not in removed]
    schema = remaining + rng.sample(EXTRA_COLUMNS, rng.randint(0, len(EXTRA_COLUMNS)))
    rng.shuffle(schema)

    assert missing_columns(schema) == sorted(removed)
    # A full schema (plus unrelated extras) reports nothing missing.
    assert missing_columns(READ_COLUMNS + list(EXTRA_COLUMNS)) == []


RATIOS = (0.70, 0.15, 0.15)


def drives_by_city(frame: pd.DataFrame) -> dict[str, list[str]]:
    return {city: sorted(group["source_recording"].unique())
            for city, group in frame.groupby("source_city")}


@pytest.mark.parametrize("seed", SEEDS)
def test_property_7_seed_sensitivity(seed):
    """Feature: ranked-city-manifest, Property 7: Seed sensitivity

    **Validates: Requirements 2.7**
    """
    import hashlib

    from tools.freeze_manifest import assign_splits, rank_key, split_key

    frame = make_candidates(seed)
    for scenario_id in frame["scenario_id"]:
        assert rank_key(seed, scenario_id) == hashlib.sha256(
            f"{seed}:{scenario_id}".encode()).hexdigest()[:16]
    for drive in frame["source_recording"].unique():
        assert split_key(seed, drive) == hashlib.sha256(
            f"{seed}:split:{drive}".encode()).hexdigest()

    # Same seed -> identical ranking and splits, also when the input rows come in another order.
    ranking, _ = build_ranking(frame, seed)
    shuffled = frame.sample(frac=1.0, random_state=seed + 1).reset_index(drop=True)
    pd.testing.assert_frame_equal(build_ranking(frame, seed)[0], ranking)
    pd.testing.assert_frame_equal(build_ranking(shuffled, seed)[0], ranking)

    drives = drives_by_city(frame)
    splits = assign_splits(drives, seed, RATIOS)
    assert assign_splits(drives, seed, RATIOS) == splits
    reversed_drives = {city: list(reversed(values)) for city, values in reversed(drives.items())}
    assert assign_splits(reversed_drives, seed, RATIOS) == splits


def test_property_7_different_seeds_usually_change_output():
    """Feature: ranked-city-manifest, Property 7: Seed sensitivity

    Aggregated over the whole seed range so no single case depends on a hash coincidence.

    **Validates: Requirements 2.7**
    """
    from tools.freeze_manifest import assign_splits

    ranking_trials = ranking_changes = split_trials = split_changes = 0
    for seed in SEEDS:
        frame = make_candidates(seed)
        kept, _ = drop_mini(frame[frame["benchmark_candidate"]])
        sizes = kept.groupby(["source_city", "benchmark_scenario_type"]).size()
        if not sizes.empty and sizes.max() >= 3:
            ranking_trials += 1
            first = list(rank_and_filter(kept, seed)["scenario_id"])
            second = list(rank_and_filter(kept, seed + 1)["scenario_id"])
            ranking_changes += first != second

        drives = drives_by_city(frame)
        if drives and max(len(values) for values in drives.values()) >= 4:
            split_trials += 1
            split_changes += (assign_splits(drives, seed, RATIOS)
                              != assign_splits(drives, seed + 1, RATIOS))

    assert ranking_trials >= 50 and ranking_changes >= ranking_trials // 2
    assert split_trials >= 20 and split_changes >= split_trials // 4


@pytest.mark.parametrize("seed", SEEDS)
def test_property_3_ranking_order_and_dense_group_ranks(seed):
    """Feature: ranked-city-manifest, Property 3: Ranking order and dense group ranks

    **Validates: Requirements 2.1, 2.4, 8.5**
    """
    from tools.freeze_manifest import rank_key

    frame = make_candidates(seed)
    ranking, _ = build_ranking(frame, seed)
    assert ranking["group_rank"].dtype == "int64"

    for _, group in ranking.groupby(["source_city", "benchmark_scenario_type"]):
        ordered = group.sort_values("group_rank", kind="mergesort")
        assert list(ordered["group_rank"]) == list(range(1, len(ordered) + 1))
        keys = [(rank_key(seed, scenario_id), scenario_id)
                for scenario_id in ordered["scenario_id"]]
        assert keys == sorted(keys)


def reference_rank_and_filter(candidates: pd.DataFrame, seed: int) -> list[tuple[str, int]]:
    """Naive model of the overlap filter: plain loops and a pairwise half-open overlap check.

    Returns ``(scenario_id, group_rank)`` for accepted rows in processing order.
    """
    import hashlib

    rows = candidates.to_dict("records")
    accepted_windows: dict[str, list[tuple[int, int]]] = {}
    result: list[tuple[str, int]] = []
    for city in sorted({row["source_city"] for row in rows}):
        city_rows = [row for row in rows if row["source_city"] == city]
        counts: dict[str, int] = {}
        for row in city_rows:
            counts[row["benchmark_scenario_type"]] = counts.get(row["benchmark_scenario_type"], 0) + 1
        for scenario_type in sorted(counts, key=lambda t: (counts[t], t)):
            group = [row for row in city_rows if row["benchmark_scenario_type"] == scenario_type]
            group.sort(key=lambda row: (
                hashlib.sha256(f"{seed}:{row['scenario_id']}".encode()).hexdigest()[:16],
                row["scenario_id"],
            ))
            rank = 0
            for row in group:
                start, end = int(row["window_start_us"]), int(row["window_end_us"])
                windows = accepted_windows.setdefault(row["source_recording"], [])
                overlaps = False
                for other_start, other_end in windows:
                    if start < other_end and other_start < end:
                        overlaps = True
                        break
                if overlaps:
                    continue
                windows.append((start, end))
                rank += 1
                result.append((row["scenario_id"], rank))
    return result


@pytest.mark.parametrize("seed", SEEDS)
def test_property_4_overlap_filter_equals_reference_model(seed):
    """Feature: ranked-city-manifest, Property 4: Overlap filter equals the reference model

    **Validates: Requirements 2.2, 2.3, 8.4**
    """
    frame = make_candidates(seed)
    kept, _ = drop_mini(frame[frame["benchmark_candidate"]])
    ranking = rank_and_filter(kept, seed)

    expected = reference_rank_and_filter(kept, seed)
    assert list(ranking["scenario_id"]) == [scenario_id for scenario_id, _ in expected]
    assert list(ranking["group_rank"]) == [rank for _, rank in expected]

    # No two accepted windows in one drive overlap (pairwise, half-open).
    for _, drive in ranking.groupby("source_recording"):
        windows = list(zip(drive["window_start_us"], drive["window_end_us"]))
        for i in range(len(windows)):
            for j in range(i + 1, len(windows)):
                (a_start, a_end), (b_start, b_end) = windows[i], windows[j]
                assert not (a_start < b_end and b_start < a_end)
