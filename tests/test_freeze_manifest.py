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


# --- Example tests for the overlap filter (Req 2.2, 2.3) ---------------------------------

EXAMPLE_SEED = 7
EXAMPLE_CITY = "pittsburgh"
DRIVE_A = "2021.06.10.12.00.00_veh-26"
DRIVE_B = "2021.06.11.12.00.00_veh-27"


def row(name: str, start_s: int, end_s: int, *, scenario_type: str = BENCHMARK_TYPES[0],
        drive: str = DRIVE_A, city: str = EXAMPLE_CITY) -> dict:
    """One candidate row with all READ_COLUMNS; window ``[start_s, end_s)`` in seconds from T0."""
    log_name = f"{drive}_00001_00100"
    token = hashlib_token(name)
    start = T0 + start_s * SECOND
    return {
        "scenario_id": f"{log_name}:{token}",
        "log_name": log_name,
        "db_file": f"{log_name}.db",
        "scenario_token": token,
        "anchor_timestamp_us": start,
        "benchmark_scenario_type": scenario_type,
        "all_scenario_tags": scenario_type,
        "map_name": MAP_NAMES[city],
        "source_recording": drive,
        "window_start_us": start,
        "window_end_us": T0 + end_s * SECOND,
        "source_city": city,
        "benchmark_candidate": True,
        "shares_mini_recording": False,
        "in_mini_split": False,
        "eligibility_version": VERSION,
    }


def hashlib_token(name: str) -> str:
    import hashlib

    return hashlib.sha256(name.encode()).hexdigest()[:16]


def frame_of(*rows: dict) -> pd.DataFrame:
    frame = pd.DataFrame(list(rows), columns=READ_COLUMNS)
    for column in INT_COLUMNS:
        frame[column] = frame[column].astype("int64")
    for column in BOOL_COLUMNS:
        frame[column] = frame[column].astype(bool)
    return frame


def test_overlap_filter_accepts_touching_windows():
    first, second = row("a", 0, 10), row("b", 10, 20)
    ranking = rank_and_filter(frame_of(first, second), EXAMPLE_SEED)
    assert set(ranking["scenario_id"]) == {first["scenario_id"], second["scenario_id"]}
    assert sorted(ranking["group_rank"]) == [1, 2]


def test_overlap_filter_rejects_nested_window():
    from tools.freeze_manifest import rank_key

    outer, inner = row("outer", 0, 20), row("inner", 5, 10)
    ranking = rank_and_filter(frame_of(outer, inner), EXAMPLE_SEED)
    # Whichever ranks first claims the drive time; the other is contained in or contains it.
    winner = min((outer, inner), key=lambda r: (rank_key(EXAMPLE_SEED, r["scenario_id"]),
                                                  r["scenario_id"]))
    assert list(ranking["scenario_id"]) == [winner["scenario_id"]]
    assert list(ranking["group_rank"]) == [1]


def test_overlap_filter_rarer_type_claims_window_first():
    from tools.freeze_manifest import rank_key

    rare_type, common_type = BENCHMARK_TYPES[0], BENCHMARK_TYPES[1]
    commons = [row(f"common-{i}", 0, 0, scenario_type=common_type) for i in range(2)]
    best, other = sorted(commons, key=lambda r: (rank_key(EXAMPLE_SEED, r["scenario_id"]),
                                                 r["scenario_id"]))
    # The common type's best-ranked candidate overlaps the rare type's only candidate.
    best.update(window_start_us=T0, window_end_us=T0 + 15 * SECOND, anchor_timestamp_us=T0)
    other.update(window_start_us=T0 + 100 * SECOND, window_end_us=T0 + 115 * SECOND,
                 anchor_timestamp_us=T0 + 100 * SECOND)
    rare = row("rare", 5, 20, scenario_type=rare_type)
    # Same window as the rare one but in another drive: drives do not interact.
    elsewhere = row("elsewhere", 5, 20, scenario_type=common_type, drive=DRIVE_B)

    ranking = rank_and_filter(frame_of(best, other, rare, elsewhere), EXAMPLE_SEED)
    accepted = set(ranking["scenario_id"])
    assert rare["scenario_id"] in accepted
    assert best["scenario_id"] not in accepted
    assert other["scenario_id"] in accepted
    assert elsewhere["scenario_id"] in accepted
    # Rare group is processed first and keeps rank 1; common ranks stay dense after the skip.
    by_id = ranking.set_index("scenario_id")["group_rank"]
    assert by_id[rare["scenario_id"]] == 1
    common_ranks = ranking.loc[ranking["benchmark_scenario_type"] == common_type, "group_rank"]
    assert sorted(common_ranks) == [1, 2]
    assert ranking["scenario_id"].iloc[0] == rare["scenario_id"]


def random_ratios(rng: random.Random) -> tuple[float, float, float]:
    """Valid (train, val, test) ratios: the default, ones with a zero component, or all positive.

    Normalized so the sum is 1 within 1e-9: the largest component absorbs the rounding residue,
    so zero components stay exactly 0.
    """
    kind = rng.randrange(4)
    if kind == 0:
        return RATIOS
    weights = [rng.uniform(0.01, 1.0) for _ in range(3)]
    if kind == 1:
        weights[rng.randrange(3)] = 0.0
    elif kind == 2 and rng.random() < 0.5:
        zeros = rng.sample(range(3), 2)
        for index in zeros:
            weights[index] = 0.0
    total = sum(weights)
    ratios = [weight / total for weight in weights]
    largest = max(range(3), key=lambda index: ratios[index])
    ratios[largest] = 1.0 - sum(ratios[index] for index in range(3) if index != largest)
    assert abs(sum(ratios) - 1.0) <= 1e-9 and min(ratios) >= 0.0
    return tuple(ratios)


def random_drives(rng: random.Random) -> dict[str, list[str]]:
    """1-4 cities from CITIES, each with 0-250 unique drives (unique across cities too).

    About a third of the cities get 0-4 drives so the small-n branch of ``split_counts`` is hit.
    """
    cities = rng.sample(CITIES, rng.randint(1, 4))
    return {
        city: [f"{city}-drive-{index:03d}-veh-{rng.randrange(10, 60)}"
               for index in range(rng.randint(0, 4) if rng.random() < 1 / 3
                                  else rng.randint(0, 250))]
        for city in cities
    }


@pytest.mark.parametrize("seed", SEEDS)
def test_property_8_split_partition_and_coverage(seed):
    """Feature: ranked-city-manifest, Property 8: Split partition and coverage

    **Validates: Requirements 3.1, 3.2, 3.3**
    """
    from tools.freeze_manifest import SPLITS, assign_splits, split_counts

    rng = random.Random(seed)
    drives = random_drives(rng)
    ratios = random_ratios(rng)
    splits = assign_splits(drives, seed, ratios)

    # Every drive gets exactly one split, and nothing else is assigned.
    all_drives = [drive for values in drives.values() for drive in values]
    assert set(splits) == set(all_drives)
    assert set(splits.values()) <= set(SPLITS)

    for city, values in drives.items():
        n = len(values)
        counts = tuple(sum(splits[drive] == split for drive in values) for split in SPLITS)
        assert sum(counts) == n
        assert counts == split_counts(n, ratios)
        for split, ratio, count in zip(SPLITS, ratios, counts):
            if ratio == 0:
                assert count == 0, (city, n, ratios, split)
            elif n >= 3:
                assert count >= 1, (city, n, ratios, split)


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (0, (0, 0, 0)),
        (1, (1, 0, 0)),
        (2, (2, 0, 0)),
        (3, (1, 1, 1)),
        (4, (2, 1, 1)),
        (7, (5, 1, 1)),
        (98, (68, 15, 15)),     # las_vegas
        (193, (135, 29, 29)),   # pittsburgh
        (208, (146, 31, 31)),   # boston
        (329, (231, 49, 49)),   # singapore
    ],
)
def test_split_counts_default_ratios(n, expected):
    from tools.freeze_manifest import split_counts

    assert split_counts(n, (0.70, 0.15, 0.15)) == expected


@pytest.mark.parametrize(
    ("n", "ratios", "expected"),
    [
        (1, (0.0, 0.5, 0.5), (0, 1, 0)),
        (2, (0.0, 0.5, 0.5), (0, 1, 1)),
        (1, (0.5, 0.0, 0.5), (1, 0, 0)),
        (2, (0.5, 0.0, 0.5), (1, 0, 1)),
        (1, (1.0, 0.0, 0.0), (1, 0, 0)),
        (2, (1.0, 0.0, 0.0), (2, 0, 0)),
        (1, (0.0, 0.0, 1.0), (0, 0, 1)),
        (7, (0.0, 0.5, 0.5), (0, 4, 3)),
        (7, (0.8, 0.2, 0.0), (6, 1, 0)),
        (208, (1.0, 0.0, 0.0), (208, 0, 0)),
    ],
)
def test_split_counts_zero_ratio_gets_no_drives(n, ratios, expected):
    from tools.freeze_manifest import split_counts

    counts = split_counts(n, ratios)
    assert counts == expected
    assert sum(counts) == n
    assert all(count == 0 for count, ratio in zip(counts, ratios) if ratio == 0)


def random_unit_weights(rng: random.Random, k: int) -> list[float]:
    """``k`` non-negative values summing to 1 within 1e-9 (last one absorbs the residue)."""
    weights = [rng.uniform(0.01, 1.0) for _ in range(k)]
    total = sum(weights)
    values = [weight / total for weight in weights]
    values[-1] = 1.0 - sum(values[:-1])
    return values


@pytest.mark.parametrize("seed", SEEDS)
def test_property_10_ratio_validation(seed):
    """Feature: ranked-city-manifest, Property 10: Ratio validation

    **Validates: Requirements 3.5**
    """
    from tools.freeze_manifest import validate_ratios

    rng = random.Random(seed)
    valid = list(random_ratios(rng))
    largest = max(range(3), key=lambda index: valid[index])

    # Valid: non-negative and summing to 1, also when perturbed within the 1e-9 tolerance.
    validate_ratios(tuple(valid))
    validate_ratios(valid)
    nudged = list(valid)
    nudged[largest] += rng.uniform(-5e-10, 5e-10)
    validate_ratios(tuple(nudged))

    # Negative component while the sum stays exactly 1: only negativity is wrong.
    negative = list(valid)
    index = rng.randrange(3)
    other = (index + rng.randint(1, 2)) % 3
    amount = rng.uniform(1e-6, 0.5)
    negative[index] = -amount
    negative[other] = 1.0 - sum(negative[i] for i in range(3) if i != other)
    assert abs(sum(negative) - 1.0) <= 1e-9
    with pytest.raises(ValueError):
        validate_ratios(tuple(negative))

    # Non-negative but the sum is off by more than 1e-9 in either direction.
    bad_sum = list(valid)
    delta = 10 ** rng.uniform(-8, -0.5)
    if rng.random() < 0.5:
        bad_sum[rng.randrange(3)] += delta
    else:
        bad_sum[largest] -= min(delta, valid[largest] / 2)
    assert abs(sum(bad_sum) - 1.0) > 1e-9 and min(bad_sum) >= 0.0
    with pytest.raises(ValueError):
        validate_ratios(tuple(bad_sum))

    # Non-finite component.
    non_finite = list(valid)
    non_finite[rng.randrange(3)] = rng.choice([float("nan"), float("inf"), float("-inf")])
    with pytest.raises(ValueError):
        validate_ratios(tuple(non_finite))

    # Wrong length, even though the values are non-negative and sum to 1.
    with pytest.raises(ValueError):
        validate_ratios(tuple(random_unit_weights(rng, rng.choice([2, 4]))))


@pytest.mark.parametrize("seed", SEEDS)
def test_property_11_manifest_format_contract(seed, tmp_path):
    """Feature: ranked-city-manifest, Property 11: Manifest format contract

    Odd seeds upper-case the input tokens to show ``to_manifest`` lowercases them.

    **Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 8.2**
    """
    import hashlib

    from tools.freeze_manifest import (
        MANIFEST_COLUMNS, assign_splits, to_manifest, write_csv_atomic,
    )

    src = str(Path(__file__).resolve().parents[1] / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from causal_planner.data.scenarios import load_scenarios

    frame = make_candidates(seed)
    if seed % 2:
        frame["scenario_token"] = frame["scenario_token"].str.upper()
    ranking, _ = build_ranking(frame, seed)
    splits = assign_splits(drives_by_city(ranking), seed, RATIOS)
    manifest = to_manifest(ranking, splits)

    assert list(manifest.columns) == MANIFEST_COLUMNS
    assert len(manifest) == len(ranking)

    source = ranking.set_index("scenario_id")
    by_id = source.loc[manifest["scenario_id"]]
    assert (manifest["scenario_type"].to_numpy() == by_id["benchmark_scenario_type"].to_numpy()).all()
    assert (manifest["scenario_token"].to_numpy()
            == by_id["scenario_token"].str.lower().to_numpy()).all()
    assert manifest["scenario_token"].str.fullmatch(r"[0-9a-f]{16}").all()
    assert (manifest["db_file"] == manifest["log_name"] + ".db").all()
    assert manifest["scenario_id"].is_unique
    assert (manifest["split"] == manifest["source_recording"].map(splits)).all()

    keys = list(zip(manifest["source_city"], manifest["scenario_type"], manifest["group_rank"]))
    assert keys == sorted(keys)

    data = manifest.to_csv(index=False, lineterminator="\n").encode("utf-8")
    assert b"\r" not in data
    assert data.split(b"\n", 1)[0].decode() == ",".join(MANIFEST_COLUMNS)

    # The real writer produces the same bytes and reports their hash.
    path = tmp_path / "manifest.csv"
    digest = write_csv_atomic(manifest, path)
    written = path.read_bytes()
    assert written == data
    assert digest == hashlib.sha256(written).hexdigest()

    if manifest.empty:
        return  # load_scenarios rejects empty manifests by design.
    loaded = load_scenarios(path)
    assert [r["scenario_id"] for r in loaded] == list(manifest["scenario_id"])
    assert list(loaded[0]) == MANIFEST_COLUMNS


@pytest.mark.parametrize("seed", SEEDS)
def test_property_6_cut_nesting_and_exact_prefix(seed):
    """Feature: ranked-city-manifest, Property 6: Cut nesting and exact prefix

    **Validates: Requirements 2.6, 5.1**
    """
    from tools.freeze_manifest import assign_splits, make_cut, to_manifest

    frame = make_candidates(seed)
    ranking, _ = build_ranking(frame, seed)
    manifest = to_manifest(ranking, assign_splits(drives_by_city(ranking), seed, RATIOS))

    rng = random.Random(seed)
    n1, n2 = sorted(rng.sample(range(1, 41), 2))
    cut1, cut2 = make_cut(manifest, n1), make_cut(manifest, n2)

    # Exact prefix: the rows with group_rank <= N, same values, manifest order kept.
    for n, cut in ((n1, cut1), (n2, cut2)):
        expected = manifest.loc[manifest["group_rank"] <= n].reset_index(drop=True)
        pd.testing.assert_frame_equal(cut, expected)
        assert list(cut.columns) == list(manifest.columns)

    # Nesting: cut(N1) is the group_rank <= N1 part of cut(N2), identical row values and order.
    assert set(cut1["scenario_id"]) <= set(cut2["scenario_id"])
    nested = cut2.loc[cut2["group_rank"] <= n1].reset_index(drop=True)
    pd.testing.assert_frame_equal(cut1, nested)

    # N at or above the largest group size returns the whole manifest.
    if not manifest.empty:
        largest = int(manifest["group_rank"].max())
        pd.testing.assert_frame_equal(make_cut(manifest, largest),
                                      manifest.reset_index(drop=True))

    # Invalid N: below 1, non-integers, and bools.
    for bad in (0, -rng.randint(1, 40), 1.0, float(n1), 2.5, "3", None, True, False):
        with pytest.raises(ValueError):
            make_cut(manifest, bad)


@pytest.mark.parametrize("seed", SEEDS)
def test_property_9_splits_independent_of_cut(seed):
    """Feature: ranked-city-manifest, Property 9: Splits are independent of the cut

    **Validates: Requirements 3.4, 8.6**
    """
    from tools.freeze_manifest import assign_splits, make_cut, to_manifest

    frame = make_candidates(seed)
    ranking, _ = build_ranking(frame, seed)
    manifest = to_manifest(ranking, assign_splits(drives_by_city(ranking), seed, RATIOS))

    # One split per drive in the full manifest.
    assert (manifest.groupby("source_recording")["split"].nunique() <= 1).all()
    full_split = manifest.groupby("source_recording")["split"].first().to_dict()

    rng = random.Random(seed)
    for n in rng.sample(range(1, 41), 5):
        cut = make_cut(manifest, n)
        # Every row keeps the split its drive has in the full manifest.
        assert (cut["split"] == cut["source_recording"].map(full_split)).all(), n
        # One split per drive inside the cut.
        assert (cut.groupby("source_recording")["split"].nunique() <= 1).all(), n


def manifest_bytes(frame: pd.DataFrame, seed: int, cut: int | None = None) -> bytes:
    """CSV bytes of the manifest (or of ``make_cut(manifest, cut)``) built from ``frame``."""
    from tools.freeze_manifest import assign_splits, make_cut, to_manifest

    ranking, _ = build_ranking(frame, seed)
    manifest = to_manifest(ranking, assign_splits(drives_by_city(ranking), seed, RATIOS))
    if cut is not None:
        manifest = make_cut(manifest, cut)
    return manifest.to_csv(index=False, lineterminator="\n").encode("utf-8")


@pytest.mark.parametrize("seed", SEEDS)
def test_property_5_determinism(seed):
    """Feature: ranked-city-manifest, Property 5: Determinism

    **Validates: Requirements 2.5**
    """
    frame = make_candidates(seed)
    # Row-shuffled copy with a non-default (shuffled, offset) index.
    shuffled = frame.sample(frac=1.0, random_state=seed + 1)
    shuffled.index = shuffled.index * 7 + 1000

    full = manifest_bytes(frame, seed)
    assert manifest_bytes(frame, seed) == full
    assert manifest_bytes(shuffled, seed) == full

    cut = manifest_bytes(frame, seed, cut=5)
    assert manifest_bytes(frame, seed, cut=5) == cut
    assert manifest_bytes(shuffled, seed, cut=5) == cut


@pytest.mark.parametrize("seed", SEEDS)
def test_property_12_group_table_and_shortfall_accounting(seed):
    """Feature: ranked-city-manifest, Property 12: Group table and shortfall accounting

    **Validates: Requirements 5.2, 5.3**
    """
    from collections import Counter

    from tools.freeze_manifest import assign_splits, group_table, make_cut, to_manifest

    frame = make_candidates(seed)
    kept, _ = drop_mini(frame[frame["benchmark_candidate"]])
    ranking = rank_and_filter(kept, seed)
    manifest = to_manifest(ranking, assign_splits(drives_by_city(ranking), seed, RATIOS))

    rng = random.Random(seed)
    ns = rng.sample(range(1, 41), rng.randint(1, 3))
    table = group_table(kept, manifest, {n: make_cut(manifest, n) for n in ns})

    candidate_sizes = Counter(zip(kept["source_city"], kept["benchmark_scenario_type"]))
    ranked_sizes = Counter(zip(manifest["source_city"], manifest["scenario_type"]))

    pairs = [(city, kind) for city in sorted(CITIES) for kind in sorted(BENCHMARK_TYPES)]
    assert len(pairs) == 56
    assert [(entry["source_city"], entry["scenario_type"]) for entry in table] == pairs

    for entry, pair in zip(table, pairs):
        ranked = ranked_sizes.get(pair, 0)
        assert entry["candidates"] == candidate_sizes.get(pair, 0)
        assert entry["ranked"] == ranked
        assert set(entry["cuts"]) == {str(n) for n in ns}
        for n in ns:
            cut = entry["cuts"][str(n)]
            assert cut["selected"] == min(n, ranked), (pair, n)
            assert cut["shortfall"] == n - cut["selected"], (pair, n)


def assert_plain_types(value, path: str = "provenance") -> None:
    """Fail unless ``value`` is built only from dict/list/str/int/float/bool/None."""
    if isinstance(value, dict):
        for key, item in value.items():
            assert type(key) is str, (path, key)
            assert_plain_types(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            assert_plain_types(item, f"{path}[{index}]")
    else:
        assert value is None or type(value) in (str, int, float, bool), (path, type(value))


@pytest.mark.parametrize("seed", SEEDS)
def test_property_13_provenance_counts_are_consistent(seed):
    """Feature: ranked-city-manifest, Property 13: Provenance counts are consistent

    **Validates: Requirements 3.6, 6.2, 6.3**
    """
    import json

    from tools.freeze_manifest import (
        assign_splits, build_provenance, make_cut, manifest_counts, split_coverage, to_manifest,
    )

    frame = make_candidates(seed)
    kept, mini_excluded = drop_mini(frame[frame["benchmark_candidate"]])
    ranking = rank_and_filter(kept, seed)
    manifest = to_manifest(ranking, assign_splits(drives_by_city(ranking), seed, RATIOS))
    n = random.Random(seed).randint(1, 40)
    cut = make_cut(manifest, n)

    for current in (manifest, cut):
        counts = manifest_counts(current)
        for name, column in (("rows_by_split_city", "source_city"),
                             ("rows_by_split_type", "scenario_type")):
            table = counts[name]
            assert set(table) == {"train", "val", "test"}, name
            assert sum(sum(row.values()) for row in table.values()) == len(current), name
            for split, row in table.items():
                part = current.loc[current["split"] == split]
                expected = part[column].value_counts().to_dict()
                assert {k: v for k, v in row.items() if v} == expected, (name, split)

        drives = counts["drives_by_split_city"]
        assert set(drives) == {"train", "val", "test"}
        assert (sum(sum(row.values()) for row in drives.values())
                == current["source_recording"].nunique())
        for split, row in drives.items():
            part = current.loc[current["split"] == split]
            expected = part.groupby("source_city")["source_recording"].nunique().to_dict()
            assert {k: v for k, v in row.items() if v} == expected, split

        coverage = split_coverage(current)
        assert set(coverage) == {"train", "val", "test"}
        for split, entry in coverage.items():
            missing = entry["groups_without_rows"]
            assert entry["groups_with_rows"] + len(missing) == 56, split
            assert missing == sorted(missing), split
            part = current.loc[current["split"] == split]
            present = {f"{c}/{t}" for c, t in zip(part["source_city"], part["scenario_type"])}
            assert entry["groups_with_rows"] == len(present), split
            for label in missing:
                city, kind = label.split("/")
                assert city in CITIES and kind in BENCHMARK_TYPES, label
                assert label not in present, label

    provenance = build_provenance(
        generated_utc="2024-01-01T00:00:00Z",
        git_commit=None if seed % 2 else "0" * 40,
        seed=seed,
        split_ratios=RATIOS,
        eligibility_path="docs/data/trainval_eligibility.parquet",
        eligibility_sha256="a" * 64,
        eligibility_bytes=1234,
        eligibility_version="v1",
        candidates=kept,
        mini_excluded_rows=mini_excluded,
        ranking=manifest,
        ranking_path="configs/scenarios/ranking.csv",
        ranking_sha256="b" * 64,
        ranking_committed=False,
        cuts={n: cut},
        cut_paths={n: f"configs/scenarios/cut_{n}.csv"},
        cut_sha256={n: "c" * 64},
    )
    assert_plain_types(provenance)
    json.dumps(provenance, sort_keys=True)
    assert provenance["eligibility"]["candidate_rows"] == len(kept) + mini_excluded
    assert provenance["ranking"]["rows"] == len(manifest)
    assert provenance["manifests"][f"configs/scenarios/cut_{n}.csv"]["rows"] == len(cut)
    assert provenance["drives_by_split_city"] == manifest_counts(manifest)["drives_by_split_city"]


def test_single_drive_group_missing_from_two_splits():
    """A Las-Vegas-like group whose only candidate sits in one drive is covered by one split only.

    **Validates: Requirements 3.6**
    """
    from tools.freeze_manifest import SPLITS, assign_splits, split_coverage, to_manifest

    city, rare_type, common_type = "las_vegas", "high_lateral_acceleration", "changing_lane"
    drives = [f"2021.07.{day:02d}.12.00.00_veh-30" for day in (10, 11, 12)]
    rows = [row("lv-rare", 0, 15, scenario_type=rare_type, drive=drives[0], city=city)]
    for drive in drives:
        rows.append(row(f"lv-common-{drive}", 100, 115, scenario_type=common_type,
                        drive=drive, city=city))

    ranking = rank_and_filter(frame_of(*rows), EXAMPLE_SEED)
    assert len(ranking) == len(rows)
    splits = assign_splits(drives_by_city(ranking), EXAMPLE_SEED, RATIOS)
    # Three drives at 70/15/15 give one drive per split.
    assert sorted(splits.values()) == sorted(SPLITS)
    coverage = split_coverage(to_manifest(ranking, splits))

    label = f"{city}/{rare_type}"
    covered = [split for split in SPLITS if label not in coverage[split]["groups_without_rows"]]
    assert covered == [splits[drives[0]]]
    for split in SPLITS:
        assert (label in coverage[split]["groups_without_rows"]) == (split != splits[drives[0]])
        # The common type has a drive in every split, so it is never reported missing.
        assert f"{city}/{common_type}" not in coverage[split]["groups_without_rows"]


CORRUPTIONS = {
    "loader_error": "scenario loader accepts manifest",
    "row_not_in_candidates": "rows match eligibility candidates",
    "changed_window": "rows match eligibility candidates",
    "overlapping_window": "no overlapping windows within a drive",
    "rank_gap": "group_rank is 1..k per group",
    "mixed_split": "one split per drive",
    "wrong_sha": "sha256 matches provenance",
    "mini_drive": "no mini recordings",
}
CLEAN_SHA = "x" * 64


def corrupt(kind: str, manifest: pd.DataFrame, candidates: pd.DataFrame, seed: int) -> dict:
    """``run_checks`` keyword arguments with one corruption of ``kind`` applied (inputs untouched).

    Each corruption is built so it breaks only its own check (see CORRUPTIONS).
    """
    args = dict(manifest=manifest.copy(), candidates=candidates.copy(), mini_recordings=[],
                recorded_sha=CLEAN_SHA, actual_sha=CLEAN_SHA, loader_error=None)
    if kind == "loader_error":
        args["loader_error"] = "boom"
        return args
    if kind == "wrong_sha":
        args["recorded_sha"] = "y" * 64
        return args
    if manifest.empty:
        pytest.skip("empty manifest has no row to corrupt")
    rng = random.Random(seed)
    target = manifest.iloc[rng.randrange(len(manifest))]
    if kind == "row_not_in_candidates":
        # The manifest is unchanged; its row simply has no eligibility candidate any more.
        args["candidates"] = candidates.loc[candidates["scenario_id"] != target["scenario_id"]]
    elif kind == "changed_window":
        # Move one window past every other window of its drive, so it overlaps nothing.
        drive_end = manifest.loc[manifest["source_recording"] == target["source_recording"],
                                 "window_end_us"].max()
        start = int(drive_end) + 1000 * SECOND
        index = target.name
        args["manifest"].loc[index, "window_start_us"] = start
        args["manifest"].loc[index, "window_end_us"] = start + WINDOW_US
    elif kind == "overlapping_window":
        # A new row 1 s after an existing window in the same drive, in its own group (rank 1),
        # with the drive's split, and also present among the candidates.
        extra = target.copy()
        extra["scenario_id"] = f"{target['scenario_id']}-overlap"
        extra["scenario_type"] = "corrupt_overlap_type"
        extra["group_rank"] = 1
        extra["window_start_us"] = int(target["window_start_us"]) + SECOND
        extra["window_end_us"] = int(target["window_end_us"]) + SECOND
        args["manifest"] = pd.concat([manifest, extra.to_frame().T], ignore_index=True)
        candidate = candidates.loc[candidates["scenario_id"] == target["scenario_id"]].copy()
        candidate["scenario_id"] = extra["scenario_id"]
        candidate["window_start_us"] = extra["window_start_us"]
        candidate["window_end_us"] = extra["window_end_us"]
        args["candidates"] = pd.concat([candidates, candidate], ignore_index=True)
    elif kind == "rank_gap":
        group = manifest.loc[(manifest["source_city"] == target["source_city"])
                             & (manifest["scenario_type"] == target["scenario_type"])]
        top = group["group_rank"].idxmax()
        args["manifest"].loc[top, "group_rank"] = int(manifest.loc[top, "group_rank"]) + 1
    elif kind == "mixed_split":
        sizes = manifest["source_recording"].value_counts()
        drives = sorted(sizes.index[sizes >= 2])
        if not drives:
            pytest.skip("no drive with two or more rows")
        rows = manifest.index[manifest["source_recording"] == rng.choice(drives)]
        flip = rows[rng.randrange(len(rows))]
        current = manifest.loc[flip, "split"]
        args["manifest"].loc[flip, "split"] = rng.choice(
            [split for split in ("train", "val", "test") if split != current])
    elif kind == "mini_drive":
        args["mini_recordings"] = [f"unrelated-mini-drive-{seed}", target["source_recording"]]
    else:
        raise AssertionError(kind)
    return args


def clean_manifest(seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    from tools.freeze_manifest import assign_splits, to_manifest

    frame = make_candidates(seed)
    kept, _ = drop_mini(frame[frame["benchmark_candidate"]])
    ranking = rank_and_filter(kept, seed)
    return to_manifest(ranking, assign_splits(drives_by_city(ranking), seed, RATIOS)), kept


@pytest.mark.parametrize("kind", sorted(CORRUPTIONS))
@pytest.mark.parametrize("seed", SEEDS)
def test_property_14_check_detects_each_corruption(seed, kind):
    """Feature: ranked-city-manifest, Property 14: Check detects each corruption

    **Validates: Requirements 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 8.10**
    """
    from tools.freeze_manifest import CHECK_NAMES, run_checks

    manifest, kept = clean_manifest(seed)
    clean = run_checks(manifest, kept, [], CLEAN_SHA, CLEAN_SHA, None)
    assert list(clean) == list(CHECK_NAMES)
    assert all(clean.values()), clean

    results = run_checks(**corrupt(kind, manifest, kept, seed))
    assert list(results) == list(CHECK_NAMES)
    failed = [name for name, passed in results.items() if not passed]
    assert failed == [CORRUPTIONS[kind]], (kind, failed)


def test_property_14_every_corruption_is_exercised_on_many_seeds():
    """Feature: ranked-city-manifest, Property 14: Check detects each corruption

    Guards against the per-seed skips hiding a corruption kind.

    **Validates: Requirements 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 8.10**
    """
    non_empty = multi_row_drive = 0
    for seed in SEEDS:
        manifest, _ = clean_manifest(seed)
        non_empty += not manifest.empty
        multi_row_drive += bool((manifest["source_recording"].value_counts() >= 2).any())
    assert non_empty >= 80 and multi_row_drive >= 50, (non_empty, multi_row_drive)


# --- File helpers (Req 4.6, 6.4, 6.5, 7.2) --------------------------------------------------


def test_write_csv_atomic_lf_only_hash_and_input_untouched(tmp_path):
    import hashlib

    from tools.freeze_manifest import file_sha256, write_csv_atomic

    frame = frame_of(row("a", 0, 15), row("b", 20, 35))
    before = frame.copy()
    path = tmp_path / "nested" / "out.csv"
    digest = write_csv_atomic(frame, path)

    data = path.read_bytes()
    assert b"\r" not in data and data.endswith(b"\n")
    assert data.count(b"\n") == len(frame) + 1
    assert data.split(b"\n", 1)[0].decode() == ",".join(READ_COLUMNS)
    assert digest == file_sha256(path) == hashlib.sha256(data).hexdigest()
    pd.testing.assert_frame_equal(frame, before)
    assert not list(path.parent.glob("*.tmp"))


def test_write_json_atomic_sorted_keys_lf_and_trailing_newline(tmp_path):
    import json

    from tools.freeze_manifest import write_json_atomic

    obj = {"zeta": 1, "alpha": {"y": [1, 2], "b": None}, "mid": "text"}
    path = tmp_path / "provenance.json"
    write_json_atomic(obj, path)

    data = path.read_bytes()
    assert b"\r" not in data
    assert data.endswith(b"}\n") and not data.endswith(b"\n\n")
    assert data == (json.dumps(obj, indent=2, sort_keys=True) + "\n").encode("utf-8")
    assert json.loads(data) == obj
    text = data.decode("utf-8")
    assert text.index('"alpha"') < text.index('"mid"') < text.index('"zeta"')
    assert text.index('"b"') < text.index('"y"')
    assert not list(tmp_path.glob("*.tmp"))


class _FailingFrame:
    """Stands in for a DataFrame whose ``to_csv`` writes part of the output, then raises."""

    def to_csv(self, stream, **kwargs):
        stream.write("partial,output\n")
        raise RuntimeError("disk full")


def test_write_csv_atomic_removes_temp_file_on_failure(tmp_path):
    from tools.freeze_manifest import write_csv_atomic

    path = tmp_path / "out.csv"
    with pytest.raises(RuntimeError, match="disk full"):
        write_csv_atomic(_FailingFrame(), path)
    assert not path.exists()
    assert list(tmp_path.iterdir()) == []


def test_write_atomic_failure_keeps_existing_target(tmp_path):
    from tools.freeze_manifest import _write_atomic

    path = tmp_path / "out.json"
    path.write_bytes(b"old\n")

    def write(stream):
        stream.write("new")
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        _write_atomic(path, write)
    assert path.read_bytes() == b"old\n"
    assert [p.name for p in tmp_path.iterdir()] == ["out.json"]


def test_file_sha256_matches_hashlib(tmp_path):
    import hashlib

    from tools.freeze_manifest import file_sha256

    path = tmp_path / "blob.bin"
    data = bytes(range(256)) * 1000
    path.write_bytes(data)
    assert file_sha256(path) == hashlib.sha256(data).hexdigest()
    empty = tmp_path / "empty.bin"
    empty.write_bytes(b"")
    assert file_sha256(empty) == hashlib.sha256(b"").hexdigest()


def test_git_commit_returns_none_when_git_is_missing(monkeypatch):
    import tools.freeze_manifest as fm

    def missing(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(fm.subprocess, "run", missing)
    assert fm.git_commit() is None


def test_git_commit_returns_none_on_nonzero_exit(monkeypatch):
    import subprocess

    import tools.freeze_manifest as fm

    monkeypatch.setattr(fm.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(
        a[0] if a else [], 128, stdout="", stderr="fatal: not a git repository"))
    assert fm.git_commit() is None


def test_git_commit_returns_stripped_hash_on_success(monkeypatch):
    import subprocess

    import tools.freeze_manifest as fm

    commit = "0123456789abcdef0123456789abcdef01234567"
    monkeypatch.setattr(fm.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(
        a[0] if a else [], 0, stdout=commit + "\n", stderr=""))
    assert fm.git_commit() == commit


def test_read_candidates_filters_to_candidates_and_returns_version(tmp_path):
    from tools.freeze_manifest import read_candidates

    rows = [row("a", 0, 15), row("b", 20, 35), row("c", 40, 55)]
    rows[1]["benchmark_candidate"] = False
    frame = frame_of(*rows)
    frame["extra_column"] = 1
    path = tmp_path / "eligibility.parquet"
    frame.to_parquet(path, index=False)

    candidates, version = read_candidates(path)
    assert version == VERSION
    assert list(candidates.columns) == READ_COLUMNS
    assert list(candidates["scenario_id"]) == [rows[0]["scenario_id"], rows[2]["scenario_id"]]
    assert candidates["benchmark_candidate"].all()


def test_read_candidates_reports_missing_columns(tmp_path):
    from tools.freeze_manifest import InputError, read_candidates

    path = tmp_path / "eligibility.parquet"
    frame_of(row("a", 0, 15)).drop(columns=["source_city", "db_file"]).to_parquet(path, index=False)
    with pytest.raises(InputError, match="Missing columns: db_file, source_city") as info:
        read_candidates(path)
    assert info.value.missing == ["db_file", "source_city"]


def test_read_candidates_missing_file(tmp_path):
    from tools.freeze_manifest import InputError, read_candidates

    with pytest.raises(InputError, match="not found"):
        read_candidates(tmp_path / "absent.parquet")


def test_read_candidates_rejects_multiple_versions(tmp_path):
    from tools.freeze_manifest import InputError, read_candidates

    rows = [row("a", 0, 15), row("b", 20, 35)]
    rows[1]["eligibility_version"] = "eligibility-v2"
    path = tmp_path / "eligibility.parquet"
    frame_of(*rows).to_parquet(path, index=False)
    with pytest.raises(InputError, match="eligibility-v1, eligibility-v2"):
        read_candidates(path)


def test_mini_recordings_includes_non_candidate_rows(tmp_path):
    from tools.freeze_manifest import mini_recordings

    rows = [
        row("clean", 0, 15, drive=DRIVE_A),
        row("shares", 0, 15, drive=DRIVE_B),
        row("in-mini", 0, 15, drive="2021.06.12.12.00.00_veh-28"),
    ]
    rows[1]["shares_mini_recording"] = True
    rows[1]["benchmark_candidate"] = False
    rows[2]["in_mini_split"] = True
    path = tmp_path / "eligibility.parquet"
    frame_of(*rows).to_parquet(path, index=False)
    assert mini_recordings(path) == {DRIVE_B, "2021.06.12.12.00.00_veh-28"}


# --- End-to-end `build` (Req 2.5, 5.2, 5.3, 6.1, 6.4, 6.5, 7.1, 7.3, 9.1) -------------------


E2E_PER_GROUP = (2, 5)


def e2e_table() -> tuple[pd.DataFrame, set[str]]:
    """Synthetic eligibility table whose mini flags are set per drive (all rows of two drives).

    Uses the first seed whose frame has at least four drives and 40 rows, so the cuts, the mini
    guard and the shortfalls all have something to do. Returns the frame and the mini drives.
    """
    for seed in SEEDS:
        frame = make_candidates(seed, mini_rate=0)
        drives = sorted(frame["source_recording"].unique())
        if len(drives) >= 4 and len(frame) >= 40:
            break
    else:
        raise AssertionError("no seed gives a large enough synthetic table")
    mini = set(drives[:2])
    frame.loc[frame["source_recording"].isin(mini), "shares_mini_recording"] = True
    return frame, mini


def e2e_paths(root: Path) -> dict:
    output_dir = root / "out"
    return {
        "output_dir": output_dir,
        "ranking": root / "full" / "t_full.csv",
        "cuts": {n: output_dir / f"t_top{n}.csv" for n in E2E_PER_GROUP},
        "provenance": output_dir / "t_top2_top5.json",
    }


def e2e_argv(eligibility: Path, paths: dict, *extra: str) -> list[str]:
    argv = ["build", "--eligibility", str(eligibility), "--output-dir", str(paths["output_dir"]),
            "--ranking-output", str(paths["ranking"]), "--prefix", "t"]
    for n in E2E_PER_GROUP:
        argv += ["--per-group", str(n)]
    return argv + list(extra)


def all_files(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.fixture
def e2e_build(tmp_path, monkeypatch):
    """Run ``build`` once into ``tmp_path`` with ``git_commit`` failing; return the run details."""
    import tools.freeze_manifest as fm

    monkeypatch.setattr(fm, "git_commit", lambda: None)
    frame, mini = e2e_table()
    eligibility = tmp_path / "eligibility.parquet"
    frame.to_parquet(eligibility, index=False)
    paths = e2e_paths(tmp_path / "run1")
    code = fm.main(e2e_argv(eligibility, paths))
    return {"code": code, "frame": frame, "mini": mini, "eligibility": eligibility,
            "paths": paths, "tmp_path": tmp_path}


def test_build_end_to_end_writes_ranking_cuts_and_provenance(e2e_build):
    import hashlib
    import json

    from tools.freeze_manifest import MANIFEST_COLUMNS, read_manifest

    assert e2e_build["code"] == 0
    paths, frame, mini = e2e_build["paths"], e2e_build["frame"], e2e_build["mini"]
    for path in [paths["ranking"], *paths["cuts"].values(), paths["provenance"]]:
        assert path.is_file(), path
        assert b"\r" not in path.read_bytes(), path

    ranking = read_manifest(paths["ranking"])
    assert list(ranking.columns) == MANIFEST_COLUMNS
    assert not ranking.empty
    assert not set(ranking["source_recording"]) & mini
    candidates = frame[frame["benchmark_candidate"]]
    assert set(ranking["scenario_id"]) <= set(candidates["scenario_id"])
    ranks = ranking["group_rank"].astype(int)
    for _, group in ranks.groupby([ranking["source_city"], ranking["scenario_type"]]):
        assert sorted(group) == list(range(1, len(group) + 1))

    cuts = {n: read_manifest(path) for n, path in paths["cuts"].items()}
    for n, cut in cuts.items():
        expected = ranking.loc[ranks <= n].reset_index(drop=True)
        pd.testing.assert_frame_equal(cut, expected)

    text = paths["provenance"].read_bytes().decode("utf-8")
    provenance = json.loads(text)
    assert text == json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    assert provenance["git_commit"] is None
    assert provenance["per_group"] == list(E2E_PER_GROUP)

    eligibility = provenance["eligibility"]
    assert eligibility["sha256"] == hashlib.sha256(e2e_build["eligibility"].read_bytes()).hexdigest()
    assert eligibility["bytes"] == e2e_build["eligibility"].stat().st_size
    assert eligibility["version"] == VERSION
    assert eligibility["candidate_rows"] == len(candidates)
    assert eligibility["mini_excluded_rows"] == int(candidates["source_recording"].isin(mini).sum())
    assert eligibility["mini_excluded_rows"] > 0

    assert provenance["ranking"]["sha256"] == hashlib.sha256(paths["ranking"].read_bytes()).hexdigest()
    assert provenance["ranking"]["rows"] == len(ranking)
    assert provenance["ranking"]["committed"] is False
    blocks = {block["per_group"]: block for block in provenance["manifests"].values()}
    assert sorted(blocks) == list(E2E_PER_GROUP)
    for n, block in blocks.items():
        assert block["sha256"] == hashlib.sha256(paths["cuts"][n].read_bytes()).hexdigest()
        assert block["rows"] == len(cuts[n])

    # Group table: all 56 pairs, counts consistent with the written CSVs (Req 5.2, 5.3).
    groups = provenance["groups"]
    assert len(groups) == len(CITIES) * len(BENCHMARK_TYPES)
    kept = candidates[~candidates["source_recording"].isin(mini)]
    for entry in groups:
        key = (entry["source_city"], entry["scenario_type"])
        in_kept = (kept["source_city"] == key[0]) & (kept["benchmark_scenario_type"] == key[1])
        assert entry["candidates"] == int(in_kept.sum())
        in_ranking = (ranking["source_city"] == key[0]) & (ranking["scenario_type"] == key[1])
        assert entry["ranked"] == int(in_ranking.sum())
        for n, cut in cuts.items():
            selected = int(((cut["source_city"] == key[0]) & (cut["scenario_type"] == key[1])).sum())
            assert entry["cuts"][str(n)] == {"selected": selected, "shortfall": n - selected}
            assert selected == min(n, entry["ranked"])
    assert sum(entry["cuts"]["5"]["shortfall"] > 0 for entry in groups) > 0


def test_build_end_to_end_is_deterministic_across_fresh_directories(e2e_build, monkeypatch):
    import tools.freeze_manifest as fm

    assert e2e_build["code"] == 0
    monkeypatch.setattr(fm, "git_commit", lambda: None)
    first = e2e_build["paths"]
    second = e2e_paths(e2e_build["tmp_path"] / "run2")
    assert fm.main(e2e_argv(e2e_build["eligibility"], second)) == 0
    assert second["ranking"].read_bytes() == first["ranking"].read_bytes()
    for n in E2E_PER_GROUP:
        assert second["cuts"][n].read_bytes() == first["cuts"][n].read_bytes()


def test_build_end_to_end_second_run_refuses_and_leaves_files_identical(e2e_build, capsys):
    import tools.freeze_manifest as fm

    assert e2e_build["code"] == 0
    run_root = e2e_build["tmp_path"] / "run1"
    before = all_files(run_root)
    capsys.readouterr()
    assert fm.main(e2e_argv(e2e_build["eligibility"], e2e_build["paths"])) == 3
    assert "Refusing to overwrite" in capsys.readouterr().err
    assert all_files(run_root) == before


def test_build_refuses_protected_provenance_path(tmp_path, monkeypatch, capsys):
    import tools.freeze_manifest as fm

    monkeypatch.setattr(fm, "git_commit", lambda: None)
    frame, _ = e2e_table()
    eligibility = tmp_path / "eligibility.parquet"
    frame.to_parquet(eligibility, index=False)
    scenarios = fm.PROJECT_ROOT / "configs" / "scenarios"
    protected = {name: (scenarios / name).read_bytes() for name in fm.PROTECTED_FILES}
    listing = sorted(p.name for p in scenarios.iterdir())

    paths = e2e_paths(tmp_path / "run")
    target = scenarios / "pittsburgh_65_sample.json"
    assert fm.main(e2e_argv(eligibility, paths, "--provenance", str(target))) == 3
    assert "Refusing to overwrite" in capsys.readouterr().err

    assert {name: (scenarios / name).read_bytes() for name in fm.PROTECTED_FILES} == protected
    assert sorted(p.name for p in scenarios.iterdir()) == listing
    assert not (tmp_path / "run").exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["eligibility.parquet"]


# --- End-to-end `check` (Req 8.1, 8.9, 8.10, 9.1) -----------------------------------------


def run_check(manifest: Path, provenance: Path, eligibility: Path, capsys) -> tuple[int, list[str]]:
    """Run ``check`` through ``main`` and return its exit code and printed lines."""
    import tools.freeze_manifest as fm

    capsys.readouterr()
    code = fm.main(["check", "--manifest", str(manifest), "--provenance", str(provenance),
                    "--eligibility", str(eligibility)])
    return code, capsys.readouterr().out.splitlines()


def test_check_end_to_end_passes_on_fresh_output(e2e_build, capsys):
    from tools.freeze_manifest import CHECK_NAMES

    assert e2e_build["code"] == 0
    paths = e2e_build["paths"]
    expected = [f"[PASS] {name}" for name in CHECK_NAMES]
    for manifest in [*paths["cuts"].values(), paths["ranking"]]:
        code, lines = run_check(manifest, paths["provenance"], e2e_build["eligibility"], capsys)
        assert code == 0, (manifest, lines)
        assert lines == expected, manifest


def e2e_corruption(kind: str, build: dict, root: Path) -> tuple[Path, Path]:
    """Apply one corruption of ``kind`` to the top-5 cut written by ``build``.

    The corrupted CSV replaces the cut in place (provenance entries are keyed by its path). A
    provenance copy and, when needed, a modified eligibility table are written under ``root``.
    Unless the corruption is ``wrong_sha``, the copied provenance records the corrupted file's
    SHA-256, so only the intended check can fail. Returns (provenance, eligibility) paths.
    """
    import hashlib
    import json

    from tools.freeze_manifest import read_manifest

    root.mkdir()
    cut_path = build["paths"]["cuts"][5]
    frame = build["frame"].copy()
    manifest = read_manifest(cut_path)
    target = manifest.iloc[0]
    eligibility = build["eligibility"]
    new_table = None

    if kind == "row_not_in_candidates":
        new_table = frame.loc[frame["scenario_id"] != target["scenario_id"]]
    elif kind == "changed_window":
        drive = manifest["source_recording"] == target["source_recording"]
        start = int(manifest.loc[drive, "window_end_us"].astype("int64").max()) + 1000 * SECOND
        manifest.loc[0, "window_start_us"] = str(start)
        manifest.loc[0, "window_end_us"] = str(start + WINDOW_US)
    elif kind == "overlapping_window":
        extra = target.copy()
        extra["scenario_id"] = f"{target['scenario_id']}-overlap"
        extra["scenario_type"] = "corrupt_overlap_type"
        extra["group_rank"] = "1"
        extra["window_start_us"] = str(int(target["window_start_us"]) + SECOND)
        extra["window_end_us"] = str(int(target["window_end_us"]) + SECOND)
        manifest = pd.concat([manifest, extra.to_frame().T], ignore_index=True)
        candidate = frame.loc[frame["scenario_id"] == target["scenario_id"]].copy()
        assert len(candidate) == 1
        candidate["scenario_id"] = extra["scenario_id"]
        candidate["window_start_us"] = int(extra["window_start_us"])
        candidate["window_end_us"] = int(extra["window_end_us"])
        new_table = pd.concat([frame, candidate], ignore_index=True)
    elif kind == "rank_gap":
        group = manifest.index[(manifest["source_city"] == target["source_city"])
                               & (manifest["scenario_type"] == target["scenario_type"])]
        top = manifest.loc[group, "group_rank"].astype("int64").idxmax()
        manifest.loc[top, "group_rank"] = str(int(manifest.loc[top, "group_rank"]) + 1)
    elif kind == "mixed_split":
        sizes = manifest["source_recording"].value_counts()
        drives = sorted(sizes.index[sizes >= 2])
        assert drives, "the e2e cut needs a drive with two or more rows"
        flip = manifest.index[manifest["source_recording"] == drives[0]][0]
        current = manifest.loc[flip, "split"]
        manifest.loc[flip, "split"] = next(s for s in ("train", "val", "test") if s != current)
    elif kind == "mini_drive":
        # Flag the whole drive as sharing a mini recording; candidates and windows stay as they are.
        new_table = frame.copy()
        new_table.loc[new_table["source_recording"] == target["source_recording"],
                      "shares_mini_recording"] = True
    elif kind == "loader_error":
        manifest.loc[0, "scenario_token"] = "not-a-token"
    elif kind != "wrong_sha":
        raise AssertionError(kind)

    if kind not in ("row_not_in_candidates", "mini_drive", "wrong_sha"):
        cut_path.write_bytes(manifest.to_csv(index=False, lineterminator="\n").encode("utf-8"))
    if new_table is not None:
        eligibility = root / "eligibility.parquet"
        new_table.to_parquet(eligibility, index=False)

    provenance = json.loads(build["paths"]["provenance"].read_text(encoding="utf-8"))
    entries = [entry for entry in provenance["manifests"].values() if entry["per_group"] == 5]
    assert len(entries) == 1
    entries[0]["sha256"] = ("0" * 64 if kind == "wrong_sha"
                            else hashlib.sha256(cut_path.read_bytes()).hexdigest())
    provenance_path = root / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return provenance_path, eligibility


@pytest.mark.parametrize("kind", sorted(CORRUPTIONS))
def test_check_end_to_end_fails_on_each_corruption(e2e_build, kind, capsys):
    """Each Property 14 corruption applied to the written files fails exactly its own check."""
    from tools.freeze_manifest import CHECK_NAMES

    assert e2e_build["code"] == 0
    cut_path = e2e_build["paths"]["cuts"][5]
    original = cut_path.read_bytes()
    try:
        provenance, eligibility = e2e_corruption(kind, e2e_build, e2e_build["tmp_path"] / "corrupt")
        code, lines = run_check(cut_path, provenance, eligibility, capsys)
    finally:
        cut_path.write_bytes(original)

    assert code == 1, lines
    assert len(lines) == len(CHECK_NAMES), lines
    failed = [line for line in lines if line.startswith("[FAIL] ")]
    expected = CORRUPTIONS[kind]
    assert any(line.startswith(f"[FAIL] {expected}: ") for line in failed), (kind, lines)
    assert len(failed) == 1, (kind, failed)


def test_check_end_to_end_missing_manifest_is_an_error(e2e_build, capsys):
    assert e2e_build["code"] == 0
    paths = e2e_build["paths"]
    code, _ = run_check(paths["output_dir"] / "absent.csv", paths["provenance"],
                        e2e_build["eligibility"], capsys)
    assert code == 2


def test_check_end_to_end_missing_provenance_fails_sha_check(e2e_build, capsys):
    assert e2e_build["code"] == 0
    paths = e2e_build["paths"]
    code, lines = run_check(paths["cuts"][5], e2e_build["tmp_path"] / "absent.json",
                            e2e_build["eligibility"], capsys)
    assert code == 1
    failed = [line for line in lines if line.startswith("[FAIL] ")]
    assert len(failed) == 1, failed
    assert failed[0].startswith("[FAIL] sha256 matches provenance: provenance file not found")


# --- Input errors (Req 1.3, 3.5, 5.4) -------------------------------------------------------


def assert_no_build_outputs(tmp_path: Path, paths: dict) -> None:
    """Nothing was written: no output dir, ranking or provenance; only the input parquet is left."""
    assert not paths["output_dir"].exists()
    assert not paths["ranking"].exists()
    assert not paths["ranking"].parent.exists()
    assert not paths["provenance"].exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["eligibility.parquet"]


def test_build_missing_column_returns_2_and_writes_nothing(tmp_path, monkeypatch, capsys):
    import tools.freeze_manifest as fm

    monkeypatch.setattr(fm, "git_commit", lambda: None)
    frame, _ = e2e_table()
    eligibility = tmp_path / "eligibility.parquet"
    frame.drop(columns=["map_name"]).to_parquet(eligibility, index=False)
    paths = e2e_paths(tmp_path / "run")
    capsys.readouterr()
    assert fm.main(e2e_argv(eligibility, paths)) == 2
    assert "Missing columns: map_name" in capsys.readouterr().err
    assert_no_build_outputs(tmp_path, paths)


@pytest.mark.parametrize(
    "extra",
    [
        ("--split-ratios", "0.5", "0.5", "0.5"),
        ("--split-ratios", "1.2", "-0.1", "-0.1"),
        ("--per-group", "0"),
        ("--per-group", "-3"),
    ],
    ids=["ratios_sum_above_1", "negative_ratio", "per_group_0", "per_group_negative"],
)
def test_build_invalid_arguments_return_2_and_write_nothing(tmp_path, monkeypatch, extra):
    import tools.freeze_manifest as fm

    monkeypatch.setattr(fm, "git_commit", lambda: None)
    frame, _ = e2e_table()
    eligibility = tmp_path / "eligibility.parquet"
    frame.to_parquet(eligibility, index=False)
    paths = e2e_paths(tmp_path / "run")
    assert fm.main(e2e_argv(eligibility, paths, *extra)) == 2
    assert_no_build_outputs(tmp_path, paths)


def test_build_missing_eligibility_file_returns_2_and_writes_nothing(tmp_path, monkeypatch):
    import tools.freeze_manifest as fm

    monkeypatch.setattr(fm, "git_commit", lambda: None)
    paths = e2e_paths(tmp_path / "run")
    assert fm.main(e2e_argv(tmp_path / "absent.parquet", paths)) == 2
    assert not paths["output_dir"].exists()
    assert not paths["ranking"].parent.exists()
    assert list(tmp_path.iterdir()) == []
