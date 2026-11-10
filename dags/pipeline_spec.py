"""Task graph and shell commands of the nuplan_pipeline DAG.

Plain Python (no Airflow import) so the graph and the commands can be unit-tested
on any machine. ``nuplan_pipeline.py`` turns every ``Step`` into an Airflow task.

Every enabled data step runs its tool's ``build`` and then its ``check``; the shell
``&&`` makes a failed build or check exit non-zero, which fails the task and blocks
everything downstream. Steps whose tools are not ready yet are deferred: they exit with
``SKIP_EXIT_CODE`` and show up as *skipped*, never as a fake success.

Commands read their paths from environment variables. ``DATA_ENV`` and
``PLANNER_ENV`` map each variable to the Jinja template the DAG renders from Airflow
Variables (``var.value``) and the run configuration (``dag_run.conf``); see
dags/README.md for every Variable and run option.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Tuple

DAG_ID = "nuplan_pipeline"
PLANNERS = ("idm", "pdm-closed", "pdm-hybrid", "urban-driver", "gc-pgp")
CITIES = ("boston", "las_vegas", "pittsburgh", "singapore")
SIMULATION_POOL = "simulation"
SIMULATION_POOL_SLOTS = 5
# BashOperator turns this exit code into a skipped task.
SKIP_EXIT_CODE = 99
WEATHER_RETRIES = 3
PLANNER_RETRIES = 1

# Paths as seen by the Airflow worker.
DATA_ENV = {
    "DATA_ROOT": "{{ var.value.DATA_ROOT }}",
    "PROC_ROOT": "{{ var.value.PROC_ROOT }}",
    "ZIPS_DIR": "{{ var.value.ZIPS_DIR }}",
    "SCRATCH": "{{ var.value.SCRATCH }}",
    "MINI_ZIP": "{{ var.value.get('MINI_ZIP', var.value.ZIPS_DIR ~ '/nuplan-v1.1_mini.zip') }}",
    # Check reports default to docs/data inside the repository, which is mounted read-only.
    "REPORT_ROOT": "{{ var.value.get('REPORT_ROOT', var.value.PROC_ROOT ~ '/reports') }}",
    "CITY_FLAGS": "{{ city_flags(dag_run.conf, var.value.get('CITIES', '')) }}",
    "MANIFEST": "{{ dag_run.conf.get('manifest') or var.value.get('MANIFEST', "
                "'configs/scenarios/city4_ranked_top5.csv') }}",
    # Same defaults as PLANNER_ENV so build_labels can read planner artifacts without the
    # simulation pool. Must be a writable mount (repo /opt/repo is read-only in Compose).
    "ARTIFACT_ROOT": "{{ var.value.get('ARTIFACT_ROOT', '/artifacts') }}",
    "PLANNER_SAMPLE": "{{ dag_run.conf.get('sample') or var.value.get('PLANNER_SAMPLE', "
                      "'configs/scenarios/pittsburgh_65_sample.csv') }}",
    "EXPERIMENT": "{{ dag_run.conf.get('experiment') or 'airflow_' ~ ts_nodash }}",
    # Smoke: first N trainval logs (writes scenario_map_smoke.parquet). Empty runs the full set.
    "MAP_LIMIT_FLAGS": "{{ map_limit_flags(dag_run.conf) }}",
}
# The planner container is started by the host's Docker daemon, so HOST_* are host paths.
PLANNER_ENV = {
    "PLANNER_SAMPLE": DATA_ENV["PLANNER_SAMPLE"],
    "DATASET_SPLIT": "{{ dag_run.conf.get('dataset_split') or var.value.get('DATASET_SPLIT', "
                     "'train_pittsburgh') }}",
    # The runner refuses to overwrite an experiment folder, so every run gets a fresh name.
    "EXPERIMENT": DATA_ENV["EXPERIMENT"],
    "PLANNER_LIMIT_FLAGS": "{{ planner_limit_flags(dag_run.conf) }}",
    "HOST_REPO_DIR": "{{ var.value.HOST_REPO_DIR }}",
    "HOST_DATA_ROOT": "{{ var.value.HOST_DATA_ROOT }}",
    "HOST_ARTIFACT_ROOT": "{{ var.value.get('HOST_ARTIFACT_ROOT', "
                          "var.value.HOST_REPO_DIR ~ '/artifacts/additional_planners') }}",
    "HOST_CHECKPOINT_ROOT": "{{ var.value.get('HOST_CHECKPOINT_ROOT', "
                            "var.value.HOST_REPO_DIR ~ '/checkpoints') }}",
}


def city_flags(conf: Optional[Mapping], default: str = "") -> str:
    """`--city X` flags from conf["cities"] (list or comma string), else the CITIES Variable.

    The result is word-split by the shell, so only known city names are accepted.
    """
    value = (conf or {}).get("cities") or default
    names = value.split(",") if isinstance(value, str) else list(value)
    names = [str(name).strip() for name in names if str(name).strip()]
    unknown = sorted(set(names) - set(CITIES))
    if unknown:
        raise ValueError(f"Unknown cities {unknown}; expected some of {list(CITIES)}")
    return " ".join(f"--city {name}" for name in names)


def planner_limit_flags(conf: Optional[Mapping]) -> str:
    """`--limit N` from conf["planner_limit"] for smoke runs; empty runs the whole sample."""
    value = (conf or {}).get("planner_limit")
    if value in (None, ""):
        return ""
    limit = int(value)
    if limit < 1:
        raise ValueError(f"planner_limit must be a positive integer, got {value!r}")
    return f"--limit {limit}"


def map_limit_flags(conf: Optional[Mapping]) -> str:
    """`--limit-logs N` for map smoke runs.

    Uses conf["map_limit_logs"] when set; otherwise reuses conf["planner_limit"] so a normal
    smoke config also keeps map extraction short. Empty runs every trainval log.
    """
    value = (conf or {}).get("map_limit_logs")
    if value in (None, ""):
        value = (conf or {}).get("planner_limit")
    if value in (None, ""):
        return ""
    limit = int(value)
    if limit < 1:
        raise ValueError(f"map_limit_logs must be a positive integer, got {value!r}")
    return f"--limit-logs {limit}"


@dataclass(frozen=True)
class Step:
    task_id: str
    command: str
    upstream: Tuple[str, ...] = ()
    retries: int = 0
    pool: Optional[str] = None
    enabled: bool = True


# The Airflow image runs the tools from their own virtualenv (TOOLS_PYTHON), apart from
# Airflow's pinned dependencies; elsewhere `python` is used.
PYTHON = '"${TOOLS_PYTHON:-python}"'


def tool_command(module: str, subcommands: List[str], flags: str) -> str:
    """`python -m tools.<module> <sub> <flags>` for each subcommand, chained with &&."""
    return " && ".join(f"{PYTHON} -m tools.{module} {sub} {flags}" for sub in subcommands)


def deferred(task_id: str, upstream: Tuple[str, ...], reason: str) -> Step:
    message = f"{task_id} is deferred: {reason}".replace('"', "'")
    return Step(task_id, f'echo "{message}"; exit {SKIP_EXIT_CODE}', upstream, enabled=False)


def planner_task_id(planner: str) -> str:
    return "planner_" + planner.replace("-", "_")


def planner_command(planner: str) -> str:
    compose_env = (
        'NUPLAN_HOST_DATA_ROOT="$HOST_DATA_ROOT" '
        'ADDITIONAL_PLANNER_HOST_ARTIFACT_ROOT="$HOST_ARTIFACT_ROOT" '
        'PLANNER_HOST_CHECKPOINT_ROOT="$HOST_CHECKPOINT_ROOT"'
    )
    return (
        f'{compose_env} docker compose --project-directory "$HOST_REPO_DIR" '
        "-f infra/docker/compose.planners.yaml run --rm planner "
        f"python -m tools.pdm_mini_reproducer --planner {planner} "
        '--sample "$PLANNER_SAMPLE" --dataset-split "$DATASET_SPLIT" '
        '--experiment "$EXPERIMENT" $PLANNER_LIMIT_FLAGS'
    )


def pipeline_steps() -> List[Step]:
    archives = '--archives-dir "$ZIPS_DIR"'
    processed = '--data-root "$DATA_ROOT" --processed "$PROC_ROOT"'
    mini = '--mini-zip "$MINI_ZIP" --scratch "$SCRATCH"'
    ego_flags = f'{processed} {mini} --report-dir "$REPORT_ROOT/ego_agent"'
    mini_table = '"$PROC_ROOT/features/ego_agent_mini_68.parquet"'

    steps = [
        Step("extract_logs", tool_command(
            "extract_nuplan_logs", ["extract", "check"],
            f'{archives} --data-root "$DATA_ROOT" $CITY_FLAGS')),
        # Reads the ZIPs directly, so it does not wait for extract_logs.
        Step("scenario_catalog", tool_command(
            "build_scenario_catalog", ["build", "check"],
            f'{archives} --output "$PROC_ROOT" --scratch "$SCRATCH" $CITY_FLAGS')),
        # Its check compares tag rows per log against the catalog shards.
        Step("eligibility", tool_command(
            "build_scenario_eligibility", ["build", "check"],
            f'{processed} {mini} --report-dir "$REPORT_ROOT/eligibility" $CITY_FLAGS'),
            upstream=("extract_logs", "scenario_catalog")),
        # check needs the mini_68 table; build-mini always re-extracts, so run it only once.
        Step("ego_agent_features",
             f"{{ [ -f {mini_table} ] || {PYTHON} -m tools.extract_ego_agent_features build-mini {ego_flags}; }} && "
             + tool_command("extract_ego_agent_features", ["build", "check"], ego_flags),
             upstream=("eligibility",)),
        # fetch is a no-op when open_meteo_raw.json is cached, so the API is called at most once.
        Step("weather_features", tool_command(
            "extract_weather_features", ["fetch", "build", "check"],
            f'{processed} --report-dir "$REPORT_ROOT/weather"'),
            upstream=("eligibility",), retries=WEATHER_RETRIES),
        # Map geometry + traffic lights at the ego start pose (298-39).
        Step("map_features", tool_command(
            "extract_map_features", ["build", "check"],
            '--ego "$PROC_ROOT/features/scenario_ego_agent.parquet" '
            '--mini-ego "$PROC_ROOT/features/scenario_ego_agent_mini_68.parquet" '
            '--alignment "$PROC_ROOT/features/scenario_ego_agent_alignment.parquet" '
            '--eligibility "$PROC_ROOT/scenario_eligibility.parquet" '
            '--db-dir "$DATA_ROOT/nuplan-v1.1/trainval" --mini-zip "$MINI_ZIP" '
            '--maps-dir "$DATA_ROOT/maps" --out-dir "$PROC_ROOT/features" '
            '--reference-dir "$PROC_ROOT/map_reference" '
            '--scratch "$SCRATCH" --report-dir "$REPORT_ROOT/map_features" $MAP_LIMIT_FLAGS'),
            upstream=("ego_agent_features",)),
        # Join ego/weather/map into scenario_features.parquet + leakage check (298-40).
        Step("integrate_features", tool_command(
            "build_scenario_features", ["build", "check"],
            '--ego "$PROC_ROOT/features/scenario_ego_agent.parquet" '
            '--ego-mini "$PROC_ROOT/features/scenario_ego_agent_mini_68.parquet" '
            '--weather "$PROC_ROOT/weather/scenario_weather.parquet" '
            '--weather-mini "$PROC_ROOT/weather/scenario_weather_mini_68.parquet" '
            '--map "$PROC_ROOT/features/scenario_map.parquet" '
            '--map-mini "$PROC_ROOT/features/scenario_map_mini_68.parquet" '
            '--eligibility "$PROC_ROOT/scenario_eligibility.parquet" '
            '--out-dir "$PROC_ROOT" --report-dir "$REPORT_ROOT/scenario_features"'),
            upstream=("ego_agent_features", "weather_features", "map_features")),
        Step("freeze_manifest",
             f'{PYTHON} -m tools.freeze_manifest check --manifest "$MANIFEST" '
             '--eligibility "$PROC_ROOT/scenario_eligibility.parquet"',
             upstream=("eligibility",)),
    ]
    steps += [
        Step(planner_task_id(p), planner_command(p), upstream=("freeze_manifest",),
             retries=PLANNER_RETRIES, pool=SIMULATION_POOL)
        for p in PLANNERS
    ]
    label_flags = (
        '--output-root "$ARTIFACT_ROOT" --experiment "$EXPERIMENT" '
        '--sample "$PLANNER_SAMPLE" --features "$PROC_ROOT/scenario_features.parquet" '
        '--out-dir "$PROC_ROOT/labels/$EXPERIMENT" --report-dir "$REPORT_ROOT/labels"'
    )
    steps += [
        # Score difference vs IDM from finished planner runs (298-51). --overwrite so Airflow
        # retries and re-triggers can replace the previous experiment's tables.
        Step("build_labels",
             f"{PYTHON} -m tools.build_planner_labels build {label_flags} --overwrite && "
             f"{PYTHON} -m tools.build_planner_labels check {label_flags}",
             upstream=tuple(planner_task_id(p) for p in PLANNERS)),
        deferred("model_dataset", ("integrate_features", "build_labels"),
                 "the model dataset join (features + labels + split) is not written yet"),
        deferred("publish_drive", ("model_dataset",),
                 "Drive publishing (rclone copy --immutable) is not automated yet"),
    ]
    return steps


def steps_by_id(steps: Optional[List[Step]] = None) -> Dict[str, Step]:
    steps = pipeline_steps() if steps is None else steps
    return {step.task_id: step for step in steps}
