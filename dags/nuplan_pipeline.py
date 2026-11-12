"""nuplan_pipeline: the nuPlan data pipeline and the five-planner comparison as one DAG.

Data: extract_logs + scenario_catalog -> eligibility -> ego/agent, weather, map features
-> integrate_features. Simulation: freeze_manifest -> five planners -> build_labels.
Both meet in model_dataset -> publish_drive. Each task runs a tool's build then check;
a failed check fails the task and blocks everything downstream. Deferred steps are
skipped until their tools reach main. Setup and run options: dags/README.md.
"""

from __future__ import annotations

import os
from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator

from pipeline_spec import (
    DAG_ID, DATA_ENV, PLANNER_ENV, SKIP_EXIT_CODE, Step, city_flags, map_limit_flags,
    pipeline_steps, planner_limit_flags,
)

# Repository path inside the Airflow containers (infra/docker/compose.airflow.yaml).
REPO_DIR = os.environ.get("NUPLAN_REPO_DIR", "/opt/repo")


def task_env(step: Step):
    if not step.enabled:
        return None
    return {**DATA_ENV, **PLANNER_ENV} if step.pool else dict(DATA_ENV)


with DAG(
    DAG_ID,
    description="nuPlan data pipeline and five-planner comparison (Team 4)",
    doc_md=__doc__,
    start_date=datetime(2026, 11, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 0},
    user_defined_macros={
        "city_flags": city_flags,
        "planner_limit_flags": planner_limit_flags,
        "map_limit_flags": map_limit_flags,
    },
    tags=["nuplan", "team4"],
) as dag:
    steps = pipeline_steps()
    tasks = {
        step.task_id: BashOperator(
            task_id=step.task_id,
            bash_command=step.command,
            cwd=REPO_DIR,
            env=task_env(step),
            append_env=True,
            retries=step.retries,
            pool=step.pool or "default_pool",
            skip_on_exit_code=SKIP_EXIT_CODE,
            # Skipped (deferred) upstream steps do not block; failed ones do.
            trigger_rule="none_failed",
        )
        for step in steps
    }
    for step in steps:
        for upstream in step.upstream:
            tasks[upstream] >> tasks[step.task_id]
