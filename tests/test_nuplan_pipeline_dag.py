"""Checks for the nuplan_pipeline Airflow DAG (dags/).

The task graph, commands and run options are tested from dags/pipeline_spec.py, which
needs no Airflow. Every generated tool command is parsed by that tool's real argument
parser, so a renamed or missing flag fails here instead of in the middle of a run.
The DAG module itself is checked only where Airflow is installed.
"""

import argparse
import contextlib
import importlib
import importlib.util
import io
from pathlib import Path
import re
import shlex
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
DAGS = ROOT / "dags"
sys.path.insert(0, str(DAGS))

import pipeline_spec as spec  # noqa: E402

DATA_TASKS = (
    "extract_logs", "scenario_catalog", "eligibility", "ego_agent_features", "weather_features",
    "map_features", "integrate_features", "build_labels",
)
PLANNER_TASKS = tuple(spec.planner_task_id(p) for p in spec.PLANNERS)
DEFERRED_TASKS = ("model_dataset", "publish_drive")
EXPECTED_UPSTREAM = {
    "extract_logs": set(),
    "scenario_catalog": set(),
    "eligibility": {"extract_logs", "scenario_catalog"},
    "ego_agent_features": {"eligibility"},
    "weather_features": {"eligibility"},
    "map_features": {"ego_agent_features"},
    "integrate_features": {"ego_agent_features", "weather_features", "map_features"},
    "freeze_manifest": {"eligibility"},
    **{task: {"freeze_manifest"} for task in PLANNER_TASKS},
    "build_labels": set(PLANNER_TASKS),
    "model_dataset": {"integrate_features", "build_labels"},
    "publish_drive": {"model_dataset"},
}
SAMPLE_ENV = {
    "DATA_ROOT": "/data", "PROC_ROOT": "/processed", "ZIPS_DIR": "/zips", "SCRATCH": "/scratch",
    "MINI_ZIP": "/zips/nuplan-v1.1_mini.zip", "REPORT_ROOT": "/processed/reports",
    "CITY_FLAGS": "--city pittsburgh", "MANIFEST": "configs/scenarios/city4_ranked_top5.csv",
    "ARTIFACT_ROOT": "/opt/repo/artifacts/additional_planners",
    "PLANNER_SAMPLE": "configs/scenarios/pittsburgh_65_sample.csv", "DATASET_SPLIT": "train_pittsburgh",
    "EXPERIMENT": "airflow_smoke", "PLANNER_LIMIT_FLAGS": "--limit 1",
    "HOST_REPO_DIR": "/host/repo", "HOST_DATA_ROOT": "/host/data",
    "HOST_ARTIFACT_ROOT": "/host/artifacts", "HOST_CHECKPOINT_ROOT": "/host/checkpoints",
}
SEPARATORS = {"&&", "||", ";", "}"}


def expand(command):
    """Substitute SAMPLE_ENV into a command the way bash would (unquoted flags word-split)."""
    command = re.sub(r"\$\{(\w+):-([^}]*)\}", lambda m: SAMPLE_ENV.get(m.group(1), m.group(2)), command)
    return re.sub(r"\$(\w+)", lambda m: SAMPLE_ENV[m.group(1)], command)


def tool_calls(command):
    """[(module, argv)] for every `python -m tools.<module> ...` in a shell command."""
    lexer = shlex.shlex(expand(command), posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    tokens = list(lexer)
    calls = []
    for i, token in enumerate(tokens):
        if token == "python" and tokens[i + 1:i + 2] == ["-m"] and tokens[i + 2].startswith("tools."):
            end = next((j for j in range(i + 3, len(tokens)) if tokens[j] in SEPARATORS), len(tokens))
            calls.append((tokens[i + 2].split(".", 1)[1], tokens[i + 3:end]))
    return calls


class _Parsed(Exception):
    def __init__(self, namespace):
        super().__init__()
        self.namespace = namespace


def parse_tool_args(module_name, argv):
    """Run the tool's own argument parsing on argv and stop before any work is done."""
    module = importlib.import_module(f"tools.{module_name}")
    real_parse = argparse.ArgumentParser.parse_args

    def capture(parser, args=None, namespace=None):
        raise _Parsed(real_parse(parser, args, namespace))

    stderr = io.StringIO()
    with mock.patch.object(argparse.ArgumentParser, "parse_args", capture), \
            mock.patch.object(sys, "argv", [module_name, *argv]), contextlib.redirect_stderr(stderr):
        try:
            module.main()
        except _Parsed as parsed:
            return parsed.namespace
        except SystemExit as exc:
            raise AssertionError(f"tools.{module_name} rejected {argv}: {stderr.getvalue()}") from exc
    raise AssertionError(f"tools.{module_name} did not parse its arguments")


class PipelineGraphTest(unittest.TestCase):
    def setUp(self):
        self.steps = spec.steps_by_id()

    def test_task_ids_and_dependencies(self):
        self.assertEqual(len(spec.pipeline_steps()), len(self.steps), "duplicate task ids")
        self.assertEqual(EXPECTED_UPSTREAM, {tid: set(s.upstream) for tid, s in self.steps.items()})

    def test_graph_is_acyclic(self):
        done, pending = set(), dict(EXPECTED_UPSTREAM)
        while pending:
            ready = [tid for tid, ups in pending.items() if ups <= done]
            self.assertTrue(ready, f"cycle among {sorted(pending)}")
            for tid in ready:
                done.add(tid)
                del pending[tid]

    def test_retries_and_pools(self):
        for tid, step in self.steps.items():
            with self.subTest(task=tid):
                expected = {"weather_features": spec.WEATHER_RETRIES}.get(
                    tid, spec.PLANNER_RETRIES if tid in PLANNER_TASKS else 0)
                self.assertEqual(expected, step.retries)
                self.assertEqual(spec.SIMULATION_POOL if tid in PLANNER_TASKS else None, step.pool)

    def test_deferred_steps_skip_instead_of_succeeding(self):
        deferred = {tid for tid, step in self.steps.items() if not step.enabled}
        self.assertEqual(set(DEFERRED_TASKS), deferred)
        for tid in deferred:
            command = self.steps[tid].command
            self.assertTrue(command.endswith(f"exit {spec.SKIP_EXIT_CODE}"), command)
            self.assertNotIn("python", command)


class PipelineCommandTest(unittest.TestCase):
    def setUp(self):
        self.steps = spec.steps_by_id()

    def test_data_steps_build_then_check(self):
        for tid in DATA_TASKS:
            with self.subTest(task=tid):
                calls = tool_calls(self.steps[tid].command)
                self.assertEqual(1, len({module for module, _ in calls}), calls)
                subcommands = [argv[0] for _, argv in calls]
                self.assertEqual("check", subcommands[-1])
                self.assertIn(subcommands[-2], {"build", "extract"})
                self.assertIn(" && ", self.steps[tid].command)

    def test_freeze_manifest_only_checks(self):
        calls = tool_calls(self.steps["freeze_manifest"].command)
        self.assertEqual([("freeze_manifest", "check")], [(m, argv[0]) for m, argv in calls])

    def test_every_tool_command_parses_with_the_real_cli(self):
        for tid, step in self.steps.items():
            for module, argv in tool_calls(step.command):
                with self.subTest(task=tid, tool=module, command=argv[:1]):
                    args = parse_tool_args(module, argv)
                    report_dir = getattr(args, "report_dir", None)
                    if report_dir is not None:
                        self.assertEqual(Path("/processed/reports"), Path(report_dir).parent,
                                         "check reports must not be written into the repository")

    def test_planner_commands_run_each_planner_in_the_planner_container(self):
        for planner in spec.PLANNERS:
            with self.subTest(planner=planner):
                command = self.steps[spec.planner_task_id(planner)].command
                self.assertIn("-f infra/docker/compose.planners.yaml run --rm planner", command)
                ((module, argv),) = tool_calls(command)
                self.assertEqual("pdm_mini_reproducer", module)
                args = parse_tool_args(module, argv)
                self.assertEqual((planner, 1, "airflow_smoke"), (args.planner, args.limit, args.experiment))
                self.assertEqual("train_pittsburgh", args.dataset_split)

    def test_every_variable_used_is_provided(self):
        for tid, step in self.steps.items():
            if not step.enabled:
                continue
            provided = set(spec.DATA_ENV) | (set(spec.PLANNER_ENV) if step.pool else set())
            used = set(re.findall(r"\$\{?(\w+)", step.command)) - {"TOOLS_PYTHON"}
            with self.subTest(task=tid):
                self.assertLessEqual(used, provided)


class RunOptionTest(unittest.TestCase):
    def test_city_flags(self):
        self.assertEqual("", spec.city_flags({}, ""))
        self.assertEqual("--city pittsburgh", spec.city_flags({}, "pittsburgh"))
        self.assertEqual("--city boston --city singapore", spec.city_flags({"cities": "boston, singapore"}))
        self.assertEqual("--city las_vegas", spec.city_flags({"cities": ["las_vegas"]}, "boston"))
        for bad in ("pittsburgh; rm -rf /", "--limit 1", "vegas"):
            with self.subTest(cities=bad), self.assertRaises(ValueError):
                spec.city_flags({"cities": bad})

    def test_planner_limit_flags(self):
        self.assertEqual("", spec.planner_limit_flags({}))
        self.assertEqual("", spec.planner_limit_flags(None))
        self.assertEqual("--limit 1", spec.planner_limit_flags({"planner_limit": 1}))
        self.assertEqual("--limit 5", spec.planner_limit_flags({"planner_limit": "5"}))
        for bad in (0, -1, "1; rm -rf /"):
            with self.subTest(limit=bad), self.assertRaises(ValueError):
                spec.planner_limit_flags({"planner_limit": bad})


class _Values:
    """Stand-in for Airflow's `var.value`: attribute access fails on a missing Variable."""

    def __init__(self, values):
        self._values = values

    def __getattr__(self, name):
        try:
            return self._values[name]
        except KeyError:
            raise KeyError(f"Variable {name} does not exist") from None

    def get(self, name, default=None):
        return self._values.get(name, default)


REQUIRED_VARIABLES = {"DATA_ROOT": "/data", "PROC_ROOT": "/processed", "ZIPS_DIR": "/zips",
                      "SCRATCH": "/scratch", "HOST_REPO_DIR": "/host/repo", "HOST_DATA_ROOT": "/host/data"}


@unittest.skipUnless(importlib.util.find_spec("jinja2"), "jinja2 is not installed")
class TemplateTest(unittest.TestCase):
    def render(self, conf, variables=REQUIRED_VARIABLES):
        import jinja2

        env = jinja2.Environment(undefined=jinja2.StrictUndefined)
        context = {"var": {"value": _Values(variables)}, "dag_run": {"conf": conf},
                   "ts_nodash": "20261106T120000", "city_flags": spec.city_flags,
                   "planner_limit_flags": spec.planner_limit_flags}
        return {key: env.from_string(template).render(context)
                for key, template in {**spec.DATA_ENV, **spec.PLANNER_ENV}.items()}

    def test_defaults(self):
        env = self.render({})
        self.assertEqual("/zips/nuplan-v1.1_mini.zip", env["MINI_ZIP"])
        self.assertEqual("/processed/reports", env["REPORT_ROOT"])
        self.assertEqual("", env["CITY_FLAGS"])
        self.assertEqual("", env["PLANNER_LIMIT_FLAGS"])
        self.assertEqual("airflow_20261106T120000", env["EXPERIMENT"])
        self.assertEqual("configs/scenarios/city4_ranked_top5.csv", env["MANIFEST"])
        self.assertEqual("configs/scenarios/pittsburgh_65_sample.csv", env["PLANNER_SAMPLE"])
        self.assertEqual("train_pittsburgh", env["DATASET_SPLIT"])
        self.assertEqual("/host/repo/artifacts/additional_planners", env["HOST_ARTIFACT_ROOT"])
        self.assertEqual("/host/repo/checkpoints", env["HOST_CHECKPOINT_ROOT"])

    def test_smoke_run_configuration(self):
        env = self.render({"cities": "pittsburgh", "planner_limit": 1, "experiment": "airflow_smoke"})
        self.assertEqual("--city pittsburgh", env["CITY_FLAGS"])
        self.assertEqual("--limit 1", env["PLANNER_LIMIT_FLAGS"])
        self.assertEqual("airflow_smoke", env["EXPERIMENT"])

    def test_missing_required_variable_fails(self):
        for name in REQUIRED_VARIABLES:
            with self.subTest(variable=name), self.assertRaises(KeyError):
                self.render({}, {k: v for k, v in REQUIRED_VARIABLES.items() if k != name})


class ComposeTest(unittest.TestCase):
    def setUp(self):
        import yaml

        self.compose = yaml.safe_load((ROOT / "infra" / "docker" / "compose.airflow.yaml").read_text())
        self.airflow = self.compose["x-airflow"]

    def test_required_variables_are_set_for_airflow(self):
        templates = " ".join({**spec.DATA_ENV, **spec.PLANNER_ENV}.values())
        required = set(re.findall(r"var\.value\.(\w+)", templates)) - {"get"}
        self.assertEqual(set(REQUIRED_VARIABLES), required)
        for name in required:
            self.assertIn(f"AIRFLOW_VAR_{name}", self.airflow["environment"])

    def test_repository_mounted_read_only_as_dags_folder(self):
        repo = next(v for v in self.airflow["volumes"] if isinstance(v, dict) and v["target"] == "/opt/repo")
        self.assertTrue(repo["read_only"])
        self.assertEqual("/opt/repo/dags", self.airflow["environment"]["AIRFLOW__CORE__DAGS_FOLDER"])

    def test_init_creates_the_simulation_pool(self):
        command = " ".join(self.compose["services"]["airflow-init"]["command"])
        self.assertIn(f"airflow pools set {spec.SIMULATION_POOL} {spec.SIMULATION_POOL_SLOTS} ", command)


@unittest.skipUnless(importlib.util.find_spec("airflow"), "Airflow is not installed")
class AirflowDagTest(unittest.TestCase):
    def test_dag_matches_the_spec(self):
        from airflow.models import DagBag

        bag = DagBag(dag_folder=str(DAGS), include_examples=False)
        self.assertEqual({}, bag.import_errors)
        dag = bag.get_dag(spec.DAG_ID)
        self.assertIsNone(dag.schedule_interval)
        self.assertFalse(dag.catchup)
        self.assertEqual(1, dag.max_active_runs)
        steps = spec.steps_by_id()
        self.assertEqual(set(steps), set(dag.task_ids))
        for task in dag.tasks:
            step = steps[task.task_id]
            with self.subTest(task=task.task_id):
                self.assertEqual(set(step.upstream), task.upstream_task_ids)
                self.assertEqual(step.retries, task.retries)
                self.assertEqual(step.pool or "default_pool", task.pool)
                self.assertEqual("none_failed", task.trigger_rule)
                self.assertEqual(step.command, task.bash_command)


if __name__ == "__main__":
    unittest.main()
