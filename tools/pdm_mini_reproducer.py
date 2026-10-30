"""Backward-compatible command and imports for matched PDM/IDM runs."""

if __package__ in (None, ""):
    from _bootstrap import bootstrap
else:
    from ._bootstrap import bootstrap

bootstrap()

from causal_planner.data.scenarios import DEFAULT_SAMPLE, PROJECT_ROOT, load_scenarios
from causal_planner.simulation.runners import (
    PLANNERS, SEARCH_PATH, build_idm_command, build_pdm_command as build_command,
    checkout_revision, experiment_run_root, pdm_main as main,
)
from causal_planner.simulation.validation import summarize_run, validate_results


if __name__ == "__main__":
    raise SystemExit(main())
