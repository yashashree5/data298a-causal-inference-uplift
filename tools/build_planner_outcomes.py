"""Backward-compatible command and imports for matched outcome assembly."""

if __package__ in (None, ""):
    from _bootstrap import bootstrap
else:
    from ._bootstrap import bootstrap

bootstrap()

from causal_planner.data.outcomes import (
    OUTCOMES, build_table, exactly_one, experiment_metadata, main,
    recorded_run_id, save_csv, wait_for_runs,
)
from causal_planner.data.scenarios import DEFAULT_SAMPLE, load_scenarios
from causal_planner.simulation.runners import PLANNERS
from causal_planner.simulation.validation import MATCHED_CONFIG, validate_pair, validate_results


if __name__ == "__main__":
    main()
