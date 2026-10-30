#!/usr/bin/env python3
"""Backward-compatible command and imports for the IDM mini runner."""

if __package__ in (None, ""):
    from _bootstrap import bootstrap
else:
    from ._bootstrap import bootstrap

bootstrap()

from causal_planner.data.scenarios import (
    DEFAULT_SAMPLE, PROJECT_ROOT, REQUIRED_COLUMNS, TOKEN_PATTERN, load_scenarios,
)
from causal_planner.simulation.runners import (
    DEFAULT_DEVKIT_ROOT, DEFAULT_OUTPUT_ROOT, build_idm_command as build_command,
    hydra_list, idm_main as main, parse_idm_args as parse_args,
)


if __name__ == "__main__":
    raise SystemExit(main())
