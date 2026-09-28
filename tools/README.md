# Tools

[`idm_mini_reproducer.py`](idm_mini_reproducer.py) starts our IDM mini experiment.
It gives everyone on the team the same entry point and simulation settings.

The script:

1. Reads and validates the saved [68-scenario CSV](../configs/scenarios/idm_mini_sample.csv).
2. Passes its log names and scenario tokens to the nuPlan devkit.
3. Runs the official IDM planner in closed-loop reactive simulation.
4. Records the launch command and saves outputs under `artifacts/idm_mini/`.

In Docker, the runner uses `/artifacts/idm_mini/`. The `/artifacts` mount points
to `NUPLAN_HOST_ARTIFACT_ROOT` from `.env`, which defaults to the repository's
`./artifacts`. A custom host output directory is honored too. Outside Docker,
the default is the repository's `artifacts/idm_mini/`; `--output-root` can
override either default.

The IDM algorithm, simulator, and scoring code are supplied by nuPlan inside
Docker. This script connects our selected sample to those components.

After completing the [local setup](../README.md#local-setup), run these commands
from the repository root:

```bash
make idm-mini-dry-run  # Validate the sample and print the command; no simulation.
make idm-mini          # Start the simulation for all 68 selected scenarios.
```

The runner prints the output directory when finished. Check nuPlan's
`runner_report.parquet` for completed/failed scenarios and the `aggregator_metric/`
folder for scores. The script does not yet print a score comparison with 0.76.

See the [sample README](../configs/scenarios/README.md) for how the 68 scenarios
were selected.
