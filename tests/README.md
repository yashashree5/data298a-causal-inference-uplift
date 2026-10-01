# Tests

[`test_idm_mini_reproducer.py`](test_idm_mini_reproducer.py) checks our sample
loader and simulation command before spending time on an actual IDM run.

It checks that:

- The saved sample contains 68 unique scenario IDs across 14 categories.
- A scenario limit selects the requested number of rows.
- The command selects IDM, reactive simulation, and the requested scenario tokens.
- A CSV missing required columns is rejected.

Run from the repository root:

```bash
make check
```

This also checks that the EDA notebook is valid JSON. These quick checks need
Python 3, but no Docker or downloaded nuPlan databases. They do not run a
simulation or verify the 0.76 score.

## PDM-Closed checks

`test_pdm_mini_reproducer.py` checks that PDM and matched IDM use identical
evaluation settings except for the planner. It also tests result validation:
missing/extra/duplicate scenarios, failed simulations, wrong categories or
planners, invalid scores, and incorrect aggregates. These are small synthetic
tests, not simulations and not evidence of driving performance.

Run all fast tests with `make check`. Run the real one-scenario simulation with
`make pdm-mini-smoke`; see [tools](../tools/README.md) for the matched IDM run.
