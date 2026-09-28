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
