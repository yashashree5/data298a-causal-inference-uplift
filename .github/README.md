# GitHub Actions

[`workflows/ci.yml`](workflows/ci.yml) runs on every pull request to `main` and
on every push to `main`, on **Linux** and **Windows**:

1. The unit-test suite (`pytest`)
2. The mini_68 validator (`python -m tools.validate_scenario_subset`)
3. JSON sanity checks for the EDA notebook and the draft feature schema

Both jobs use Python 3.11 and install the shared root `requirements.txt`, including
the editable package with its `modeling` extra. This installs scikit-learn so the
modeling tests run in CI. The pip cache tracks both `requirements.txt` and
`pyproject.toml`; update the shared requirements when adding dependencies.

A failed step fails the check. To block merges until it is green, the repository
owner must require the `CI / ubuntu-latest` and `CI / windows-latest` checks
under branch protection for `main`.

## Known temporary deselections

CI retains two map-feature test exclusions introduced for Linear 298-39 / PR #30.
An excluded test can pass in a newer local environment; these exclusions do not
establish its current result. Remove the `--deselect` lines from `ci.yml` after
verifying both supported CI platforms. Record local runs separately.
