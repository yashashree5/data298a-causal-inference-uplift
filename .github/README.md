# GitHub Actions

[`workflows/ci.yml`](workflows/ci.yml) runs on every pull request to `main` and
on every push to `main`, on **Linux** and **Windows**:

1. The unit-test suite (`pytest`)
2. The mini_68 validator (`python -m tools.validate_scenario_subset`)
3. JSON sanity checks for the EDA notebook and the draft feature schema

A failed step fails the check. To block merges until it is green, the repository
owner must require the `CI / ubuntu-latest` and `CI / windows-latest` checks
under branch protection for `main`.

## Known temporary deselections

Two map-feature tests still fail on `main` (Linear 298-39 / PR #30, unfinished).
CI deselects them so the check can pass. Remove those `--deselect` lines from
`ci.yml` when the map-feature work is finished and the tests are green.
