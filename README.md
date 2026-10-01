# Causal Inference and Uplift Engine for Product Experimentation

**DATA 298A — MSDA Project I**  
**Section 22 | Team 4 | Topic 20**

## Team

- Jim He
- Ameya Khond
- Prathamesh Mankar
- Tejas Sawant
- Yashashree Shinde

---

## Project Overview

Autonomous-driving planners are commonly evaluated using aggregate benchmark
scores across many driving scenarios. While these scores are useful for comparing
overall planner performance, they can hide scenario-specific improvements and
regressions.

A candidate planner may improve performance at intersections while performing
worse during highway merges or complex pedestrian interactions.

This project explores an **AI-based planner experimentation and regression
intelligence system** that goes beyond average planner scores and investigates:

- Where does a candidate planner improve performance?
- Where does it introduce regressions?
- Which scenario characteristics are associated with these differences?
- Which scenarios should autonomous-driving engineers prioritize for further
  simulation and validation?

The current application domain is autonomous-driving planner evaluation using
the **nuPlan planning benchmark**.

---

## Course Topic

**Topic 20 — Causal Inference and Uplift Engine for Product Experimentation**

The project applies the core idea of heterogeneous effects to autonomous-driving
planner experimentation.

Instead of only asking:

> Is the candidate planner better on average?

we investigate:

> Under which driving conditions does changing the planner improve or degrade
> performance?

---

## Current Project Status

The current review branch, **`test/idm-mini-reproduction`**, contains a completed local
comparison of **IDM and PDM-Closed**. These are upstream driving planners, not
the causal/uplift models the team will develop. Nothing needs to be merged into
`main` to review or run this branch. The original IDM work and PDM-Closed
comparison are consolidated here; teammates need only this branch.

The `mini_68` experiment uses the same fixed 68 scenarios for both planners,
covering 14 official challenge categories, 38 driving logs, and four map locations.
Both runs use nuPlan 1.2.2, closed-loop reactive agents, simulation seed 0, and
the same scoring configuration.

| Planner | Successfully scored | Simulation failures | Official aggregate |
| --- | ---: | ---: | ---: |
| IDM | 68/68 | 0 | 0.7600884182496237 |
| PDM-Closed | 68/68 | 0 | 0.9004246949123553 |

These results describe **our selected mini sample**, not the paper's original
benchmark cohort. The sample is approximately category-balanced, not a random
representation of all driving. A zero score is a valid outcome, not necessarily
a simulation failure.

**Completed:** dataset EDA, saved sample, Docker runners, matched simulations,
result validation, and a 136-row outcome table in Parquet and CSV.
**Next:** extract pre-simulation condition features, join them to the outcomes,
then develop and evaluate the causal/uplift models. Those models are not yet
implemented here. PlanTF and PLUTO are on hold and are not part of this experiment.

### Start Here for Team Review

- [Sample selection](configs/scenarios/README.md): why 68 scenarios, the 14
  categories, sampling rules, and coverage limitations.
- [EDA notebook](notebooks/nuplan_mini_eda.ipynb): database inventory and scenario
  exploration. Rerunning EDA is not required to use the committed sample.
- [Tools](tools/README.md): simulation commands, validation, and table export.
- [Docker](infra/docker/README.md): pinned dependencies and data mounts.
- [Tests](tests/README.md): what the quick tests check, and what they do not.

To review without downloading nuPlan or building Docker, run from the repository
root with Python 3:

```bash
make check
python3 -m tools.pdm_mini_reproducer --dry-run
python3 -m tools.pdm_mini_reproducer --planner idm --dry-run
```

These commands check the code/manifest and print simulation commands; they do
not simulate driving or reproduce the scores. For a real run, use
[Local Setup](#local-setup).

---

## Dataset and Benchmark

### nuPlan

The local experiment uses the **nuPlan v1.1 mini dataset** and nuPlan's
closed-loop reactive evaluation framework.

nuPlan provides:

- real-world driving logs;
- HD maps;
- ego-vehicle trajectories;
- surrounding-agent trajectories;
- traffic-light information;
- categorized driving scenarios;
- planner implementations;
- closed-loop simulation;
- standardized planner evaluation metrics.

The dataset contains driving data collected across multiple cities, including:

- Boston
- Pittsburgh
- Las Vegas
- Singapore

Large raw nuPlan files will **not** be committed to this repository.

Download the mini databases and maps from the
[official nuPlan download page](https://www.nuscenes.org/nuplan#download), then
follow [Local Setup](#local-setup). Use nuPlan mini, not nuScenes mini.

---

## Published Baseline

The current published baseline selected for reproduction is the
**Intelligent Driver Model (IDM) Planner** from the nuPlan benchmark.

The published nuPlan benchmark reports an IDM closed-loop reactive score of
**0.76 in Table III**.

Baseline reproduction is separate from the models developed by the team.

Its purpose is to verify that our:

- nuPlan environment,
- scenario selection,
- simulation configuration,
- planner setup,
- and metric aggregation

are functioning correctly before conducting the main experiments.

---

## Proposed Experiment

The planner comparison is complete for the selected mini scenarios. The next
stage connects these paired outcomes to scenario conditions for modeling.

Conceptually:

```text
                 nuPlan Scenario
                       |
              Scenario Features
                       |
             +---------+---------+
             |                   |
             v                   v
      Baseline Planner     Candidate Planner
             |                   |
             v                   v
        Outcome A            Outcome B
             |                   |
             +---------+---------+
                       |
                       v
             Planner Difference
                       |
                       v
               AI Modeling Layer
                       |
                       v
            Regression Intelligence
```

---

## IDM Reproduction Test Using nuPlan Mini

This branch uses the nuPlan v1.1 mini split to test the IDM reproduction
workflow locally. Initial data exploration is a supporting step: it verifies the
dataset structure and produces a reproducible scenario sample for the IDM test.
Because the mini split is not the complete benchmark evaluation set, this test
does not by itself reproduce the published `0.76` result.

In the mini split, one SQLite `.db` file represents one driving log. Each
database contains one row in the `log` table with metadata such as the vehicle,
recording date, location, and map version. The initial local inspection found:

- 64 SQLite databases representing 64 driving logs;
- the same 12-table schema in every database;
- 1,364 scenes in total, where each scene is a snippet of up to 20 seconds;
- 518,999 LiDAR frames in total; and
- scenario tags attached to selected LiDAR frames.

The main data hierarchy is:

```text
nuPlan mini split
└── SQLite database / driving log
    └── scene, up to 20 seconds
        └── LiDAR frame and timestamp
            ├── ego pose
            ├── detected objects and tracks
            ├── traffic-light status
            └── zero or more scenario tags
```

A scenario tag identifies a LiDAR frame as an event anchor. The nuPlan devkit
combines that anchor with a configured extraction window to construct a
simulation scenario. The scenario is therefore not stored as one database row;
it is assembled from multiple time-indexed rows when requested by the devkit.

The completed mini evaluation followed these steps:

1. inventory the 64 mini databases and validate their schemas;
2. summarize scenario categories, multilabel anchors, and temporal overlap;
3. create and document a deterministic mini-dataset sample;
4. run the IDM planner on that sample with the intended closed-loop reactive
   configuration;
5. validate the run artifacts, scenario-level metrics, failures, and aggregate
   score.

### Verified Mini Result

On 2026-09-28, run `idm_mini_20260928T212421Z` completed and scored **68 of 68
scenarios**, covering 14 categories and 38 driving logs, with **zero simulation
failures**. The scored scenario IDs and categories matched the saved sample
exactly, with none missing or added. Simulation took **24 minutes 26 seconds**.

The official nuPlan aggregate score was **0.7600884182496237**, which rounds to
**0.76**. This is numerical agreement on our fixed mini sample, not a reproduction
on the paper's original evaluation cohort. The reference is Table III of
[Karnchanachari et al. (2024), Towards Learning-Based Planning: The nuPlan
Benchmark for Real-World Autonomous Driving](https://arxiv.org/abs/2403.04133).

The run used the official `idm_planner` defaults, `closed_loop_reactive_agents`,
the `nuplan_mini` builder, the challenge-category filter restricted to our saved
sample, a sequential worker, and simulation seed 0. The devkit commit was
`ce3c323af01c0d7ec5672f7832ef53f9c679aab0` (v1.2), using nuPlan v1.1 mini data.
The sample's hash and selection rules are recorded in the
[sample README](configs/scenarios/README.md).

nuPlan computes a score for each scenario and averages those 68 scores. Thirteen
scenarios scored zero despite completing successfully. The run also logged 18
IDM route-fallback warnings; those scenarios remain included in the result.

Full evidence remains local under
`artifacts/idm_mini/idm_mini_20260928T212421Z/`: `run_manifest.json`, and nested
nuPlan outputs containing `runner_report.parquet`, `aggregator_metric/*.parquet`,
the resolved Hydra configuration, and logs. The aggregate file contains a
`final_score` row with the reported score. These large generated files are
Git-ignored; this summary is included in the repository for teammates.

Large nuPlan databases and generated analysis artifacts will remain local and
will not be committed to this repository.

### Repository Structure for the Mini Test

```text
infra/docker/                         Docker environment and pinned devkit
notebooks/nuplan_mini_eda.ipynb       Scenario inventory and sample selection
configs/scenarios/idm_mini_sample.csv Committed 68-scenario test manifest
tools/pdm_mini_reproducer.py          Matched IDM and PDM-Closed entrypoint
tools/build_planner_outcomes.py      Validated paired-run table and CSV export
tools/idm_mini_reproducer.py          Preserved original IDM-only runner
tests/                               Fast command and result-validation tests
external/                            Optional local upstream clones, Git ignored
artifacts/                             Generated EDA and simulation outputs
```

The committed scenario manifest is the handoff between EDA and simulation. It
contains identifiers and metadata for the exact 68 scenarios selected locally;
it does not contain raw driving data. The reproducer reads those identifiers and
passes them to the nuPlan scenario filter.

See [the sample selection README](configs/scenarios/README.md) for the eligibility
rules, sampling seed, overlap exclusion, explanation of the 68-scenario count,
and limits on how representative the sample is.

### Completed 68-Scenario IDM/PDM-Closed Comparison

PDM-Closed and IDM work are together on `test/idm-mini-reproduction`. PDM reuses the saved mini
sample and adds a separate Docker image without changing the original IDM
environment. See [Docker setup](infra/docker/README.md) and
[PDM runner commands](tools/README.md#pdm-closed-mini-comparison) to run both
planners over the same 68 scenarios.

The full matched experiment completed **68/68 scenarios for each
planner**, with no failures or missing/duplicate pairs. IDM scored
**0.7600884182496237** and PDM-Closed **0.9004246949123553** on this fixed mini
sample. Both used the same nuPlan 1.2.2 image and reactive evaluation settings.
This is not a claim about the original full benchmark cohort.

Artifacts are organized by experiment setup:

```text
artifacts/
├── eda/
├── mini_68/
│   ├── idm/
│   ├── pdm_closed/
│   ├── planner_outcomes.parquet
│   ├── planner_outcomes.csv
│   └── experiment.json
└── idm_mini/               # Preserved original nuPlan 1.2.0 reproduction
```

The combined table contains 136 scenario-planner rows. Initial-condition model
features have **not** been extracted. The JSON records setup/source provenance;
timestamps remain internal metadata rather than outer directory names. Original
simulation logs and nuPlan's internal folders are preserved unchanged. Rerunning
into an existing planner folder is refused; use an explicit new experiment name
for another setup. See [runner documentation](tools/README.md) for commands.

Each table row represents **one scenario evaluated by one planner**, with
identifiers, scenario metadata, planner, outcomes, and provenance. The unique
key is `(scenario_id, planner)` within this experiment. There is no stored
difference column: derive PDM-Closed minus IDM by pairing the two rows for a
scenario. The 136 rows represent 68 pairs, not 136 independent scenarios.

For the next modeling stage, extract conditions available at simulation start,
not future trajectory information. Scenario-category tags can describe future
events and are not automatically safe input features. Keep all scenarios from a
driving log together when defining train/test splits.

### What Teammates Receive Through Git

The code, documentation, Docker definitions, EDA notebook, and selected-scenario
CSV are tracked. **The raw databases/maps, `.env`, upstream clones, and generated
`artifacts/` are not.** A fresh clone will not contain the 136-row result table.

To inspect existing results without rerunning, obtain `planner_outcomes.parquet`,
the optional CSV copy, and `experiment.json` from the experiment owner through
the team's chosen sharing channel. For a full audit or to rebuild the table,
also obtain the `idm/` and `pdm_closed/` source-run directories, including their
manifests, environments, reports, resolved configurations, and metrics. No shared
artifact location is configured in this repository yet.

### Local Setup

Requirements: Python 3 for the fast checks, Make, Docker with Compose, and the
licensed nuPlan mini databases plus maps. **No AWS account, GPU, or pretrained
checkpoint is needed for IDM/PDM-Closed.** Images use Linux/amd64, including
emulation on Apple Silicon; build and simulation times vary by machine.

If `.env` does not exist, copy `.env.example` to `.env`. Do not overwrite an
existing `.env`. Set `NUPLAN_HOST_DATA_ROOT` to an absolute directory with this
layout:

```text
nuplan-dataset/
├── maps/
└── nuplan-v1.1/
    └── splits/
        └── mini/            # 64 .db files
```

Run the commands below from the repository root. `NUPLAN_HOST_ARTIFACT_ROOT`
defaults to `./artifacts`, relative to that root. To store runs elsewhere, set
it to an absolute host directory in `.env`.

```bash
make check
make pdm-docker-build
make pdm-mini-dry-run
make pdm-mini
make idm-mini-matched
```

The final two commands run all 68 scenarios under `artifacts/mini_68/`. Use
`idm-mini-matched` for this comparison so IDM and PDM share the same environment.
Confirm `valid: true`, 68 scored scenarios, and zero failures in each planner's
`result.json`. Then use the [table-building command](tools/README.md#long-format-planner-outcome-table)
to generate Parquet, CSV, and `experiment.json`; simulation alone does not create
the combined table. Existing result folders/exports are never overwritten.

### Original IDM-Only Reproduction (Historical)

The original nuPlan 1.2.0 workflow is retained separately. It is not the matched
1.2.2 comparison above:

```bash
make docker-build
make idm-mini-dry-run
make idm-mini
```

`make idm-mini-dry-run` validates the committed manifest and prints the nuPlan
command without starting a simulation. `make idm-mini` runs all manifest rows
and writes generated output under `artifacts/idm_mini/` by default, or under
`idm_mini/` inside your custom `NUPLAN_HOST_ARTIFACT_ROOT`. The container always
uses `/artifacts/idm_mini/` for these runs. Existing run artifacts are unchanged.

The Makefile sets the Compose project directory to the repository root. If
invoking Compose directly, use the same setting:

```bash
docker compose --project-directory . -f infra/docker/compose.yaml config
```
