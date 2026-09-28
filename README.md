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

The project is currently in the **scoping, literature-review, and technical
feasibility phase**.

Following feedback from the project presentation, the team is investigating
recent AI approaches, particularly **Transformer and attention-based
architectures**, for modeling complex driving scenarios and planner-performance
differences.

The final four model architectures have **not yet been frozen**.

Current work includes:

- refining the project scope;
- reviewing recent Transformer-based autonomous-driving research;
- investigating the nuPlan dataset and scenario representation;
- studying the IDM published baseline;
- evaluating candidate planner options;
- defining evaluation metrics;
- assessing compute and simulation requirements;
- refining the end-to-end system architecture.

---

## Dataset and Benchmark

### nuPlan

The project currently plans to use the **nuPlan autonomous-driving planning
benchmark**.

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

Instructions and scripts for obtaining and preparing the required subset will
be maintained under `data/`.

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

The planned experiment compares a baseline planner with a candidate planner on
selected nuPlan scenarios.

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
tools/idm_mini_reproducer.py          IDM mini simulation entrypoint
tests/test_idm_mini_reproducer.py     Fast manifest and command tests
external/nuplan-devkit/               Optional local devkit clone, Git ignored
artifacts/                             Generated EDA and simulation outputs
```

The committed scenario manifest is the handoff between EDA and simulation. It
contains identifiers and metadata for the exact 68 scenarios selected locally;
it does not contain raw driving data. The reproducer reads those identifiers and
passes them to the nuPlan scenario filter.

See [the sample selection README](configs/scenarios/README.md) for the eligibility
rules, sampling seed, overlap exclusion, explanation of the 68-scenario count,
and limits on how representative the sample is.

### Local Setup

Copy `.env.example` to `.env` and set `NUPLAN_HOST_DATA_ROOT` to the directory
containing `maps/` and `nuplan-v1.1/`.

Run the commands below from the repository root. `NUPLAN_HOST_ARTIFACT_ROOT`
defaults to `./artifacts`, relative to that root. To store runs elsewhere, set
it to an absolute host directory in `.env`.

```bash
cp .env.example .env
make check
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
