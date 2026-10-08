# AV Planner Regression Intelligence Platform

### Causal Inference and Uplift Engine for Product Experimentation

**DATA 298A / 298B — Master's Capstone Project**<br>
**Section 22 | Team 4 | Topic 20**

---

## Overview

Autonomous-driving motion planners are evaluated across large collections of simulated driving scenarios. Aggregate benchmark scores are useful for measuring overall performance, but they can hide important scenario-specific regressions.

A candidate planner may improve average performance while becoming worse in specific situations such as highway merges, dense intersections, lane changes, or pedestrian interactions.

This project investigates an **AV Planner Regression Intelligence Platform** that goes beyond overall benchmark scores to answer three practical questions:

> **Where does a planner change improve performance?**<br>
> **Where does it introduce regressions?**<br>
> **Which scenarios should engineers investigate first?**

The system is intended as an offline experimentation and validation tool for autonomous-driving planning, simulation, and validation teams.

---

## Problem Statement

Consider an autonomous-driving team evaluating a new planner release.

An aggregate benchmark might report:

```text
Baseline Planner   0.76
Candidate Planner  0.82
```

The candidate appears better overall.

However, its behavior may actually look like:

```text
Dense intersections       Improvement
Pedestrian interactions   Improvement
Normal lane following     Similar
Highway merges            Regression
High-speed lane changes   Regression
```

Those regressions can be hidden by the aggregate score.

Our project focuses on identifying and prioritizing these **scenario-specific planner differences** rather than relying only on average benchmark performance.

---

## Project Objective

Our objective is to build an end-to-end experimentation and regression-intelligence system that can:

1. evaluate baseline and candidate planners across autonomous-driving scenarios;
2. characterize the driving conditions associated with each scenario;
3. identify scenario-specific planner improvements and regressions;
4. study modern AI methods for learning patterns in these differences;
5. prioritize important regression scenarios for engineering investigation;
6. present the results through an interpretable validation interface.

The goal is **not simply to determine which planner has the highest average score**.

The goal is to understand:

> **Under what driving conditions does a planner change help or hurt?**

---

## Target Users

The intended users are engineers working on:

- autonomous-driving motion planning;
- simulation;
- planner validation;
- regression testing;
- safety evaluation;
- release analysis.

A useful system should help these users focus limited simulation and debugging resources on scenarios where a planner change appears most consequential.

---

## Dataset and Benchmark

### nuPlan

The current benchmark selected for the project is **nuPlan**, a large-scale autonomous-driving planning benchmark containing real-world driving logs and a closed-loop simulation framework.

nuPlan provides information including:

- ego-vehicle states;
- surrounding traffic agents;
- trajectories;
- road and map information;
- traffic context;
- categorized driving scenarios;
- planner implementations;
- closed-loop simulation;
- standardized planner evaluation metrics.

The benchmark contains more than 1,200 hours of driving data collected across:

- Boston;
- Pittsburgh;
- Las Vegas;
- Singapore.

Raw nuPlan data will **not** be stored in this repository.

Dataset acquisition, versioning, scenario selection, and preprocessing instructions will be documented separately.

---

## Published Baseline

The current published reference point is the **Intelligent Driver Model (IDM) Planner** reported in the nuPlan benchmark.

The published benchmark reports:

```text
IDM Planner
Closed-loop reactive score: 0.76
nuPlan benchmark paper — Table III
```

Baseline reproduction is separate from the models developed by our team.

Its purpose is to verify that our:

- simulation environment;
- planner configuration;
- scenario selection;
- metric computation;
- result aggregation

are working correctly before conducting the main experiments.

---

## Completed 68-Scenario Planner Comparison

This repository includes a completed local comparison of **IDM and PDM-Closed**.
These are upstream driving planners used to validate the experiment pipeline,
not the causal/uplift models the team will develop.

The `mini_68` experiment evaluates both planners on the same fixed 68 scenarios,
covering 14 official challenge categories, 38 driving logs, and four map
locations. Both runs use nuPlan 1.2.2, closed-loop reactive agents, simulation
seed 0, and the same scoring configuration.

| Planner | Successfully scored | Simulation failures | Official aggregate |
|---|---:|---:|---:|
| IDM | 68/68 | 0 | 0.7600884182496237 |
| PDM-Closed | 68/68 | 0 | 0.9004246949123553 |

These results describe the selected, approximately category-balanced mini
sample. They are not a reproduction of the original benchmark cohort or a claim
about performance across all nuPlan scenarios. A zero score is a valid outcome,
not necessarily a simulation failure.

The completed work includes dataset exploration, a deterministic scenario
manifest, Docker runners, matched simulations, result validation, and a
136-row outcome table in Parquet and CSV. Each row represents one planner on one
scenario; the 136 rows form 68 matched pairs.

### Review the Experiment

- [Sample selection](configs/scenarios/README.md) documents the 68 scenarios,
  sampling rules, and coverage limitations.
- [EDA notebook](notebooks/nuplan_mini_eda.ipynb) contains the database inventory
  and scenario exploration.
- [Outcome CSV](artifacts/mini_68/planner_outcomes.csv) and
  [Parquet](artifacts/mini_68/planner_outcomes.parquet) contain the matched
  scenario-planner outcomes.
- [Experiment metadata](artifacts/mini_68/experiment.json) records the shared
  setup and source provenance.
- [Tools](tools/README.md), [Docker](infra/docker/README.md), and
  [tests](tests/README.md) document reproduction and validation.

Teammates can inspect the committed sample and results without downloading
nuPlan or rerunning simulations. From the repository root, the fast validation
suite is:

```bash
make check
python3 -m tools.pdm_mini_reproducer --dry-run
python3 -m tools.pdm_mini_reproducer --planner idm --dry-run
```

These commands validate the code and manifest and print the simulation commands;
they do not reproduce the scores. Full rerun instructions are in the
[tools documentation](tools/README.md).

---

## Core Experiment

The project studies planner changes using the same or controlled sets of driving scenarios.

Conceptually:

```text
                         nuPlan
                           |
                           v
                    Scenario Library
                           |
                           v
                   Scenario Selection
                           |
                  +--------+--------+
                  |                 |
                  v                 v
          Baseline Planner    Candidate Planner
                  |                 |
                  v                 v
             Outcome A          Outcome B
                  |                 |
                  +--------+--------+
                           |
                           v
                 Experiment Results
                           |
                           v
                Scenario Intelligence
                           |
                           v
               Regression Intelligence
                           |
              +------------+------------+
              |            |            |
              v            v            v
           Detect       Characterize   Prioritize
         regressions     conditions    scenarios
              +------------+------------+
                           |
                           v
                  Engineering Review
```

The exact candidate planner is still being evaluated as part of the technical-feasibility phase.

---

## Scenario Intelligence

A major part of the project is understanding **what kind of driving situation is being evaluated**.

Potential scenario information includes:

- ego speed and state;
- road geometry;
- intersections;
- merges;
- lane changes;
- surrounding vehicles;
- pedestrians;
- traffic density;
- relative motion;
- agent interactions;
- temporal trajectories;
- scenario complexity.

This allows the project to move beyond:

> "Planner B is better."

toward statements such as:

> "Planner B improves performance in dense low-speed intersections but shows regressions in particular high-speed merging conditions."

---

## AI and Modeling Direction

Our original proposal investigated heterogeneous treatment-effect approaches including:

- Gradient-Boosted T-Learner;
- Causal Forest;
- Doubly Robust Learner;
- Policy Learning.

Following feedback from our project presentation, we expanded the technical investigation toward **recent AI architectures**, particularly Transformer and attention-based approaches capable of representing temporal and multi-agent driving scenarios.

Current research areas include:

- scene-level Transformers;
- temporal Transformers;
- trajectory Transformers;
- agent-interaction attention;
- graph/interaction Transformers;
- modern conditional-effect estimation;
- regression-ranking and prioritization methods.

### Important

The final model architectures are **not yet frozen**.

Model selection will be based on:

- fit to the project objective;
- compatibility with nuPlan;
- available scenario representation;
- reproducibility;
- computational requirements;
- availability of reference implementations;
- measurable improvement over simpler approaches.

The project is defined by the **planner-regression problem**, not by any single model architecture.

---

## Evaluation

Evaluation will occur at multiple levels.

### 1. Planner Evaluation

Measure the actual performance of baseline and candidate planners using appropriate nuPlan metrics.

### 2. Model Evaluation

Evaluate whether the learned system correctly identifies or estimates scenario-specific planner differences.

The original proposal included:

- PEHE;
- treatment-effect RMSE;
- policy regret.

These metrics are being reviewed as the final learning task is refined.

### 3. Regression Retrieval

A key product-level question is:

> If an engineer can investigate only a limited number of scenarios, can our system surface important regressions better than simple or random prioritization?

Potential metrics include regression-retrieval precision at fixed review budgets.

### 4. Robustness and Failure Analysis

Later experiments may investigate performance across:

- scenario categories;
- cities;
- traffic complexity;
- repeated simulation runs;
- simple versus difficult scenarios;
- different planner configurations.

The final evaluation protocol will be committed before comparative model results are used.

---

## System Architecture

The current high-level architecture is:

```text
                    +--------------------+
                    |    nuPlan Data     |
                    +----------+---------+
                               |
                               v
                    +--------------------+
                    | Data Ingestion &   |
                    | Validation         |
                    +----------+---------+
                               |
                               v
                    +--------------------+
                    | Scenario Selection |
                    | & Representation   |
                    +----------+---------+
                               |
                               v
                  +--------------------------+
                  | Planner Experimentation  |
                  +------------+-------------+
                               |
                    +----------+----------+
                    |                     |
                    v                     v
              Baseline Planner      Candidate Planner
                    |                     |
                    +----------+----------+
                               |
                               v
                    +--------------------+
                    | Experiment Results |
                    +----------+---------+
                               |
                               v
                    +--------------------+
                    | AI / Modeling      |
                    | Layer              |
                    +----------+---------+
                               |
                               v
                    +--------------------+
                    | Regression         |
                    | Intelligence       |
                    +----------+---------+
                               |
                               v
                    +--------------------+
                    | Scenario Ranking   |
                    | & Analysis         |
                    +----------+---------+
                               |
                               v
                    +--------------------+
                    | AV Validation      |
                    | Dashboard          |
                    +--------------------+
```

This architecture will evolve as technical feasibility is validated.

---

## Expected Product

Instead of returning only an overall planner score, the final system is intended to provide information such as:

```text
Candidate Planner Validation
================================================

Overall Performance
Baseline Planner       0.76
Candidate Planner      0.81

Scenario Analysis
------------------------------------------------
Dense intersections       Improvement
Pedestrian interactions   Improvement
Normal lane following     Similar
Highway merges            Regression
High-speed lane changes   Regression

Priority Regression Review
------------------------------------------------
Scenario 08172             HIGH
Scenario 02918             HIGH
Scenario 07128             MEDIUM
```

The exact interface and metrics will evolve with the project.

---

## Project Scope

### In Scope

- nuPlan data and scenario processing;
- planner experimentation;
- published baseline reproduction;
- candidate planner comparison;
- scenario representation;
- modern AI/modeling research;
- scenario-specific regression analysis;
- regression prioritization;
- evaluation and failure analysis;
- experiment reproducibility;
- offline engineering prototype/dashboard.

### Out of Scope

This project does **not** attempt to:

- build a complete autonomous-driving stack;
- design a production motion planner from scratch;
- deploy software to a real autonomous vehicle;
- replace real-world safety validation;
- make real-time safety-critical driving decisions;
- claim real-world safety improvements based only on simulation.

---

## Project Phases

### M1 — Problem Definition & Technical Feasibility

Current focus:

- finalize product/problem definition;
- validate nuPlan suitability;
- investigate scenario data;
- verify published baseline requirements;
- investigate candidate planner options;
- survey modern AI architectures;
- define evaluation strategy;
- finalize system architecture.

### M2 — Data & Planner Experiment Pipeline

Completed for the fixed `mini_68` development experiment:

- configure reproducible nuPlan environment;
- select development scenario subset;
- implement scenario extraction;
- run baseline planner;
- run candidate planner;
- store experiment results;
- validate the paired experiment pipeline.

### M3 — First Model

Planned work:

- freeze model input/output contract;
- select first model architecture;
- implement training pipeline;
- train on development data;
- evaluate on held-out scenarios;
- establish first model results.

### M4 — Expanded Modeling & Evaluation

Planned work:

- additional model architectures;
- comparative experiments;
- robustness analysis;
- ablation studies;
- scenario-family analysis;
- failure analysis.

### M5 — Regression Intelligence Prototype

Planned work:

- regression detection;
- scenario prioritization;
- engineering review workflow;
- visualization/dashboard;
- end-to-end integration;
- reproducibility validation.

---

## Team

| Member | Primary Ownership |
|---|---|
| **Ameya Khond** | Data and nuPlan scenario pipeline |
| **Jim He** | Planner simulation and baseline experimentation |
| **Prathamesh Mankar** | AI/model research and model development |
| **Tejas Sawant** | Evaluation, benchmarking, and experimental analysis |
| **Yashashree Shinde** | System architecture, integration, product, and project documentation |

Responsibilities may evolve as the technical design is finalized, while individual work remains traceable through Linear issues and GitHub pull requests.

---

## Repository Structure

```text
.
├── README.md
├── Makefile
├── pyproject.toml              # Optional editable package installation
├── artifacts/
│   ├── mini_68/               # Committed matched outcome snapshot
│   └── pittsburgh_v1/         # Planning README only; no results yet
├── configs/
│   ├── scenarios/             # Fixed scenario manifest and selection notes
│   └── features/              # Draft feature-schema scaffold
├── src/
│   └── causal_planner/
│       ├── simulation/
│       │   ├── runners.py     # Existing planner commands and execution
│       │   └── validation.py  # Reports and matched-run compatibility
│       └── data/
│           ├── scenarios.py  # Manifest loading and validation
│           ├── outcomes.py   # Validated planner outcome assembly
│           ├── features.py   # Scaffold: extraction not implemented yet
│           └── dataset.py    # Scaffold: joins/splits not implemented yet
├── docs/
│   └── research/
├── infra/
│   └── docker/                # Pinned nuPlan environments
├── notebooks/                 # Dataset exploration
├── tests/                     # Fast validation tests
└── tools/                     # Compatible CLI entry points into src/
```

Raw nuPlan data, complete simulation outputs, local environment files, and
upstream repositories remain excluded from Git.

Existing Makefile commands and `tools.*` imports still work without installing
the package or rebuilding Docker. Reusable implementation now lives in `src/`;
the tools delegate to it. See [repository structure and migration](docs/repository_structure.md)
for the file mapping and optional editable installation.

`tools/extract_scenario_features.py` and `tools/build_model_dataset.py` are
explicit scaffolds: `--help` works, but execution exits with a not-implemented
message. No feature data or modeling table has been generated. The future
`configs/scenarios/pittsburgh_sample.csv` will be created after cataloging and
validating a Pittsburgh cohort; the current runners still use the mini split.

---

## Project Management

Project work is tracked through **Linear and GitHub**.

Linear is used for:

- project milestones;
- task ownership;
- acceptance criteria;
- blockers;
- status tracking.

GitHub is used for:

- implementation;
- experiment code;
- documentation;
- pull requests;
- code review;
- reproducibility artifacts.

Each substantive unit of work should be associated with an owned Linear issue and corresponding GitHub evidence.

---

## Current Status

**Phase: M2 planner experiment pipeline completed; M3 modeling preparation**

Current work is focused on:

- extracting pre-simulation scenario-condition features;
- joining those features to the matched planner outcomes;
- defining leakage-safe train/test splits by driving log;
- reviewing recent Transformer/attention-based approaches;
- defining the evaluation protocol;
- refining the system architecture.

The causal/uplift models and regression-intelligence interface are not yet
implemented. PlanTF and PLUTO are not part of the completed `mini_68`
experiment.

---

## Reference

Karnchanachari, N., Geromichalos, D., Tan, K. S., Li, N., Eriksen, C., Yaghoubi, S., Mehdipour, N., Bernasconi, G., Fong, W. K., Guo, Y., & Caesar, H. (2024). *Towards learning-based planning: The nuPlan benchmark for real-world autonomous driving*. IEEE International Conference on Robotics and Automation (ICRA), 629–636.

---

## Disclaimer

This project is an academic research and engineering prototype developed as part of the SJSU MSDA capstone sequence. It is intended for offline experimentation and research and is not a production autonomous-driving or safety system.
