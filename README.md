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
