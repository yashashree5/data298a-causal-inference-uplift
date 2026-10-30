# Planner checkpoints

Place the official tuPlan Garage checkpoint files here for local runs. The
files are intentionally ignored by Git and mounted read-only in the additional
planner container.

Required filenames:

| Planner | Filename |
| --- | --- |
| PDM-Hybrid | `pdm_offset_checkpoint.ckpt` |
| UrbanDriver | `urbandriver_checkpoint.ckpt` |
| GC-PGP | `gc_pgp_checkpoint.ckpt` |

The upstream checkpoint folder is linked from the
[tuPlan Garage README](https://github.com/autonomousvision/tuplan_garage#2-training).
Record the downloaded file hashes before running. Each validated run also saves
the checkpoint filename, size, and SHA-256 in `run_manifest.json`.

To keep weights outside the repository, set `PLANNER_HOST_CHECKPOINT_ROOT` to an
absolute host directory containing the same filenames.
