# Docker environments

- `Dockerfile` / `compose.yaml`: the original IDM mini environment. Keep it
  unchanged so the recorded IDM run remains reproducible.
- `Dockerfile.pdm` / `compose.pdm.yaml`: PDM-Closed and a matched IDM baseline,
  both using nuPlan 1.2.2. This image builds on the original CPU image and adds
  the pinned tuPlan Garage checkout and its extra dependencies.

Run Make targets from the repository root, with dataset paths set in `.env`
as described in the main README. Docker mounts the dataset read-only. The
temporary map-lock directory is writable; maps and databases are not modified.
Both environments honor `NUPLAN_HOST_ARTIFACT_ROOT` (default `./artifacts`).

## PDM-Closed pins

| Component | Revision |
| --- | --- |
| nuPlan 1.2.2 | `a581fbcab373db99a13f191f94521c75c256e080` |
| tuPlan Garage | `b51d5d04fac1bd4389653b9ab2ff73ea88f435a3` |

The Dockerfile fetches upstream source directly and verifies each commit. No
planner code is copied into this repository. The extra package versions are in
`requirements.pdm.txt`; the base environment versions are in `requirements.txt`.
No GPU, AWS account, or pretrained checkpoint is needed for PDM-Closed.

Sources: [tuPlan Garage](https://github.com/autonomousvision/tuplan_garage),
[upstream requirements](https://github.com/autonomousvision/tuplan_garage/blob/b51d5d04fac1bd4389653b9ab2ff73ea88f435a3/requirements.txt).
Upstream licenses remain with their checkouts inside the image.

```bash
make pdm-docker-build  # Builds the base image first, then the PDM image
make pdm-mini-dry-run
make pdm-mini-smoke   # First saved scenario only
make idm-mini-matched RUN_ARGS=--limit=1  # Same scenario, same image
```

The images use Linux/amd64, including on Apple Silicon via Docker emulation.
Keep planner dependencies isolated here instead of installing them in the host
Python environment. See the [runner documentation](../../tools/README.md) for
full-sample commands and output validation.
