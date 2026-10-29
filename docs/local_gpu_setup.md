# Local GPU workstation setup (replaces Colab)

Tested on Windows 11, Intel Core Ultra 9 285K (24 cores), 63 GB RAM, RTX 5090 (32 GB), two 4 TB NVMe drives.

## Layout

All nuPlan data lives under `D:\nuplan`:

```text
D:\nuplan\
  downloads\                           # every original archive (train cities, mini, maps); never modified
  dataset\                             # NUPLAN_HOST_DATA_ROOT in .env (mounted as /data in Docker)
    maps\                              # extracted nuplan-maps-v1.0.zip (4 city map.gpkg files)
    nuplan-v1.1\
      trainval\<log_name>.db           # all 7,353 train DBs (441 GB), flat; devkit `nuplan` builder path
      trainval_extraction.json         # per-log source archive/entry and size, written by `check`
  scratch\                             # temporary per-log DB extraction for the catalog (auto-cleaned)
  processed\
    catalog\<city>\<log>.parquet       # resumable catalog shards
    catalog\failures.json
    scenario_catalog.parquet           # combined, validated catalog
    scenario_catalog_summary.json
```

`trainval/` is flat because the pinned devkit's `nuplan` scenario builder reads
`${NUPLAN_DATA_ROOT}/nuplan-v1.1/trainval`, and log names are unique across cities.
The mini split is kept zipped in `downloads\`; extract it to `nuplan-v1.1\splits\mini\` only if
the `mini_68` experiment needs rerunning.

## Extract the train logs

```powershell
.\.venv\Scripts\python -m tools.extract_nuplan_logs extract   # resumable; skips complete files
.\.venv\Scripts\python -m tools.extract_nuplan_logs check     # verifies every DB size, writes provenance
```

Each DB is written to `<name>.db.partial` and renamed only after the full read (ZIP CRC check)
succeeds. Measured on this machine: 441 GB in about 6 minutes with 6 workers.

## Python environment

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python -m pip install pandas numpy pyarrow matplotlib pyyaml pytest jupyter ipykernel tqdm requests pyproj
.\.venv\Scripts\python -m ipykernel install --user --name nuplan-gpu --display-name "Python (nuplan-gpu)"
```

The RTX 5090 (Blackwell) requires a CUDA 12.8+ PyTorch build; the `cu128` wheel index provides it.
Select the **Python (nuplan-gpu)** kernel for notebooks.

## Baseline checks (previously notebook cell 3)

```powershell
.\.venv\Scripts\python -m tools.build_paired_dataset --overwrite
.\.venv\Scripts\python tools/validate_scenario_subset.py
.\.venv\Scripts\python -m pytest -q
```

`.gitattributes` forces LF line endings. Without it, Git on Windows converts the frozen
manifest to CRLF and its SHA-256 provenance check fails.

## Download the remaining archives from Drive

```powershell
rclone config create gdrive drive scope=drive.readonly   # one-time browser login
powershell -ExecutionPolicy Bypass -File tools/download_nuplan_drive.ps1
```

Archives already in `Downloads` are skipped. Re-running resumes and skips completed files.

## Scenario catalog (previously notebook cell 11)

```powershell
.\.venv\Scripts\python -m tools.build_scenario_catalog build            # all archives found
.\.venv\Scripts\python -m tools.build_scenario_catalog build --city singapore
.\.venv\Scripts\python -m tools.build_scenario_catalog check            # validate + combine
```

`build` skips completed shards and records failures; re-run it to retry. `check` compares
shards against every DB entry in the discovered archives, checks duplicate keys and missing
fields, and reports `COMPLETE` or `INCOMPLETE` with the specific problems.

Some logs have no `scenario_tag` rows, so their shards are empty. In Boston, 132 of 1,647 logs
are empty (verified directly against the source DB).

## Notes

- Catalog extraction is CPU/disk bound (Boston: about 35 s at about 48 logs/s with 20 workers).
  The GPU is for model training, not for catalog building or nuPlan closed-loop simulation.
- Planner simulations still run through the Docker images in `infra/docker` (Linux/amd64,
  CPU). On Windows this needs Docker Desktop with the WSL 2 backend.
