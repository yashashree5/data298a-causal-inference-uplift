"""Load and validate the fixed scenario manifests."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SAMPLE = PROJECT_ROOT / "configs" / "scenarios" / "idm_mini_sample.csv"

REQUIRED_COLUMNS = {
    "scenario_id",
    "log_name",
    "db_file",
    "scenario_token",
    "scenario_type",
    "map_name",
}
TOKEN_PATTERN = re.compile(r"^[0-9a-f]{16}$")


def load_scenarios(sample_path: Path, limit: Optional[int] = None) -> List[Dict[str, str]]:
    """Load and validate scenario rows from the committed manifest."""
    with sample_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        available_columns = set(reader.fieldnames or [])
        missing_columns = REQUIRED_COLUMNS - available_columns
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Scenario manifest is missing required columns: {missing}")
        scenarios = list(reader)

    if not scenarios:
        raise ValueError("Scenario manifest contains no scenarios")

    seen_ids = set()
    for row in scenarios:
        scenario_id = row["scenario_id"]
        token = row["scenario_token"].lower()
        if scenario_id in seen_ids:
            raise ValueError(f"Duplicate scenario_id: {scenario_id}")
        if not TOKEN_PATTERN.fullmatch(token):
            raise ValueError(f"Invalid scenario token for {scenario_id}: {token}")
        if row["db_file"] != f"{row['log_name']}.db":
            raise ValueError(f"Database and log name do not match for {scenario_id}")
        seen_ids.add(scenario_id)
        row["scenario_token"] = token

    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        scenarios = scenarios[:limit]

    return scenarios
