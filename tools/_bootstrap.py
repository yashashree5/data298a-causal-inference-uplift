"""Allow existing checkout commands to run without installing the package."""

from pathlib import Path
import sys


def bootstrap():
    source = str(Path(__file__).resolve().parents[1] / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
