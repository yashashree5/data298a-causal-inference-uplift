"""Scaffold for extracting pre-simulation features; extraction is not implemented.

The future extractor will read a fixed manifest and the original nuPlan
databases/maps, use the actual simulation start as its feature cutoff, and
produce one row per scenario. See docs/repository_structure.md.
"""

import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Planned scenario feature extractor (not implemented yet)."
    )
    parser.parse_args()
    parser.exit(
        2,
        "Feature extraction is not implemented yet. "
        "See docs/repository_structure.md; no data was written.\n",
    )
