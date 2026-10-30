"""Scaffold for feature/outcome assembly; dataset building is not implemented.

The future builder will validate scenario membership and provenance, pivot
paired outcomes, join features, and keep each driving log in a single split.
It must not fit preprocessing on held-out data. The existing outcome cleaning,
pairing, and grouped-fold workflow remains in tools.build_paired_dataset;
joining the extracted feature tables is separate work.
"""

import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Planned modeling dataset builder (not implemented yet)."
    )
    parser.parse_args()
    parser.exit(
        2,
        "This package placeholder is not implemented yet. Use "
        "tools.build_paired_dataset for existing outcome cleaning and pairing. "
        "See docs/repository_structure.md; no data was written.\n",
    )
