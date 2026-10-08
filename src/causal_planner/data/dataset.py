"""Scaffold for feature/outcome assembly; dataset building is not implemented.

The future builder will validate scenario membership and provenance, pivot
paired outcomes, join features, and keep each driving log in a single split.
It must not fit preprocessing on held-out data.
"""

import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Planned modeling dataset builder (not implemented yet)."
    )
    parser.parse_args()
    parser.exit(
        2,
        "Modeling dataset assembly is not implemented yet. "
        "See docs/repository_structure.md; no data was written.\n",
    )
