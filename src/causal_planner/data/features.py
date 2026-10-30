"""Scaffold for extracting pre-simulation features; extraction is not implemented.

Implemented extractors remain in tools.extract_ego_agent_features and
tools.extract_weather_features. This placeholder writes no data and does not
replace their YAML field contract. See docs/repository_structure.md.
"""

import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Planned scenario feature extractor (not implemented yet)."
    )
    parser.parse_args()
    parser.exit(
        2,
        "This package placeholder is not implemented yet. Use "
        "tools.extract_ego_agent_features or tools.extract_weather_features. "
        "See docs/repository_structure.md; no data was written.\n",
    )
