"""Scaffold command for future scenario feature extraction."""

if __package__ in (None, ""):
    from _bootstrap import bootstrap
else:
    from ._bootstrap import bootstrap

bootstrap()

from causal_planner.data.features import main


if __name__ == "__main__":
    main()
