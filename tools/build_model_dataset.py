"""Scaffold command for future modeling dataset assembly."""

if __package__ in (None, ""):
    from _bootstrap import bootstrap
else:
    from ._bootstrap import bootstrap

bootstrap()

from causal_planner.data.dataset import main


if __name__ == "__main__":
    main()
