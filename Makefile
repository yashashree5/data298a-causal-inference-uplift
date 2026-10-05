.PHONY: check docker-build idm-mini-dry-run idm-mini

COMPOSE := docker compose --project-directory . -f infra/docker/compose.yaml
PDM_COMPOSE := docker compose --project-directory . -f infra/docker/compose.pdm.yaml

check:
	python3 -m unittest discover -s tests -v
	python3 -m json.tool notebooks/nuplan_mini_eda.ipynb >/dev/null

docker-build:
	$(COMPOSE) build nuplan

idm-mini-dry-run:
	$(COMPOSE) run --rm nuplan python tools/idm_mini_reproducer.py --dry-run

idm-mini:
	$(COMPOSE) run --rm nuplan python tools/idm_mini_reproducer.py

.PHONY: pdm-docker-build pdm-mini-dry-run pdm-mini-smoke pdm-mini idm-mini-matched
# RUN_ARGS=--limit=1 also works for the matched IDM run.
RUN_ARGS ?=

pdm-docker-build: docker-build
	$(PDM_COMPOSE) build pdm

pdm-mini-dry-run:
	$(PDM_COMPOSE) run --rm pdm python -m tools.pdm_mini_reproducer --dry-run $(RUN_ARGS)

pdm-mini-smoke:
	$(PDM_COMPOSE) run --rm pdm python -m tools.pdm_mini_reproducer --limit 1

pdm-mini:
	$(PDM_COMPOSE) run --rm pdm python -m tools.pdm_mini_reproducer $(RUN_ARGS)

idm-mini-matched:
	$(PDM_COMPOSE) run --rm pdm python -m tools.pdm_mini_reproducer --planner idm $(RUN_ARGS)
