.PHONY: check docker-build idm-mini-dry-run idm-mini

COMPOSE := docker compose --project-directory . -f infra/docker/compose.yaml

check:
	python3 -m unittest discover -s tests -v
	python3 -m json.tool notebooks/nuplan_mini_eda.ipynb >/dev/null

docker-build:
	$(COMPOSE) build nuplan

idm-mini-dry-run:
	$(COMPOSE) run --rm nuplan python tools/idm_mini_reproducer.py --dry-run

idm-mini:
	$(COMPOSE) run --rm nuplan python tools/idm_mini_reproducer.py
