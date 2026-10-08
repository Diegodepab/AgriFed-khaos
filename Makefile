.PHONY: build up down shell audit test lint format logs clean

AGRIFED_UID ?= $(shell id -u)
AGRIFED_GID ?= $(shell id -g)
AGRIFED_DATASET_ROOT ?= ../enfermedadeshoja_v1.0/EnfermedadesHoja_v1.0
AGRIFED_PROJECT_EXTRAS ?= vision,fl,dev

export AGRIFED_UID
export AGRIFED_GID
export AGRIFED_DATASET_ROOT
export AGRIFED_PROJECT_EXTRAS

build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

shell:
	docker compose exec agrifed bash

audit:
	docker compose exec agrifed python -m agrifed_khaos.cli

test:
	docker compose exec agrifed pytest

lint:
	docker compose exec agrifed ruff check src tests
	docker compose exec agrifed mypy src tests

format:
	docker compose exec agrifed ruff format src tests

logs:
	docker compose logs -f agrifed

clean:
	docker compose down -v --remove-orphans