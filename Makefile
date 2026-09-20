# TerraSpectra monorepo task runner
SHELL := /bin/bash
MODULES_PY := contracts/python pipeline model api

.PHONY: help setup test lint fmt stub-model sample-cube up up-gpu down

help: ## Show targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n",$$1,$$2}'

setup: ## Install all module dependencies
	@for m in $(MODULES_PY); do echo "== $$m"; (cd $$m && uv sync --all-extras) || exit 1; done
	cd dashboard && pnpm install
	cp -n .env.example .env || true
	uvx pre-commit install || true

test: ## Run every module's tests
	@for m in $(MODULES_PY); do echo "== $$m"; (cd $$m && uv run pytest -q) || exit 1; done
	cd dashboard && pnpm test

lint: ## Lint every module
	@for m in pipeline model api; do echo "== $$m"; (cd $$m && uv run ruff check . && uv run ruff format --check . && uv run mypy) || exit 1; done
	cd dashboard && pnpm lint && pnpm typecheck

fmt: ## Auto-format Python and TS
	@for m in pipeline model api; do (cd $$m && uv run ruff check --fix . && uv run ruff format .); done
	cd dashboard && pnpm format

stub-model: ## Write contract stub model to models/model.pt
	cd contracts/python && uv run python ../fixtures/stub_model.py ../../models/model.pt

sample-cube: ## Write a synthetic C1 cube to data/synthetic_cube.tif
	mkdir -p data && cd contracts/python && uv run python ../fixtures/synthetic_cube.py ../../data/synthetic_cube.tif --size 512

up: ## Start the full stack (CPU)
	docker compose up --build

up-gpu: ## Start the full stack with GPU worker
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build

down: ## Stop the stack
	docker compose down
