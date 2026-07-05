# streamllm dev tasks. Run `make help` for the list.
.DEFAULT_GOAL := help
PY := PYTHONPATH=src python3

.PHONY: help install test lint format typecheck check

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Editable install with dev + test extras
	pip install -e ".[test,dev]"

test: ## Run the CPU-only test suite (no downloads)
	$(PY) -m pytest -q

lint: ## Ruff lint
	ruff check src tests bench web/server.py

format: ## Ruff format (writes)
	ruff format src tests bench

typecheck: ## Mypy on the package
	mypy src

check: ## All gates: lint + format-check + typecheck + tests
	ruff check src tests bench web/server.py
	ruff format --check src tests bench
	mypy src
	$(PY) -m pytest -q
