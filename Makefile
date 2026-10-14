export UV_CACHE_DIR := $(PWD)/.uv-cache
export HF_HOME := $(PWD)/.hf-cache

.PHONY: install run debug clean lint lint-strict restarto

install:
	mkdir -p $(UV_CACHE_DIR) $(HF_HOME)
	uv sync

run:
	uv run python -m src

debug:
	uv run python -m pdb -m src

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 .
	uv run mypy . --strict

restarto:
	rm -rf .venv

clean:
	rm -rf __pycache__ .mypy_cache .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
