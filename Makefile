.PHONY: help install install-dev lint format type-check clean test

help:
	@echo "AtlasPatch Development Commands"
	@echo "===================================="
	@echo "  make install       - Install the package in development mode"
	@echo "  make install-dev   - Install package with development dependencies"
	@echo "  make lint          - Run ruff linter"
	@echo "  make format        - Format code with ruff"
	@echo "  make type-check    - Run mypy type checking"
	@echo "  make test          - Run pytest tests"
	@echo "  make test-cov      - Run tests with coverage report"
	@echo "  make check         - Run lint, format check, and type checking"
	@echo "  make clean         - Remove build artifacts and cache files"
	@echo "  make pre-commit    - Install pre-commit hooks"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

lint:
	ruff check .

format:
	ruff format .

format-check:
	ruff format --check .

type-check:
	mypy atlas_patch --ignore-missing-imports

test:
	pytest

test-cov:
	pytest --cov=atlas_patch --cov-report=html --cov-report=term

check: lint format-check type-check
	@echo "All checks passed!"

pre-commit:
	pre-commit install

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/
	rm -rf dist/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	@echo "Cleaned up build artifacts and cache files"
