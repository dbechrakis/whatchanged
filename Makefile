PY := .venv/bin/python

.PHONY: install lint types test check bench docs

install:
	python3 -m venv .venv
	$(PY) -m pip install -e ".[dev,docs]"

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

types:
	$(PY) -m mypy

test:
	$(PY) -m pytest

check: lint types test

bench:
	$(PY) benchmarks/run.py

docs:
	$(PY) -m mkdocs build --strict
