PYTHON ?= python3.12
VENV ?= .venv
BIN := $(VENV)/bin

.PHONY: help setup samples ingest serve eval test lint up clean-data

help:
	@echo "setup       create $(VENV) and install DeedLens with dev extras"
	@echo "samples     regenerate the fictional sample PDFs        (Phase 2)"
	@echo "ingest      ingest PDFs in data/raw                      (Phase 3)"
	@echo "serve       run the API with auto-reload                 (Phase 5)"
	@echo "eval        run the answer evaluation on the samples     (Phase 7)"
	@echo "test        run unit tests (no key, no network)"
	@echo "lint        ruff check"
	@echo "up          docker compose up --build                    (Phase 8)"
	@echo "clean-data  delete data/ after confirmation"

setup:
	test -d $(VENV) || $(PYTHON) -m venv $(VENV)
	@echo "Intel Mac? Run first: $(BIN)/pip install torch==2.2.2 'numpy<2'"
	$(BIN)/pip install -e ".[dev]"

samples:
	$(BIN)/python samples/make_samples.py

ingest serve eval up:
	@echo "'make $@' is not implemented yet (see docs/BUILD_PROMPT.md §19)."

test:
	$(BIN)/pytest -q

lint:
	$(BIN)/ruff check .

clean-data:
	@read -p "Delete ./data (all ingested documents and indexes)? [y/N] " ans; \
	if [ "$$ans" = "y" ]; then rm -rf data && echo "data/ deleted"; else echo "kept"; fi
