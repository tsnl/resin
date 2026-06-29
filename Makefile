UV ?= uv
PATHS := src tests examples

.PHONY: check format check-ruff check-basedpyright check-pytest

check: check-ruff check-basedpyright check-pytest

format:
	$(UV) run ruff format $(PATHS)
	$(UV) run ruff check --fix $(PATHS)

check-ruff:
	$(UV) run ruff check $(PATHS)

check-basedpyright:
	$(UV) run basedpyright --level warning --warnings $(PATHS)

check-pytest:
	$(UV) run pytest tests