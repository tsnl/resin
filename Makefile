default: tests format-check typecheck


#
# Tests:
#

.PHONY: test

test:
	uv run --extra dev -- pytest -v --tb=short tests/

#
# Ruff:
#

.PHONY: format format-check

format:
	uv run --with zero --extra dev -- ruff format .
	uv run --with zero --extra dev -- ruff check --select I --fix .

format-check:
	uv run --with zero --extra dev -- ruff check --select I .

#
# Type-check:
#

.PHONY: check

check:
	uv run --with zero -- pyright
