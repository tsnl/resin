#
# Tests:
#

.PHONY: tests

tests:
	uv run --with zero -- pytest -v --tb=short tests/

#
# Ruff:
#

.PHONY: format format-check

format:
	uv run --with zero -- ruff format .

format-check:
	uv run --with zero -- ruff check .
	
#
# Typecheck:
#

.PHONY: typecheck

typecheck:
	uv run --with zero -- pyright
