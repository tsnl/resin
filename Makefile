default: sync

#
# Sync:
#

.PHONY: sync
sync: shaders
	uv sync --extra dev

#
# Shaders:
#

.PHONY: shaders
shaders:
	uv run --with zero --extra dev tools/build-shaders.py

#
# Tests:
#

.PHONY: tests
tests: shaders
	uv run --extra dev python -m pytest -v --tb=short tests/

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

.PHONY: typecheck

typecheck:
	uv run --with zero -- pyright

#
# Build:
#

.PHONY: wheel
wheel: shaders
	uv build
