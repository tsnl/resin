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
	uv run --with zfw --extra dev modules/zfw_core/tools/build-shaders.py

#
# Tests:
#

.PHONY: tests
tests: shaders
	uv run --extra dev python -m pytest -vs --tb=short .

#
# Ruff:
#

.PHONY: format format-check

format:
	uv run --with zfw --extra dev -- ruff format .
	uv run --with zfw --extra dev -- ruff check --select I --fix .

format-check:
	uv run --with zfw --extra dev -- ruff check --select I .

#
# Type-check:
#

.PHONY: typecheck

typecheck:
	uv run --with zfw -- pyright

#
# Build:
#

.PHONY: wheel
wheel: shaders
	uv build
