default: wheel

#
# Sync
#

.PHONY: sync
sync: bundled-data
	uv sync --extra dev

.PHONY: bundled-data
bundled-data:
	uv run --extra dev python ./build-bundled-data.py

#
# Sandbox:
#

.PHONY: sandbox
sandbox: sync
	uv run --extra dev zfw-sandbox

#
# Tests:
#

.PHONY: tests
tests: sync
	uv run --extra dev python -m pytest -vs --tb=short .

#
# Ruff:
#

.PHONY: format
format:
	uv run --with zfw --extra dev -- ruff format .
	uv run --with zfw --extra dev -- ruff check --select I --fix .

.PHONY: check-formatting
check-formatting:
	uv run --with zfw --extra dev -- ruff check --select I .

#
# Type-check:
#

.PHONY: check-typing
check-typing:
	uv run --with zfw -- pyright

#
# Build:
#

.PHONY: wheel
wheel: sync check-formatting check-typing tests
	uv build
