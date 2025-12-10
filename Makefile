default: wheel

#
# Sync
#

.PHONY: sync
sync: bundled-data
	uv sync --all-extras

.PHONY: bundled-data
bundled-data:
	uv run --package zfw --extra dev python ./build-bundled-data.py

#
# Sandbox:
#

.PHONY: sandbox
sandbox: sync
	uv run --package zfw_sandbox zfw-sandbox --debug

#
# Tests:
#

.PHONY: tests
tests: sync
	uv run --package zfw --extra dev python -m pytest -vs --tb=short .

#
# Ruff:
#

.PHONY: format
format:
	uv run --package zfw --extra dev -- ruff format .
	uv run --package zfw --extra dev -- ruff check --select I --fix .

.PHONY: check-formatting
check-formatting:
	uv run --package zfw --extra dev -- ruff check --select I .

#
# Type-check:
#

.PHONY: check-typing
check-typing:
	uv run --package zfw --extra dev -- pyright

#
# Build:
#

.PHONY: wheel
wheel: sync check-formatting check-typing tests
	uv build --all
