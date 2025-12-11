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
# Lint, Check: Format, Lint, Typecheck:
#

.PHONY: check
check: format-check lint-check type-check

.PHONY: format format-check
format:
	uv run --package zfw --extra dev -- ruff format .
format-check:
	uv run --package zfw --extra dev -- ruff format --check

.PHONY: lint lint-check
lint: format
	uv run --package zfw --extra dev -- ruff check --fix .
lint-check:
	uv run --package zfw --extra dev -- ruff check .

.PHONY: type-check
type-check:
	uv run --package zfw --extra dev -- pyright

#
# Build:
#

.PHONY: wheel
wheel: sync check-formatting check-typing tests
	uv build --all
