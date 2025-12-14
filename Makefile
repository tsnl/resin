default: wheel

#
# Run:
#

.PHONY: sandbox
sandbox: sync
	uv run --package zfw_sandbox zfw-sandbox --debug

.PHONY: tests
tests: sync
	uv run --package zfw --extra dev python -m pytest -vs --tb=short .

#
# Develop:
#

.PHONY: sync
sync: build
	uv sync --all-extras

.PHONY: build
build:
	uv run --package zfw_build zfw-build -p modules/zfw
	uv run --package zfw_build zfw-build -p modules/zfw_sandbox

.PHONY: check
check:
	uv run --package zfw --extra dev -- ruff format --check .
	uv run --package zfw --extra dev -- ruff check .
	uv run --package zfw --extra dev -- pyright

.PHONY: format format-check
format:
	uv run --package zfw --extra dev -- ruff format .
	uv run --package zfw --extra dev -- ruff check --fix .

#
# Deploy:
#

.PHONY: wheel
wheel: sync check tests
	uv build --all
