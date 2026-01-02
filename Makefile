BUNDLED_DATA=src/zfw/bundled_data

#
# Default:
#

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
# Build:
#

.PHONY: build
build: build-fonts

# Fonts:
#

.PHONY: build-fonts
build-fonts:
	mkdir -p $(BUNDLED_DATA)/fonts
	cp res/fonts/Inter/Inter.ttf $(BUNDLED_DATA)/fonts/
	cp res/fonts/Lora/Lora.ttf $(BUNDLED_DATA)/fonts/
	cp res/fonts/SourceCodePro/SourceCodePro.ttf $(BUNDLED_DATA)/fonts/
	uv run --package zfw_bmfont_cooker zfw-bmfont-cooker

#
# Deploy:
#

.PHONY: wheel
wheel: sync check tests
	uv build --all
