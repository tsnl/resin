BUNDLED_DATA=src/zfw/bundled_data

#
# Default:
#

default: wheel

#
# Run:
#

.PHONY: sandbox
sandbox: sync build
	uv run --package zfw_sandbox zfw-sandbox --debug

.PHONY: tests
tests: sync build
	uv run --package zfw --extra dev python -m pytest -vs --tb=short .

.PHONY: bench
tests-profiling: sync build
	uv run --package zfw --extra dev python -m pytest --profile -vs --tb=short .
	uv run flameprof --width 4096 prof/combined.prof > prof/combined.svg
	uv run flameprof --width 4096 prof/test_basic_draw_2d.prof > prof/test_basic_draw_2d.svg


#
# Develop:
#

.PHONY: sync
sync:
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
build-fonts: \
	$(BUNDLED_DATA)/fonts/monospaced.zfw_atlas \
	$(BUNDLED_DATA)/fonts/SourceCodePro.ttf \
	$(BUNDLED_DATA)/fonts/sans-serif.zfw_atlas \
	$(BUNDLED_DATA)/fonts/Inter.ttf \
	$(BUNDLED_DATA)/fonts/serif.zfw_atlas \
	$(BUNDLED_DATA)/fonts/Lora.ttf

$(BUNDLED_DATA)/fonts/monospaced.zfw_atlas: sync $(BUNDLED_DATA)/fonts
	uv run --package zfw_bmfont_cooker zfw-bmfont-cooker $@
$(BUNDLED_DATA)/fonts/SourceCodePro.ttf: $(BUNDLED_DATA)/fonts
	cp res/fonts/SourceCodePro/SourceCodePro.ttf $<

$(BUNDLED_DATA)/fonts/sans-serif.zfw_atlas: sync $(BUNDLED_DATA)/fonts
	uv run --package zfw_bmfont_cooker zfw-bmfont-cooker $@
$(BUNDLED_DATA)/fonts/Inter.ttf: $(BUNDLED_DATA)/fonts
	cp res/fonts/Inter/Inter.ttf $<

$(BUNDLED_DATA)/fonts/serif.zfw_atlas: sync $(BUNDLED_DATA)/fonts
	uv run --package zfw_bmfont_cooker zfw-bmfont-cooker $@
$(BUNDLED_DATA)/fonts/Lora.ttf: $(BUNDLED_DATA)/fonts
	cp res/fonts/Lora/Lora.ttf $<

$(BUNDLED_DATA)/fonts:
	mkdir -p $@


#
# Deploy:
#

.PHONY: wheel
wheel: sync build check tests
	uv build --all
