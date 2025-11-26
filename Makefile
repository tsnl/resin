default: tests format-check typecheck

#
# Configuration:
#

VULKAN_SDK_VERSION=1.4.328.1

#
# Setup:
#

.PHONY: setup clean

setup:
	uv run --extra dev --with setup -- setup setup --vulkan-sdk-version $(VULKAN_SDK_VERSION)

clean:
	uv run --extra dev --with setup -- setup clean --vulkan-sdk-version $(VULKAN_SDK_VERSION) || true

#
# Tests:
#

.PHONY: test

test:
	source build/vulkan-sdk/${VULKAN_SDK_VERSION}/setup-env.sh && \
	uv run --extra dev -- pytest -v --tb=short tests/

#
# Ruff:
#

.PHONY: format format-check

format:
	uv run --with zero -- ruff format .

format-check:
	uv run --with zero -- ruff check .
	
#
# Type-check:
#

.PHONY: check

check:
	uv run --with zero -- pyright
