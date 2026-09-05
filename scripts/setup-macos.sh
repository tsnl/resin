#!/usr/bin/env bash
set -euo pipefail

if [[ $(uname -s) != Darwin ]]; then
    printf '%s\n' 'This script requires macOS.' >&2
    exit 1
fi

if ! command -v brew >/dev/null 2>&1; then
    printf '%s\n' 'Install Homebrew from https://brew.sh, then rerun this script.' >&2
    exit 1
fi

if ! xcrun --find clang >/dev/null 2>&1; then
    printf '%s\n' 'Run xcode-select --install, finish installing the command-line tools, then rerun this script.' >&2
    exit 1
fi

brew install cmake rustup shaderc molten-vk vulkan-loader vulkan-tools

# shellcheck disable=SC2016
printf '%s\n' \
    '' \
    'Dependencies installed. Run these commands in your shell from the repository root:' \
    '' \
    'export PATH="$(brew --prefix rustup)/bin:$PATH"' \
    'export DYLD_LIBRARY_PATH="$(brew --prefix vulkan-loader)/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"' \
    'export VK_DRIVER_FILES="$(brew --prefix molten-vk)/etc/vulkan/icd.d/MoltenVK_icd.json"' \
    '' \
    'git submodule update --init' \
    'cargo run -- examples/window.resin'
