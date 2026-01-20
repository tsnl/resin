#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT=$(dirname "$(realpath "${BASH_SOURCE[0]}")")

pushd "${PROJECT_ROOT}/assets" > /dev/null 2>&1
    tar -xf kenney.tar.zstd
popd > /dev/null 2>&1
