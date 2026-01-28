#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

quiet-pushd () {
    pushd "$1" > /dev/null
}
quiet-popd () {
    popd > /dev/null
}
quiet-run () {
    # Run a command quietly, printing output only on failure.
    local TEMPFILE
    TEMPFILE="$(mktemp)"
    if ! "$@" > "${TEMPFILE}" 2>&1; then
        cat "${TEMPFILE}"
        rm "${TEMPFILE}"
        return 1
    fi
    rm "${TEMPFILE}"
    return 0
}

git-repo-setup () {
    quiet-pushd "${ROOT}"
        quiet-run git lfs install && quiet-run git lfs pull
        quiet-run git submodule update --init --recursive
    quiet-popd
}

build-wgpu-native () {
    local OUTPUT_DIR
    
    quiet-pushd "${ROOT}/deps/wgpu-native"
        quiet-run make lib-native-release
    quiet-popd

    OUTPUT_DIR="${ROOT}/src/resin_gen/bundled_data/wgpu/"
    mkdir -p "${OUTPUT_DIR}/lib"
    mkdir -p "${OUTPUT_DIR}/include"
    
    cp "${ROOT}/deps/wgpu-native/target/release/libwgpu_native.a" "${OUTPUT_DIR}/lib/libwgpu_native.a"
    cp "${ROOT}/deps/wgpu-native/ffi/webgpu-headers/webgpu.h" "${OUTPUT_DIR}/include/webgpu.h"
}

main () {
    git-repo-setup
    build-wgpu-native
    echo "Done" >&2
}

main "$@" >&2

