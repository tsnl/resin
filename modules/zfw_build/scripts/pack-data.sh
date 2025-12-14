#!/usr/bin/env bash

set -euo pipefail

SCRIPT="${BASH_SOURCE[0]}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

error () {
    echo -e "\e[1;31mERROR\e[0m $*" >&2
}

info () {
    echo -e "\e[1;34mINFO\e[0m $*" >&2
}

main () {
    local INPUT
    local OUTPUT
    
    if [ "$#" -ne 1 ]; then
        error "${SCRIPT} <path-to-folder>"
        return 1
    fi

    INPUT="$1"

    if ! [ -d "${INPUT}" ]; then
        error "Input path is not a directory: ${INPUT}"
        return 1
    fi

    if [ -f "${INPUT}/DONE" ]; then
        rm -f "${INPUT}/DONE"
    fi

    info "Compressing ${INPUT} to ${OUTPUT}"
    OUTPUT="${REPO_ROOT}/modules/compressed_data/$(basename "${INPUT}").tar.zst"
    tar cf - "${INPUT}" | zstd --ultra -22 > "${OUTPUT}"

    return 0    
}