#!/bin/bash
# Sponza benchmark runner
# Renders Sponza from multiple viewpoints and reports timing statistics.
#
# Usage: ./scripts/sponza.sh
#
# Output:
#   - Images saved to output/sponza-benchmark/<viewpoint>/
#   - Timing report saved to output/sponza-benchmark/timing-report.json

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "Running Sponza benchmark..."
echo ""

uv run python scripts/sponza-benchmark.py "$@"
