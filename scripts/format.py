"""Format and lint-fix the resin codebase with ruff."""

import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

_FORMAT_PATHS = [
    "src",
    "tests",
    "examples",
    "scripts",
]


def _tool_executable(name: str) -> str:
    executable = shutil.which(name)
    if executable is None:
        print(f"error: {name} not found on PATH", file=sys.stderr)
        raise SystemExit(1)
    return executable


def _run_ruff(subcommand: str, args: Sequence[str]) -> int:
    print(f"=== ruff {subcommand} ===", flush=True)
    result = subprocess.run(
        [_tool_executable("ruff"), subcommand, *args],
        cwd=_REPO_ROOT,
        check=False,
    )
    return result.returncode


def main() -> int:
    paths = tuple(_FORMAT_PATHS)
    failures: list[str] = []

    if _run_ruff("format", paths) != 0:
        failures.append("ruff format")

    if _run_ruff("check", ("--fix", *paths)) != 0:
        failures.append("ruff check --fix")

    if failures:
        print(f"Failed: {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
