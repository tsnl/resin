"""Run static analysis checks for the resin codebase."""

import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

_CHECK_PATHS = [
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


def _run_check(name: str, args: Sequence[str]) -> int:
    print(f"=== {name} ===", flush=True)
    result = subprocess.run(
        [_tool_executable(name), *args],
        cwd=_REPO_ROOT,
        check=False,
    )
    return result.returncode


_ALL_CHECKS: dict[str, tuple[str, ...]] = {
    "ruff": (
        "check",
        *_CHECK_PATHS,
    ),
    "basedpyright": (
        "--level",
        "warning",
        "--warnings",
        *_CHECK_PATHS,
    ),
    "pytest": ("tests",),
}


def main(argv: Sequence[str] | None = None) -> int:
    selected = list(argv if argv is not None else sys.argv[1:])
    unknown = [name for name in selected if name not in _ALL_CHECKS]
    if unknown:
        names = ", ".join(_ALL_CHECKS)
        print(
            f"error: unknown check(s): {', '.join(unknown)} (expected: {names})",
            file=sys.stderr,
        )
        return 1

    checks = (
        [(name, _ALL_CHECKS[name]) for name in selected]
        if selected
        else list(_ALL_CHECKS.items())
    )

    failures: list[str] = []
    for name, args in checks:
        if _run_check(name, args) != 0:
            failures.append(name)

    if failures:
        print(f"Failed checks: {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
