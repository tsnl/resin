"""
Cross-platform Makefile replacement using PyInvoke.
See: https://www.pyinvoke.org/
"""

from pathlib import Path
import shutil
import sys
import platform

import rich
import rich.prompt

from invoke import task, Context  # type: ignore

#
# Configuration
#

BUNDLED_DATA = "src/resin/bundled_data"
INTERACTIVE = True
PTY = platform.system() != "Windows" and INTERACTIVE


#
# Clean:
#


@task
def clean(c: Context, force: bool = False):
    if INTERACTIVE and not force:
        confirm = rich.prompt.Confirm.ask(
            "[bold yellow]WARNING[/bold yellow] Running `git clean -fxd .` will delete all untracked files.",
            default=False,
        )
        if not confirm:
            rich.print(
                "[bold yellow]WARNING[/bold yellow] Aborting clean.", file=sys.stderr
            )
            return

    _run(c, "git clean -fxd .")


#
# Build:
#


@task
def build_fonts(c: Context):
    _ensure_fonts_output_dir()
    _emit_bitmap_fonts(c)
    _copy_ttf_font_files()


def _ensure_fonts_output_dir():
    output_dir = Path(f"{BUNDLED_DATA}/fonts")
    output_dir.mkdir(parents=True, exist_ok=True)


def _emit_bitmap_fonts(c: Context):
    for name in ["monospaced", "sans-serif", "serif"]:
        output_path = Path(f"{BUNDLED_DATA}/fonts/{name}.resin_atlas")
        if output_path.is_dir():
            continue

        _uv_run(c, f"resin-bmfont-cooker {output_path}")


def _copy_ttf_font_files():
    for font_name in ["SourceCodePro", "Inter", "Lora"]:
        output_path = Path(f"{BUNDLED_DATA}/fonts/{font_name}.ttf")
        if output_path.is_file():
            continue

        input_path = Path(f"res/fonts/{font_name}/{font_name}.ttf")
        shutil.copy(input_path, output_path)


@task(pre=[build_fonts])
def build(_: Context): ...


#
# Run:
#


@task(pre=[build])
def sponza(c: Context):
    _uv_run(c, "scripts/sponza-benchmark.py")


@task(pre=[build])
def editor(c: Context):
    _uv_run(c, "resin-editor --debug")


@task(pre=[build])
def tests(
    c: Context,
    profile: bool = False,
    filter: str = "",
):
    cmd = "pytest -vs --tb=short ."

    if filter:
        cmd += f" -k '{filter}'"
    if profile:
        cmd += " --profile"

    _uv_run(c, cmd)

    if profile:
        _uv_run(c, "flameprof --width 4096 prof/combined.prof > prof/combined.svg")


@task(pre=[build])
def test_mitsuba_ref(c: Context):
    """Run the Mitsuba reference comparison test."""
    _uv_run(c, "python scripts/test-mitsuba-ref.py")


#
# Develop:
#


@task
def sync(c: Context):
    _run(c, "uv sync --all-extras")


@task
def check(c: Context):
    _uv_run(c, "ruff format --check .")
    _uv_run(c, "ruff check .")
    _uv_run(c, "pyright")


@task
def format(c: Context):
    _uv_run(c, "ruff format .")
    _uv_run(c, "ruff check --fix .")


#
# Deploy:
#


@task(pre=[sync, build, check, tests])
def wheel(c: Context):
    _run(c, "uv build --all")


#
# Implementation:
#


def _uv_run(c: Context, command: str):
    _run(c, f"uv run --all-packages {command}")


def _run(c: Context, command: str):
    c.run(command, pty=PTY)
