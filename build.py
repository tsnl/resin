#!/usr/bin/env -S uv run

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent

COMPILE_FLAGS = [
    "-std=c11",
    "-Wall",
    "-Wextra",
    "-Iresin_gen/bundled_data/wgpu/include",
    "-Iresin_runtime/include",
]


def quiet_run(args: list[str], cwd: Path | str | None = None) -> None:
    """Run a command quietly, printing output only on failure."""
    result = subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=Path(cwd) if cwd else ROOT,
    )
    if result.returncode != 0:
        sys.stderr.buffer.write(result.stdout)
        sys.exit(1)


def generate_compile_flags() -> None:
    """Generate compile_flags.txt for LSP support."""
    (ROOT / "compile_flags.txt").write_text("\n".join(COMPILE_FLAGS) + "\n")


def git_repo_setup() -> None:
    """Set up git LFS and submodules."""
    quiet_run(["git", "lfs", "install"])
    quiet_run(["git", "lfs", "pull"])
    quiet_run(["git", "submodule", "update", "--init", "--recursive"])


def copy_wgpu_native() -> None:
    """Build wgpu-native and copy artifacts."""
    wgpu_dir = ROOT / "deps" / "wgpu-native"
    quiet_run(["make", "lib-native-release"], cwd=wgpu_dir)

    output_dir = ROOT / "resin_gen" / "bundled_data" / "wgpu"
    lib_dir = output_dir / "lib"
    include_dir = output_dir / "include"

    lib_dir.mkdir(parents=True, exist_ok=True)
    include_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(
        wgpu_dir / "target" / "release" / "libwgpu_native.a",
        lib_dir / "libwgpu_native.a",
    )
    shutil.copy(
        wgpu_dir / "ffi" / "webgpu-headers" / "webgpu.h",
        include_dir / "webgpu.h",
    )


def build_resin_runtime() -> None:
    """Build resin_runtime as a static library."""

    runtime_dir = ROOT / "resin_runtime"
    dst = ROOT / "resin_gen" / "bundled_data" / "resin_runtime"

    inc_dir = dst / "include"
    shutil.rmtree(inc_dir, ignore_errors=True)
    shutil.copytree(runtime_dir / "include", inc_dir)

    lib_dir = dst / "lib"
    lib_dir.mkdir(parents=True, exist_ok=True)

    # Find all C source files
    src_files = list((runtime_dir / "src").glob("*.c"))

    with tempfile.TemporaryDirectory() as tmpdir:
        obj_files: list[Path] = []

        # Compile each source file to object file
        for src_file in src_files:
            obj_file = Path(tmpdir) / (src_file.stem + ".o")
            obj_files.append(obj_file)
            quiet_run(
                [
                    "cc",
                    "-c",
                    *COMPILE_FLAGS,
                    str(src_file),
                    "-o",
                    str(obj_file),
                ]
            )

        # Create static library
        lib_file = lib_dir / "libresin_runtime.a"
        quiet_run(["ar", "rcs", str(lib_file), *[str(o) for o in obj_files]])


def main() -> None:
    generate_compile_flags()
    git_repo_setup()
    copy_wgpu_native()
    build_resin_runtime()
    print("Done", file=sys.stderr)


if __name__ == "__main__":
    main()
