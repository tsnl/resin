#!/usr/bin/env python3
"""Build script to compile Slang shaders to SPIR-V.

This script compiles all .slang files in the shader/ directory to SPIR-V
and places the output in zero/data/shader/ and tests/gpu_test/data/.
"""

import shutil
import subprocess
import sys
from pathlib import Path


def main():
    """Compile all Slang shaders to SPIR-V."""
    # Find project root
    project_root = Path(__file__).parent.parent
    shader_dir = project_root / "shader"

    # Output directories
    package_shader_dir = project_root / "zero" / "data" / "shader"
    test_shader_dir = project_root / "tests" / "gpu_test" / "data"

    # Ensure output directories exist
    package_shader_dir.mkdir(parents=True, exist_ok=True)
    test_shader_dir.mkdir(parents=True, exist_ok=True)

    # Check if slangc is available
    slangc_path = shutil.which("slangc")
    if not slangc_path:
        print("ERROR: slangc not found in PATH", file=sys.stderr)
        print("Please install Slang shader compiler:", file=sys.stderr)
        print("  https://github.com/shader-slang/slang/releases", file=sys.stderr)
        sys.exit(1)

    print(f"Found slangc: {slangc_path}")
    print(f"Compiling shaders from: {shader_dir}")

    # Find all .slang files
    slang_files = list(shader_dir.glob("*.slang"))
    if not slang_files:
        print(f"WARNING: No .slang files found in {shader_dir}")
        return

    # Compile each shader
    for slang_file in slang_files:
        print(f"\nCompiling: {slang_file.name}")

        # Compile vertex shader
        for output_dir, dir_name in [
            (package_shader_dir, "package"),
            (test_shader_dir, "tests"),
        ]:
            vertex_output = output_dir / f"{slang_file.stem}.vert.spv"
            try:
                subprocess.run(
                    [
                        slangc_path,
                        str(slang_file),
                        "-target",
                        "spirv",
                        "-stage",
                        "vertex",
                        "-entry",
                        "vertexMain",
                        "-o",
                        str(vertex_output),
                    ],
                    check=True,
                    capture_output=True,
                )
                print(f"  ✓ {dir_name}: {vertex_output.relative_to(project_root)}")
            except subprocess.CalledProcessError as e:
                print(
                    f"  ✗ ERROR compiling vertex shader for {dir_name}:",
                    file=sys.stderr,
                )
                print(f"    {e.stderr.decode()}", file=sys.stderr)
                sys.exit(1)

            # Compile fragment shader
            fragment_output = output_dir / f"{slang_file.stem}.frag.spv"
            try:
                subprocess.run(
                    [
                        slangc_path,
                        str(slang_file),
                        "-target",
                        "spirv",
                        "-stage",
                        "fragment",
                        "-entry",
                        "fragmentMain",
                        "-o",
                        str(fragment_output),
                    ],
                    check=True,
                    capture_output=True,
                )
                print(f"  ✓ {dir_name}: {fragment_output.relative_to(project_root)}")
            except subprocess.CalledProcessError as e:
                print(
                    f"  ✗ ERROR compiling fragment shader for {dir_name}:",
                    file=sys.stderr,
                )
                print(f"    {e.stderr.decode()}", file=sys.stderr)
                sys.exit(1)

    print(f"\n✓ Successfully compiled {len(slang_files)} shader(s)")


if __name__ == "__main__":
    main()
