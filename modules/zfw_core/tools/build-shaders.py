#!/usr/bin/env python3

"""
Build script to compile Slang shaders to SPIR-V.

Edit the 'SHADERS' or 'TARGET_ROOT_MAP' variables below to modify which shaders
are built and where they are output.
"""

import shutil
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Literal, TypeAlias


ShaderStage: TypeAlias = Literal["vertex", "fragment"]
Target: TypeAlias = Literal["zfw_core", "zfw_core_tests", "zfw_sandbox"]


@dataclass
class Shader:
    source: Path
    stages: dict[ShaderStage, str]
    targets: list[Target]


def slangc(path: Path, stage: ShaderStage, entry: str, output: Path):
    args = [
        "slangc",
        str(path),
        "-target",
        "spirv",
        "-stage",
        stage,
        "-entry",
        entry,
        "-o",
        str(output),
    ]
    print(" ".join(args))
    res = subprocess.run(args, check=False, capture_output=False)
    if res.returncode != 0:
        exit(res.returncode)


def get_output_path_suffix(shader: Shader, stage: ShaderStage) -> Path:
    return shader.source.with_suffix(f".{stage[:4]}.spv")


def compile_shaders(shaders: list[Shader], target_root_map: dict[Target, Path]):
    # Make the intermediate build directory:
    build_root = Path("build/build-shaders")
    build_root.mkdir(parents=True, exist_ok=True)

    # Build each shader, outputting to the intermediate build directory:
    for shader in shaders:
        for stage in shader.stages:
            input_path = shader.source

            output_suffix = get_output_path_suffix(shader, stage)
            build_output_path = build_root / output_suffix

            build_output_path.parent.mkdir(parents=True, exist_ok=True)
            slangc(
                path=input_path,
                stage=stage,
                entry=shader.stages[stage],
                output=build_output_path,
            )

    # Copy each shader to one or more target output directories:
    for shader in shaders:
        # Gather all unique target output roots for this shader:
        target_output_roots = {target_root_map[target] for target in shader.targets}

        # Copy each stage's file to each target output root:
        for target_output_root in target_output_roots:
            for stage in shader.stages:
                output_suffix = get_output_path_suffix(shader, stage)
                build_output_path = build_root / output_suffix
                target_output_path = target_output_root / output_suffix

                target_output_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(build_output_path, target_output_path)


def main():
    compile_shaders(SHADERS, TARGET_ROOT_MAP)


SHADERS: list[Shader] = [
    Shader(
        source=Path("modules/zfw_core/shaders/tests/triangle.slang"),
        stages={"vertex": "vertexMain", "fragment": "fragmentMain"},
        targets=["zfw_core_tests", "zfw_sandbox"],
    ),
    Shader(
        source=Path("modules/zfw_core/shaders/tests/tinted_bitmap.slang"),
        stages={"vertex": "vertexMain", "fragment": "fragmentMain"},
        targets=["zfw_core_tests"],
    ),
    # Shader(
    #     source=Path("shaders/zfw/r2d.slang"),
    #     stages={"vertex": "vertexMain", "fragment": "fragmentMain"},
    #     targets=["zfw"],
    # ),
]

TARGET_ROOT_MAP: dict[Target, Path] = {
    "zfw_core": Path("modules/zfw_core/src/bundled_data"),
    "zfw_core_tests": Path("modules/zfw_core/tests/gpu_test/data"),
    "zfw_sandbox": Path("modules/zfw_sandbox/src/bundled_data"),
}


if __name__ == "__main__":
    main()
