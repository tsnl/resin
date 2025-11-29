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
Target: TypeAlias = Literal["zero", "tests", "zero_sandbox"]


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
    subprocess.run(args, check=True, capture_output=True)


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
        source=Path("shaders/tests/triangle.slang"),
        stages={"vertex": "vertexMain", "fragment": "fragmentMain"},
        targets=["tests", "zero_sandbox"],
    ),
    Shader(
        source=Path("shaders/tests/tinted_bitmap.slang"),
        stages={"vertex": "vertexMain", "fragment": "fragmentMain"},
        targets=["tests"],
    ),
    Shader(
        source=Path("shaders/zero/render2d.slang"),
        stages={"vertex": "vertexMain", "fragment": "fragmentMain"},
        targets=["zero"],
    ),
]

TARGET_ROOT_MAP: dict[Target, Path] = {
    "zero": Path("zero/bundled_data"),
    "zero_sandbox": Path("zero/bundled_data"),
    "tests": Path("tests/gpu_test/data"),
}


if __name__ == "__main__":
    main()
