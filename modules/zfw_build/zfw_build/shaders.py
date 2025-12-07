import shutil
import subprocess
from pathlib import Path
from typing import Literal, TypeAlias
from dataclasses import dataclass


@dataclass
class Shader:
    source: Path
    stages: dict[ShaderStage, str]
    targets: list[str]


ShaderStage: TypeAlias = Literal["vertex", "fragment"]


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


def get_output_path_suffix(
    package_root: Path,
    shader: Shader,
    stage: ShaderStage,
) -> Path:
    return shader.source.with_suffix(f".{stage[:4]}.spv").relative_to(package_root)


def compile_shaders(
    package_root: Path,
    shaders: list[Shader],
    targets: dict[str, Path],
):
    # Make the intermediate build directory:
    build_root = Path("build/build-shaders")
    build_root.mkdir(parents=True, exist_ok=True)

    # Build each shader, outputting to the intermediate build directory:
    for shader in shaders:
        for stage in shader.stages:
            input_path = shader.source

            output_suffix = get_output_path_suffix(package_root, shader, stage)
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
        # Check all targets are valid
        for target in shader.targets:
            if target not in targets:
                raise ValueError(
                    f"Unknown target '{target}' for shader '{shader.source}'"
                )

        # Gather all unique target output roots for this shader:
        target_output_roots = {targets[target] for target in shader.targets}

        # Copy each stage's file to each target output root:
        for target_output_root in target_output_roots:
            for stage in shader.stages:
                output_suffix = get_output_path_suffix(package_root, shader, stage)
                build_output_path = build_root / output_suffix
                target_output_path = target_output_root / output_suffix

                target_output_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(build_output_path, target_output_path)
