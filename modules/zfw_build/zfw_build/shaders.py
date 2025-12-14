import subprocess
from pathlib import Path
from typing import Literal, TypeAlias

import pydantic


ShaderStage: TypeAlias = Literal["vertex", "fragment"]


class Shader(pydantic.BaseModel):
    source: str
    """Path to the shader source file, relative to the package root."""

    stages: dict["ShaderStage", str]
    """Mapping of shader stage to entry point name in the source file."""

    def __post_init__(self):
        assert not Path(self.source).is_absolute()


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
    shader: Shader,
    stage: ShaderStage,
) -> Path:
    return Path(shader.source).with_suffix(f".{stage[:4]}.spv")


def build_shaders(
    package_root: Path,
    shaders: list[Shader],
    package_output_path: Path,
):
    # Make the intermediate build directory:
    build_root = Path("build/build-shaders")
    build_root.mkdir(parents=True, exist_ok=True)

    # Build each shader, outputting to the intermediate build directory:
    for shader in shaders:
        for stage in shader.stages:
            input_path = package_root / shader.source
            assert input_path.is_file(), f"Shader source not found: {input_path}"

            output_suffix = get_output_path_suffix(shader, stage)
            output_path = package_output_path / output_suffix

            output_path.parent.mkdir(parents=True, exist_ok=True)
            slangc(
                path=input_path,
                stage=stage,
                entry=shader.stages[stage],
                output=output_path,
            )
