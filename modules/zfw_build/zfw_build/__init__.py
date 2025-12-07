__all__ = [
    "Shader",
    "compile_shaders",
]

from dataclasses import dataclass
from pathlib import Path

from .shaders import Shader, compile_shaders


@dataclass
class Input:
    root: Path
    targets: dict[str, Path]
    shaders: list[Shader]


def run(input_: Input):
    compile_shaders(input_.root, input_.shaders, input_.targets)
