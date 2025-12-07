from abc import ABC
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

from .shaders import Shader, compile_shaders


class AssetBuildHookBase(BuildHookInterface, ABC):
    ROOT: Path
    TARGETS: dict[str, Path]
    SHADERS: list[Shader]

    def initialize(self, version: str, build_data: dict) -> None:
        """
        This occurs immediately before each build.
        Any modifications to the build data will be seen by the build target.
        """

        _ = version
        _ = build_data

        compile_shaders(self.ROOT, self.SHADERS, self.TARGETS)

    def finalize(self, version: str, build_data: dict, artifact_path: str) -> None:
        """
        This occurs immediately after each build.
        """
        _ = version
        _ = build_data
        _ = artifact_path

    def clean(self, versions: list[str]) -> None:
        """
        This occurs before the build process if the -c / --clean flag was passed,
        or when invoking the clean command.
        """
