__all__ = [
    "Shader",
    "build_shaders",
]

import argparse
from pathlib import Path
import tomllib as toml

import pydantic

from .shaders import Shader, build_shaders
from .datapacks import build_datapacks


REPO_ROOT = Path(__file__).parent.parent.parent.parent


class Config(pydantic.BaseModel):
    shaders: list[Shader] = []
    """Shaders to compile into the package's bundled data directory."""

    data_prefix: str = "data"
    """Prefix under 'bundled_data_path' where data packs are extracted/copied."""

    data_packs: list[str] = []
    """List of '.tar.zstd' files or folders to extract/copy into the data prefix."""


def load_pyproject_config(project_root: Path) -> tuple[str, Config]:
    pyproject_toml_path = project_root / "pyproject.toml"
    with open(pyproject_toml_path, "rb") as f:
        pyproject_dict = toml.load(f)

    project_name = pyproject_dict.get("project", {}).get("name")
    if project_name is None:
        raise ValueError("pyproject.toml does not contain [project] section with name")

    zfw_build_config_dict = pyproject_dict.get("tool", {}).get("zfw_build")
    if zfw_build_config_dict is None:
        raise ValueError("pyproject.toml does not contain [tool.zfw_build] section")

    return project_name, Config(**zfw_build_config_dict)


def main():
    ap = argparse.ArgumentParser(
        description="Build tool for zfw Python projects.",
    )
    ap.add_argument(
        "-p",
        "--package",
        type=Path,
        default=".",
        help="Path to the root of the Python package to build.",
    )
    args = ap.parse_args()

    package_name, config = load_pyproject_config(args.package)

    # Shaders:
    build_shaders(
        package_root=args.package,
        shaders=config.shaders,
        package_output_path=(args.package / package_name / "bundled_data"),
    )

    # Data packs:
    build_datapacks(
        file_path_list=[args.package / suffix for suffix in config.data_packs],
        dst_parent_dir=(
            args.package / package_name / "bundled_data" / config.data_prefix
        ),
    )
