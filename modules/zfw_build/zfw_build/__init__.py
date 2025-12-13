__all__ = [
    "Shader",
    "compile_shaders",
]

from pathlib import Path

from .shaders import Shader, compile_shaders
from .unpack import extract_tar_zstd


REPO_ROOT = Path(__file__).parent.parent.parent.parent


def unpack_compressed_data():
    extract_tar_zstd(
        REPO_ROOT / "modules/compressed_data/KenneyGameAssetsAllInOne-3_3_0.tar.zstd",
        REPO_ROOT / "data",
    )


def zfw_core_shaders():
    package_root = REPO_ROOT / "modules/zfw"

    shaders: list[Shader] = [
        Shader(
            source="shaders/zfw/r2d.slang",
            stages={"vertex": "vertexMain", "fragment": "fragmentMain"},
        ),
    ]

    compile_shaders(
        package_root=package_root,
        shaders=shaders,
        package_output_path=(package_root / "zfw/bundled_data"),
    )


def main():
    unpack_compressed_data()
    zfw_core_shaders()
