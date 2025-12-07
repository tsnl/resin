from pathlib import Path

import zfw_build


class CustomBuildHook(zfw_build.AssetBuildHookBase):
    ROOT = Path(__file__).parent

    TARGETS: dict[str, Path] = {
        "main": ROOT / "zfw_core" / "bundled_data",
        "test": ROOT / "tests" / "gpu_test" / "data",
    }

    SHADERS: list[zfw_build.Shader] = [
        zfw_build.Shader(
            source=(ROOT / "shaders/tests/triangle.slang"),
            stages={"vertex": "vertexMain", "fragment": "fragmentMain"},
            targets=["test"],
        ),
        zfw_build.Shader(
            source=(ROOT / "shaders/tests/tinted_bitmap.slang"),
            stages={"vertex": "vertexMain", "fragment": "fragmentMain"},
            targets=["test"],
        ),
        # zfw_build.Shader(
        #     source=Path("shaders/zfw/r2d.slang"),
        #     stages={"vertex": "vertexMain", "fragment": "fragmentMain"},
        #     targets=["main"],
        # ),
    ]
