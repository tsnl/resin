from pathlib import Path

import zfw_build

ROOT = Path(__file__).parent


def zfw_core_main():
    zfw_core_root = ROOT / "modules/zfw_core"

    targets: dict[str, Path] = {
        "zfw_core": zfw_core_root / "zfw_core/bundled_data",
    }

    shaders: list[zfw_build.Shader] = [
        zfw_build.Shader(
            source=(zfw_core_root / "shaders/zfw/r2d.slang"),
            stages={"vertex": "vertexMain", "fragment": "fragmentMain"},
            targets=["zfw_core"],
        ),
    ]

    zfw_build.run(
        input_=zfw_build.Input(
            root=zfw_core_root,
            targets=targets,
            shaders=shaders,
        )
    )


def main():
    zfw_core_main()


if __name__ == "__main__":
    main()
