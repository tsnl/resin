#!/usr/bin/env python3
"""
Test script for comparing resin and mitsuba renderers using resin-replay.

This script renders test scenes with both renderers for visual comparison.
Mitsuba must be installed separately (requires Python <= 3.13).
"""

import importlib.util
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import PIL.Image

from resin import setup_logging

LOG = logging.getLogger(__name__)

FRAME_WIDTH = 1024
FRAME_HEIGHT = 1024

ENV_MAP_PATH = "tests/data/glTF-Sample-Environments/field.hdr"

TEST_SCENES = [
    {
        "name": "damaged-helmet",
        "gltf": "tests/data/glTF-Sample-Assets/Models/DamagedHelmet/glTF/DamagedHelmet.gltf",
        "camera_transform": np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, -3.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        "fov": 45.0,
    },
    {
        "name": "avocado",
        "gltf": "tests/data/glTF-Sample-Assets/Models/Avocado/glTF/Avocado.gltf",
        "camera_transform": np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, -0.25],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        "fov": 60.0,
    },
]


def check_mitsuba_available() -> bool:
    """Check if mitsuba is available."""
    return importlib.util.find_spec("mitsuba") is not None


def run_resin_replay(
    gltf_path: str,
    output_dir: Path,
    camera_id: str,
    camera_transform: np.ndarray,
    fov: float,
    renderer: str,
    env_map_path: str,
) -> subprocess.CompletedProcess:
    """Run resin-replay with the given parameters."""
    matrix_values = ",".join(f"{v:.8g}" for v in camera_transform.flatten())
    csv_input = f"{camera_id},{matrix_values}\n"

    result = subprocess.run(
        [
            "uv",
            "run",
            "resin-replay",
            gltf_path,
            "--output",
            str(output_dir),
            "--renderers",
            renderer,
            "--environment-map",
            env_map_path,
            "--fov",
            str(fov),
            "--width",
            str(FRAME_WIDTH),
            "--height",
            str(FRAME_HEIGHT),
            "--samples",
            "64",
            "--accumulator-frames",
            "32",
            "-v",
        ],
        input=csv_input,
        capture_output=True,
        text=True,
        cwd=str(Path.cwd()),
    )

    return result


def verify_output(output_path: Path, expected_size: tuple[int, int]) -> bool:
    """Verify the output image exists and has the expected size."""
    if not output_path.exists():
        LOG.error(f"Output file not found: {output_path}")
        return False

    with PIL.Image.open(output_path) as img:
        if img.size != expected_size:
            LOG.error(f"Unexpected image size: {img.size}, expected {expected_size}")
            return False
        if img.mode not in ("RGB", "RGBA"):
            LOG.error(f"Unexpected image mode: {img.mode}")
            return False

    return True


def main() -> int:
    setup_logging(level=logging.INFO)

    output_dir = Path("output/resin/test_mitsuba_ref")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mitsuba_available = check_mitsuba_available()
    if not mitsuba_available:
        LOG.warning("Mitsuba not installed, will only test resin renderer")

    success = True

    for scene_config in TEST_SCENES:
        scene_name = scene_config["name"]
        gltf_path = scene_config["gltf"]
        camera_transform = scene_config["camera_transform"]
        fov = scene_config["fov"]

        # Test resin renderer
        LOG.info(f"Testing resin renderer for {scene_name}")
        scene_output_dir = output_dir / scene_name

        result = run_resin_replay(
            gltf_path=gltf_path,
            output_dir=scene_output_dir,
            camera_id=scene_name,
            camera_transform=camera_transform,
            fov=fov,
            renderer="resin",
            env_map_path=ENV_MAP_PATH,
        )

        if result.returncode != 0:
            LOG.error(
                f"resin-replay failed for {scene_name} with resin renderer:\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}"
            )
            success = False
            continue

        expected_output = scene_output_dir / scene_name / "color-00000-resin.png"
        if not verify_output(expected_output, (FRAME_WIDTH, FRAME_HEIGHT)):
            success = False
            continue

        LOG.info(f"Successfully rendered {scene_name} with resin: {expected_output}")

        # Test mitsuba renderer if available
        if mitsuba_available:
            LOG.info(f"Testing mitsuba renderer for {scene_name}")
            mitsuba_output_dir = output_dir / f"{scene_name}-mitsuba"

            result = run_resin_replay(
                gltf_path=gltf_path,
                output_dir=mitsuba_output_dir,
                camera_id=f"{scene_name}-mitsuba",
                camera_transform=camera_transform,
                fov=fov,
                renderer="mitsuba",
                env_map_path=ENV_MAP_PATH,
            )

            if result.returncode != 0:
                LOG.error(
                    f"resin-replay failed for {scene_name} with mitsuba renderer:\n"
                    f"stdout: {result.stdout}\n"
                    f"stderr: {result.stderr}"
                )
                success = False
                continue

            expected_output = (
                mitsuba_output_dir / f"{scene_name}-mitsuba" / "color-00000-mitsuba.png"
            )
            if not verify_output(expected_output, (FRAME_WIDTH, FRAME_HEIGHT)):
                success = False
                continue

            LOG.info(
                f"Successfully rendered {scene_name} with mitsuba: {expected_output}"
            )

    if success:
        LOG.info("All tests passed!")
        return 0
    else:
        LOG.error("Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
