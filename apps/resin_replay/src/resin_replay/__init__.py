"""
resin-replay: CLI tool for rendering glTF scenes with camera trajectories.

Supports rendering with either the resin (Draw3d) renderer or Mitsuba,
or both for comparison.
"""

__all__ = ["main"]

import csv
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Literal

import click
import numpy as np
import numpy.typing as npt
import PIL.Image
import wgpu

from resin import (
    Draw3dCamera,
    Draw3dGeometry,
    Draw3dMaterial,
    Draw3dRenderer,
    Draw3dScene,
    Draw3dTexture,
    help_request_wgpu_device,
    load_gltf,
    load_image,
    setup_logging,
    write_mitsuba_scene,
)

LOG = logging.getLogger(__name__)

RendererType = Literal["resin", "mitsuba", "both"]


def _naughty_dog_curve(
    x: npt.NDArray[np.floating],
    A: float,
    B: float,
    C: float,
    D: float,
    E: float,
    F: float,
) -> npt.NDArray[np.floating]:
    """Attempt to better capture Uncharted 2 / Naughty Dog filmic curve."""
    return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F


def _naughty_dog_tonemap(color: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Apply Naughty Dog / Uncharted 2 filmic tonemapping."""
    A = 0.22  # Shoulder strength
    B = 0.30  # Linear strength
    C = 0.10  # Linear angle
    D = 0.20  # Toe strength
    E = 0.01  # Toe numerator
    F = 0.30  # Toe denominator
    white = 11.2
    num = _naughty_dog_curve(color, A, B, C, D, E, F)
    denom = _naughty_dog_curve(np.array([white], dtype=color.dtype), A, B, C, D, E, F)
    return num / denom


def _linear_to_srgb(color: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Convert linear RGB to sRGB with proper gamma curve."""
    low = color * 12.92
    high = 1.055 * np.power(np.maximum(color, 0.0), 1.0 / 2.4) - 0.055
    return np.where(color <= 0.0031308, low, high)


def _apply_tonemapping(image: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Apply full tonemapping pipeline matching Resin's postprocess shader."""
    # Apply exposure (matching draw_3d.wgsl exposure of 1.5)
    exposed = image * 1.5
    # Apply Naughty Dog tonemapping
    tonemapped = _naughty_dog_tonemap(exposed)
    # Apply linear to sRGB gamma correction
    gamma_corrected = _linear_to_srgb(tonemapped)
    return gamma_corrected


FRAME_WIDTH = 1024
FRAME_HEIGHT = 1024


def _parse_csv_cameras(
    csv_input: str,
) -> dict[str, list[npt.NDArray[np.float32]]]:
    """Parse CSV input containing camera transforms.

    Each row has: camera_id, followed by 16 values for 4x4 row-major transform.

    Returns:
        Dict mapping camera_id to list of 4x4 transform matrices.
    """
    cameras: dict[str, list[npt.NDArray[np.float32]]] = defaultdict(list)

    reader = csv.reader(csv_input.strip().splitlines())
    for row_num, row in enumerate(reader, 1):
        if not row or row[0].startswith("#"):
            continue

        if len(row) < 17:
            raise ValueError(
                f"Row {row_num}: Expected camera_id + 16 matrix values, got {len(row)} values"
            )

        camera_id = row[0].strip()
        try:
            matrix_values = [float(v) for v in row[1:17]]
        except ValueError as e:
            raise ValueError(f"Row {row_num}: Invalid matrix value: {e}") from e

        transform = np.array(matrix_values, dtype=np.float32).reshape((4, 4))
        cameras[camera_id].append(transform)

    return dict(cameras)


def _create_readback_buffer(device, w: int, h: int):
    """Create a buffer for reading back rendered data from GPU."""
    return device.create_buffer(
        size=w * h * 4 * 2,  # rgba16float
        usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
        label="ReadbackBuffer",
    )


def _render_resin_frame(
    gpu_device,
    renderer: Draw3dRenderer,
    scene: Draw3dScene,
    w: int,
    h: int,
) -> npt.NDArray[np.float32]:
    """Render a scene using the resin renderer and read back the result."""
    texture = renderer.get_output_image()
    readback_buffer = _create_readback_buffer(gpu_device, w, h)

    for _ in range(renderer.accumulator_frame_count):
        command_encoder = gpu_device.create_command_encoder(label="CommandEncoder")
        renderer.record(scene, command_encoder)
        command_encoder.copy_texture_to_buffer(
            source=wgpu.TexelCopyTextureInfo(
                texture=texture,
                mip_level=0,
                origin=(0, 0, 0),
                aspect=wgpu.TextureAspect.all,
            ),
            destination=wgpu.TexelCopyBufferInfo(
                bytes_per_row=w * 4 * 2,  # rgba16float
                rows_per_image=h,
                buffer=readback_buffer,
            ),
            copy_size=texture.size,
        )
        gpu_device.queue.submit([command_encoder.finish()])

    readback_buffer.map_sync(wgpu.MapMode.READ)
    data = (
        np.asarray(readback_buffer.read_mapped())
        .view(dtype=np.float16)
        .astype(np.float32)
    )
    readback_buffer.unmap()

    return data.reshape((h, w, 4))


def _save_image(data: npt.NDArray[np.float32], output_path: Path) -> None:
    """Save float32 RGBA image data to PNG file."""
    # Clamp and convert to uint8
    img_data = (np.clip(data, 0.0, 1.0) * 255.0).astype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    PIL.Image.fromarray(img_data).convert("RGBA").save(output_path)


def _render_with_resin(
    gltf_path: Path,
    cameras: dict[str, list[npt.NDArray[np.float32]]],
    output_dir: Path,
    environment_map_path: Path,
    fov_y_deg: float,
    width: int,
    height: int,
    samples_per_pixel: int,
    accumulator_frame_count: int,
) -> None:
    """Render all camera frames using the resin (Draw3d) renderer."""
    LOG.info("Initializing resin renderer...")

    # Initialize GPU
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    gpu_device = help_request_wgpu_device(adapter)

    # Create renderer
    renderer = Draw3dRenderer(
        gpu_device,
        gpu_device.queue,
        (width, height),
        samples_per_pixel=samples_per_pixel,
        accumulator_frame_count=accumulator_frame_count,
    )

    # Load glTF scene
    LOG.info(f"Loading glTF: {gltf_path}")
    gltf_scene = load_gltf(gltf_path)

    # Convert to Draw3d objects
    meshes: dict = {}
    for (geom_res, mat_res), transforms in gltf_scene.items():
        geometry = Draw3dGeometry.from_resource(geom_res, renderer)
        material = Draw3dMaterial.from_resource(mat_res, renderer)
        meshes[(geometry, material)] = transforms

    # Render each camera trajectory
    total_frames = sum(len(transforms) for transforms in cameras.values())
    frame_idx = 0

    for camera_id, transforms in cameras.items():
        LOG.info(f"Rendering camera '{camera_id}' ({len(transforms)} frames)")

        for idx, transform in enumerate(transforms):
            renderer.reset()

            # Re-upload meshes after reset
            meshes_for_frame: dict = {}
            for (geom_res, mat_res), instance_transforms in gltf_scene.items():
                geometry = Draw3dGeometry.from_resource(geom_res, renderer)
                material = Draw3dMaterial.from_resource(mat_res, renderer)
                meshes_for_frame[(geometry, material)] = instance_transforms

            # Re-upload environment map after reset
            env_data = load_image(environment_map_path)
            env_tex_for_frame = Draw3dTexture(
                renderer, data=env_data, usage="environment"
            )

            scene = Draw3dScene(
                camera=Draw3dCamera(
                    transform=transform,
                    fov_y_rad=np.radians(fov_y_deg),
                    aspect_ratio=width / height,
                ),
                meshes=meshes_for_frame,
                environment_map=env_tex_for_frame,
            )

            # Render
            data = _render_resin_frame(gpu_device, renderer, scene, width, height)

            # Save output
            output_path = output_dir / camera_id / f"color-{idx:05d}-resin.png"
            _save_image(data, output_path)

            frame_idx += 1
            LOG.info(f"  Frame {idx + 1}/{len(transforms)} saved to {output_path}")

    LOG.info(f"Resin rendering complete: {total_frames} frames")


def _render_with_mitsuba(
    gltf_path: Path,
    cameras: dict[str, list[npt.NDArray[np.float32]]],
    output_dir: Path,
    environment_map_path: Path,
    fov_y_deg: float,
    width: int,
    height: int,
    samples_per_pixel: int,
) -> None:
    """Render all camera frames using Mitsuba."""
    import mitsuba as mi

    # Set Mitsuba variant
    mi.set_variant("scalar_rgb")

    LOG.info("Initializing Mitsuba renderer...")

    # Create a temporary directory for the Mitsuba scene
    mitsuba_scene_dir = output_dir / ".mitsuba_scene"
    mitsuba_scene_dir.mkdir(parents=True, exist_ok=True)

    # Render each camera trajectory
    total_frames = sum(len(transforms) for transforms in cameras.values())

    for camera_id, transforms in cameras.items():
        LOG.info(f"Rendering camera '{camera_id}' ({len(transforms)} frames)")

        for idx, transform in enumerate(transforms):
            # Write Mitsuba scene with this camera transform
            scene_path = write_mitsuba_scene(
                gltf_path=gltf_path,
                output_dir=mitsuba_scene_dir,
                camera_transform=transform,
                camera_fov_y_deg=fov_y_deg,
                film_width=width,
                film_height=height,
                environment_map_path=environment_map_path,
                samples_per_pixel=samples_per_pixel,
            )

            # Load and render the scene
            scene = mi.load_file(str(scene_path))
            image = mi.render(scene)

            # Convert to numpy and apply tonemapping
            image_np = np.array(image, dtype=np.float32)

            # Apply tonemapping pipeline matching Resin's postprocess shader
            image_np = _apply_tonemapping(image_np)

            # Clamp after tonemapping
            image_np = np.clip(image_np, 0.0, 1.0)

            # Add alpha channel
            if image_np.shape[2] == 3:
                alpha = np.ones((*image_np.shape[:2], 1), dtype=np.float32)
                image_np = np.concatenate([image_np, alpha], axis=2)

            # Save output
            output_path = output_dir / camera_id / f"color-{idx:05d}-mitsuba.png"
            _save_image(image_np, output_path)

            LOG.info(f"  Frame {idx + 1}/{len(transforms)} saved to {output_path}")

    LOG.info(f"Mitsuba rendering complete: {total_frames} frames")


@click.command()
@click.argument("gltf_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=Path("./output/resin_replay/"),
    help="Output directory for rendered images.",
)
@click.option(
    "--renderers",
    "-r",
    type=click.Choice(["resin", "mitsuba", "both"]),
    default="resin",
    help="Which renderer(s) to use.",
)
@click.option(
    "--environment-map",
    "-e",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to HDR environment map (required).",
)
@click.option(
    "--fov",
    type=float,
    default=45.0,
    help="Vertical field of view in degrees.",
)
@click.option(
    "--width",
    type=int,
    default=FRAME_WIDTH,
    help="Output image width.",
)
@click.option(
    "--height",
    type=int,
    default=FRAME_HEIGHT,
    help="Output image height.",
)
@click.option(
    "--samples",
    type=int,
    default=64,
    help="Samples per pixel.",
)
@click.option(
    "--accumulator-frames",
    type=int,
    default=32,
    help="Number of accumulator frames (resin only).",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging.",
)
def main(
    gltf_path: Path,
    output: Path,
    renderers: RendererType,
    environment_map: Path,
    fov: float,
    width: int,
    height: int,
    samples: int,
    accumulator_frames: int,
    verbose: bool,
) -> None:
    """Render glTF scenes with camera trajectories.

    Reads camera transforms from CSV on stdin. Each row should contain:
    camera_id,m00,m01,m02,m03,m10,m11,m12,m13,m20,m21,m22,m23,m30,m31,m32,m33

    Where the 16 values represent a 4x4 row-major camera transform matrix.
    Camera IDs can be repeated for multiple frames in a trajectory.

    Output is organized as:
    <output>/<camera_id>/color-<frame_idx>-<renderer>.png
    """
    setup_logging(level=logging.DEBUG if verbose else logging.INFO)

    LOG.info(f"resin-replay: Rendering {gltf_path}")
    LOG.info(f"Output directory: {output}")
    LOG.info(f"Renderer(s): {renderers}")

    # Read camera transforms from stdin
    if sys.stdin.isatty():
        LOG.error(
            "No camera data provided on stdin. Pipe CSV camera data to this command."
        )
        sys.exit(1)

    csv_input = sys.stdin.read()
    cameras = _parse_csv_cameras(csv_input)

    if not cameras:
        LOG.error("No valid camera transforms found in input.")
        sys.exit(1)

    total_frames = sum(len(transforms) for transforms in cameras.values())
    LOG.info(f"Loaded {len(cameras)} camera(s) with {total_frames} total frame(s)")

    # Create output directory
    output.mkdir(parents=True, exist_ok=True)

    # Render with selected renderer(s)
    if renderers in ("resin", "both"):
        _render_with_resin(
            gltf_path=gltf_path,
            cameras=cameras,
            output_dir=output,
            environment_map_path=environment_map,
            fov_y_deg=fov,
            width=width,
            height=height,
            samples_per_pixel=samples,
            accumulator_frame_count=accumulator_frames,
        )

    if renderers in ("mitsuba", "both"):
        _render_with_mitsuba(
            gltf_path=gltf_path,
            cameras=cameras,
            output_dir=output,
            environment_map_path=environment_map,
            fov_y_deg=fov,
            width=width,
            height=height,
            samples_per_pixel=samples,
        )

    LOG.info("Done!")


if __name__ == "__main__":
    main()
