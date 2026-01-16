#!/usr/bin/env python3
"""
Sponza benchmark script for measuring path tracer performance.

Renders the Sponza scene from multiple canonical viewpoints, measures timing
statistics, and saves reference images for comparison.
"""

import json
import logging
import shutil
import sys
import time
from pathlib import Path

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
)

LOG = logging.getLogger(__name__)

FRAME_WIDTH = 1024
FRAME_HEIGHT = 1024
SAMPLES_PER_PIXEL = 1
ACCUMULATOR_FRAMES = 1

SPONZA_GLTF = "tests/data/glTF-Sample-Assets/Models/Sponza/glTF/Sponza.gltf"
ENV_MAP_PATH = "tests/data/glTF-Sample-Environments/field.hdr"

# Canonical Sponza viewpoints
# Sponza is approximately: 30m long (Y), 15m wide (X), 12m tall (Z)
# Origin is roughly at center of courtyard
VIEWPOINTS = [
    {
        "name": "courtyard-entry",
        "description": "Near entrance looking down main axis",
        # Position at one end, looking down the courtyard
        "camera_transform": np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, -12.0],
                [0.0, 0.0, 1.0, 2.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        "fov": 60.0,
    },
]


def create_readback_buffer(device: wgpu.GPUDevice, w: int, h: int) -> wgpu.GPUBuffer:
    """Create a buffer for reading back rendered data from GPU."""
    return device.create_buffer(
        size=w * h * 4 * 2,  # rgba16float
        usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
        label="ReadbackBuffer",
    )


def readback_texture(
    gpu_device: wgpu.GPUDevice,
    texture: wgpu.GPUTexture,
    w: int,
    h: int,
) -> npt.NDArray[np.float32]:
    """Read back a texture from GPU to CPU as float32 RGBA."""
    readback_buffer = create_readback_buffer(gpu_device, w, h)
    command_encoder = gpu_device.create_command_encoder(label="AOVReadback")
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


def render_frame(
    gpu_device: wgpu.GPUDevice,
    renderer: Draw3dRenderer,
    scene: Draw3dScene,
    w: int,
    h: int,
) -> tuple[npt.NDArray[np.float32], float]:
    """Render a scene and return the image data and total render time in seconds."""
    texture = renderer.get_output_image()
    readback_buffer = create_readback_buffer(gpu_device, w, h)

    start_time = time.perf_counter()

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

    # map_sync blocks until GPU work is complete
    readback_buffer.map_sync(wgpu.MapMode.READ)
    end_time = time.perf_counter()
    data = (
        np.asarray(readback_buffer.read_mapped())
        .view(dtype=np.float16)
        .astype(np.float32)
    )
    readback_buffer.unmap()

    return data.reshape((h, w, 4)), end_time - start_time


def save_image(data: npt.NDArray[np.float32], output_path: Path) -> None:
    """Save float32 RGBA image data to PNG file."""
    img_data = (np.clip(data, 0.0, 1.0) * 255.0).astype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    PIL.Image.fromarray(img_data).convert("RGBA").save(output_path)


def main() -> int:
    setup_logging(level=logging.INFO)

    LOG.info("Sponza Benchmark")
    LOG.info("================")
    LOG.info(f"Resolution: {FRAME_WIDTH}x{FRAME_HEIGHT}")
    LOG.info(f"Samples per pixel: {SAMPLES_PER_PIXEL}")
    LOG.info(f"Accumulator frames: {ACCUMULATOR_FRAMES}")

    output_dir = Path("output/sponza-benchmark")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize GPU
    LOG.info("Initializing GPU...")
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    gpu_device = help_request_wgpu_device(adapter)

    # Create renderer
    LOG.info("Creating renderer...")
    renderer = Draw3dRenderer(
        gpu_device,
        gpu_device.queue,
        (FRAME_WIDTH, FRAME_HEIGHT),
        samples_per_pixel=SAMPLES_PER_PIXEL,
        accumulator_frame_count=ACCUMULATOR_FRAMES,
    )

    # Enable debug AOVs for validation
    renderer.set_debug_flags(
        emit_surface_color=True,
        emit_surface_normal=True,
        emit_emissive=True,
        emit_orm=True,
    )

    # Load Sponza
    gltf_path = Path(SPONZA_GLTF)
    if not gltf_path.exists():
        LOG.error(f"Sponza model not found: {gltf_path}")
        return 1

    LOG.info(f"Loading Sponza model: {gltf_path}")
    load_start = time.perf_counter()
    gltf_scene = load_gltf(gltf_path)
    load_time = time.perf_counter() - load_start
    LOG.info(f"Model loaded in {load_time:.2f}s")

    # Convert to Draw3d objects
    meshes: dict[tuple[Draw3dGeometry, Draw3dMaterial], npt.NDArray[np.float32]] = {}
    for (geom_res, mat_res), transforms in gltf_scene.items():
        geometry = Draw3dGeometry.from_resource(geom_res, renderer)
        material = Draw3dMaterial.from_resource(mat_res, renderer)
        meshes[(geometry, material)] = transforms

    # Check environment map exists
    env_path = Path(ENV_MAP_PATH)
    if not env_path.exists():
        LOG.error(f"Environment map not found: {env_path}")
        return 1

    # Benchmark results
    results: list[dict] = []

    LOG.info("")
    LOG.info("Rendering viewpoints...")
    LOG.info("-" * 60)

    for viewpoint in VIEWPOINTS:
        name = viewpoint["name"]
        description = viewpoint["description"]
        camera_transform = viewpoint["camera_transform"]
        fov = viewpoint["fov"]

        LOG.info(f"\n{name}: {description}")

        # Reset renderer for fresh accumulation
        renderer.reset()

        # Re-enable debug AOVs after reset (reset clears them)
        renderer.set_debug_flags(
            emit_surface_color=True,
            emit_surface_normal=True,
            emit_emissive=True,
            emit_orm=True,
        )

        # Re-upload meshes after reset
        meshes_for_frame: dict[
            tuple[Draw3dGeometry, Draw3dMaterial], npt.NDArray[np.float32]
        ] = {}
        for (geom_res, mat_res), instance_transforms in gltf_scene.items():
            geometry = Draw3dGeometry.from_resource(geom_res, renderer)
            material = Draw3dMaterial.from_resource(mat_res, renderer)
            meshes_for_frame[(geometry, material)] = instance_transforms

        # Re-upload environment map after reset
        env_data_fresh = load_image(env_path)
        env_tex_for_frame = Draw3dTexture(
            renderer, data=env_data_fresh, usage="environment"
        )

        scene = Draw3dScene(
            camera=Draw3dCamera(
                transform=camera_transform,
                fov_y_rad=np.radians(fov),
                aspect_ratio=FRAME_WIDTH / FRAME_HEIGHT,
            ),
            meshes=meshes_for_frame,
            environment_map=env_tex_for_frame,
        )

        # Render and time
        image_data, render_time = render_frame(
            gpu_device, renderer, scene, FRAME_WIDTH, FRAME_HEIGHT
        )

        # Calculate metrics
        total_samples = FRAME_WIDTH * FRAME_HEIGHT * SAMPLES_PER_PIXEL
        samples_per_sec = total_samples / render_time
        ms_per_frame = (render_time / ACCUMULATOR_FRAMES) * 1000

        # Save image
        output_path = output_dir / name / "color-00000-resin.png"
        save_image(image_data, output_path)

        # Save AOV debug outputs
        aov_textures = [
            ("surface_color", renderer.get_frame_surface_color_image()),
            ("surface_normal", renderer.get_frame_surface_normal_image()),
            ("surface_orm", renderer.get_frame_surface_orm_image()),
            ("surface_emissive", renderer.get_frame_surface_emissive_image()),
        ]
        for aov_name, aov_texture in aov_textures:
            aov_path = output_dir / name / f"{aov_name}-00000-resin.png"
            aov_data = readback_texture(gpu_device, aov_texture, FRAME_WIDTH, FRAME_HEIGHT)
            save_image(aov_data, aov_path)
            LOG.info(f"  AOV saved: {aov_path}")

        # Record results
        result = {
            "viewpoint": name,
            "description": description,
            "render_time_sec": render_time,
            "samples_per_sec": samples_per_sec,
            "ms_per_accumulator_frame": ms_per_frame,
            "total_samples": total_samples,
            "image_path": str(output_path),
        }
        results.append(result)

        LOG.info(f"  Render time: {render_time:.2f}s")
        LOG.info(f"  Samples/sec: {samples_per_sec / 1e6:.2f}M")
        LOG.info(f"  ms/frame: {ms_per_frame:.1f}")
        LOG.info(f"  Image saved: {output_path}")

    # Summary
    LOG.info("")
    LOG.info("=" * 60)
    LOG.info("SUMMARY")
    LOG.info("=" * 60)

    total_time = sum(r["render_time_sec"] for r in results)
    avg_samples_per_sec = sum(r["samples_per_sec"] for r in results) / len(results)
    avg_ms_per_frame = sum(r["ms_per_accumulator_frame"] for r in results) / len(
        results
    )

    LOG.info(f"Total render time: {total_time:.2f}s")
    LOG.info(f"Average samples/sec: {avg_samples_per_sec / 1e6:.2f}M")
    LOG.info(f"Average ms/frame: {avg_ms_per_frame:.1f}")

    # Save timing report
    report = {
        "benchmark": "sponza",
        "resolution": [FRAME_WIDTH, FRAME_HEIGHT],
        "samples_per_pixel": SAMPLES_PER_PIXEL,
        "accumulator_frames": ACCUMULATOR_FRAMES,
        "model_load_time_sec": load_time,
        "total_render_time_sec": total_time,
        "average_samples_per_sec": avg_samples_per_sec,
        "average_ms_per_accumulator_frame": avg_ms_per_frame,
        "viewpoints": results,
    }

    report_path = output_dir / "timing-report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    LOG.info(f"\nTiming report saved: {report_path}")
    LOG.info("Benchmark complete!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
