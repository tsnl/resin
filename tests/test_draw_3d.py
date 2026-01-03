from typing import Generator
import pytest
import wgpu
from PIL import Image
import os
import ctypes
import numpy as np
from conftest import GpuFixture

from zfw import (
    Draw3dRenderer,
    Draw3dFrame,
    Draw3dScene,
    Draw3dCamera,
    load_gltf,
)


FRAME_W = 1024
FRAME_H = 1024


def _create_readback_buffer(device: wgpu.GPUDevice, w: int, h: int) -> wgpu.GPUBuffer:
    """Create a buffer for reading back rendered data from GPU."""
    return device.create_buffer(
        size=w * h * 4 * ctypes.sizeof(ctypes.c_float),
        usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
        label="ReadbackBuffer",
    )


def _render_and_readback(
    gpu: GpuFixture,
    renderer: Draw3dRenderer,
    frame: Draw3dFrame,
    scene: Draw3dScene,
    w: int,
    h: int,
) -> np.ndarray:
    """Render a scene and read back the result as a numpy array.

    Returns:
        Array of shape (H, W, 4) with float32 values.
    """
    readback_buffer = _create_readback_buffer(gpu.device, w, h)

    command_encoder = gpu.device.create_command_encoder(label="CommandEncoder")
    renderer.record(scene, frame, command_encoder)
    command_encoder.copy_texture_to_buffer(
        source=wgpu.TexelCopyTextureInfo(
            texture=frame.get_output_image(),
            mip_level=0,
            origin=(0, 0, 0),
            aspect=wgpu.TextureAspect.all,
        ),
        destination=wgpu.TexelCopyBufferInfo(
            bytes_per_row=w * 4 * ctypes.sizeof(ctypes.c_float),
            rows_per_image=h,
            buffer=readback_buffer,
        ),
        copy_size=frame.get_output_image().size,
    )
    gpu.queue.submit([command_encoder.finish()])

    readback_buffer.map_sync(wgpu.MapMode.READ)
    data = np.asarray(readback_buffer.read_mapped()).view(dtype=np.float32)
    readback_buffer.unmap()

    return data.reshape((h, w, 4))


def _save_debug_image(data: np.ndarray, filename: str, format: str = "RGBA") -> None:
    """Save float32 image data to PNG file.

    Args:
        data: Array of shape (H, W, C) with float32 values in [0, 1].
        filename: Output path relative to output/draw_3d/.
        format: Image format ("RGB", "RGBA", or "L" for grayscale).
    """
    img_data = (data * 255.0).astype(np.uint8)
    output_path = f"output/draw_3d/{filename}"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img = Image.fromarray(img_data, format)
    img.save(output_path)


@pytest.fixture(scope="module")
def renderer(gpu: GpuFixture) -> Generator[Draw3dRenderer, None, None]:
    yield Draw3dRenderer(gpu.device, gpu.queue, (FRAME_W, FRAME_H))


def test_basic_draw_3d(gpu: GpuFixture, renderer: Draw3dRenderer):
    frame = Draw3dFrame(renderer)

    meshes = load_gltf(
        renderer=renderer,
        # path="tests/data/glTF-Sample-Assets/Models/Avocado/glTF/Avocado.gltf",
        path="tests/data/glTF-Sample-Assets/Models/Cube/glTF/Cube.gltf",
    )
    scene = Draw3dScene(
        camera=Draw3dCamera(
            transform=np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, -1000.0],
                    [0.0, 0.0, 1.0, 0.0],
                ],
                dtype=np.float32,
            ),
            fov_y_rad=np.radians(60.0),
            aspect_ratio=FRAME_W / FRAME_H,
        ),
        meshes=meshes,
    )

    data = _render_and_readback(gpu, renderer, frame, scene, FRAME_W, FRAME_H)
    _save_debug_image(data, "basic_render_test.png")


def test_primary_ray_generation(gpu: GpuFixture, renderer: Draw3dRenderer):
    """Test that primary rays are generated correctly with proper FOV coverage."""
    frame = Draw3dFrame(renderer)

    # Simple camera: identity transform (at origin, looking down +Y)
    scene = Draw3dScene(
        camera=Draw3dCamera(
            transform=np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                ],
                dtype=np.float32,
            ),
            fov_y_rad=np.radians(60.0),
            aspect_ratio=FRAME_W / FRAME_H,
        ),
        meshes={},  # No geometry needed for ray direction test
    )

    # Enable primary ray direction debug flag
    frame.set_debug_flags(emit_primary_ray_direction=True)
    data = _render_and_readback(gpu, renderer, frame, scene, FRAME_W, FRAME_H)

    # Extract RGB (ray direction mapped to [0,1])
    data = data[:, :, :3]
    _save_debug_image(data, "primary_ray_test.png", format="RGB")

    # Convert back to direction vectors: [0,1] -> [-1,1]
    directions = data * 2.0 - 1.0

    # Test center pixel: should point straight forward (+Y)
    center_dir = directions[FRAME_H // 2, FRAME_W // 2]
    center_dir_normalized = center_dir / np.linalg.norm(center_dir)
    assert np.allclose(center_dir_normalized, [0.0, 1.0, 0.0], atol=0.01), (
        f"Center ray should point down +Y, got {center_dir_normalized}"
    )

    # Test corner pixels: should have significant deviation
    # For 60deg FOV, tan(30deg) = 0.577, so at corners we expect:
    # X component ~= 0.577 (right) or -0.577 (left)
    # Z component ~= 0.577 (top) or -0.577 (bottom)
    # Y component should always be positive and ~1.0

    # Top-left corner
    tl_dir = directions[0, 0]
    tl_dir_normalized = tl_dir / np.linalg.norm(tl_dir)
    assert tl_dir_normalized[0] < -0.4, (
        f"Top-left X should be negative, got {tl_dir_normalized}"
    )
    assert tl_dir_normalized[1] > 0.5, (
        f"Top-left Y should be positive, got {tl_dir_normalized}"
    )
    assert tl_dir_normalized[2] > 0.4, (
        f"Top-left Z should be positive (up), got {tl_dir_normalized}"
    )

    # Top-right corner
    tr_dir = directions[0, FRAME_W - 1]
    tr_dir_normalized = tr_dir / np.linalg.norm(tr_dir)
    assert tr_dir_normalized[0] > 0.4, (
        f"Top-right X should be positive, got {tr_dir_normalized}"
    )
    assert tr_dir_normalized[1] > 0.5, (
        f"Top-right Y should be positive, got {tr_dir_normalized}"
    )
    assert tr_dir_normalized[2] > 0.4, (
        f"Top-right Z should be positive (up), got {tr_dir_normalized}"
    )

    # Bottom-left corner
    bl_dir = directions[FRAME_H - 1, 0]
    bl_dir_normalized = bl_dir / np.linalg.norm(bl_dir)
    assert bl_dir_normalized[0] < -0.4, (
        f"Bottom-left X should be negative, got {bl_dir_normalized}"
    )
    assert bl_dir_normalized[1] > 0.5, (
        f"Bottom-left Y should be positive, got {bl_dir_normalized}"
    )
    assert bl_dir_normalized[2] < -0.4, (
        f"Bottom-left Z should be negative (down), got {bl_dir_normalized}"
    )

    # Bottom-right corner
    br_dir = directions[FRAME_H - 1, FRAME_W - 1]
    br_dir_normalized = br_dir / np.linalg.norm(br_dir)
    assert br_dir_normalized[0] > 0.4, (
        f"Bottom-right X should be positive, got {br_dir_normalized}"
    )
    assert br_dir_normalized[1] > 0.5, (
        f"Bottom-right Y should be positive, got {br_dir_normalized}"
    )
    assert br_dir_normalized[2] < -0.4, (
        f"Bottom-right Z should be negative (down), got {br_dir_normalized}"
    )

    # Verify FOV spread: measure angle from center to corner
    corner_angle = np.arccos(np.dot(center_dir_normalized, tl_dir_normalized))
    expected_corner_angle = np.radians(30.0 * np.sqrt(2))  # ~42.4 degrees diagonally
    assert np.abs(corner_angle - expected_corner_angle) < np.radians(5.0), (
        f"Corner angle {np.degrees(corner_angle):.1f}° should be ~42.4°"
    )


def test_depth_visualization(gpu: GpuFixture, renderer: Draw3dRenderer):
    """Test depth visualization for debugging ray-triangle intersections."""
    frame = Draw3dFrame(renderer)

    meshes = load_gltf(
        renderer=renderer,
        path="tests/data/glTF-Sample-Assets/Models/Cube/glTF/Cube.gltf",
    )

    # Camera at Y=-5, looking forward (+Y) toward cube at origin
    scene = Draw3dScene(
        camera=Draw3dCamera(
            transform=np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, -5.0],
                    [0.0, 0.0, 1.0, 0.0],
                ],
                dtype=np.float32,
            ),
            fov_y_rad=np.radians(60.0),
            aspect_ratio=FRAME_W / FRAME_H,
            max_distance=10.0,  # Cube should be at distance ~5
        ),
        meshes=meshes,
    )

    # Enable depth debug flag
    frame.set_debug_flags(emit_closest_hit_depth_in_r=True)
    data = _render_and_readback(gpu, renderer, frame, scene, FRAME_W, FRAME_H)

    # Extract red channel (normalized depth)
    depth_normalized = data[:, :, 0]
    _save_debug_image(
        np.stack([depth_normalized, depth_normalized, depth_normalized], axis=-1),
        "depth_test.png",
        format="RGB",
    )

    # Analyze depth statistics
    hit_mask = depth_normalized < 1.0  # Pixels that hit something
    hit_count = np.sum(hit_mask)
    total_pixels = FRAME_W * FRAME_H
    hit_percentage = 100.0 * hit_count / total_pixels

    if hit_count > 0:
        hit_depths = depth_normalized[hit_mask] * scene.camera.max_distance
        min_depth = np.min(hit_depths)
        max_depth = np.max(hit_depths)
        mean_depth = np.mean(hit_depths)

        print(f"Depth test results:")
        print(f"  Hit pixels: {hit_count}/{total_pixels} ({hit_percentage:.1f}%)")
        print(f"  Depth range: {min_depth:.2f} to {max_depth:.2f}")
        print(f"  Mean depth: {mean_depth:.2f}")

        # Check center region for hits (cube should be visible there)
        center_region = depth_normalized[
            FRAME_H // 2 - 50 : FRAME_H // 2 + 50,
            FRAME_W // 2 - 50 : FRAME_W // 2 + 50,
        ]
        center_hits = np.sum(center_region < 1.0)
        center_total = center_region.size
        center_hit_pct = 100.0 * center_hits / center_total

        print(
            f"  Center region hits: {center_hits}/{center_total} ({center_hit_pct:.1f}%)"
        )

        # For a cube at distance 5, we expect some hits
        # The cube is 2x2x2 at origin, so should be visible
        # Assert that we get at least some hits in the center region
        assert center_hit_pct > 10.0, (
            f"Expected significant hits in center region, got {center_hit_pct:.1f}%"
        )
    else:
        print(f"Depth test results: No hits detected")
        # This might indicate a problem with the ray tracer
        print("WARNING: No ray-triangle intersections detected!")
