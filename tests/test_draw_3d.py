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


@pytest.fixture(scope="module")
def renderer(gpu: GpuFixture) -> Generator[Draw3dRenderer, None, None]:
    yield Draw3dRenderer(gpu.device, gpu.queue, (FRAME_W, FRAME_H))


def test_basic_draw_3d(gpu: GpuFixture, renderer: Draw3dRenderer):
    frame = Draw3dFrame(renderer)

    # DEBUG:
    frame.set_debug_flags(emit_closest_hit_depth_in_r=True)

    readback_buffer = gpu.device.create_buffer(
        size=FRAME_W * FRAME_H * 4 * ctypes.sizeof(ctypes.c_float),
        usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
        label="BasicDraw3dTest.ReadbackBuffer",
    )

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

    command_encoder = gpu.device.create_command_encoder(
        label="BasicDraw2dTest.CommandEncoder"
    )

    renderer.record(scene, frame, command_encoder)
    command_encoder.copy_texture_to_buffer(
        source=wgpu.TexelCopyTextureInfo(
            texture=frame.get_output_image(),
            mip_level=0,
            origin=(0, 0, 0),
            aspect=wgpu.TextureAspect.all,
        ),
        destination=wgpu.TexelCopyBufferInfo(
            bytes_per_row=FRAME_W * 4 * ctypes.sizeof(ctypes.c_float),
            rows_per_image=FRAME_H,
            buffer=readback_buffer,
        ),
        copy_size=frame.get_output_image().size,
    )

    gpu.queue.submit([command_encoder.finish()])

    readback_buffer.map_sync(wgpu.MapMode.READ)
    data = np.asarray(readback_buffer.read_mapped()).view(dtype=np.float32)
    readback_buffer.unmap()

    # Reshape to (1024, 1024, 4)
    data = data.reshape((1024, 1024, 4))

    # Apply tonemapping
    img_data = (data * 255.0).astype(np.uint8)
    output_path = "output/draw_3d/basic_render_test.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    img = Image.fromarray(img_data, "RGBA")
    img.save(output_path)


def test_primary_ray_generation(gpu: GpuFixture, renderer: Draw3dRenderer):
    """Test that primary rays are generated correctly with proper FOV coverage."""
    # Use smaller image for easier testing
    frame = Draw3dFrame(renderer)
    readback_buffer = gpu.device.create_buffer(
        size=FRAME_W * FRAME_H * 4 * ctypes.sizeof(ctypes.c_float),
        usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
        label="PrimaryRayTest.ReadbackBuffer",
    )

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

    command_encoder = gpu.device.create_command_encoder(
        label="PrimaryRayTest.CommandEncoder"
    )

    # Enable primary ray direction debug flag
    frame.set_debug_flags(
        emit_primary_ray_direction=True,
    )
    renderer.record(scene, frame, command_encoder)
    command_encoder.copy_texture_to_buffer(
        source=wgpu.TexelCopyTextureInfo(
            texture=frame.get_output_image(),
            mip_level=0,
            origin=(0, 0, 0),
            aspect=wgpu.TextureAspect.all,
        ),
        destination=wgpu.TexelCopyBufferInfo(
            bytes_per_row=FRAME_W * 4 * ctypes.sizeof(ctypes.c_float),
            rows_per_image=FRAME_H,
            buffer=readback_buffer,
        ),
        copy_size=frame.get_output_image().size,
    )

    gpu.queue.submit([command_encoder.finish()])

    readback_buffer.map_sync(wgpu.MapMode.READ)
    data = np.asarray(readback_buffer.read_mapped()).view(dtype=np.float32)
    readback_buffer.unmap()

    # Reshape to (H, W, 4), extract RGB (ray direction mapped to [0,1])
    data = data.reshape((FRAME_H, FRAME_W, 4))[:, :, :3]

    # Save debug image
    img_data = (data * 255.0).astype(np.uint8)
    output_path = "output/draw_3d/primary_ray_test.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img = Image.fromarray(img_data, "RGB")
    img.save(output_path)

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

    # print("Primary ray generation test passed!")
    # print(f"  Center: {center_dir_normalized}")
    # print(f"  TL: {tl_dir_normalized}")
    # print(f"  TR: {tr_dir_normalized}")
    # print(f"  BL: {bl_dir_normalized}")
    # print(f"  BR: {br_dir_normalized}")
    # print(f"  Corner angle: {np.degrees(corner_angle):.1f}°")
