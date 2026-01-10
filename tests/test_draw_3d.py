import time
from typing import Generator

import numpy as np
import os
import pytest
import rich
import wgpu
import PIL.Image

from zfw import (
    Draw3dRenderer,
    Draw3dFrame,
    Draw3dScene,
    Draw3dCamera,
    Draw3dGeometry,
    Draw3dTexture,
    Draw3dMaterial,
    load_gltf,
    logger,
    load_image,
)

FRAME_W = 1024
FRAME_H = 1024


def _create_readback_buffer(device: wgpu.GPUDevice, w: int, h: int) -> wgpu.GPUBuffer:
    """Create a buffer for reading back rendered data from GPU."""
    return device.create_buffer(
        size=w * h * 4 * 2,  # rgba16float
        usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
        label="ReadbackBuffer",
    )


def _render_and_readback(
    gpu_device: wgpu.GPUDevice,
    renderer: Draw3dRenderer,
    frame: Draw3dFrame,
    scene: Draw3dScene,
    w: int,
    h: int,
    measure_runtime: bool = False,
) -> np.ndarray:
    """Render a scene and read back the result as a numpy array.

    Returns:
        Array of shape (H, W, 4) with float32 values.
    """
    readback_buffer = _create_readback_buffer(gpu_device, w, h)

    repeat_count = 1 if not measure_runtime else 1

    start_time = time.monotonic_ns()

    for _ in range(repeat_count):
        command_encoder = gpu_device.create_command_encoder(label="CommandEncoder")

        # Reset accumulators in case this frame is re-used on a different scene:
        frame.reset(command_encoder)

        renderer.record(scene, frame, command_encoder)
        command_encoder.copy_texture_to_buffer(
            source=wgpu.TexelCopyTextureInfo(
                texture=frame.get_output_image(),
                mip_level=0,
                origin=(0, 0, 0),
                aspect=wgpu.TextureAspect.all,
            ),
            destination=wgpu.TexelCopyBufferInfo(
                bytes_per_row=w * 4 * 2,  # rgba16float
                rows_per_image=h,
                buffer=readback_buffer,
            ),
            copy_size=frame.get_output_image().size,
        )
        gpu_device.queue.submit([command_encoder.finish()])

    end_time = time.monotonic_ns()

    if measure_runtime:
        elapsed_ms = (end_time - start_time) * 1e-6
        avg_ms = elapsed_ms / repeat_count
        LOG.info(f"Render took ~{avg_ms:.2f} ms")
        rich.print(
            f"[dark_blue]render x {repeat_count} took ~{avg_ms:.2f} ms per render[/dark_blue]",
            end=" ",
        )

    readback_buffer.map_sync(wgpu.MapMode.READ)
    data = (
        np.asarray(readback_buffer.read_mapped())
        .view(dtype=np.float16)
        .astype(np.float32)
    )
    readback_buffer.unmap()

    return data.reshape((h, w, 4))


def _save_debug_image(data: np.ndarray, filename: str, format: str = "RGBA") -> None:
    """Save float32 image data to PNG file.

    Args:
        data: Array of shape (H, W, C) with float32 values in [0, 1].
        filename: Output path relative to output/zfw/draw_3d/.
        format: Image format ("RGB", "RGBA", or "L" for grayscale).
    """
    img_data = (data * 255.0).astype(np.uint8)
    output_path = f"output/zfw/test_draw_3d/{filename}"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    PIL.Image.fromarray(img_data).convert(format).save(output_path)


@pytest.fixture(scope="module")
def renderer(gpu_device: wgpu.GPUDevice) -> Generator[Draw3dRenderer, None, None]:
    yield Draw3dRenderer(gpu_device, gpu_device.queue, (FRAME_W, FRAME_H))


def test_basic_draw_3d(gpu_device: wgpu.GPUDevice, renderer: Draw3dRenderer):
    frame = Draw3dFrame(renderer)
    frame.samples_per_pixel = 2048

    scenes = {
        "two_avocados": _load_two_avocados_scene(renderer),
        "damaged_helmet": _load_damaged_helmet_scene(renderer),
        "flight_helmet": _load_flight_helmet_scene(renderer),
    }

    # Load Field environment map
    hdr_data = load_image("tests/data/glTF-Sample-Environments/field.hdr")
    env_map_texture = Draw3dTexture(
        renderer,
        data=hdr_data,
        usage="environment",
    )

    for scene_name, scene in scenes.items():
        scene.environment_map = env_map_texture

        data = _render_and_readback(
            gpu_device,
            renderer,
            frame,
            scene,
            FRAME_W,
            FRAME_H,
            measure_runtime=True,
        )
        _save_debug_image(data, f"test_basic_draw_3d-{scene_name}.png")


def _load_two_avocados_scene(renderer: Draw3dRenderer) -> Draw3dScene:
    """Helper to load a scene with two avocado models."""

    resource_meshes = load_gltf(
        gltf_path="tests/data/glTF-Sample-Assets/Models/Avocado/glTF/Avocado.gltf",
    )

    # Convert resource types to Draw3d objects using renderer cache
    meshes: dict = {}
    for (geom_res, mat_res), transforms in resource_meshes.items():
        geometry = Draw3dGeometry.from_resource(geom_res, renderer)
        material = Draw3dMaterial.from_resource(mat_res, renderer)
        meshes[(geometry, material)] = transforms

    assert len(meshes) == 1
    mesh, transforms = next(iter(meshes.items()))
    assert transforms.shape[0] == 1
    model_transform = transforms[0]

    meshes = {
        mesh: np.array(
            [
                model_transform
                @ np.array(
                    [
                        [1.0, 0.0, 0.0, 0.05],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    dtype=np.float32,
                ),
                model_transform
                @ np.array(
                    [
                        [1.0, 0.0, 0.0, -0.05],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    dtype=np.float32,
                ),
            ]
        )
    }

    scene = Draw3dScene(
        camera=Draw3dCamera(
            transform=np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, -0.25],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
            fov_y_rad=np.radians(60.0),
            aspect_ratio=FRAME_W / FRAME_H,
            max_distance=0.50,
        ),
        meshes=meshes,
    )

    return scene


def _load_damaged_helmet_scene(renderer: Draw3dRenderer) -> Draw3dScene:
    """Helper to load a scene with the damaged helmet model."""

    resource_meshes = load_gltf(
        gltf_path="tests/data/glTF-Sample-Assets/Models/DamagedHelmet/glTF/DamagedHelmet.gltf",
    )

    # Convert resource types to Draw3d objects using renderer cache
    meshes: dict = {}
    for (geom_res, mat_res), transforms in resource_meshes.items():
        geometry = Draw3dGeometry.from_resource(geom_res, renderer)
        material = Draw3dMaterial.from_resource(mat_res, renderer)
        meshes[(geometry, material)] = transforms

    scene = Draw3dScene(
        camera=Draw3dCamera(
            transform=np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, -3.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
            fov_y_rad=np.radians(45.0),
            aspect_ratio=FRAME_W / FRAME_H,
            max_distance=5.0,
        ),
        meshes=meshes,
    )

    return scene


def _load_flight_helmet_scene(renderer: Draw3dRenderer) -> Draw3dScene:
    resource_meshes = load_gltf(
        gltf_path="tests/data/glTF-Sample-Assets/Models/FlightHelmet/glTF/FlightHelmet.gltf",
    )

    # Convert resource types to Draw3d objects using renderer cache
    meshes: dict = {}
    for (geom_res, mat_res), transforms in resource_meshes.items():
        geometry = Draw3dGeometry.from_resource(geom_res, renderer)
        material = Draw3dMaterial.from_resource(mat_res, renderer)
        meshes[(geometry, material)] = transforms

    scene = Draw3dScene(
        camera=Draw3dCamera(
            transform=np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, -1.5],
                    [0.0, 0.0, 1.0, 0.30],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            fov_y_rad=np.radians(45.0),
            aspect_ratio=FRAME_W / FRAME_H,
            max_distance=5.0,
        ),
        meshes=meshes,
    )

    return scene


def test_primary_ray_generation(gpu_device: wgpu.GPUDevice, renderer: Draw3dRenderer):
    """Test that primary rays are generated correctly with proper FOV coverage."""
    frame = Draw3dFrame(renderer)

    # Simple camera: identity transform (at origin, looking down +Y)
    scene = Draw3dScene(
        camera=Draw3dCamera(
            transform=np.eye(4, dtype=np.float32),
            fov_y_rad=np.radians(60.0),
            aspect_ratio=FRAME_W / FRAME_H,
        ),
        meshes={},  # No geometry needed for ray direction test
    )

    # Enable primary ray direction debug flag
    frame.set_debug_flags(emit_primary_ray_direction=True)
    data = _render_and_readback(gpu_device, renderer, frame, scene, FRAME_W, FRAME_H)

    # Extract RGB (ray direction mapped to [0,1])
    data = data[:, :, :3]
    _save_debug_image(data, "test_primary_ray_generation.png", format="RGB")

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


def test_depth_visualization(gpu_device: wgpu.GPUDevice, renderer: Draw3dRenderer):
    """Test depth visualization for debugging ray-triangle intersections."""
    frame = Draw3dFrame(renderer)

    resource_meshes = load_gltf(
        gltf_path="tests/data/glTF-Sample-Assets/Models/Cube/glTF/Cube.gltf",
    )

    # Convert resource types to Draw3d objects using renderer cache
    meshes: dict = {}
    for (geom_res, mat_res), transforms in resource_meshes.items():
        geometry = Draw3dGeometry.from_resource(geom_res, renderer)
        material = Draw3dMaterial.from_resource(mat_res, renderer)
        meshes[(geometry, material)] = transforms

    # Camera at Y=-5, looking forward (+Y) toward cube at origin
    scene = Draw3dScene(
        camera=Draw3dCamera(
            transform=np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, -5.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
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
    data = _render_and_readback(gpu_device, renderer, frame, scene, FRAME_W, FRAME_H)

    # Save output:
    _save_debug_image(data, "test_depth_visualization.png", format="RGBA")

    # Extract channels
    depth_normalized = data[:, :, 0]

    # Check center region for hits (cube should be visible there)
    center_region = depth_normalized[
        FRAME_H // 2 - 50 : FRAME_H // 2 + 50,
        FRAME_W // 2 - 50 : FRAME_W // 2 + 50,
    ]
    center_hits = np.sum(center_region < 1.0)
    center_total = center_region.size
    center_hit_pct = 100.0 * center_hits / center_total

    # For a cube at distance 5, we expect some hits
    # The cube is 2x2x2 at origin, so should be visible
    # Assert that we get at least some hits in the center region
    assert center_hit_pct > 10.0, (
        f"Expected significant hits in center region, got {center_hit_pct:.1f}%"
    )


def test_world_position_visualization(
    gpu_device: wgpu.GPUDevice, renderer: Draw3dRenderer
):
    """Test world position visualization to see what's actually being hit."""
    frame = Draw3dFrame(renderer)

    resource_meshes = load_gltf(
        gltf_path="tests/data/glTF-Sample-Assets/Models/Cube/glTF/Cube.gltf",
    )

    # Convert resource types to Draw3d objects using renderer cache
    meshes: dict = {}
    for (geo_res, mat_res), transforms in resource_meshes.items():
        geometry = Draw3dGeometry.from_resource(geo_res, renderer)
        material = Draw3dMaterial.from_resource(mat_res, renderer)
        meshes[(geometry, material)] = transforms

    # Camera at Y=-5, looking forward (+Y) toward cube at origin
    scene = Draw3dScene(
        camera=Draw3dCamera(
            transform=np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, -5.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
            fov_y_rad=np.radians(60.0),
            aspect_ratio=FRAME_W / FRAME_H,
        ),
        meshes=meshes,
    )

    # Enable world position debug flag
    frame.set_debug_flags(emit_hit_world_position=True)
    data = _render_and_readback(gpu_device, renderer, frame, scene, FRAME_W, FRAME_H)

    _save_debug_image(data, "test_world_position.png", format="RGBA")


def test_coordinate_system_offset_px(
    gpu_device: wgpu.GPUDevice, renderer: Draw3dRenderer
):
    """Test that positive X camera offset shifts the depth centroid left."""
    frame = Draw3dFrame(renderer)

    resource_meshes = load_gltf(
        gltf_path="tests/data/glTF-Sample-Assets/Models/Cube/glTF/Cube.gltf",
    )

    # Convert resource types to Draw3d objects using renderer cache
    meshes: dict = {}
    for (geom_res, mat_res), transforms in resource_meshes.items():
        geometry = Draw3dGeometry.from_resource(geom_res, renderer)
        material = Draw3dMaterial.from_resource(mat_res, renderer)
        meshes[(geometry, material)] = transforms

    # Camera at Y=-5 with X offset of +0.50, looking forward (+Y) toward cube at origin
    scene = Draw3dScene(
        camera=Draw3dCamera(
            transform=np.array(
                [
                    [1.0, 0.0, 0.0, 0.5],
                    [0.0, 1.0, 0.0, -5.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
            fov_y_rad=np.radians(60.0),
            aspect_ratio=FRAME_W / FRAME_H,
            max_distance=10.0,
        ),
        meshes=meshes,
    )

    # Enable depth debug flag
    frame.set_debug_flags(emit_closest_hit_depth_in_r=True)
    data = _render_and_readback(gpu_device, renderer, frame, scene, FRAME_W, FRAME_H)

    # Save output
    _save_debug_image(data, "test_coordinate_system_offset_px.png", format="RGBA")

    # Extract depth channel (R) and alpha
    alpha_channel = data[:, :, 3]

    # Find pixels with hits (alpha > 0)
    hit_mask = alpha_channel > 0
    assert np.sum(hit_mask) > 0, "Expected some hits on the cube"

    # Calculate centroid of depth pixels
    _, x_indices = np.where(hit_mask)
    centroid_x = np.mean(x_indices)

    # Positive X offset should shift centroid to the left (smaller X pixel coordinate)
    # Center of frame is at FRAME_W / 2
    assert centroid_x < FRAME_W / 2, (
        f"Positive X camera offset should shift cube left, "
        f"centroid at X={centroid_x:.1f} should be < {FRAME_W / 2}"
    )


def test_coordinate_system_offset_py(
    gpu_device: wgpu.GPUDevice, renderer: Draw3dRenderer
):
    """Test that positive Y camera offset makes the cube appear larger."""
    frame = Draw3dFrame(renderer)

    resource_meshes = load_gltf(
        gltf_path="tests/data/glTF-Sample-Assets/Models/Cube/glTF/Cube.gltf",
    )

    # Convert resource types to Draw3d objects using renderer cache
    meshes: dict = {}
    for (geom_res, mat_res), transforms in resource_meshes.items():
        geometry = Draw3dGeometry.from_resource(geom_res, renderer)
        material = Draw3dMaterial.from_resource(mat_res, renderer)
        meshes[(geometry, material)] = transforms

    # Camera at Y=-4.5 (closer by 0.5), looking forward (+Y) toward cube at origin
    scene = Draw3dScene(
        camera=Draw3dCamera(
            transform=np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, -4.5],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
            fov_y_rad=np.radians(60.0),
            aspect_ratio=FRAME_W / FRAME_H,
            max_distance=10.0,
        ),
        meshes=meshes,
    )

    # Enable depth debug flag
    frame.set_debug_flags(emit_closest_hit_depth_in_r=True)
    data = _render_and_readback(gpu_device, renderer, frame, scene, FRAME_W, FRAME_H)

    # Save output
    _save_debug_image(data, "test_coordinate_system_offset_py.png", format="RGBA")

    # Extract alpha channel
    alpha_channel = data[:, :, 3]

    # Find bounding box of non-zero alpha pixels
    hit_mask = alpha_channel > 0
    assert np.sum(hit_mask) > 0, "Expected some hits on the cube"

    y_indices, x_indices = np.where(hit_mask)
    bbox_width = np.max(x_indices) - np.min(x_indices)
    bbox_height = np.max(y_indices) - np.min(y_indices)
    bbox_area = bbox_width * bbox_height

    # Store the bbox area for comparison
    # For a reference, render at Y=-5 (baseline)
    scene_baseline = Draw3dScene(
        camera=Draw3dCamera(
            transform=np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, -5.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
            fov_y_rad=np.radians(60.0),
            aspect_ratio=FRAME_W / FRAME_H,
            max_distance=10.0,
        ),
        meshes=meshes,
    )

    data_baseline = _render_and_readback(
        gpu_device, renderer, frame, scene_baseline, FRAME_W, FRAME_H
    )
    alpha_baseline = data_baseline[:, :, 3]
    hit_mask_baseline = alpha_baseline > 0

    y_baseline, x_baseline = np.where(hit_mask_baseline)
    bbox_width_baseline = np.max(x_baseline) - np.min(x_baseline)
    bbox_height_baseline = np.max(y_baseline) - np.min(y_baseline)
    bbox_area_baseline = bbox_width_baseline * bbox_height_baseline

    # Moving camera closer (positive Y) should make the cube appear larger
    assert bbox_area > bbox_area_baseline, (
        f"Positive Y camera offset (closer) should make cube larger, "
        f"area={bbox_area} should be > baseline={bbox_area_baseline}"
    )


def test_coordinate_system_offset_pz(
    gpu_device: wgpu.GPUDevice, renderer: Draw3dRenderer
):
    """Test that positive Z camera offset shifts the depth centroid downward."""
    frame = Draw3dFrame(renderer)

    resource_meshes = load_gltf(
        gltf_path="tests/data/glTF-Sample-Assets/Models/Cube/glTF/Cube.gltf",
    )

    # Convert resource types to Draw3d objects using renderer cache
    meshes: dict = {}
    for (geom_res, mat_res), transforms in resource_meshes.items():
        geometry = Draw3dGeometry.from_resource(geom_res, renderer)
        material = Draw3dMaterial.from_resource(mat_res, renderer)
        meshes[(geometry, material)] = transforms

    # Camera at Y=-5 with Z offset of +0.5, looking forward (+Y) toward cube at origin
    scene = Draw3dScene(
        camera=Draw3dCamera(
            transform=np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, -5.0],
                    [0.0, 0.0, 1.0, 0.5],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
            fov_y_rad=np.radians(60.0),
            aspect_ratio=FRAME_W / FRAME_H,
            max_distance=10.0,
        ),
        meshes=meshes,
    )

    # Enable depth debug flag
    frame.set_debug_flags(emit_closest_hit_depth_in_r=True)
    data = _render_and_readback(gpu_device, renderer, frame, scene, FRAME_W, FRAME_H)

    # Save output
    _save_debug_image(data, "test_coordinate_system_offset_pz.png", format="RGBA")

    # Extract depth channel (R) and alpha
    alpha_channel = data[:, :, 3]

    # Find pixels with hits (alpha > 0)
    hit_mask = alpha_channel > 0
    assert np.sum(hit_mask) > 0, "Expected some hits on the cube"

    # Calculate centroid of depth pixels
    y_indices, x_indices = np.where(hit_mask)
    centroid_y = np.mean(y_indices)

    # Positive Z offset should shift centroid downward (larger Y pixel coordinate)
    # Center of frame is at FRAME_H / 2
    assert centroid_y > FRAME_H / 2, (
        f"Positive Z camera offset should shift cube down, "
        f"centroid at Y={centroid_y:.1f} should be > {FRAME_H / 2}"
    )


def test_environment_map_basic(gpu_device: wgpu.GPUDevice, renderer: Draw3dRenderer):
    """Test environment map rendering with an empty scene."""
    frame = Draw3dFrame(renderer)

    # Load HDR environment map
    hdr_data = load_image("tests/data/glTF-Sample-Environments/helipad.hdr")
    # HDR files are loaded as RGB float32 in linear color space
    assert hdr_data.ndim == 3
    assert hdr_data.shape[2] == 3
    height, width = hdr_data.shape[:2]

    # Convert to Draw3dTexture
    env_map_texture = Draw3dTexture(
        renderer,
        data=hdr_data,
        usage="environment",
    )

    # Create empty scene with environment map
    scene = Draw3dScene(
        camera=Draw3dCamera(
            transform=np.eye(4, dtype=np.float32),  # Identity transform (at origin)
            fov_y_rad=np.radians(60.0),
            aspect_ratio=FRAME_W / FRAME_H,
        ),
        meshes={},  # Empty scene - no geometry
        environment_map=env_map_texture,
    )

    # Render
    data = _render_and_readback(gpu_device, renderer, frame, scene, FRAME_W, FRAME_H)

    # Save debug output
    _save_debug_image(data, "test_environment_map_basic.png", format="RGBA")

    # Verify we got non-black pixels (environment map was sampled)
    # Check that the mean brightness is above some threshold
    brightness = np.mean(data[:, :, :3])
    assert brightness > 0.01, (
        f"Environment map should produce visible output, got mean brightness {brightness}"
    )

    # Verify we have some variation in the image (not all the same color)
    std_dev = np.std(data[:, :, :3])
    assert std_dev > 0.01, (
        f"Environment map should have color variation, got std dev {std_dev}"
    )


LOG = logger(__name__)
