"""
Tests for the draw_3d module.
"""

from pathlib import Path

import numpy as np
import PIL.Image
import pytest

from .basic import BaseResource, logger
from .gpu import (
    GpuContext,
    GpuDevice,
    GpuImage,
    GpuBuffer,
    GpuCommandEncoder,
    GpuImageMeta,
    GpuBufferMeta,
)
from .draw_3d import (
    Draw3dContext,
    Draw3dRenderer,
    Draw3dGeometry,
    Draw3dMaterial,
    Draw3dCameraIntrinsics,
    VERTEX_DTYPE,
)

TEST_IMAGE_W, TEST_IMAGE_H = 1280, 720

LOG = logger(__name__)


def make_cube_geometry(renderer: Draw3dRenderer) -> Draw3dGeometry:
    """
    Create a unit cube centered at the origin with proper normals.

    Each face has its own vertices with face normals (not shared vertices).
    """
    # Define the 8 corners of a unit cube centered at origin
    # Corners: (-0.5, -0.5, -0.5) to (0.5, 0.5, 0.5)
    half = 0.5

    # We need 6 faces * 4 vertices = 24 vertices (each face has unique normals)
    # Face order: +X, -X, +Y, -Y, +Z, -Z

    vertices = []

    # +X face (right): normal (1, 0, 0)
    vertices.extend(
        [
            ((half, -half, -half), (1, 0, 0), (0, 0)),  # bottom-left
            ((half, half, -half), (1, 0, 0), (1, 0)),  # bottom-right
            ((half, half, half), (1, 0, 0), (1, 1)),  # top-right
            ((half, -half, half), (1, 0, 0), (0, 1)),  # top-left
        ]
    )

    # -X face (left): normal (-1, 0, 0)
    vertices.extend(
        [
            ((-half, -half, half), (-1, 0, 0), (0, 0)),
            ((-half, half, half), (-1, 0, 0), (1, 0)),
            ((-half, half, -half), (-1, 0, 0), (1, 1)),
            ((-half, -half, -half), (-1, 0, 0), (0, 1)),
        ]
    )

    # +Y face (top): normal (0, 1, 0)
    vertices.extend(
        [
            ((-half, half, -half), (0, 1, 0), (0, 0)),
            ((half, half, -half), (0, 1, 0), (1, 0)),
            ((half, half, half), (0, 1, 0), (1, 1)),
            ((-half, half, half), (0, 1, 0), (0, 1)),
        ]
    )

    # -Y face (bottom): normal (0, -1, 0)
    vertices.extend(
        [
            ((-half, -half, half), (0, -1, 0), (0, 0)),
            ((half, -half, half), (0, -1, 0), (1, 0)),
            ((half, -half, -half), (0, -1, 0), (1, 1)),
            ((-half, -half, -half), (0, -1, 0), (0, 1)),
        ]
    )

    # +Z face (front): normal (0, 0, 1)
    vertices.extend(
        [
            ((-half, -half, half), (0, 0, 1), (0, 0)),
            ((half, -half, half), (0, 0, 1), (1, 0)),
            ((half, half, half), (0, 0, 1), (1, 1)),
            ((-half, half, half), (0, 0, 1), (0, 1)),
        ]
    )

    # -Z face (back): normal (0, 0, -1)
    vertices.extend(
        [
            ((half, -half, -half), (0, 0, -1), (0, 0)),
            ((-half, -half, -half), (0, 0, -1), (1, 0)),
            ((-half, half, -half), (0, 0, -1), (1, 1)),
            ((half, half, -half), (0, 0, -1), (0, 1)),
        ]
    )

    # Convert to numpy array
    vertex_data = np.array(vertices, dtype=VERTEX_DTYPE)

    # Indices: 6 faces * 2 triangles * 3 vertices = 36 indices
    # Each face uses vertices [4*i, 4*i+1, 4*i+2, 4*i+3]
    # Two triangles per face: (0,1,2) and (0,2,3)
    indices = []
    for face in range(6):
        base = face * 4
        indices.extend(
            [
                base + 0,
                base + 1,
                base + 2,  # first triangle
                base + 0,
                base + 2,
                base + 3,  # second triangle
            ]
        )
    index_data = np.array(indices, dtype=np.uint16)

    return Draw3dGeometry(
        renderer=renderer,
        vertex_data=vertex_data,
        index_data=index_data,
    )


class Draw3dTestEngine(BaseResource):
    """Test engine for 3D rendering tests."""

    def __init__(self):
        super().__init__(parent_resource=None)

        self.gpu_context = GpuContext(
            app_name="zfw draw_3d_test",
            enable_debug_layer_support=True,
            enable_present_support=False,
        )

        self.gpu_device = GpuDevice(
            context=self.gpu_context,
            physical_device=self.gpu_context.enumerate_physical_devices()[0],
            surface=None,
        )

        self.draw_3d_context = Draw3dContext()
        self.renderer = Draw3dRenderer(
            context=self.draw_3d_context,
            gpu_device=self.gpu_device,
        )

        self.target = GpuImage(
            device=self.gpu_device,
            usages=["color-attachment", "transfer-src"],
            meta=GpuImageMeta(
                shape=(TEST_IMAGE_H, TEST_IMAGE_W, 4),
                dtype=np.uint8,
                color_space="srgb",
            ),
        )

    def _on_dispose(self) -> None:
        self.target.dispose()
        self.renderer.dispose()
        self.draw_3d_context.dispose()
        self.gpu_device.dispose()
        self.gpu_context.dispose()

    def readback(self) -> np.ndarray:
        """Read back the rendered image from GPU to CPU."""
        buffer = GpuBuffer(
            device=self.gpu_device,
            usages=["copy-dst", "staging"],
            meta=GpuBufferMeta(
                element_count=(TEST_IMAGE_H * TEST_IMAGE_W * 4),
                element_dtype=np.uint8,
            ),
        )
        encoder = GpuCommandEncoder(device=self.gpu_device, queue_type="transfer")
        encoder.transition_image_layout(
            image=self.target,
            layout="transfer-src-optimal",
        )
        encoder.copy_image_to_buffer(src=self.target, dst=buffer)
        encoder.submit().wait()

        result = buffer.memory.read(dtype=np.uint8).reshape(
            (TEST_IMAGE_H, TEST_IMAGE_W, 4)
        )
        buffer.dispose()
        return result


def test_draw_3d_red_cube():
    """
    Test rendering a simple red cube.

    The cube is positioned at the origin, and the camera is placed at (0, 0, 3)
    looking toward the origin. The cube should appear red with Lambertian shading
    using an overhead light direction (0, 0, 1).
    """
    engine = Draw3dTestEngine()

    # Create cube geometry
    cube = make_cube_geometry(engine.renderer)

    # Create red material (color tint will be multiplied with the default white texture)
    red_material = Draw3dMaterial(
        renderer=engine.renderer,
        color_tint=(1.0, 0.0, 0.0),  # Red
    )

    # Camera transform: position at (0, 0, 3), looking at origin
    # This is the world transform of the camera (not view matrix)
    camera_transform = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 3.0],  # Camera at z=3
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    # Camera intrinsics
    camera_intrinsics = Draw3dCameraIntrinsics(
        vertical_fov=np.radians(60.0),  # 60 degree vertical FOV
        clip_near=0.1,
        clip_far=100.0,
    )

    # Model transform: identity (cube at origin)
    # Rotate slightly so we can see multiple faces
    angle_y = np.radians(30.0)
    angle_x = np.radians(20.0)
    cos_y, sin_y = np.cos(angle_y), np.sin(angle_y)
    cos_x, sin_x = np.cos(angle_x), np.sin(angle_x)

    # Rotation around Y axis
    rot_y = np.array(
        [
            [cos_y, 0.0, sin_y, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-sin_y, 0.0, cos_y, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    # Rotation around X axis
    rot_x = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, cos_x, -sin_x, 0.0],
            [0.0, sin_x, cos_x, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    model_transform = rot_x @ rot_y

    # Create meshes dict: single cube instance
    meshes: dict[tuple[Draw3dGeometry, Draw3dMaterial], np.ndarray] = {
        (cube, red_material): np.array([model_transform], dtype=np.float32),
    }

    # Draw
    command_encoder = GpuCommandEncoder(
        device=engine.gpu_device,
        queue_type="graphics",
    )
    engine.renderer.draw(
        command_encoder=command_encoder,
        meshes=meshes,
        camera_transform=camera_transform,
        camera_intrinsics=camera_intrinsics,
        target=engine.target,
    )
    command_encoder.submit().wait()

    # Read back and save image
    image = engine.readback()

    output_dir = Path("output/zfw/draw_3d_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "test_draw_3d_red_cube.png"
    PIL.Image.fromarray(image).save(output_path)
    LOG.info(f"Saved rendered image to {output_path}")

    # Clean up
    red_material.dispose()
    cube.dispose()
    engine.dispose()

    # Test passes if we reach here without errors
    assert True


if __name__ == "__main__":
    pytest.main(["-v", __file__])
