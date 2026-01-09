from pathlib import Path

import numpy as np
import wgpu

from zfw import (
    Draw3dGeometry,
    Draw3dMaterial,
    Draw3dRenderer,
    GeometryResource,
    MaterialResource,
    load_gltf,
)


def test_gltf_loader_basic(gpu_device: wgpu.GPUDevice):
    """Test loading a basic glTF file (Box model)."""
    # Create renderer
    renderer = Draw3dRenderer(
        device=gpu_device,
        queue=gpu_device.queue,
        target_size_wh_px=(512, 512),
    )

    # Load the Box glTF model
    box_path = Path("tests/data/glTF-Sample-Assets/Models/Box/glTF/Box.gltf")
    meshes = load_gltf(box_path)

    # Verify we got a valid meshes dict
    assert isinstance(meshes, dict)
    assert len(meshes) > 0

    # Check that keys are (geometry_resource, material_resource) tuples
    for key, transforms in meshes.items():
        assert isinstance(key, tuple)
        assert len(key) == 2
        geometry_resource, material_resource = key

        # Verify types
        assert isinstance(geometry_resource, GeometryResource)
        assert isinstance(material_resource, MaterialResource)

        # Convert to Draw3d objects
        geometry = Draw3dGeometry.from_resource(geometry_resource, renderer)
        material = Draw3dMaterial.from_resource(material_resource, renderer)

        # Verify converted types
        assert isinstance(geometry, Draw3dGeometry)
        assert isinstance(material, Draw3dMaterial)

        # Verify transform array shape (N, 4, 4)
        assert isinstance(transforms, np.ndarray)
        assert transforms.ndim == 3
        assert transforms.shape[1:] == (4, 4)
        assert transforms.dtype == np.float32

        # For Box model, expect at least one instance
        assert transforms.shape[0] >= 1
