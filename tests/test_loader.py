from pathlib import Path

import numpy as np

from conftest import GpuFixture

from zfw import Draw3dGeometry, Draw3dMaterial, Draw3dRenderer, load_gltf


def test_gltf_loader_basic(gpu: GpuFixture):
    """Test loading a basic glTF file (Box model)."""
    # Create renderer
    renderer = Draw3dRenderer(
        device=gpu.device,
        queue=gpu.queue,
        target_size_wh_px=(512, 512),
    )

    # Load the Box glTF model
    box_path = Path("tests/data/glTF-Sample-Assets/Models/Box/glTF/Box.gltf")
    meshes = load_gltf(renderer, box_path)

    # Verify we got a valid meshes dict
    assert isinstance(meshes, dict)
    assert len(meshes) > 0

    # Check that keys are (geometry, material) tuples
    for key, transforms in meshes.items():
        assert isinstance(key, tuple)
        assert len(key) == 2
        geometry, material = key

        # Verify types
        assert isinstance(geometry, Draw3dGeometry)
        assert isinstance(material, Draw3dMaterial)

        # Verify transform array shape (N, 3, 4)
        assert isinstance(transforms, np.ndarray)
        assert transforms.ndim == 3
        assert transforms.shape[1:] == (3, 4)
        assert transforms.dtype == np.float32

        # For Box model, expect at least one instance
        assert transforms.shape[0] >= 1
