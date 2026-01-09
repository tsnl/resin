from pathlib import Path

import numpy as np

from conftest import GpuFixture

from zfw import (
    Draw3dGeometry,
    Draw3dMaterial,
    Draw3dRenderer,
    Draw3dTexture,
    GeometryResource,
    MaterialResource,
    load_gltf,
)


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
        geometry = Draw3dGeometry(
            renderer,
            v_p_array=geometry_resource.v_p_array,
            v_n_array=geometry_resource.v_n_array,
            v_t_array=geometry_resource.v_t_array,
            t_indices=geometry_resource.t_indices,
        )
        material = Draw3dMaterial(
            renderer,
            color_factor=material_resource.color_factor,
            color_texture=(
                Draw3dTexture(
                    renderer,
                    data=material_resource.color_map.data,
                    usage="color",
                )
                if material_resource.color_map
                else None
            ),
            normal_texture=(
                Draw3dTexture(
                    renderer,
                    data=material_resource.normal_map.data,
                    usage="normal",
                )
                if material_resource.normal_map
                else None
            ),
            metalness_factor=material_resource.metalness_factor,
            metalness_texture=(
                Draw3dTexture(
                    renderer,
                    data=material_resource.metalness_map.data,
                    usage="metalness",
                )
                if material_resource.metalness_map
                else None
            ),
            roughness_factor=material_resource.roughness_factor,
            roughness_texture=(
                Draw3dTexture(
                    renderer,
                    data=material_resource.roughness_map.data,
                    usage="roughness",
                )
                if material_resource.roughness_map
                else None
            ),
        )

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
