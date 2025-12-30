__all__ = [
    "GltfScene",
    "load_gltf",
]

import base64
import io
from pathlib import Path

import numpy as np
import PIL.Image
import pygltflib

from .basic import BaseResource, expect, logger
from .draw_3d import (
    Draw3dGeometry,
    Draw3dMaterial,
    Draw3dRenderer,
    VERTEX_DTYPE,
)
from .gpu import GpuImage
from .images import load_rgba_image


LOG = logger(__name__)

# Coordinate system transformation matrix: glTF (Y-up, Z-forward) to Z-up, Y-forward.
# This is a -90° rotation around the X-axis.
# Maps: X -> X, Y -> Z, Z -> -Y
GLTF_TO_ZUP_MATRIX = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)


#
# GLTF loader
#


class GltfScene(BaseResource):
    """
    A loaded glTF scene ready for rendering.

    Contains a meshes dict mapping (geometry, material) pairs to instance
    transforms, suitable for passing directly to Draw3dRenderer.draw().
    """

    meshes: dict[tuple[Draw3dGeometry, Draw3dMaterial], np.ndarray]
    """
    Mapping of (geometry, material) pairs to per-instance transforms.
    Each transform array has shape (N, 4, 4) with N instance transforms in row-major order.
    """

    geometries: list[Draw3dGeometry]
    """All geometry objects created for this scene (for disposal)."""

    materials: list[Draw3dMaterial]
    """All material objects created for this scene (for disposal)."""

    images: list[GpuImage]
    """All GPU images created for this scene (for disposal)."""

    def __init__(
        self,
        *,
        meshes: dict[tuple[Draw3dGeometry, Draw3dMaterial], np.ndarray],
        geometries: list[Draw3dGeometry],
        materials: list[Draw3dMaterial],
        images: list[GpuImage],
        parent_resource: BaseResource | None = None,
    ):
        super().__init__(parent_resource=parent_resource)

        self.meshes = meshes
        self.geometries = geometries
        self.materials = materials
        self.images = images

    def _on_dispose(self) -> None:
        """Dispose all resources created for this scene."""
        for geometry in self.geometries:
            geometry.dispose()
        for material in self.materials:
            material.dispose()
        for image in self.images:
            image.dispose()


def load_gltf(
    renderer: Draw3dRenderer,
    path: Path | str,
    *,
    transform_coordinate_system: bool = False,
) -> list[GltfScene]:
    """
    Load a glTF file and return a list of scenes.

    Each scene contains a meshes dict mapping (geometry, material) pairs to
    instance transforms, suitable for passing to Draw3dRenderer.draw().

    :param renderer: The Draw3dRenderer to create resources with.
    :param path: Path to the glTF or GLB file.
    :param transform_coordinate_system: If True, transform from glTF's coordinate
        system (Y-up, Z-forward, X-right) to Z-up, Y-forward, X-right. This applies
        a -90° rotation around the X-axis to all geometry and instance transforms.
    :return: List of GltfScene objects, one per scene in the glTF file.
    """
    path = Path(path)
    LOG.info(f"Loading glTF file: {path}")

    gltf = pygltflib.GLTF2().load(str(path))

    # Check if glTF loaded successfully
    if gltf is None:
        raise ValueError(f"Failed to load glTF file: {path}")

    # Load all binary data
    blob_data = _load_blob_data(gltf, path)

    # Create shared resources: images, materials, geometries
    gpu_images = _load_images(renderer, gltf, blob_data, path.parent)
    materials = _load_materials(renderer, gltf, gpu_images)
    geometries = _load_geometries(renderer, gltf, blob_data)

    # Process each scene
    scenes: list[GltfScene] = []

    scene_indices = range(len(gltf.scenes)) if gltf.scenes else []
    if not scene_indices:
        LOG.warning("No scenes found in glTF file")
        return scenes

    for scene_idx in scene_indices:
        scene = gltf.scenes[scene_idx]
        meshes = _process_scene(
            gltf, scene, geometries, materials, transform_coordinate_system
        )

        # Only include geometries/materials/images that are actually used in this scene
        used_geometries = set()
        used_materials = set()
        for geom, mat in meshes.keys():
            used_geometries.add(geom)
            used_materials.add(mat)

        scenes.append(
            GltfScene(
                meshes=meshes,
                geometries=list(used_geometries),
                materials=list(used_materials),
                images=gpu_images,  # All images shared across scenes
            )
        )

    LOG.info(
        f"Loaded {len(scenes)} scene(s) with {len(geometries)} geometries, "
        f"{len(materials)} materials, {len(gpu_images)} images"
    )

    return scenes


def _load_blob_data(gltf: pygltflib.GLTF2, path: Path) -> list[bytes]:
    """Load all buffer data from the glTF file."""
    blob_data: list[bytes] = []

    for buffer_idx, buffer in enumerate(gltf.buffers or []):
        if buffer.uri is None:
            # Embedded GLB binary chunk
            if (bb := gltf.binary_blob()) is not None:
                assert isinstance(bb, bytes)
                blob_data.append(bb)
            else:
                raise ValueError(f"Buffer {buffer_idx} has no URI and no binary blob")
        elif buffer.uri.startswith("data:"):
            # Base64 embedded data
            header, data = buffer.uri.split(",", 1)
            blob_data.append(base64.b64decode(data))
        else:
            # External file
            buffer_path = path.parent / buffer.uri
            blob_data.append(buffer_path.read_bytes())

    return blob_data


def _get_accessor_data(
    gltf: pygltflib.GLTF2,
    blob_data: list[bytes],
    accessor_idx: int,
) -> np.ndarray:
    """Extract numpy array data from a glTF accessor."""
    accessor = gltf.accessors[accessor_idx]
    buffer_view = gltf.bufferViews[expect(accessor.bufferView)]

    # Component type mapping
    component_types: dict[int, np.dtype] = {
        5120: np.dtype(np.int8),
        5121: np.dtype(np.uint8),
        5122: np.dtype(np.int16),
        5123: np.dtype(np.uint16),
        5125: np.dtype(np.uint32),
        5126: np.dtype(np.float32),
    }
    dtype = component_types[accessor.componentType]

    # Type to component count mapping
    type_counts: dict[str, int] = {
        "SCALAR": 1,
        "VEC2": 2,
        "VEC3": 3,
        "VEC4": 4,
        "MAT2": 4,
        "MAT3": 9,
        "MAT4": 16,
    }
    component_count = type_counts[accessor.type]

    # Get raw bytes
    buffer_bytes = blob_data[buffer_view.buffer]
    byte_offset = (buffer_view.byteOffset or 0) + (accessor.byteOffset or 0)
    byte_stride = buffer_view.byteStride

    if byte_stride is None or byte_stride == 0:
        # Tightly packed data
        byte_length = accessor.count * component_count * dtype.itemsize
        raw_bytes = buffer_bytes[byte_offset : byte_offset + byte_length]
        data = np.frombuffer(raw_bytes, dtype=dtype)
        if component_count > 1:
            data = data.reshape(accessor.count, component_count)
    else:
        # Strided data - need to extract element by element
        element_size = component_count * dtype.itemsize
        data = np.zeros((accessor.count, component_count), dtype=dtype)
        for i in range(accessor.count):
            offset = byte_offset + i * byte_stride
            element_bytes = buffer_bytes[offset : offset + element_size]
            element = np.frombuffer(element_bytes, dtype=dtype)
            data[i] = element
        if component_count == 1:
            data = data.flatten()

    return data


def _load_images(
    renderer: Draw3dRenderer,
    gltf: pygltflib.GLTF2,
    blob_data: list[bytes],
    base_path: Path,
) -> list[GpuImage]:
    """Load all images from the glTF file as GPU textures."""
    gpu_images: list[GpuImage] = []

    for image_idx, image in enumerate(gltf.images or []):
        image_data: np.ndarray | None = None

        if image.bufferView is not None:
            # Image data embedded in buffer
            buffer_view = gltf.bufferViews[image.bufferView]
            buffer_bytes = blob_data[buffer_view.buffer]
            byte_offset = buffer_view.byteOffset or 0
            byte_length = buffer_view.byteLength
            raw_bytes = buffer_bytes[byte_offset : byte_offset + byte_length]

            pil_image = PIL.Image.open(io.BytesIO(raw_bytes)).convert("RGBA")
            image_data = np.array(pil_image, dtype=np.float32) / 255.0

        elif (image_uri := image.uri) is not None:
            if image_uri.startswith("data:"):
                # Base64 embedded image
                header, data = image_uri.split(",", 1)  # type: ignore
                raw_bytes = base64.b64decode(data)
                pil_image = PIL.Image.open(io.BytesIO(raw_bytes)).convert("RGBA")
                image_data = np.array(pil_image, dtype=np.float32) / 255.0
            else:
                # External file
                image_path = base_path / image.uri
                image_data = load_rgba_image(
                    image_path,
                    file_color_space="srgb",
                    output_color_space="linear",
                )

        if image_data is not None:
            gpu_image = GpuImage(
                device=renderer.gpu_device,
                usages=["texture-binding"],
                data=image_data,
            )
            gpu_images.append(gpu_image)
            LOG.debug(f"Loaded image {image_idx}: {image_data.shape}")
        else:
            LOG.warning(f"Could not load image {image_idx}")
            # Create a default white image as placeholder
            gpu_images.append(
                GpuImage(
                    device=renderer.gpu_device,
                    usages=["texture-binding"],
                    data=np.ones((1, 1, 4), dtype=np.float32),
                )
            )

    return gpu_images


def _load_materials(
    renderer: Draw3dRenderer,
    gltf: pygltflib.GLTF2,
    gpu_images: list[GpuImage],
) -> list[Draw3dMaterial]:
    """Load all materials from the glTF file."""
    materials: list[Draw3dMaterial] = []

    for material_idx, material in enumerate(gltf.materials or []):
        color_tint = (1.0, 1.0, 1.0)
        color_image: GpuImage | None = None
        normal_image: GpuImage | None = None
        metalness_tint = 1.0
        roughness_tint = 1.0
        metalness_image: GpuImage | None = None
        roughness_image: GpuImage | None = None

        # PBR metallic-roughness workflow
        pbr = material.pbrMetallicRoughness
        if pbr is not None:
            # Base color
            if pbr.baseColorFactor is not None:
                color_tint = (
                    pbr.baseColorFactor[0],
                    pbr.baseColorFactor[1],
                    pbr.baseColorFactor[2],
                )

            if pbr.baseColorTexture is not None:
                tex_idx = pbr.baseColorTexture.index
                if tex_idx is not None and gltf.textures:
                    texture = gltf.textures[tex_idx]
                    if texture.source is not None and texture.source < len(gpu_images):
                        color_image = gpu_images[texture.source]

            # Metallic-roughness
            if pbr.metallicFactor is not None:
                metalness_tint = pbr.metallicFactor
            if pbr.roughnessFactor is not None:
                roughness_tint = pbr.roughnessFactor

            if pbr.metallicRoughnessTexture is not None:
                tex_idx = pbr.metallicRoughnessTexture.index
                if tex_idx is not None and gltf.textures:
                    texture = gltf.textures[tex_idx]
                    if texture.source is not None and texture.source < len(gpu_images):
                        # glTF stores metallic in B channel, roughness in G channel
                        # For now, use the same image for both
                        metalness_image = gpu_images[texture.source]
                        roughness_image = gpu_images[texture.source]

        # Normal map
        if material.normalTexture is not None:
            tex_idx = material.normalTexture.index
            if tex_idx is not None and gltf.textures:
                texture = gltf.textures[tex_idx]
                if texture.source is not None and texture.source < len(gpu_images):
                    normal_image = gpu_images[texture.source]

        draw_material = Draw3dMaterial(
            renderer=renderer,
            color_tint=color_tint,
            color_image=color_image,
            normal_image=normal_image,
            metalness_tint=metalness_tint,
            metalness_image=metalness_image,
            roughness_tint=roughness_tint,
            roughness_image=roughness_image,
        )
        materials.append(draw_material)
        LOG.debug(f"Loaded material {material_idx}: {material.name}")

    # If no materials, create a default one
    if not materials:
        materials.append(Draw3dMaterial(renderer=renderer))

    return materials


def _load_geometries(
    renderer: Draw3dRenderer,
    gltf: pygltflib.GLTF2,
    blob_data: list[bytes],
) -> dict[tuple[int, int], Draw3dGeometry]:
    """
    Load all unique geometries (mesh primitives) from the glTF file.

    Returns a dict mapping (mesh_index, primitive_index) to Draw3dGeometry.
    Only TRIANGLES topology is supported.
    """
    geometries: dict[tuple[int, int], Draw3dGeometry] = {}

    for mesh_idx, mesh in enumerate(gltf.meshes or []):
        for prim_idx, primitive in enumerate(mesh.primitives):
            # Only support TRIANGULAR mode (4 = TRIANGLES)
            mode = primitive.mode if primitive.mode is not None else 4
            if mode != 4:
                LOG.warning(
                    f"Skipping primitive {mesh_idx}.{prim_idx}: "
                    f"unsupported mode {mode} (only TRIANGLES=4 supported)"
                )
                continue

            # Get vertex attributes
            attributes = primitive.attributes

            # Position (required)
            if attributes.POSITION is None:
                LOG.warning(
                    f"Skipping primitive {mesh_idx}.{prim_idx}: no POSITION attribute"
                )
                continue

            positions = _get_accessor_data(gltf, blob_data, attributes.POSITION)
            vertex_count = len(positions)

            # Normal (optional, generate flat normals if missing)
            if attributes.NORMAL is not None:
                normals = _get_accessor_data(gltf, blob_data, attributes.NORMAL)
            else:
                normals = np.zeros((vertex_count, 3), dtype=np.float32)
                normals[:, 2] = 1.0  # Default to +Z normal

            # Texture coordinates (optional)
            if attributes.TEXCOORD_0 is not None:
                texcoords = _get_accessor_data(gltf, blob_data, attributes.TEXCOORD_0)
            else:
                texcoords = np.zeros((vertex_count, 2), dtype=np.float32)

            # Build vertex array
            vertex_data = np.zeros(vertex_count, dtype=VERTEX_DTYPE)
            vertex_data["position"] = positions.astype(np.float32)
            vertex_data["normal"] = normals.astype(np.float32)
            vertex_data["texcoord0"] = texcoords.astype(np.float32)

            # Get or generate indices
            if primitive.indices is not None:
                indices = _get_accessor_data(gltf, blob_data, primitive.indices)
            else:
                # Generate sequential indices
                indices = np.arange(vertex_count, dtype=np.uint32)

            # Normalize indices to uint16
            if indices.max() > 65535:
                LOG.warning(
                    f"Primitive {mesh_idx}.{prim_idx} has indices > 65535, "
                    f"truncating (may cause visual artifacts)"
                )
            index_data = indices.astype(np.uint16)

            # Create geometry
            geometry = Draw3dGeometry(
                renderer=renderer,
                vertex_data=vertex_data,
                index_data=index_data,
            )
            geometries[(mesh_idx, prim_idx)] = geometry
            LOG.debug(
                f"Loaded geometry {mesh_idx}.{prim_idx}: "
                f"{vertex_count} vertices, {len(index_data)} indices"
            )

    return geometries


def _process_scene(
    gltf: pygltflib.GLTF2,
    scene: pygltflib.Scene,
    geometries: dict[tuple[int, int], Draw3dGeometry],
    materials: list[Draw3dMaterial],
    transform_coordinate_system: bool,
) -> dict[tuple[Draw3dGeometry, Draw3dMaterial], np.ndarray]:
    """
    Process a glTF scene and collect all mesh instances with world transforms.

    Returns a meshes dict suitable for Draw3dRenderer.draw().
    """
    # Collect all instances: (geometry, material) -> list of transforms
    instances: dict[tuple[Draw3dGeometry, Draw3dMaterial], list[np.ndarray]] = {}

    def traverse_node(node_idx: int, parent_transform: np.ndarray) -> None:
        """Recursively traverse nodes, accumulating transforms."""
        node = gltf.nodes[node_idx]

        # Compute local transform
        local_transform = _get_node_transform(node)

        # Compute world transform
        world_transform = parent_transform @ local_transform

        # If this node has a mesh, add instances for each primitive
        if node.mesh is not None:
            mesh = gltf.meshes[node.mesh]
            for prim_idx, primitive in enumerate(mesh.primitives):
                key = (node.mesh, prim_idx)
                if key not in geometries:
                    continue  # Skipped primitive (e.g., non-triangle topology)

                geometry = geometries[key]

                # Get material (default to first material if none specified)
                material_idx = (
                    primitive.material if primitive.material is not None else 0
                )
                if material_idx >= len(materials):
                    material_idx = 0
                material = materials[material_idx]

                instance_key = (geometry, material)
                if instance_key not in instances:
                    instances[instance_key] = []
                instances[instance_key].append(world_transform.copy())

        # Traverse children
        for child_idx in node.children or []:
            traverse_node(child_idx, world_transform)

    # Start traversal from scene root nodes
    # If transforming coordinate system, start with the conversion matrix
    # so all transforms are in the new coordinate system
    if transform_coordinate_system:
        root_transform = GLTF_TO_ZUP_MATRIX.copy()
    else:
        root_transform = np.eye(4, dtype=np.float32)

    for root_idx in scene.nodes or []:
        traverse_node(root_idx, root_transform)

    # Convert instance lists to numpy arrays
    meshes: dict[tuple[Draw3dGeometry, Draw3dMaterial], np.ndarray] = {}
    for key, transform_list in instances.items():
        meshes[key] = np.array(transform_list, dtype=np.float32)

    return meshes


def _get_node_transform(node: pygltflib.Node) -> np.ndarray:
    """Get the local transform matrix for a node."""
    if node.matrix is not None:
        # Matrix is stored column-major in glTF, but we need row-major
        matrix = np.array(node.matrix, dtype=np.float32).reshape(4, 4).T
        return matrix

    # Build transform from TRS components
    transform = np.eye(4, dtype=np.float32)

    # Scale
    if node.scale is not None:
        scale = np.array(node.scale, dtype=np.float32)
        transform = transform @ np.diag([scale[0], scale[1], scale[2], 1.0])

    # Rotation (quaternion: x, y, z, w)
    if node.rotation is not None:
        qx, qy, qz, qw = node.rotation
        rot_matrix = _quaternion_to_matrix(qx, qy, qz, qw)
        transform = rot_matrix @ transform

    # Translation
    if node.translation is not None:
        translation = np.array(node.translation, dtype=np.float32)
        trans_matrix = np.eye(4, dtype=np.float32)
        trans_matrix[:3, 3] = translation
        transform = trans_matrix @ transform

    return transform


def _quaternion_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    """Convert a quaternion to a 4x4 rotation matrix (row-major)."""
    # Normalize quaternion
    n = np.sqrt(x * x + y * y + z * z + w * w)
    if n > 0:
        x, y, z, w = x / n, y / n, z / n, w / n

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy), 0],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx), 0],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy), 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
