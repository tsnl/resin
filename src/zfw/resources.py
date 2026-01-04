"""
Resources = file-based data assets used by ZFW, such as images, 3D models, 3D materials, etc.
"""

__all__ = [
    "COOKED_ATLAS_PATH_SUFFIX",
    "CookedAtlas",
    "CookedAtlasGlyphCacheKey",
    "CookedAtlasGlyphInfo",
    "GeometryResource",
    "MaterialResource",
    "load_gltf",
    "load_image",
    "load_rgba_image",
    "load_rgba_image_from_bytes",
]

import base64
import io
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Literal

import numpy as np
import orjson
import pydantic
import PIL.Image
import pygltflib
import jaxtyping as jt

from .excepts import LogicError
from .basic import ColorSpace, Font, FontSize, FontWeight, logger
from .images import convert_color


LOG = logger(__name__)


#
# Resource Types for 3D Geometry and Materials
#


class GeometryResource:
    """Contains NumPy arrays needed to construct a Draw3dGeometry."""

    v_p_array: jt.Float32[np.ndarray, "nv 3"]
    """Vertex positions array."""

    v_n_array: jt.Float32[np.ndarray, "nv 3"]
    """Vertex normals array."""

    v_t_array: jt.Float32[np.ndarray, "nv 2"]
    """Vertex texture coordinates array."""

    t_indices: jt.UInt32[np.ndarray, "nt 3"]
    """Triangle indices array."""

    def __init__(
        self,
        *,
        v_p_array: jt.Float32[np.ndarray, "nv 3"],
        v_n_array: jt.Float32[np.ndarray, "nv 3"],
        v_t_array: jt.Float32[np.ndarray, "nv 2"],
        t_indices: jt.UInt32[np.ndarray, "nt 3"],
    ) -> None:
        self.v_p_array = v_p_array
        self.v_n_array = v_n_array
        self.v_t_array = v_t_array
        self.t_indices = t_indices


class MaterialResource:
    """Contains NumPy arrays and factors needed to construct a Draw3dMaterial."""

    color_map: jt.Float32[np.ndarray, "h w 3"] | None
    """Base color texture (sRGB converted to linear)."""

    color_factor: tuple[float, float, float]
    """Base color factor."""

    normal_map: jt.Float32[np.ndarray, "h w 3"] | None
    """Normal map texture (linear space)."""

    metalness_map: jt.Float32[np.ndarray, "h w 1"] | None
    """Metalness texture (linear space)."""

    metalness_factor: float
    """Metalness factor."""

    roughness_map: jt.Float32[np.ndarray, "h w 1"] | None
    """Roughness texture (linear space)."""

    roughness_factor: float
    """Roughness factor."""

    def __init__(
        self,
        *,
        color_map: jt.Float32[np.ndarray, "h w 3"] | None = None,
        color_factor: tuple[float, float, float] = (1.0, 1.0, 1.0),
        normal_map: jt.Float32[np.ndarray, "h w 3"] | None = None,
        metalness_map: jt.Float32[np.ndarray, "h w 1"] | None = None,
        metalness_factor: float = 1.0,
        roughness_map: jt.Float32[np.ndarray, "h w 1"] | None = None,
        roughness_factor: float = 1.0,
    ) -> None:
        self.color_map = color_map
        self.color_factor = color_factor
        self.normal_map = normal_map
        self.metalness_map = metalness_map
        self.metalness_factor = metalness_factor
        self.roughness_map = roughness_map
        self.roughness_factor = roughness_factor


#
# Image Loading
#
# FIXME: PIL only reads uint8 images. Need a custom image reader (and probably BC
# support).
#


def load_image(
    *,
    file_path: Path,
    expected_channel_count: Literal[1, 4],
    input_color_space: ColorSpace = "srgb",
    output_color_space: ColorSpace = "linear",
) -> np.ndarray:
    match expected_channel_count:
        case 1:
            if input_color_space != "linear" or output_color_space != "linear":
                LOG.error(
                    "Color space conversion requested for mono image: "
                    "ignoring and loading as linear: please fix the caller."
                )
            return load_mono_image(file_path)
        case 4:
            return load_rgba_image(
                file_path,
                input_color_space=input_color_space,
                output_color_space=output_color_space,
            )


def load_mono_image(file_path: Path) -> np.ndarray:
    """
    Loads a mono (grayscale) image as a normalized NumPy array. Assumed linear color
    space.

    :param file_path: The path to the image file
    :return: Mono image as float32 array with shape (H, W, 1).
    """
    src = np.array(PIL.Image.open(file_path).convert("L"))
    src_normalized = src.astype(np.float32) / 255.0
    return src_normalized[:, :, np.newaxis]


def load_rgba_image(
    file_path: Path,
    input_color_space: ColorSpace = "srgb",
    output_color_space: ColorSpace = "linear",
) -> np.ndarray:
    """
    Loads an RGBA image as a normalized NumPy array.

    :param file_path: The path to the image file
    :param input_color_space: The color space of the input image.
    :param output_color_space: The desired output color space.
    :return: RGBA image as float32 array with shape (H, W, 4).
    """
    src = np.array(PIL.Image.open(file_path).convert("RGBA"))
    src_normalized = src.astype(np.float32) / 255.0
    dst_rgb = convert_color(
        src_normalized[..., :3],
        src_color_space=input_color_space,
        dst_color_space=output_color_space,
    )
    dst_alpha = src_normalized[..., 3:4]
    return np.concatenate((dst_rgb, dst_alpha), axis=-1)


def load_rgba_image_from_bytes(
    raw_bytes: bytes,
    input_color_space: ColorSpace = "srgb",
    output_color_space: ColorSpace = "linear",
) -> np.ndarray:
    """
    Loads an RGBA image from raw bytes as a normalized NumPy array.

    :param raw_bytes: The raw image bytes (e.g., PNG or JPEG data).
    :param input_color_space: The color space of the input image.
    :param output_color_space: The desired output color space.
    :return: RGBA image as float32 array with shape (H, W, 4).
    """
    src = np.array(PIL.Image.open(io.BytesIO(raw_bytes)).convert("RGBA"))
    src_normalized = src.astype(np.float32) / 255.0
    dst_rgb = convert_color(
        src_normalized[..., :3],
        src_color_space=input_color_space,
        dst_color_space=output_color_space,
    )
    dst_alpha = src_normalized[..., 3:4]
    return np.concatenate((dst_rgb, dst_alpha), axis=-1)


#
# GLTF loader
#


def load_gltf(
    path: Path | str,
    *,
    transform_coordinate_system: bool = True,
) -> dict[tuple[GeometryResource, MaterialResource], jt.Float32[np.ndarray, "N 4 4"]]:
    """
    Load a glTF file and return a meshes dict with resource types.

    Returns a meshes dict mapping (geometry_resource, material_resource) pairs to instance
    transforms (Nx4x4 arrays). Matrices are in row-major order, and should have [0, 0, 0, 1]
    in the last row to represent affine transforms in homogeneous coordinates.

    :param path: Path to the glTF or GLB file.
    :param transform_coordinate_system: If True, transform from glTF's coordinate
        system (Y-up, Z-forward, X-right) to Z-up, Y-forward, X-right. This applies
        a -90° rotation around the X-axis to all geometry and instance transforms.
    :return: Dict mapping (geometry_resource, material_resource) to instance transforms.
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
    image_sources = _load_image_sources(gltf, blob_data, path.parent)
    materials = _load_material_resources(gltf, image_sources)
    geometries = _load_geometry_resources(gltf, blob_data)

    # Process the default scene (or first scene)
    scene_idx = gltf.scene if gltf.scene is not None else 0
    if not gltf.scenes or scene_idx >= len(gltf.scenes):
        LOG.warning("No valid scene found in glTF file")
        return {}

    scene = gltf.scenes[scene_idx]
    meshes = _process_scene_resources(
        gltf, scene, geometries, materials, transform_coordinate_system
    )

    LOG.info(
        f"Loaded scene with {len(geometries)} geometries, "
        f"{len(materials)} materials, {sum(len(t) for t in meshes.values())} instances"
    )

    return meshes


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
    assert isinstance(accessor.bufferView, int)
    buffer_view = gltf.bufferViews[accessor.bufferView]

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


class _ImageSource:
    """Stores raw image bytes or file path for deferred loading with color space."""

    def __init__(
        self,
        *,
        raw_bytes: bytes | None = None,
        file_path: Path | None = None,
    ):
        self.raw_bytes = raw_bytes
        self.file_path = file_path

    def load(self, input_color_space: ColorSpace) -> np.ndarray:
        """Load the image with the specified input color space, output as linear."""
        if self.raw_bytes is not None:
            return load_rgba_image_from_bytes(
                self.raw_bytes,
                input_color_space=input_color_space,
                output_color_space="linear",
            )
        elif self.file_path is not None:
            return load_rgba_image(
                self.file_path,
                input_color_space=input_color_space,
                output_color_space="linear",
            )
        else:
            raise ValueError("ImageSource has no raw_bytes or file_path")


def _load_image_sources(
    gltf: pygltflib.GLTF2,
    blob_data: list[bytes],
    base_path: Path,
) -> list[_ImageSource]:
    """Load all image sources from the glTF file.

    Returns ImageSource objects that can be loaded later with the appropriate
    color space for each texture usage.
    """
    image_sources: list[_ImageSource] = []

    for image_idx, image in enumerate(gltf.images or []):
        source: _ImageSource | None = None

        if image.bufferView is not None:
            # Image data embedded in buffer
            buffer_view = gltf.bufferViews[image.bufferView]
            buffer_bytes = blob_data[buffer_view.buffer]
            byte_offset = buffer_view.byteOffset or 0
            byte_length = buffer_view.byteLength
            raw_bytes = buffer_bytes[byte_offset : byte_offset + byte_length]
            source = _ImageSource(raw_bytes=raw_bytes)

        elif (image_uri := image.uri) is not None:
            if image_uri.startswith("data:"):
                # Base64 embedded image
                _, data = image_uri.split(",", 1)  # type: ignore[var-annotated]
                raw_bytes = base64.b64decode(data)
                source = _ImageSource(raw_bytes=raw_bytes)
            else:
                # External file
                source = _ImageSource(file_path=base_path / image.uri)

        if source is not None:
            image_sources.append(source)
            LOG.debug(f"Found image source {image_idx}")
        else:
            LOG.warning(f"Could not find image source {image_idx}")
            # Create a placeholder that will return white
            image_sources.append(_ImageSource(raw_bytes=_WHITE_1X1_PNG))

    return image_sources


# 1x1 white PNG for placeholder images
_WHITE_1X1_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5/hPwAIAgL/4d1j8wAAAABJRU5ErkJggg=="
)


def _extract_channel_as_rgba(image_data: np.ndarray, channel: int) -> np.ndarray:
    """Extract a single channel from RGBA image data and return as single-channel array.

    The extracted channel value is placed in a single-channel array.
    """
    return image_data[:, :, channel : channel + 1]


def _load_material_resources(
    gltf: pygltflib.GLTF2,
    image_sources: list[_ImageSource],
) -> list[MaterialResource]:
    """Load all materials from the glTF file as MaterialResource objects.

    Color space handling:
    - Base color textures: loaded as sRGB (converted to linear)
    - Metallic-roughness textures: loaded as linear (no conversion)
    - Normal maps: loaded as linear (no conversion)
    """
    materials: list[MaterialResource] = []

    # Cache for loaded image data from sources
    # Maps (source_index, color_space, channel_or_none) to np.ndarray
    # channel_or_none: None for full RGBA, 0-3 for extracted channel
    image_data_cache: dict[tuple[int, ColorSpace, int | None], np.ndarray] = {}

    def get_or_load_image(
        source_idx: int,
        input_color_space: ColorSpace,
        channel: int | None = None,
    ) -> np.ndarray:
        """Get or load image data, with caching to avoid duplicates."""
        cache_key = (source_idx, input_color_space, channel)
        if cache_key in image_data_cache:
            return image_data_cache[cache_key]

        source_data = image_sources[source_idx].load(input_color_space)
        if channel is not None:
            # Extract single channel
            data = _extract_channel_as_rgba(source_data, channel)
        else:
            data = source_data

        image_data_cache[cache_key] = data
        return data

    for material_idx, material in enumerate(gltf.materials or []):
        color_factor = (1.0, 1.0, 1.0)
        color_map: np.ndarray | None = None
        normal_map: np.ndarray | None = None
        metalness_factor = 1.0
        roughness_factor = 1.0
        metalness_map: np.ndarray | None = None
        roughness_map: np.ndarray | None = None

        # PBR metallic-roughness workflow
        pbr = material.pbrMetallicRoughness
        if pbr is not None:
            # Base color
            if pbr.baseColorFactor is not None:
                color_factor = (
                    pbr.baseColorFactor[0],
                    pbr.baseColorFactor[1],
                    pbr.baseColorFactor[2],
                )

            if pbr.baseColorTexture is not None:
                tex_idx = pbr.baseColorTexture.index
                if tex_idx is not None and gltf.textures:
                    texture = gltf.textures[tex_idx]
                    if texture.source is not None and texture.source < len(
                        image_sources
                    ):
                        # Base color is stored in sRGB, need RGB only
                        rgba_data = get_or_load_image(
                            texture.source, input_color_space="srgb"
                        )
                        color_map = rgba_data[:, :, :3]

            # Metallic-roughness
            if pbr.metallicFactor is not None:
                metalness_factor = pbr.metallicFactor
            if pbr.roughnessFactor is not None:
                roughness_factor = pbr.roughnessFactor

            if pbr.metallicRoughnessTexture is not None:
                tex_idx = pbr.metallicRoughnessTexture.index
                if tex_idx is not None and gltf.textures:
                    texture = gltf.textures[tex_idx]
                    if texture.source is not None and texture.source < len(
                        image_sources
                    ):
                        # glTF stores metallic in B channel (index 2),
                        # roughness in G channel (index 1)
                        # Metallic-roughness is stored in linear space
                        metalness_map = get_or_load_image(
                            texture.source, input_color_space="linear", channel=2
                        )
                        roughness_map = get_or_load_image(
                            texture.source, input_color_space="linear", channel=1
                        )

        # Normal map (stored in linear space)
        if material.normalTexture is not None:
            tex_idx = material.normalTexture.index
            if tex_idx is not None and gltf.textures:
                texture = gltf.textures[tex_idx]
                if texture.source is not None and texture.source < len(image_sources):
                    rgba_data = get_or_load_image(
                        texture.source, input_color_space="linear"
                    )
                    normal_map = rgba_data[:, :, :3]

        material_resource = MaterialResource(
            color_factor=color_factor,
            color_map=color_map,
            normal_map=normal_map,
            metalness_factor=metalness_factor,
            metalness_map=metalness_map,
            roughness_factor=roughness_factor,
            roughness_map=roughness_map,
        )
        materials.append(material_resource)
        LOG.debug(f"Loaded material {material_idx}: {material.name}")

    # If no materials, create a default one
    if not materials:
        materials.append(MaterialResource())

    return materials


def _load_geometry_resources(
    gltf: pygltflib.GLTF2,
    blob_data: list[bytes],
) -> dict[tuple[int, int], GeometryResource]:
    """
    Load all unique geometries (mesh primitives) from the glTF file as GeometryResource objects.

    Returns a dict mapping (mesh_index, primitive_index) to GeometryResource.
    Only TRIANGLES topology is supported.
    """
    geometries: dict[tuple[int, int], GeometryResource] = {}

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
                # Wrap texture coordinates to [0, 1) range to simulate repeat wrapping
                texcoords = np.mod(texcoords, 1.0)
            else:
                texcoords = np.zeros((vertex_count, 2), dtype=np.float32)

            # Get or generate indices
            if primitive.indices is not None:
                indices = _get_accessor_data(gltf, blob_data, primitive.indices)
            else:
                # Generate sequential indices
                indices = np.arange(vertex_count, dtype=np.uint32)

            # Ensure indices are in the right format
            indices = indices.astype(np.uint32)

            # Reshape indices to triangles (Nx3)
            triangle_count = len(indices) // 3
            t_indices = indices[: triangle_count * 3].reshape(-1, 3)

            # Create geometry resource
            geometry = GeometryResource(
                v_p_array=positions.astype(np.float32),
                v_n_array=normals.astype(np.float32),
                v_t_array=texcoords.astype(np.float32),
                t_indices=t_indices,
            )
            geometries[(mesh_idx, prim_idx)] = geometry
            LOG.debug(
                f"Loaded geometry {mesh_idx}.{prim_idx}: "
                f"{vertex_count} vertices, {triangle_count} triangles"
            )

    return geometries


def _process_scene_resources(
    gltf: pygltflib.GLTF2,
    scene: pygltflib.Scene,
    geometries: dict[tuple[int, int], GeometryResource],
    materials: list[MaterialResource],
    transform_coordinate_system: bool,
) -> dict[tuple[GeometryResource, MaterialResource], np.ndarray]:
    """
    Process a glTF scene and collect all mesh instances with world transforms.

    Returns a meshes dict with resource types.
    """
    # Collect all instances: (geometry, material) -> list of transforms
    instances: dict[tuple[GeometryResource, MaterialResource], list[np.ndarray]] = {}

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

                instances[instance_key].append(world_transform)

        # Traverse children
        for child_idx in node.children or []:
            traverse_node(child_idx, world_transform)

    # Start traversal from scene root nodes
    # If transforming coordinate system, start with the conversion matrix
    if transform_coordinate_system:
        root_transform = GLTF_TO_Z_UP_MATRIX.copy()
    else:
        root_transform = np.eye(4, dtype=np.float32)

    for root_idx in scene.nodes or []:
        traverse_node(root_idx, root_transform)

    # Convert instance lists to numpy arrays with shape (N, 3, 4)
    meshes: dict[tuple[GeometryResource, MaterialResource], np.ndarray] = {}
    for key, transform_list in instances.items():
        meshes[key] = np.array(transform_list, dtype=np.float32)

    return meshes


def _get_node_transform(node: pygltflib.Node) -> np.ndarray:
    """Get the local transform matrix for a node."""
    # Matrix is stored column-major in glTF, but we need row-major
    if node.matrix is not None:
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


# Coordinate system transformation matrix: glTF (Y-up, Z-forward) to Z-up, Y-forward.
# This is a -90° rotation around the X-axis.
# Maps: X -> X, Y -> Z, Z -> -Y
GLTF_TO_Z_UP_MATRIX = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)


#
# CookedAtlas:
#


CookedAtlasType = Literal["glyph-cache"]


COOKED_ATLAS_PATH_SUFFIX: str = ".zfw_atlas"


@dataclass(kw_only=True, frozen=True)
class CookedAtlas:
    atlas_type: CookedAtlasType
    atlas_data: np.ndarray  # (h, w, 4) RGBA f32 image or (h, w, 1) mono f32 image
    image_xywh_list: list[tuple[int, int, int, int]]
    color_space: ColorSpace = "linear"
    readme_text: str | None = None
    license_text: str | None = None
    as_glyph_cache: dict[CookedAtlasGlyphCacheKey, "CookedAtlasGlyphInfo"] | None = None

    @staticmethod
    def load(
        *,
        path: Path,
        color_space: ColorSpace = "linear",
        load_readme_text: bool = False,
        load_license_text: bool = False,
        load_glyph_metrics: bool = True,
    ) -> "CookedAtlas":
        if path.suffix != COOKED_ATLAS_PATH_SUFFIX:
            raise ValueError(
                f"CookedAtlas load path must have suffix {COOKED_ATLAS_PATH_SUFFIX!r}: "
                f"{path=}"
            )
        if not path.is_dir():
            raise FileNotFoundError(f"Cooked atlas path not found: {path}")

        # Load index.json
        with open(path / "index.json", "rb") as f:
            index = CookedAtlasIndexFile(**orjson.loads(f.read()))
        image_xywh_list = index.image_xywh_list

        # Load atlas.png
        atlas_path = path / "atlas.png"
        atlas_data = load_image(
            file_path=atlas_path,
            expected_channel_count=index.channel_count,
            input_color_space=index.color_space,
            output_color_space=color_space,
        )

        # (Optional) Load README.md
        if load_readme_text:
            with open(path / "README.md", "r", encoding="utf-8") as f:
                readme_text = f.read()
        else:
            readme_text = None

        # (Optional) Load LICENSE.txt
        if load_license_text:
            with open(path / "LICENSE.txt", "r", encoding="utf-8") as f:
                license_text = f.read()
        else:
            license_text = None

        # (Optional) Load glyph metrics
        if load_glyph_metrics:
            with open(path / "glyph_metrics.json", "rb") as f:
                as_glyph_cache = {
                    key: val
                    for key, val in CookedAtlasGlyphCacheExtFile(
                        **orjson.loads(f.read())
                    ).data
                }
        else:
            as_glyph_cache = None

        return CookedAtlas(
            atlas_type="glyph-cache",
            atlas_data=atlas_data,
            image_xywh_list=image_xywh_list,
            color_space=color_space,
            readme_text=readme_text,
            license_text=license_text,
            as_glyph_cache=as_glyph_cache,
        )

    def save(self, path: Path) -> None:
        if path.suffix != COOKED_ATLAS_PATH_SUFFIX:
            raise ValueError(
                f"CookedAtlas save path must have {COOKED_ATLAS_PATH_SUFFIX!r} suffix: "
                f"{path=}"
            )

        path.mkdir(parents=True, exist_ok=True)

        # Save index.json
        with open(path / "index.json", "wb") as f:
            index = CookedAtlasIndexFile(
                cooked_atlas_type=self.atlas_type,
                image_xywh_list=self.image_xywh_list,
                channel_count=self.atlas_data.shape[2],
                color_space=self.color_space,
            )
            f.write(orjson.dumps(index.model_dump()))

        # Save atlas.png
        pil_atlas = (np.clip(self.atlas_data, 0.0, 1.0) * 255).astype(np.uint8)
        pil_atlas = pil_atlas.squeeze()
        pil_mode = "RGBA" if self.atlas_data.shape[2] == 4 else "L"
        pil_image = PIL.Image.fromarray(pil_atlas, mode=pil_mode)
        pil_image.save(
            path / "atlas.png",
            format="PNG",
            compress_level=9,
            optimize=True,
        )

        # (Optional) Save README.md
        if self.readme_text:
            with open(path / "README.md", "w", encoding="utf-8") as f:
                f.write(self.readme_text)

        # (Optional) Save LICENSE.txt
        if self.license_text:
            with open(path / "LICENSE.txt", "w", encoding="utf-8") as f:
                f.write(self.license_text)

        # (Optional) Save glyph cache extension data:
        if self.as_glyph_cache:
            if self.atlas_type != "glyph-cache":
                raise LogicError(
                    "Inconsistent CookedAtlas instance: if 'as_glyph_cache' is set, "
                    "'atlas_type' must be 'glyph_cache'."
                )
            with open(path / "glyph_metrics.json", "wb") as f:
                glyph_metrics_file = CookedAtlasGlyphCacheExtFile(
                    data=list(self.as_glyph_cache.items())
                )
                f.write(orjson.dumps(glyph_metrics_file.model_dump()))


class CookedAtlasIndexFile(pydantic.BaseModel):
    cooked_atlas_type: CookedAtlasType
    image_xywh_list: list[tuple[int, int, int, int]]
    channel_count: Literal[1, 4]
    color_space: ColorSpace


#
# CookedAtlas: GlyphCacheExt
#


class CookedAtlasGlyphCacheExtFile(pydantic.BaseModel):
    data: list[tuple[CookedAtlasGlyphCacheKey, "CookedAtlasGlyphInfo"]]


@dataclass(frozen=True, kw_only=True)
class CookedAtlasGlyphCacheKey:
    glyph_index: int
    font_name: Font
    font_size: FontSize
    font_weight: FontWeight
    scale: Fraction


class CookedAtlasGlyphInfo(pydantic.BaseModel):
    image_id: int
    ascender_26_6: int
    descender_26_6: int
    height_26_6: int
    bitmap_left: int
    bitmap_top: int
