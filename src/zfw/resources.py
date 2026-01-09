"""
Resources = file-based data assets used by ZFW, such as images, 3D models, 3D materials,
etc.
"""

__all__ = [
    "COOKED_ATLAS_PATH_SUFFIX",
    "CookedAtlas",
    "CookedAtlasGlyphCacheKey",
    "CookedAtlasGlyphInfo",
    "GeometryResource",
    "ImageResource",
    "MaterialResource",
    "load_gltf",
    "load_gltf_v2",
    "load_image",
    "load_image_from_bytes",
]

import base64
from collections import defaultdict
import hashlib
import io
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import quaternion
import orjson
import pydantic
import imageio.v3 as iio
import cv2 as cv
import pygltflib
import jaxtyping as jt

from .excepts import LogicError
from .basic import BaseDisposable, Font, FontSize, FontWeight, expect, logger
from .images import (
    ImageFormat,
    convert_image_format,
    convert_srgb_to_linear,
    normalize_image_to_f32,
    convert_linear_to_srgb,
)


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
    """
    Contains image resources and factors needed to construct a Draw3dMaterial.
    """

    color_map: "npt.NDArray[np.float32] | None"
    color_factor: tuple[float, float, float]
    normal_map: "npt.NDArray[np.float32] | None"
    metalness_map: "npt.NDArray[np.float32] | None"
    metalness_factor: float
    roughness_map: "npt.NDArray[np.float32] | None"
    roughness_factor: float

    def __init__(
        self,
        *,
        color_map: "npt.NDArray[np.float32] | None",
        normal_map: "npt.NDArray[np.float32] | None",
        metalness_map: "npt.NDArray[np.float32] | None",
        roughness_map: "npt.NDArray[np.float32] | None",
        color_factor: tuple[float, float, float] = (1.0, 1.0, 1.0),
        metalness_factor: float = 1.0,
        roughness_factor: float = 1.0,
    ) -> None:
        self.color_map = color_map
        self.color_factor = color_factor
        self.normal_map = normal_map
        self.metalness_map = metalness_map
        self.metalness_factor = metalness_factor
        self.roughness_map = roughness_map
        self.roughness_factor = roughness_factor


class ImageResource:
    """Stores image data along with metadata about format and dimensions."""

    def __init__(
        self,
        *,
        data: np.ndarray,
        width: int,
        height: int,
        depth: int,
        image_format: ImageFormat,
    ) -> None:
        self.data = data
        self.width = width
        self.height = height
        self.depth = depth
        self.image_format = image_format


#
# Image Loading
#


def load_image(
    file_path: Path | str,
    image_format: ImageFormat,
    *,
    expected_format: ImageFormat | None = None,
) -> ImageResource:
    """
    Load an image from a file path.

    If expected_format is not None, performs color/format conversion from
    image_format (input) to expected_format (output).

    :param file_path: Path to the image file.
    :param image_format: The format of the image as stored.
    :param expected_format: Optional output format. If provided, converts the image.
    :return: ImageResource with loaded data and metadata.
    """
    file_path = Path(file_path)

    # Load image using imageio (preserves bit-depth)
    raw_data = iio.imread(file_path)

    # Ensure 3D array (height, width, channels)
    if raw_data.ndim == 2:
        # Grayscale image - add channel dimension
        raw_data = raw_data[:, :, np.newaxis]

    height, width = raw_data.shape[:2]

    # Apply format conversion if needed
    if expected_format is not None and expected_format != image_format:
        converted_data = convert_image_format(raw_data, image_format, expected_format)
        output_format = expected_format
    else:
        # Just normalize to float32
        converted_data = normalize_image_to_f32(raw_data)
        output_format = image_format

    # Determine depth (number of channels)
    depth = converted_data.shape[2] if converted_data.ndim == 3 else 1

    return ImageResource(
        data=converted_data,
        width=width,
        height=height,
        depth=depth,
        image_format=output_format,
    )


def load_image_from_bytes(
    raw_bytes: bytes,
    image_format: ImageFormat,
    *,
    expected_format: ImageFormat | None = None,
) -> ImageResource:
    """
    Load an image from raw bytes.

    If expected_format is not None, performs color/format conversion from
    image_format (input) to expected_format (output).

    :param raw_bytes: Raw image bytes (e.g., PNG, JPEG data).
    :param image_format: The format of the image as stored.
    :param expected_format: Optional output format. If provided, converts the image.
    :return: ImageResource with loaded data and metadata.
    """
    # Load image using imageio from bytes
    raw_data = iio.imread(io.BytesIO(raw_bytes))

    # Ensure 3D array (height, width, channels)
    if raw_data.ndim == 2:
        # Grayscale image - add channel dimension
        raw_data = raw_data[:, :, np.newaxis]

    height, width = raw_data.shape[:2]

    # # Apply format conversion if needed
    # if expected_format is not None and expected_format != image_format:
    #     converted_data = convert_image_format(raw_data, image_format, expected_format)
    #     output_format = expected_format
    # else:
    #     # Just normalize to float32
    #     converted_data = normalize_image_to_f32(raw_data)
    #     output_format = image_format
    converted_data = normalize_image_to_f32(raw_data)
    output_format = image_format

    # Determine depth (number of channels)
    depth = converted_data.shape[2] if converted_data.ndim == 3 else 1

    return ImageResource(
        data=converted_data,
        width=width,
        height=height,
        depth=depth,
        image_format=output_format,
    )


#
# New Loader API
#


type GltfScene = dict[
    tuple[GeometryResource, MaterialResource],
    npt.NDArray[np.float32],  # (N, 4, 4)
]


def load_gltf(gltf_path: Path | str) -> GltfScene:
    """
    Load a glTF file and return a meshes dict with resource types.

    Returns a list of GltfScene objects. Each GltfScene is a dict mapping a pair of
    (geometry_resource, material_resource) to instance transforms (Nx4x4 arrays).
    Matrices are in row-major order, and should have [0, 0, 0, 1] in the last row to
    represent affine transforms in homogeneous coordinates.

    :param gltf_path: Path to the glTF or GLB file.
    :return: GltfScene object.
    """

    gltf_path = Path(gltf_path)

    # Load glTF file
    gltf = pygltflib.GLTF2().load(str(gltf_path))
    if gltf is None:
        raise ValueError(f"Failed to load glTF file: {gltf_path}")

    # Verify exactly one scene:
    if not gltf.scenes or len(gltf.scenes) == 0:
        raise ValueError("glTF file has no scenes")

    #
    # help_load_url()
    #

    def load_url(uri: str) -> bytes:
        if uri.startswith("file://"):
            file_path = Path(uri[len("file://") :])
            return file_path.read_bytes()
        elif uri.startswith("data:"):
            _, bs_base64 = uri.split(",", maxsplit=1)
            return base64.b64decode(bs_base64)
        else:
            file_path = gltf_path.parent / uri
            return file_path.read_bytes()

    #
    # load_buffer()
    #

    def load_buffer(buffer: pygltflib.Buffer) -> np.ndarray:
        """
        Loads a file from a path or raw bytes.

        If raw bytes as a numpy array are provided, they are returned as-is.
        """

        if (uri := buffer.uri) is not None:
            return np.frombuffer(load_url(uri), dtype=np.uint8)

        if (buffer_bytes := gltf.binary_blob()) is not None:
            return np.frombuffer(buffer_bytes, dtype=np.uint8)

        raise ValueError("Buffer has no URI and GLTF has no binary blob")

    def source_desc(source: str | bytes) -> str:
        match source:
            case str():
                return source
            case bytes():
                shasum = hashlib.sha1(source).hexdigest()
                return f"bytes(len={len(source)}, sha1={shasum[:7]})"
            case _:
                return f"unknown(type={type(source)})"

    buffers = [load_buffer(buffer) for buffer in (gltf.buffers or [])]

    #
    # load_buffer_view()
    #

    def load_buffer_view(buffer_view: pygltflib.BufferView) -> np.ndarray:
        buffer_bytes = buffers[buffer_view.buffer]
        byte_offset = buffer_view.byteOffset or 0
        byte_length = buffer_view.byteLength
        return buffer_bytes[byte_offset : byte_offset + byte_length]

    buffer_views = [
        load_buffer_view(buffer_view) for buffer_view in (gltf.bufferViews or [])
    ]

    #
    # load_accessor()
    #

    def load_accessor(accessor: pygltflib.Accessor) -> np.ndarray:
        assert accessor.bufferView is not None
        bs = buffer_views[accessor.bufferView]

        # Get the component type:
        component_dtype: np.dtype = {
            5120: np.dtype(np.int8),
            5121: np.dtype(np.uint8),
            5122: np.dtype(np.int16),
            5123: np.dtype(np.uint16),
            5125: np.dtype(np.uint32),
            5126: np.dtype(np.float32),
        }[accessor.componentType]

        # Get the component count:
        component_count: int = {
            "SCALAR": 1,
            "VEC2": 2,
            "VEC3": 3,
            "VEC4": 4,
            "MAT2": 4,
            "MAT3": 9,
            "MAT4": 16,
        }[accessor.type]

        # Get the element count:
        element_count = accessor.count

        # Get the total byte count per-element:
        element_bytes = component_dtype.itemsize * component_count

        # If byteStride is set, handle strided data:
        bv = gltf.bufferViews[accessor.bufferView]
        if bv.byteStride is not None and bv.byteStride != 0:
            bs = np.lib.stride_tricks.sliding_window_view(
                bs,
                window_shape=element_bytes,
            )[:: bv.byteStride, :].ravel()

        # Create array
        return np.frombuffer(
            np.ascontiguousarray(bs),
            dtype=(component_dtype, component_count),
            count=element_count,
        )

    accessors = [load_accessor(accessor) for accessor in (gltf.accessors or [])]

    #
    # load_image:
    #

    def load_image(image: pygltflib.Image) -> np.ndarray:
        # No color space conversion should be performed. According to the glTF spec,
        # the color profile in an image file should be ignored.
        # See: https://github.com/KhronosGroup/glTF-Sample-Models/issues/316
        #
        # Unfortunately, OpenCV does not support disabling color space conversion
        # on imdecode.
        #
        # This means OpenCV will automatically perform gamma correction on sRGB images,
        # trying to linearize an already linear image, resulting in incorrect colors.
        # > When using OpenCV's cv2.imread(), the library generally handles gamma
        # > correction automatically during the file reading process for common formats
        # > like PNG and JPEG, converting them from their typical sRGB space to a
        # > linear or near-linear space for processing. This is done by the image
        # > codecs (libjpeg, libpng, etc.) that OpenCV uses under the hood.
        #
        # To work around this, we try to guess whether the image uses the sRGB color
        # space in its metadata. If so, we reverse the gamma correction after loading.

        bs = np.frombuffer(load_image_raw_bytes(image), dtype=np.uint8)
        im = cv.imdecode(bs, cv.IMREAD_COLOR_RGB | cv.IMREAD_ANYDEPTH)
        im = validate_rgb_image(im)
        im = normalize_image(im)
        # im = convert_linear_to_srgb(im) if image_is_srgb(image) else im
        im = convert_srgb_to_linear(im) if image_is_srgb(image) else im
        return im

    def image_is_srgb(image: pygltflib.Image) -> bool:
        """
        Guess whether an image will be loaded by OpenCV in sRGB color space.
        """
        if image.mimeType == "image/jpeg":
            return True
        if image.mimeType == "image/png":
            return False
        if image.uri and image.uri.endswith(".jpg"):
            return True
        if image.uri and image.uri.endswith(".jpeg"):
            return True
        if image.uri and image.uri.endswith(".png"):
            return False
        return False

    def load_image_raw_bytes(image: pygltflib.Image) -> bytes:
        return (
            load_url(image.uri)
            if image.uri is not None
            else buffer_views[expect(image.bufferView)].tobytes()
        )

    def normalize_image(im: np.ndarray) -> np.ndarray:
        match im.dtype.type:
            case np.uint8:
                return im.astype(np.float32) / 255.0
            case np.uint16:
                return im.astype(np.float32) / 65535.0
            case np.float32:
                return im
            case _:
                raise LogicError(f"Unsupported image dtype: {im.dtype}")

    def validate_rgb_image(im: npt.ArrayLike) -> np.ndarray:
        im = np.asarray(im)
        if im.shape[2] != 3:
            raise ValueError(f"Expected RGB image with 3 channels, got: {im.shape[2]}")
        return im

    images = [load_image(image) for image in (gltf.images or [])]

    #
    # load_texture:
    #

    def load_texture(texture: pygltflib.Texture) -> np.ndarray:
        if texture.source is None:
            raise ValueError("Texture has no source")
        # We ignore sampler for now
        return images[texture.source]

    textures = [load_texture(texture) for texture in (gltf.textures or [])]

    #
    # load_material:
    #

    def load_material(material: pygltflib.Material) -> MaterialResource:
        if material.pbrMetallicRoughness is None:
            raise ValueError("Only PBR metallic-roughness materials are supported")

        pbr = material.pbrMetallicRoughness

        base_color_factor = (
            (
                pbr.baseColorFactor[0],
                pbr.baseColorFactor[1],
                pbr.baseColorFactor[2],
            )
            if pbr.baseColorFactor is not None
            else (1.0, 1.0, 1.0)
        )
        base_color_texture = (
            textures[pbr.baseColorTexture.index] if pbr.baseColorTexture else None
        )

        metalness_roughness_texture = (
            textures[pbr.metallicRoughnessTexture.index]
            if pbr.metallicRoughnessTexture
            else None
        )

        metalness_factor = pbr.metallicFactor if pbr.metallicFactor is not None else 1.0
        metalness_texture = (
            metalness_roughness_texture[:, :, 2:3]
            if metalness_roughness_texture is not None
            else None
        )

        roughness_factor = (
            pbr.roughnessFactor if pbr.roughnessFactor is not None else 1.0
        )
        roughness_texture = (
            metalness_roughness_texture[:, :, 1:2]
            if metalness_roughness_texture is not None
            else None
        )

        normal_scale = np.float32(
            (material.normalTexture.scale or 1.0) if material.normalTexture else 1.0
        )
        normal_texture = (
            normal_scale * textures[expect(material.normalTexture.index)]
            if material.normalTexture
            else None
        )

        return MaterialResource(
            color_map=base_color_texture,
            color_factor=base_color_factor,
            normal_map=normal_texture,
            metalness_map=metalness_texture,
            metalness_factor=metalness_factor,
            roughness_map=roughness_texture,
            roughness_factor=roughness_factor,
        )

    materials = [load_material(material) for material in (gltf.materials or [])]

    #
    # load_mesh():
    #

    def load_mesh(
        mesh: pygltflib.Mesh,
    ) -> list[tuple[GeometryResource, MaterialResource]]:
        resources: list[tuple[GeometryResource, MaterialResource]] = []

        for primitive in mesh.primitives:
            if primitive.material is None:
                raise ValueError("Primitive has no material")

            geometry_resource = load_primitive(primitive)
            material_resource = materials[primitive.material]

            resources.append((geometry_resource, material_resource))

        return resources

    def load_primitive(primitive: pygltflib.Primitive) -> GeometryResource:
        # Only support TRIANGULAR mode (4 = TRIANGLES)
        mode = primitive.mode if primitive.mode is not None else 4
        if mode != 4:
            raise ValueError(f"Only TRIANGLES mode is supported, got: {mode=}")

        attributes = primitive.attributes

        positions = accessors[expect(attributes.POSITION)]
        vertex_count = len(positions)

        normals = (
            accessors[attributes.NORMAL].astype(np.float32)  #
            if attributes.NORMAL is not None
            else np.full((vertex_count, 3), (0.0, 0.0, 1.0), dtype=np.float32)
        )
        texcoords = (
            accessors[attributes.TEXCOORD_0].astype(np.float32)
            if attributes.TEXCOORD_0 is not None
            else np.zeros((vertex_count, 2), dtype=np.float32)
        )
        indices = (
            accessors[primitive.indices].astype(np.uint32)
            if primitive.indices is not None
            else np.arange(vertex_count, dtype=np.uint32)
        )

        return GeometryResource(
            v_p_array=positions,
            v_n_array=normals,
            v_t_array=texcoords,
            t_indices=indices.reshape((-1, 3)),
        )

    #
    # load_scene:
    #

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

    def load_node(
        node: pygltflib.Node,
        parent_transform: np.ndarray,
        acc: defaultdict[
            tuple[GeometryResource, MaterialResource],
            list[npt.NDArray[np.float32]],
        ],
    ) -> None:
        local_transform = compute_node_transform(node)
        world_transform = parent_transform @ local_transform

        if node.mesh is not None:
            mesh = gltf.meshes[expect(node.mesh)]
            resources = load_mesh(mesh)
            for geometry_resource, material_resource in resources:
                acc[(geometry_resource, material_resource)].append(world_transform)

        for child in node.children or []:
            load_node(
                node=gltf.nodes[child],
                parent_transform=world_transform,
                acc=acc,
            )

    def compute_node_transform(node: pygltflib.Node) -> np.ndarray:
        if node.matrix is not None:
            return np.array(node.matrix, dtype=np.float32).reshape((4, 4)).T

        s = np.eye(3)
        if node.scale is not None:
            s *= np.array(node.scale, dtype=np.float32).reshape((1, 3))

        r = np.eye(3)
        if node.rotation is not None:
            x = [node.rotation[3], node.rotation[0], node.rotation[1], node.rotation[2]]
            q = quaternion.as_quat_array(x)
            r = quaternion.as_rotation_matrix(q).astype(np.float32)

        t = np.zeros((3,), dtype=np.float32)
        if node.translation is not None:
            t = np.array(node.translation, dtype=np.float32)

        res = np.eye(4, dtype=np.float32)
        res[:3, :3] = r @ s
        res[:3, 3] = t
        return res

    def load_scene(scene: pygltflib.Scene) -> GltfScene:
        acc = defaultdict(list)
        for root_node_idx in scene.nodes or []:
            load_node(
                node=gltf.nodes[root_node_idx],
                parent_transform=GLTF_TO_Z_UP_MATRIX,
                acc=acc,
            )
        return {k: np.stack(v, axis=0) for k, v in acc.items()}

    return load_scene(gltf.scenes[0])


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
    image_format: ImageFormat = "rgba32float"
    readme_text: str | None = None
    license_text: str | None = None
    as_glyph_cache: dict[CookedAtlasGlyphCacheKey, "CookedAtlasGlyphInfo"] | None = None

    @staticmethod
    def load(
        *,
        path: Path,
        image_format: ImageFormat = "rgba32float",
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

        # Load atlas.png with format conversion
        atlas_path = path / "atlas.png"
        resource = load_image(
            atlas_path,
            image_format=index.image_format,
            expected_format=image_format,
        )
        atlas_data = resource.data

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
            image_format=image_format,
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
                image_format=self.image_format,
            )
            f.write(orjson.dumps(index.model_dump()))

        # Save atlas.png using imageio
        atlas_uint8 = (np.clip(self.atlas_data, 0.0, 1.0) * 255).astype(np.uint8)
        atlas_uint8 = atlas_uint8.squeeze()
        iio.imwrite(path / "atlas.png", atlas_uint8)

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
    image_format: ImageFormat


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
