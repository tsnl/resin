"""
Mitsuba interoperability module for converting glTF scenes to Mitsuba XML format.

This module provides functionality to export glTF scenes loaded via load_gltf()
to Mitsuba's XML scene description format, faithfully translating the PBR
metallic-roughness material model.
"""

__all__ = [
    "gltf_to_mitsuba_xml",
    "write_mitsuba_scene",
]

import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import PIL.Image

from .resources import (
    GltfScene,
    GeometryResource,
    MaterialResource,
    load_gltf,
)


@dataclass
class MitsubaSceneConfig:
    """Configuration for Mitsuba scene export."""

    output_dir: Path
    """Directory where the scene XML and assets will be written."""

    scene_name: str = "scene"
    """Name for the scene file (without extension)."""

    integrator: Literal["path", "volpath", "direct", "aov"] = "path"
    """Mitsuba integrator type."""

    max_depth: int = 8
    """Maximum path depth for path tracing."""

    samples_per_pixel: int = 64
    """Number of samples per pixel."""


def _indent_xml(elem: ET.Element, level: int = 0) -> None:
    """Add indentation to XML elements for pretty printing."""
    indent = "\n" + "    " * level
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = indent + "    "
        if not elem.tail or not elem.tail.strip():
            elem.tail = indent
        for child in elem:
            _indent_xml(child, level + 1)
        last_child = elem[-1]
        if not last_child.tail or not last_child.tail.strip():
            last_child.tail = indent
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = indent


def _save_texture(
    data: npt.NDArray[np.float32],
    output_path: Path,
    is_linear: bool = True,
) -> None:
    """Save a texture array to a file.

    Args:
        data: Texture data as (H, W, C) float32 array in [0, 1].
        output_path: Path to save the texture.
        is_linear: Whether the data is in linear color space.
    """
    # Clamp and convert to uint8
    data_clamped = np.clip(data, 0.0, 1.0)

    # For linear data that will be used as color, we should save as-is
    # Mitsuba will handle the color space conversion
    data_uint8 = (data_clamped * 255.0).astype(np.uint8)

    # Handle single channel textures
    if data_uint8.ndim == 2:
        img = PIL.Image.fromarray(data_uint8, mode="L")
    elif data_uint8.shape[2] == 1:
        img = PIL.Image.fromarray(data_uint8[:, :, 0], mode="L")
    elif data_uint8.shape[2] == 3:
        img = PIL.Image.fromarray(data_uint8, mode="RGB")
    elif data_uint8.shape[2] == 4:
        img = PIL.Image.fromarray(data_uint8, mode="RGBA")
    else:
        raise ValueError(f"Unsupported texture shape: {data_uint8.shape}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


def _save_mesh_ply(
    geometry: GeometryResource,
    output_path: Path,
) -> None:
    """Save geometry as a PLY file for Mitsuba.

    Args:
        geometry: The geometry resource to save.
        output_path: Path to save the PLY file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    v_p = geometry.v_p_array
    v_n = geometry.v_n_array
    v_t = geometry.v_t_array
    t_indices = geometry.t_indices

    num_vertices = v_p.shape[0]
    num_faces = t_indices.shape[0]

    with open(output_path, "wb") as f:
        # Write PLY header
        header = f"""ply
format binary_little_endian 1.0
element vertex {num_vertices}
property float x
property float y
property float z
property float nx
property float ny
property float nz
property float u
property float v
element face {num_faces}
property list uchar int vertex_indices
end_header
"""
        f.write(header.encode("ascii"))

        # Write vertex data
        for i in range(num_vertices):
            # Position
            f.write(np.float32(v_p[i, 0]).tobytes())
            f.write(np.float32(v_p[i, 1]).tobytes())
            f.write(np.float32(v_p[i, 2]).tobytes())
            # Normal
            f.write(np.float32(v_n[i, 0]).tobytes())
            f.write(np.float32(v_n[i, 1]).tobytes())
            f.write(np.float32(v_n[i, 2]).tobytes())
            # UV
            f.write(np.float32(v_t[i, 0]).tobytes())
            f.write(np.float32(v_t[i, 1]).tobytes())

        # Write face data
        for i in range(num_faces):
            f.write(np.uint8(3).tobytes())  # Number of vertices per face
            f.write(np.int32(t_indices[i, 0]).tobytes())
            f.write(np.int32(t_indices[i, 1]).tobytes())
            f.write(np.int32(t_indices[i, 2]).tobytes())


def _create_texture_element(
    parent: ET.Element,
    name: str,
    texture_path: str,
    is_color: bool = False,
) -> ET.Element:
    """Create a Mitsuba texture element.

    Args:
        parent: Parent XML element.
        name: Name of the texture parameter.
        texture_path: Relative path to the texture file.
        is_color: Whether this is a color texture (needs sRGB conversion).

    Returns:
        The created texture element.
    """
    tex = ET.SubElement(parent, "texture", type="bitmap", name=name)
    ET.SubElement(tex, "string", name="filename", value=texture_path)
    if is_color:
        # Color textures are typically stored in sRGB
        ET.SubElement(tex, "boolean", name="raw", value="false")
    else:
        # Non-color data (roughness, metalness, normal) should be raw
        ET.SubElement(tex, "boolean", name="raw", value="true")
    return tex


def _create_principled_bsdf(
    parent: ET.Element,
    material: MaterialResource,
    material_id: int,
    textures_dir: str,
    output_dir: Path,
) -> ET.Element:
    """Create a Mitsuba principled BSDF from a glTF PBR material.

    The glTF PBR metallic-roughness model maps to Mitsuba's principled BSDF as follows:
    - baseColorFactor/Texture -> base_color
    - metallicFactor/Texture -> metallic
    - roughnessFactor/Texture -> roughness
    - normalTexture -> (via normalmap BSDF wrapper)

    Args:
        parent: Parent XML element.
        material: The material resource.
        material_id: Unique ID for this material (for naming textures).
        textures_dir: Relative path to textures directory.
        output_dir: Absolute path to output directory for saving textures.

    Returns:
        The created BSDF element.
    """
    # If we have a normal map, wrap with normalmap BSDF
    bsdf: ET.Element | None = None
    if material.normal_map is not None:
        bsdf = ET.SubElement(parent, "bsdf", type="normalmap")
        normal_tex_name = f"material_{material_id}_normal.png"
        normal_tex_path = f"{textures_dir}/{normal_tex_name}"

        _save_texture(
            material.normal_map,
            output_dir / "textures" / normal_tex_name,
            is_linear=True,
        )
        _create_texture_element(bsdf, "normalmap", normal_tex_path, is_color=False)

        # Create nested principled BSDF
        inner_bsdf = ET.SubElement(bsdf, "bsdf", type="principled")
    else:
        inner_bsdf = ET.SubElement(parent, "bsdf", type="principled")

    # Base color
    if material.color_map is not None:
        color_tex_name = f"material_{material_id}_basecolor.png"
        color_tex_path = f"{textures_dir}/{color_tex_name}"
        _save_texture(
            material.color_map,
            output_dir / "textures" / color_tex_name,
            is_linear=False,  # Color textures are typically sRGB
        )
        _create_texture_element(inner_bsdf, "base_color", color_tex_path, is_color=True)
    else:
        r, g, b = material.color_factor
        ET.SubElement(inner_bsdf, "rgb", name="base_color", value=f"{r}, {g}, {b}")

    # Metallic
    if material.metalness_map is not None:
        metal_tex_name = f"material_{material_id}_metallic.png"
        metal_tex_path = f"{textures_dir}/{metal_tex_name}"
        _save_texture(
            material.metalness_map,
            output_dir / "textures" / metal_tex_name,
            is_linear=True,
        )
        _create_texture_element(inner_bsdf, "metallic", metal_tex_path, is_color=False)
    else:
        ET.SubElement(
            inner_bsdf, "float", name="metallic", value=str(material.metalness_factor)
        )

    # Roughness
    if material.roughness_map is not None:
        rough_tex_name = f"material_{material_id}_roughness.png"
        rough_tex_path = f"{textures_dir}/{rough_tex_name}"
        _save_texture(
            material.roughness_map,
            output_dir / "textures" / rough_tex_name,
            is_linear=True,
        )
        _create_texture_element(inner_bsdf, "roughness", rough_tex_path, is_color=False)
    else:
        ET.SubElement(
            inner_bsdf, "float", name="roughness", value=str(material.roughness_factor)
        )

    # Set specular to 0.5 (default for dielectric F0 of 0.04)
    ET.SubElement(inner_bsdf, "float", name="specular", value="0.5")

    return bsdf if bsdf is not None else inner_bsdf


def _create_area_emitter(
    parent: ET.Element,
    material: MaterialResource,
    material_id: int,
    textures_dir: str,
    output_dir: Path,
) -> ET.Element | None:
    """Create a Mitsuba area emitter if the material has emissive properties.

    Args:
        parent: Parent XML element (the shape).
        material: The material resource.
        material_id: Unique ID for this material (for naming textures).
        textures_dir: Relative path to textures directory.
        output_dir: Absolute path to output directory for saving textures.

    Returns:
        The created emitter element, or None if not emissive.
    """
    # Check if material is emissive
    has_emissive_factor = any(c > 0.0 for c in material.emissive_factor)

    if material.emissive_map is None and not has_emissive_factor:
        return None

    emitter = ET.SubElement(parent, "emitter", type="area")

    if material.emissive_map is not None:
        # Save emissive texture as PNG (normalized)
        # The emissive_factor will be applied as a scale multiplier
        emissive_tex_name = f"material_{material_id}_emissive.png"
        emissive_tex_path = f"{textures_dir}/{emissive_tex_name}"

        _save_texture(
            material.emissive_map,
            output_dir / "textures" / emissive_tex_name,
            is_linear=False,  # Emissive textures are typically sRGB
        )

        # Create texture reference for radiance with scale
        # Use a spectrum with texture for the radiance
        tex = ET.SubElement(emitter, "texture", type="bitmap", name="radiance")
        ET.SubElement(tex, "string", name="filename", value=emissive_tex_path)
        ET.SubElement(tex, "boolean", name="raw", value="false")

        # Apply emissive factor as scale if it's not (1, 1, 1)
        r, g, b = material.emissive_factor
        if r != 1.0 or g != 1.0 or b != 1.0:
            # For non-uniform scale, we need to bake it into the texture
            # For now, use the max component as uniform scale
            scale = max(r, g, b)
            if scale > 0:
                ET.SubElement(emitter, "float", name="scale", value=str(scale))
    else:
        # Just use emissive factor as constant radiance
        r, g, b = material.emissive_factor
        ET.SubElement(emitter, "rgb", name="radiance", value=f"{r}, {g}, {b}")

    return emitter


def _matrix_to_mitsuba_string(matrix: npt.NDArray[np.float32]) -> str:
    """Convert a 4x4 row-major matrix to Mitsuba's matrix string format.

    Mitsuba expects matrices in row-major order as space-separated values.
    """
    # Flatten row-major and convert to string
    values = matrix.flatten()
    return " ".join(f"{v:.8g}" for v in values)


def gltf_to_mitsuba_xml(
    gltf_scene: GltfScene,
    config: MitsubaSceneConfig,
    camera_transform: npt.NDArray[np.float32] | None = None,
    camera_fov_y_deg: float = 45.0,
    film_width: int = 1024,
    film_height: int = 1024,
    environment_map_path: Path | str | None = None,
) -> ET.Element:
    """Convert a glTF scene to Mitsuba XML format.

    Args:
        gltf_scene: The loaded glTF scene from load_gltf().
        config: Configuration for the Mitsuba scene export.
        camera_transform: 4x4 camera transform matrix (row-major). If None, uses identity.
        camera_fov_y_deg: Vertical field of view in degrees.
        film_width: Output image width in pixels.
        film_height: Output image height in pixels.
        environment_map_path: Optional path to an HDR environment map.

    Returns:
        The root XML element of the Mitsuba scene.
    """
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create root scene element
    root = ET.Element("scene", version="2.0.0")

    # Add integrator
    integrator = ET.SubElement(root, "integrator", type=config.integrator)
    ET.SubElement(integrator, "integer", name="max_depth", value=str(config.max_depth))

    # Add sensor (camera)
    sensor = ET.SubElement(root, "sensor", type="perspective")
    ET.SubElement(sensor, "float", name="fov", value=str(camera_fov_y_deg))
    ET.SubElement(sensor, "string", name="fov_axis", value="y")

    # Camera transform
    if camera_transform is not None:
        # Camera transform is given in Resin's Z-up, Y-forward coordinate system.
        # We need to convert it to Mitsuba's Y-up system with correct view direction.
        #
        # Resin camera convention: looks in +Y direction (local frame)
        # Mitsuba camera convention: looks in +Z direction (local frame)
        #
        # Steps:
        # 1. Convert coordinate system from Z-up to Y-up
        # 2. Rotate camera 180° around Y to flip view direction (camera was looking +Z,
        #    but we need it to look -Z towards the origin)

        z_up_to_y_up = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        # camera_transform is camera-to-world in Z-up coordinates
        # Convert to Y-up coordinates
        camera_transform_y_up = (
            z_up_to_y_up @ camera_transform @ np.linalg.inv(z_up_to_y_up)
        )

        # Rotate camera 180° around Y-axis to flip view direction
        # This makes the camera look towards -Z instead of +Z
        flip_view = np.array(
            [
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        camera_transform_mitsuba = (camera_transform_y_up @ flip_view).astype(
            np.float32
        )

        # Mitsuba's to_world expects camera-to-world matrix (not inverted)
        transform = ET.SubElement(sensor, "transform", name="to_world")
        ET.SubElement(
            transform,
            "matrix",
            value=_matrix_to_mitsuba_string(camera_transform_mitsuba),
        )

    # Film
    film = ET.SubElement(sensor, "film", type="hdrfilm")
    ET.SubElement(film, "integer", name="width", value=str(film_width))
    ET.SubElement(film, "integer", name="height", value=str(film_height))
    ET.SubElement(film, "string", name="pixel_format", value="rgb")
    ET.SubElement(film, "string", name="component_format", value="float16")

    # Sampler
    sampler = ET.SubElement(sensor, "sampler", type="independent")
    ET.SubElement(
        sampler, "integer", name="sample_count", value=str(config.samples_per_pixel)
    )

    # Add environment map if provided
    if environment_map_path is not None:
        env_path = Path(environment_map_path)
        # Copy environment map to output directory
        env_dest = output_dir / "textures" / env_path.name
        env_dest.parent.mkdir(parents=True, exist_ok=True)
        if env_path != env_dest:
            shutil.copy2(env_path, env_dest)

        emitter = ET.SubElement(root, "emitter", type="envmap")
        ET.SubElement(
            emitter, "string", name="filename", value=f"textures/{env_path.name}"
        )
        # Scale for HDR environments
        ET.SubElement(emitter, "float", name="scale", value="1.0")

        # Apply coordinate system transform to environment map
        # Resin uses Z-up, Y-forward; Mitsuba uses Y-up with different longitude origin.
        #
        # Resin's envmap sampling (draw_3d.wgsl):
        #   theta = atan2(dir.x, dir.y)  // longitude from +Y axis
        #   u = (theta + pi) / (2*pi)
        #   So u=0.5 corresponds to +Y direction
        #
        # Mitsuba's default equirectangular sampling (Y-up):
        #   theta = atan2(x, z)  // longitude from +Z axis
        #   u = (theta + pi) / (2*pi)
        #   So u=0.5 corresponds to +Z direction
        #
        # When we convert Resin's +Y (forward) to Mitsuba's Y-up coords:
        #   (0, 1, 0)_resin -> (0, 0, -1)_mitsuba (Z-up to Y-up: x,y,z -> x,z,-y)
        #
        # This direction samples at u=1.0 in Mitsuba but should sample at u=0.5.
        # We need to rotate the environment 180° around Y-axis to align the longitude.
        #
        # to_world rotates the environment, so sampling direction d becomes
        # T^(-1) @ d in emitter local coords. R_y(180)^(-1) = R_y(180).
        env_transform = np.array(
            [
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        env_transform_elem = ET.SubElement(emitter, "transform", name="to_world")
        ET.SubElement(
            env_transform_elem,
            "matrix",
            value=_matrix_to_mitsuba_string(env_transform),
        )

    # Process meshes and materials
    mesh_dir = output_dir / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    material_id = 0
    mesh_id = 0

    for (geometry, material), transforms in gltf_scene.items():
        # Save mesh geometry
        mesh_filename = f"mesh_{mesh_id}.ply"
        mesh_path = mesh_dir / mesh_filename
        _save_mesh_ply(geometry, mesh_path)

        # Create shape for each instance
        for instance_idx in range(transforms.shape[0]):
            shape = ET.SubElement(root, "shape", type="ply")
            ET.SubElement(
                shape, "string", name="filename", value=f"meshes/{mesh_filename}"
            )

            # Add instance transform
            # The geometry is loaded with apply_z_up_conversion=False, so vertices
            # are already in glTF's native Y-up coordinate system (same as Mitsuba).
            # We just need to apply the instance transform directly.
            instance_transform = transforms[instance_idx]
            transform_elem = ET.SubElement(shape, "transform", name="to_world")
            ET.SubElement(
                transform_elem,
                "matrix",
                value=_matrix_to_mitsuba_string(instance_transform),
            )

            # Add material (BSDF)
            _create_principled_bsdf(
                shape,
                material,
                material_id,
                textures_dir="textures",
                output_dir=output_dir,
            )

            # Add area emitter if material is emissive
            _create_area_emitter(
                shape,
                material,
                material_id,
                textures_dir="textures",
                output_dir=output_dir,
            )

        material_id += 1
        mesh_id += 1

    # Pretty print
    _indent_xml(root)

    return root


def write_mitsuba_scene(
    gltf_path: Path | str,
    output_dir: Path | str,
    camera_transform: npt.NDArray[np.float32] | None = None,
    camera_fov_y_deg: float = 45.0,
    film_width: int = 1024,
    film_height: int = 1024,
    environment_map_path: Path | str | None = None,
    integrator: Literal["path", "volpath", "direct", "aov"] = "path",
    max_depth: int = 8,
    samples_per_pixel: int = 64,
) -> Path:
    """Load a glTF file and write a complete Mitsuba scene.

    Args:
        gltf_path: Path to the glTF file.
        output_dir: Directory where the scene will be written.
        camera_transform: 4x4 camera transform matrix (row-major).
        camera_fov_y_deg: Vertical field of view in degrees.
        film_width: Output image width in pixels.
        film_height: Output image height in pixels.
        environment_map_path: Optional path to an HDR environment map.
        integrator: Mitsuba integrator type.
        max_depth: Maximum path depth.
        samples_per_pixel: Number of samples per pixel.

    Returns:
        Path to the written scene XML file.
    """
    gltf_path = Path(gltf_path)
    output_dir = Path(output_dir)

    # Load glTF without Z-up conversion (Mitsuba uses standard Y-up coordinates)
    gltf_scene = load_gltf(gltf_path, apply_z_up_conversion=False)

    # Create config
    config = MitsubaSceneConfig(
        output_dir=output_dir,
        scene_name=gltf_path.stem,
        integrator=integrator,
        max_depth=max_depth,
        samples_per_pixel=samples_per_pixel,
    )

    # Convert to Mitsuba XML
    root = gltf_to_mitsuba_xml(
        gltf_scene=gltf_scene,
        config=config,
        camera_transform=camera_transform,
        camera_fov_y_deg=camera_fov_y_deg,
        film_width=film_width,
        film_height=film_height,
        environment_map_path=environment_map_path,
    )

    # Write XML file
    scene_path = output_dir / f"{config.scene_name}.xml"
    tree = ET.ElementTree(root)
    tree.write(scene_path, encoding="utf-8", xml_declaration=True)

    return scene_path
