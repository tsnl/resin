"""3D Gaussian Splatting helpers, DSL nodes, and reference implementations."""

from resin.lib.gaussians.blend import GaussianBlendNode, gaussian_blend
from resin.lib.gaussians.camera import FlyCamera
from resin.lib.gaussians.gnomen import GnomenCloud, make_gnomen_cloud
from resin.lib.gaussians.gpu_session import GpuForwardSession
from resin.lib.gaussians.image_io import rgb_f32_to_rgb888_bytes
from resin.lib.gaussians.linalg import (
    look_at_view_proj,
    project_points,
    quat_to_rotmat,
    scale_rot_to_cov3d,
)
from resin.lib.gaussians.preprocess import preprocess_gaussians
from resin.lib.gaussians.reference import (
    blend_gaussians_cpu,
    pad_preprocess_result,
    preprocess_gaussians_cpu,
    render_gnomen_cpu,
    sort_by_depth_cpu,
)
from resin.lib.gaussians.tiling import (
    TiledGaussianBlendNode,
    TiledLayout,
    blend_gaussians_tiled_cpu,
    build_tiled_layout,
    gaussian_blend_tiled,
)

__all__ = [
    "FlyCamera",
    "GaussianBlendNode",
    "GnomenCloud",
    "GpuForwardSession",
    "TiledGaussianBlendNode",
    "TiledLayout",
    "blend_gaussians_cpu",
    "blend_gaussians_tiled_cpu",
    "build_tiled_layout",
    "gaussian_blend",
    "gaussian_blend_tiled",
    "look_at_view_proj",
    "make_gnomen_cloud",
    "pad_preprocess_result",
    "preprocess_gaussians",
    "preprocess_gaussians_cpu",
    "project_points",
    "quat_to_rotmat",
    "render_gnomen_cpu",
    "rgb_f32_to_rgb888_bytes",
    "scale_rot_to_cov3d",
    "sort_by_depth_cpu",
]
