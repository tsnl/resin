"""3D Gaussian Splatting helpers, DSL nodes, and reference implementations."""

from resin.lib.gaussians.blend import GaussianBlendNode, gaussian_blend
from resin.lib.gaussians.gnomen import GnomenCloud, make_gnomen_cloud
from resin.lib.gaussians.linalg import (
    look_at_view_proj,
    project_points,
    quat_to_rotmat,
    scale_rot_to_cov3d,
)
from resin.lib.gaussians.preprocess import preprocess_gaussians
from resin.lib.gaussians.reference import (
    blend_gaussians_cpu,
    preprocess_gaussians_cpu,
    render_gnomen_cpu,
    sort_by_depth_cpu,
)

__all__ = [
    "GaussianBlendNode",
    "GnomenCloud",
    "blend_gaussians_cpu",
    "gaussian_blend",
    "look_at_view_proj",
    "make_gnomen_cloud",
    "preprocess_gaussians",
    "preprocess_gaussians_cpu",
    "project_points",
    "quat_to_rotmat",
    "render_gnomen_cpu",
    "scale_rot_to_cov3d",
    "sort_by_depth_cpu",
]
