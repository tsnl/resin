"""3D Gaussian Splatting helpers and reference implementations."""

from resin.lib.gs.gnomen import GnomenCloud, make_gnomen_cloud
from resin.lib.gs.linalg import (
    look_at_view_proj,
    project_points,
    quat_to_rotmat,
    scale_rot_to_cov3d,
)
from resin.lib.gs.reference import (
    blend_gaussians_cpu,
    preprocess_gaussians_cpu,
    render_gnomen_cpu,
    sort_by_depth_cpu,
)
from resin.lib.gs.render import argsort_depths, gaussian_blend

__all__ = [
    "GnomenCloud",
    "argsort_depths",
    "blend_gaussians_cpu",
    "gaussian_blend",
    "look_at_view_proj",
    "make_gnomen_cloud",
    "preprocess_gaussians_cpu",
    "project_points",
    "quat_to_rotmat",
    "render_gnomen_cpu",
    "scale_rot_to_cov3d",
    "sort_by_depth_cpu",
]