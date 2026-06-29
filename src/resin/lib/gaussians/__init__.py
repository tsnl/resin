"""3D Gaussian Splatting helpers, DSL nodes, and reference implementations."""

from resin.lib.gaussians.gnomen import GnomenCloud, make_gnomen_cloud
from resin.lib.gaussians.linalg import (
    look_at_view_proj,
    project_points,
    quat_to_rotmat,
    scale_rot_to_cov3d,
)

__all__ = [
    "GnomenCloud",
    "look_at_view_proj",
    "make_gnomen_cloud",
    "project_points",
    "quat_to_rotmat",
    "scale_rot_to_cov3d",
]
