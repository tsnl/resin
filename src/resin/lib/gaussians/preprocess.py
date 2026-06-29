"""Host-side preprocess: 3D gaussians → 2D means, depths, conics, colors, opacities."""

from resin.lib.gaussians.gnomen import GnomenCloud
from resin.lib.gaussians.reference import PreprocessResult, preprocess_gaussians_cpu


def preprocess_gaussians(
    cloud: GnomenCloud,
    *,
    width: int,
    height: int,
    view: tuple[tuple[float, ...], ...] | None = None,
    proj: tuple[tuple[float, ...], ...] | None = None,
) -> PreprocessResult:
    """Project and cull gaussians (no SH); returns 2D attributes for blending."""
    return preprocess_gaussians_cpu(
        cloud, width=width, height=height, view=view, proj=proj
    )
