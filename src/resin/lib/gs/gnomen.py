"""Procedural test gaussian point clouds for 3DGS development."""

from dataclasses import dataclass


@dataclass(frozen=True)
class GnomenCloud:
    """Small fixed-layout gaussian cloud for golden tests."""

    means: tuple[tuple[float, float, float], ...]
    scales: tuple[tuple[float, float, float], ...]
    quats: tuple[tuple[float, float, float, float], ...]
    colors: tuple[tuple[float, float, float], ...]
    opacities: tuple[float, ...]

    @property
    def count(self) -> int:
        return len(self.means)


def make_gnomen_cloud() -> GnomenCloud:
    """Three axis-aligned gaussians on -Z in front of a +Z-looking camera.

    Layout (camera at origin, looks down -Z):
      red   at (0, 0, -2)
      green at (0.4, 0, -3)
      blue  at (-0.4, 0, -4)

    Each gaussian uses unit quaternion (w=1) and moderate isotropic scale so all
    three project near the image center from the default gnomen camera.
    """
    return GnomenCloud(
        means=(
            (0.0, 0.0, -2.0),
            (0.4, 0.0, -3.0),
            (-0.4, 0.0, -4.0),
        ),
        scales=(
            (0.15, 0.15, 0.15),
            (0.12, 0.12, 0.12),
            (0.12, 0.12, 0.12),
        ),
        quats=(
            (1.0, 0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0, 0.0),
        ),
        colors=(
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        opacities=(0.9, 0.9, 0.9),
    )