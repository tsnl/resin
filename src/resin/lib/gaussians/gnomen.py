"""Procedural test gaussian point clouds for 3DGS development."""

from dataclasses import dataclass

# Samples per axis including the shared hub (so step count is _SAMPLES_PER_AXIS - 1).
# Spacing ≈ 1.2 / 99 ≈ 0.012 with radius ~0.04 → heavy overlap for continuous lines.
_SAMPLES_PER_AXIS = 100
_AXIS_LENGTH = 1.2
_ORIGIN = (0.0, 0.0, -2.5)
# Slightly fat isotropic core; tips use the same scale so the stroke is even.
_LINE_RADIUS = 0.045
_IDENTITY_QUAT = (1.0, 0.0, 0.0, 0.0)
_OPACITY = 0.92

# Elongate along each axis so blobs blend into a thick continuous stroke.
_SCALE_X = (_LINE_RADIUS * 1.8, _LINE_RADIUS, _LINE_RADIUS)
_SCALE_Y = (_LINE_RADIUS, _LINE_RADIUS * 1.8, _LINE_RADIUS)
_SCALE_Z = (_LINE_RADIUS, _LINE_RADIUS, _LINE_RADIUS * 1.8)
_SCALE_HUB = (_LINE_RADIUS * 1.4, _LINE_RADIUS * 1.4, _LINE_RADIUS * 1.4)


@dataclass(frozen=True)
class GnomenCloud:
    """Fixed-layout gaussian cloud for golden tests and the interactive viewer."""

    means: tuple[tuple[float, float, float], ...]
    scales: tuple[tuple[float, float, float], ...]
    quats: tuple[tuple[float, float, float, float], ...]
    colors: tuple[tuple[float, float, float], ...]
    opacities: tuple[float, ...]

    @property
    def count(self) -> int:
        return len(self.means)


def make_gnomen_cloud() -> GnomenCloud:
    """RGB axis gizmo: densely sampled gaussians forming thick continuous lines.

    Right-handed triad (thumb +X, index +Y, middle +Z toward the viewer when
    looking down -Z from the default camera):

      hub at (0, 0, -2.5)
      red   +X  (screen right after NDC-X correction)
      green +Y  (screen up)
      blue  +Z  (toward the camera / out of the screen)

    Count = 1 hub + 3 * (samples_per_axis - 1) = 1 + 3 * 99 = 298.
    """
    ox, oy, oz = _ORIGIN
    means: list[tuple[float, float, float]] = []
    colors: list[tuple[float, float, float]] = []
    scales: list[tuple[float, float, float]] = []

    def add_dot(
        p: tuple[float, float, float],
        color: tuple[float, float, float],
        scale: tuple[float, float, float],
    ) -> None:
        means.append(p)
        colors.append(color)
        scales.append(scale)

    # Shared hub.
    add_dot((ox, oy, oz), (1.0, 1.0, 1.0), _SCALE_HUB)

    # Dense samples along each axis (skip t=0; hub already placed).
    n_seg = _SAMPLES_PER_AXIS - 1
    for i in range(1, _SAMPLES_PER_AXIS):
        t = _AXIS_LENGTH * (i / n_seg)
        add_dot((ox + t, oy, oz), (1.0, 0.15, 0.15), _SCALE_X)
        add_dot((ox, oy + t, oz), (0.15, 1.0, 0.15), _SCALE_Y)
        add_dot((ox, oy, oz + t), (0.2, 0.4, 1.0), _SCALE_Z)

    n = len(means)
    expected = 1 + 3 * n_seg
    assert n == expected, f"expected {expected} gaussians, got {n}"

    return GnomenCloud(
        means=tuple(means),
        scales=tuple(scales),
        quats=tuple(_IDENTITY_QUAT for _ in range(n)),
        colors=tuple(colors),
        opacities=tuple(_OPACITY for _ in range(n)),
    )
