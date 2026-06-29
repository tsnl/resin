"""CPU reference implementations for 3DGS preprocess and forward blend."""

import math
from typing import TypedDict

from resin.lib.gaussians.gnomen import GnomenCloud, make_gnomen_cloud
from resin.lib.gaussians.linalg import (
    look_at_view_proj,
    mat4_mul_vec4,
    project_points,
    scale_rot_to_cov3d,
)


def _invert_sym2x2(a: float, b: float, c: float) -> tuple[float, float, float] | None:
    det = a * c - b * b
    if abs(det) < 1e-12:
        return None
    inv_det = 1.0 / det
    return (c * inv_det, -b * inv_det, a * inv_det)


def _rotate_cov3d_world_to_cam(
    cov_world: tuple[tuple[float, ...], ...],
    view: tuple[tuple[float, ...], ...],
) -> tuple[tuple[float, ...], ...]:
    """Σ_cam = R Σ_world Rᵀ with R = view's upper-left 3×3 (rows = camera axes)."""
    r = view
    # M = R @ Σ
    m = [[0.0, 0.0, 0.0] for _ in range(3)]
    for i in range(3):
        for j in range(3):
            m[i][j] = (
                r[i][0] * cov_world[0][j]
                + r[i][1] * cov_world[1][j]
                + r[i][2] * cov_world[2][j]
            )
    # Σ_cam = M @ Rᵀ
    out = [[0.0, 0.0, 0.0] for _ in range(3)]
    for i in range(3):
        for j in range(3):
            out[i][j] = m[i][0] * r[j][0] + m[i][1] * r[j][1] + m[i][2] * r[j][2]
    return (
        (out[0][0], out[0][1], out[0][2]),
        (out[1][0], out[1][1], out[1][2]),
        (out[2][0], out[2][1], out[2][2]),
    )


def _cov3d_to_conic_and_radius(
    cov3d_cam: tuple[tuple[float, ...], ...],
    mean_cam: tuple[float, float, float],
    *,
    focal_x_px: float,
    focal_y_px: float,
) -> tuple[tuple[float, float, float], float] | None:
    """Project camera-space Σ₃ to a pixel-space conic (Σ₂⁻¹) and screen radius.

    ``focal_*_px`` are in **pixels**, matching means2d and blend ``dx, dy``.
    """
    tx, ty, tz = mean_cam
    if tz <= 1e-4:
        return None
    tz_inv = 1.0 / tz
    tz2_inv = tz_inv * tz_inv
    j00 = focal_x_px * tz_inv
    j02 = -focal_x_px * tx * tz2_inv
    j11 = focal_y_px * tz_inv
    j12 = -focal_y_px * ty * tz2_inv

    c00 = cov3d_cam[0][0]
    c01 = cov3d_cam[0][1]
    c02 = cov3d_cam[0][2]
    c11 = cov3d_cam[1][1]
    c12 = cov3d_cam[1][2]
    c22 = cov3d_cam[2][2]

    # J is 2×3: [[j00, 0, j02], [0, j11, j12]].  Σ₂ = J Σ₃ Jᵀ.
    t00 = j00 * c00 + j02 * c02
    t01 = j00 * c01 + j02 * c12
    t02 = j00 * c02 + j02 * c22
    t11 = j11 * c11 + j12 * c12
    t12 = j11 * c12 + j12 * c22

    cov00 = t00 * j00 + t02 * j02 + 0.3
    cov01 = t01 * j11 + t02 * j12
    cov11 = t11 * j11 + t12 * j12 + 0.3

    conic = _invert_sym2x2(cov00, cov01, cov11)
    if conic is None:
        return None
    radius = float(math.ceil(3.0 * math.sqrt(max(cov00, cov11, 0.0))))
    return conic, radius


class PreprocessResult(TypedDict):
    means2d: tuple[tuple[float, float], ...]
    depths: tuple[float, ...]
    conics: tuple[tuple[float, float, float], ...]
    colors: tuple[tuple[float, float, float], ...]
    opacities: tuple[float, ...]
    radii: tuple[float, ...]


def preprocess_gaussians_cpu(
    cloud: GnomenCloud,
    *,
    width: int,
    height: int,
    view: tuple[tuple[float, ...], ...] | None = None,
    proj: tuple[tuple[float, ...], ...] | None = None,
) -> PreprocessResult:
    aspect = width / height
    if view is None or proj is None:
        view, proj = look_at_view_proj(aspect=aspect)

    means2d, depths = project_points(cloud.means, view, proj)
    # proj diagonals are NDC focals; convert to pixels for EWA / blend.
    focal_x_px = proj[0][0] * (0.5 * width)
    focal_y_px = proj[1][1] * (0.5 * height)

    conics: list[tuple[float, float, float]] = []
    radii: list[float] = []
    valid_means2d: list[tuple[float, float]] = []
    valid_depths: list[float] = []
    valid_colors: list[tuple[float, float, float]] = []
    valid_opacities: list[float] = []

    for i, mean in enumerate(cloud.means):
        cov_world = scale_rot_to_cov3d(cloud.scales[i], cloud.quats[i])
        cov_cam = _rotate_cov3d_world_to_cam(cov_world, view)
        cam = mat4_mul_vec4(view, (mean[0], mean[1], mean[2], 1.0))
        projected = _cov3d_to_conic_and_radius(
            cov_cam,
            (cam[0], cam[1], cam[2]),
            focal_x_px=focal_x_px,
            focal_y_px=focal_y_px,
        )
        if projected is None:
            continue
        conic, radius = projected
        c0, c1, c2 = conic
        det = c0 * c2 - c1 * c1
        if det <= 0.0:
            continue
        if radius < 1.0:
            continue

        ndc_x, ndc_y = means2d[i]
        # Flip NDC X only: view is +Z-forward but projection uses OpenGL w=-z,
        # which mirrors horizontal NDC. Y is left unflipped because that mirror
        # cancels with image-space Y-down, so +Y still reads as "up" on screen.
        px = (0.5 - ndc_x * 0.5) * width
        py = (ndc_y * 0.5 + 0.5) * height
        if (
            px + radius < 0
            or py + radius < 0
            or px - radius > width
            or py - radius > height
        ):
            continue

        valid_means2d.append((px, py))
        valid_depths.append(depths[i])
        valid_colors.append(cloud.colors[i])
        valid_opacities.append(cloud.opacities[i])
        conics.append(conic)
        radii.append(float(radius))

    return PreprocessResult(
        means2d=tuple(valid_means2d),
        depths=tuple(valid_depths),
        conics=tuple(conics),
        colors=tuple(valid_colors),
        opacities=tuple(valid_opacities),
        radii=tuple(radii),
    )


def sort_by_depth_cpu(depths: tuple[float, ...]) -> tuple[int, ...]:
    return tuple(i for i, _ in sorted(enumerate(depths), key=lambda item: item[1]))


def pad_preprocess_result(pre: PreprocessResult, count: int) -> PreprocessResult:
    """Pad a culled preprocess result to a fixed gaussian slot count.

    Extra slots are placed off-screen with huge depth and zero opacity so they
    do not affect blending. Used by interactive viewers that pin program shape.
    """
    n = len(pre["depths"])
    if n == count:
        return pre
    if n > count:
        raise ValueError(f"cannot pad {n} gaussians to smaller count {count}")

    pad = count - n
    return PreprocessResult(
        means2d=pre["means2d"] + ((-1e3, -1e3),) * pad,
        depths=pre["depths"] + (1e6,) * pad,
        conics=pre["conics"] + ((1.0, 0.0, 1.0),) * pad,
        colors=pre["colors"] + ((0.0, 0.0, 0.0),) * pad,
        opacities=pre["opacities"] + (0.0,) * pad,
        radii=pre["radii"] + (0.0,) * pad,
    )


def blend_gaussians_cpu(
    *,
    width: int,
    height: int,
    means2d: tuple[tuple[float, float], ...],
    conics: tuple[tuple[float, float, float], ...],
    colors: tuple[tuple[float, float, float], ...],
    opacities: tuple[float, ...],
    order: tuple[int, ...],
) -> tuple[float, ...]:
    image = [0.0] * (width * height * 3)
    # Per-pixel transmittance for front-to-back compositing.
    T = [1.0] * (width * height)
    for idx in order:
        mx, my = means2d[idx]
        c0, c1, c2 = conics[idx]
        r, g, b = colors[idx]
        opacity = opacities[idx]

        # c0,c2 are conic (Σ⁻¹) diagonals; extent uses 3σ from inverse scale.
        # For SPD conic, σ² ≈ 1/λ_min(conic) ≲ 1/min(c0,c2) as a safe bound.
        extent = 3.0 / math.sqrt(max(min(c0, c2), 1e-8))
        x0 = max(0, int(math.floor(mx - extent)))
        x1 = min(width, int(math.ceil(mx + extent)))
        y0 = max(0, int(math.floor(my - extent)))
        y1 = min(height, int(math.ceil(my + extent)))

        for py in range(y0, y1):
            for px in range(x0, x1):
                pix = py * width + px
                if T[pix] < 1e-4:
                    continue
                dx = px + 0.5 - mx
                dy = py + 0.5 - my
                power = -0.5 * (c0 * dx * dx + c2 * dy * dy) - c1 * dx * dy
                if power > 0.0:
                    continue
                alpha = min(0.99, opacity * math.exp(power))
                if alpha < 1.0 / 255.0:
                    continue
                weight = alpha * T[pix]
                offset = pix * 3
                image[offset] += r * weight
                image[offset + 1] += g * weight
                image[offset + 2] += b * weight
                T[pix] *= 1.0 - alpha

    return tuple(image)


def render_gnomen_cpu(
    cloud: GnomenCloud | None = None,
    *,
    width: int = 32,
    height: int = 32,
) -> tuple[float, ...]:
    cloud = make_gnomen_cloud() if cloud is None else cloud
    pre = preprocess_gaussians_cpu(cloud, width=width, height=height)
    order = sort_by_depth_cpu(pre["depths"])
    return blend_gaussians_cpu(
        width=width,
        height=height,
        means2d=pre["means2d"],
        conics=pre["conics"],
        colors=pre["colors"],
        opacities=pre["opacities"],
        order=order,
    )
