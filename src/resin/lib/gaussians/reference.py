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


def _cov3d_to_conic(
    cov3d: tuple[tuple[float, ...], ...],
    mean_cam: tuple[float, float, float],
    *,
    focal_x: float,
    focal_y: float,
) -> tuple[float, float, float] | None:
    tx, ty, tz = mean_cam
    if tz <= 1e-4:
        return None
    tz_inv = 1.0 / tz
    tz2_inv = tz_inv * tz_inv
    j00 = focal_x * tz_inv
    j02 = -focal_x * tx * tz2_inv
    j11 = focal_y * tz_inv
    j12 = -focal_y * ty * tz2_inv

    c00 = cov3d[0][0]
    c01 = cov3d[0][1]
    c02 = cov3d[0][2]
    c11 = cov3d[1][1]
    c12 = cov3d[1][2]
    c22 = cov3d[2][2]

    t00 = j00 * c00 + j02 * c02
    t01 = j00 * c01 + j02 * c12
    t02 = j00 * c02 + j02 * c22
    t12 = j11 * c11 + j12 * c12

    cov00 = t00 * j00 + t02 * j02 + 0.3
    cov01 = t01 * j00 + t12 * j02
    cov11 = t01 * j11 + t12 * j12 + 0.3
    return _invert_sym2x2(cov00, cov01, cov11)


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
    focal_y = proj[1][1]
    focal_x = proj[0][0]

    conics: list[tuple[float, float, float]] = []
    radii: list[float] = []
    valid_means2d: list[tuple[float, float]] = []
    valid_depths: list[float] = []
    valid_colors: list[tuple[float, float, float]] = []
    valid_opacities: list[float] = []

    for i, mean in enumerate(cloud.means):
        cov3d = scale_rot_to_cov3d(cloud.scales[i], cloud.quats[i])
        cam = mat4_mul_vec4(view, (mean[0], mean[1], mean[2], 1.0))
        conic = _cov3d_to_conic(
            cov3d, (cam[0], cam[1], cam[2]), focal_x=focal_x, focal_y=focal_y
        )
        if conic is None:
            continue
        c0, c1, c2 = conic
        det = c0 * c2 - c1 * c1
        if det <= 0.0:
            continue
        radius = math.ceil(3.0 * math.sqrt(max(c0, c2)))
        if radius < 1.0:
            continue

        ndc_x, ndc_y = means2d[i]
        px = (ndc_x * 0.5 + 0.5) * width
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

    pad_n = count - n
    return PreprocessResult(
        means2d=pre["means2d"] + ((-1e3, -1e3),) * pad_n,
        depths=pre["depths"] + (1e6,) * pad_n,
        conics=pre["conics"] + ((1.0, 0.0, 1.0),) * pad_n,
        colors=pre["colors"] + ((0.0, 0.0, 0.0),) * pad_n,
        opacities=pre["opacities"] + (0.0,) * pad_n,
        radii=pre["radii"] + (0.0,) * pad_n,
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

        span = 3.0 * math.sqrt(max(c0, c2))
        x0 = max(0, int(math.floor(mx - span)))
        x1 = min(width, int(math.ceil(mx + span)))
        y0 = max(0, int(math.floor(my - span)))
        y1 = min(height, int(math.ceil(my + span)))

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
