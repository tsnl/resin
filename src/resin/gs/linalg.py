"""Pure-Python linalg helpers shared by reference and DSL paths."""

import math


def quat_to_rotmat(q: tuple[float, float, float, float]) -> tuple[tuple[float, ...], ...]:
    w, x, y, z = q
    return (
        (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)),
        (2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)),
        (2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)),
    )


def mat3_mul_vec3(m: tuple[tuple[float, ...], ...], v: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    )


def mat4_mul_vec4(m: tuple[tuple[float, ...], ...], v: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    out = [0.0, 0.0, 0.0, 0.0]
    for row in range(4):
        out[row] = sum(m[row][col] * v[col] for col in range(4))
    return (out[0], out[1], out[2], out[3])


def scale_rot_to_cov3d(
    scale: tuple[float, float, float],
    quat: tuple[float, float, float, float],
) -> tuple[tuple[float, ...], ...]:
    r = quat_to_rotmat(quat)
    sx, sy, sz = scale
    # cov = R @ diag(s^2) @ R^T
    rs = (
        (r[0][0] * sx * sx, r[0][1] * sy * sy, r[0][2] * sz * sz),
        (r[1][0] * sx * sx, r[1][1] * sy * sy, r[1][2] * sz * sz),
        (r[2][0] * sx * sx, r[2][1] * sy * sy, r[2][2] * sz * sz),
    )
    return (
        (
            rs[0][0] * r[0][0] + rs[0][1] * r[0][1] + rs[0][2] * r[0][2],
            rs[0][0] * r[1][0] + rs[0][1] * r[1][1] + rs[0][2] * r[1][2],
            rs[0][0] * r[2][0] + rs[0][1] * r[2][1] + rs[0][2] * r[2][2],
        ),
        (
            rs[1][0] * r[0][0] + rs[1][1] * r[0][1] + rs[1][2] * r[0][2],
            rs[1][0] * r[1][0] + rs[1][1] * r[1][1] + rs[1][2] * r[1][2],
            rs[1][0] * r[2][0] + rs[1][1] * r[2][1] + rs[1][2] * r[2][2],
        ),
        (
            rs[2][0] * r[0][0] + rs[2][1] * r[0][1] + rs[2][2] * r[0][2],
            rs[2][0] * r[1][0] + rs[2][1] * r[1][1] + rs[2][2] * r[1][2],
            rs[2][0] * r[2][0] + rs[2][1] * r[2][1] + rs[2][2] * r[2][2],
        ),
    )


def look_at_view_proj(
    *,
    eye: tuple[float, float, float] = (0.0, 0.0, 0.0),
    center: tuple[float, float, float] = (0.0, 0.0, -1.0),
    up: tuple[float, float, float] = (0.0, 1.0, 0.0),
    fov_y_deg: float = 60.0,
    aspect: float = 1.0,
    z_near: float = 0.1,
    z_far: float = 100.0,
) -> tuple[tuple[tuple[float, ...], ...], tuple[tuple[float, ...], ...]]:
    ex, ey, ez = eye
    cx, cy, cz = center
    ux, uy, uz = up

    fx, fy, fz = cx - ex, cy - ey, cz - ez
    flen = math.sqrt(fx * fx + fy * fy + fz * fz)
    fx, fy, fz = fx / flen, fy / flen, fz / flen

    sx = fy * uz - fz * uy
    sy = fz * ux - fx * uz
    sz = fx * uy - fy * ux
    slen = math.sqrt(sx * sx + sy * sy + sz * sz)
    sx, sy, sz = sx / slen, sy / slen, sz / slen

    ux2 = sy * fz - sz * fy
    uy2 = sz * fx - sx * fz
    uz2 = sx * fy - sy * fx

    view = (
        (sx, sy, sz, -(sx * ex + sy * ey + sz * ez)),
        (ux2, uy2, uz2, -(ux2 * ex + uy2 * ey + uz2 * ez)),
        (fx, fy, fz, -(fx * ex + fy * ey + fz * ez)),
        (0.0, 0.0, 0.0, 1.0),
    )

    f = 1.0 / math.tan(math.radians(fov_y_deg) * 0.5)
    proj = (
        (f / aspect, 0.0, 0.0, 0.0),
        (0.0, f, 0.0, 0.0),
        (0.0, 0.0, z_far / (z_near - z_far), (z_far * z_near) / (z_near - z_far)),
        (0.0, 0.0, -1.0, 0.0),
    )
    return view, proj


def project_points(
    means: tuple[tuple[float, float, float], ...],
    view: tuple[tuple[float, ...], ...],
    proj: tuple[tuple[float, ...], ...],
) -> tuple[tuple[tuple[float, float], ...], tuple[float, ...]]:
    means2d: list[tuple[float, float]] = []
    depths: list[float] = []
    for mx, my, mz in means:
        cam = mat4_mul_vec4(view, (mx, my, mz, 1.0))
        clip = mat4_mul_vec4(proj, cam)
        w = clip[3] if abs(clip[3]) > 1e-8 else 1.0
        means2d.append((clip[0] / w, clip[1] / w))
        depths.append(cam[2])
    return tuple(means2d), tuple(depths)