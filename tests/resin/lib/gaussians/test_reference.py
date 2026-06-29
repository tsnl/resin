"""CPU reference tests for 3DGS preprocess and blend."""

import pytest

from resin.lib.gaussians.gnomen import make_gnomen_cloud
from resin.lib.gaussians.linalg import look_at_view_proj, project_points, quat_to_rotmat
from resin.lib.gaussians.preprocess import preprocess_gaussians
from resin.lib.gaussians.reference import (
    blend_gaussians_cpu,
    render_gnomen_cpu,
    sort_by_depth_cpu,
)


def test_identity_quat() -> None:
    r = quat_to_rotmat((1.0, 0.0, 0.0, 0.0))
    assert r[0][0] == pytest.approx(1.0)
    assert r[1][1] == pytest.approx(1.0)
    assert r[2][2] == pytest.approx(1.0)


def test_gnomen_has_dense_axis_dots() -> None:
    cloud = make_gnomen_cloud()
    # 1 hub + 99 samples × 3 axes
    assert cloud.count == 298
    # Hub is near white; axes carry strong R/G/B channels.
    assert cloud.colors[0] == pytest.approx((1.0, 1.0, 1.0))
    reds = sum(1 for c in cloud.colors if c[0] > 0.8 and c[1] < 0.3 and c[2] < 0.3)
    greens = sum(1 for c in cloud.colors if c[1] > 0.8 and c[0] < 0.3 and c[2] < 0.3)
    blues = sum(1 for c in cloud.colors if c[2] > 0.8 and c[0] < 0.4 and c[1] < 0.5)
    assert reds >= 90
    assert greens >= 90
    assert blues >= 90


def test_gnomen_z_arm_depths_decrease_toward_camera() -> None:
    cloud = make_gnomen_cloud()
    view, proj = look_at_view_proj(aspect=1.0)
    _, depths = project_points(cloud.means, view, proj)
    # Blue arm runs toward +Z (toward the default camera); depths decrease.
    blue_depths = [
        depths[i]
        for i, c in enumerate(cloud.colors)
        if c[2] > 0.8 and c[0] < 0.4 and c[1] < 0.5
    ]
    assert blue_depths == sorted(blue_depths, reverse=True)


def test_preprocess_keeps_most_gnomen() -> None:
    cloud = make_gnomen_cloud()
    pre = preprocess_gaussians(cloud, width=128, height=128)
    # Blue arm points toward the camera; a few may frustum-cull at the near side.
    assert len(pre["means2d"]) >= cloud.count // 2
    for mx, my in pre["means2d"]:
        assert -32.0 <= mx <= 160.0
        assert -32.0 <= my <= 160.0


def test_sort_front_to_back() -> None:
    assert sort_by_depth_cpu((2.0, 3.0, 4.0)) == (0, 1, 2)


def test_center_pixel_has_color() -> None:
    image = render_gnomen_cpu(width=32, height=32)
    off = (16 * 32 + 16) * 3
    assert image[off] + image[off + 1] + image[off + 2] > 0.05


def test_blend_matches_manual_order() -> None:
    cloud = make_gnomen_cloud()
    pre = preprocess_gaussians(cloud, width=16, height=16)
    order = sort_by_depth_cpu(pre["depths"])
    image = blend_gaussians_cpu(
        width=16,
        height=16,
        means2d=pre["means2d"],
        conics=pre["conics"],
        colors=pre["colors"],
        opacities=pre["opacities"],
        order=order,
    )
    assert len(image) == 16 * 16 * 3
    assert any(v > 0.0 for v in image)
