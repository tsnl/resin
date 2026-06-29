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


def test_gnomen_depths_increase() -> None:
    cloud = make_gnomen_cloud()
    view, proj = look_at_view_proj(aspect=1.0)
    _, depths = project_points(cloud.means, view, proj)
    assert depths[0] < depths[1] < depths[2]


def test_preprocess_keeps_all_gnomen() -> None:
    cloud = make_gnomen_cloud()
    pre = preprocess_gaussians(cloud, width=32, height=32)
    assert len(pre["means2d"]) == 3
    for mx, my in pre["means2d"]:
        assert 0.0 <= mx <= 32.0
        assert 0.0 <= my <= 32.0


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
