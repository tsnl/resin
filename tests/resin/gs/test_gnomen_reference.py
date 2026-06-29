"""TDD tests for gnomen fixture and CPU reference pipeline."""


import pytest

from resin.gs.gnomen import make_gnomen_cloud
from resin.gs.linalg import look_at_view_proj, project_points, quat_to_rotmat
from resin.gs.reference import (
    blend_gaussians_cpu,
    preprocess_gaussians_cpu,
    render_gnomen_cpu,
    sort_by_depth_cpu,
)


class TestGnomenFixture:
    def test_three_gaussians(self) -> None:
        cloud = make_gnomen_cloud()
        assert cloud.count == 3
        assert cloud.colors[0] == (1.0, 0.0, 0.0)
        assert cloud.means[0][2] == -2.0


class TestLinalgReference:
    def test_identity_quat_is_identity_rot(self) -> None:
        r = quat_to_rotmat((1.0, 0.0, 0.0, 0.0))
        assert r[0][0] == pytest.approx(1.0)
        assert r[1][1] == pytest.approx(1.0)
        assert r[2][2] == pytest.approx(1.0)

    def test_project_gnomen_depths_increase(self, capsys: pytest.CaptureFixture[str]) -> None:
        cloud = make_gnomen_cloud()
        view, proj = look_at_view_proj(aspect=1.0)
        _, depths = project_points(cloud.means, view, proj)
        print(f"gnomen depths: {depths}")
        captured = capsys.readouterr()
        assert "gnomen depths" in captured.out
        assert depths[0] < depths[1] < depths[2]


class TestPreprocessReference:
    def test_all_gnomen_survive_culling(self, capsys: pytest.CaptureFixture[str]) -> None:
        cloud = make_gnomen_cloud()
        pre = preprocess_gaussians_cpu(cloud, width=32, height=32)
        print(f"preprocess means2d: {pre['means2d']}")
        print(f"preprocess depths: {pre['depths']}")
        print(f"preprocess conics: {pre['conics']}")
        captured = capsys.readouterr()
        assert "preprocess means2d" in captured.out
        assert len(pre["means2d"]) == 3
        assert len(pre["conics"]) == 3
        for mx, my in pre["means2d"]:
            assert 0.0 <= mx <= 32.0
            assert 0.0 <= my <= 32.0


class TestSortReference:
    def test_front_to_back_order(self) -> None:
        order = sort_by_depth_cpu((2.0, 3.0, 4.0))
        assert order == (0, 1, 2)


class TestBlendReference:
    def test_center_pixel_has_color(self, capsys: pytest.CaptureFixture[str]) -> None:
        cloud = make_gnomen_cloud()
        pre = preprocess_gaussians_cpu(cloud, width=32, height=32)
        order = sort_by_depth_cpu(pre["depths"])
        image = blend_gaussians_cpu(
            width=32,
            height=32,
            means2d=pre["means2d"],
            conics=pre["conics"],
            colors=pre["colors"],
            opacities=pre["opacities"],
            order=order,
        )
        cx, cy = 16, 16
        offset = (cy * 32 + cx) * 3
        center = (image[offset], image[offset + 1], image[offset + 2])
        print(f"center pixel RGB: {center}")
        captured = capsys.readouterr()
        assert "center pixel RGB" in captured.out
        assert center[0] + center[1] + center[2] > 0.1

    def test_render_gnomen_full_pipeline(self, capsys: pytest.CaptureFixture[str]) -> None:
        image = render_gnomen_cpu(width=32, height=32)
        nonzero = sum(1 for v in image if v > 1e-4)
        print(f"nonzero channels: {nonzero} / {len(image)}")
        captured = capsys.readouterr()
        assert "nonzero channels" in captured.out
        assert nonzero > 0
        max_val = max(image)
        print(f"max channel: {max_val}")
        assert max_val > 0.01
        assert max_val <= 1.5