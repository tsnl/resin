"""CPU reference tests for 3DGS linalg and fixtures."""

import pytest

from resin.lib.gaussians.gnomen import make_gnomen_cloud
from resin.lib.gaussians.linalg import look_at_view_proj, project_points, quat_to_rotmat


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
