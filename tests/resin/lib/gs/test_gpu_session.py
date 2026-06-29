"""Tests for reusable GPU forward-render sessions."""

import pytest

from resin.lib.gs.camera import FlyCamera
from resin.lib.gs.gnomen import make_gnomen_cloud
from resin.lib.gs.gpu_session import GpuForwardSession
from resin.lib.gs.reference import preprocess_gaussians_cpu, render_gnomen_cpu


def test_gpu_session_matches_cpu_reference(capsys: pytest.CaptureFixture[str]) -> None:
    width = height = 32
    cloud = make_gnomen_cloud()
    pre = preprocess_gaussians_cpu(cloud, width=width, height=height)
    cpu_image = render_gnomen_cpu(cloud, width=width, height=height)
    gpu_image = GpuForwardSession(width=width, height=height).render(pre)

    cx, cy = 16, 16
    off = (cy * width + cx) * 3
    print(f"cpu center: {cpu_image[off : off + 3]}")
    print(f"gpu center: {gpu_image[off : off + 3]}")
    captured = capsys.readouterr()
    assert "cpu center" in captured.out
    assert gpu_image == pytest.approx(cpu_image, rel=1e-4, abs=1e-4)


def test_fixed_count_session_survives_culling_changes() -> None:
    cloud = make_gnomen_cloud()
    camera = FlyCamera()
    session = GpuForwardSession(width=32, height=32, fixed_count=cloud.count)
    for _ in range(20):
        camera.apply_mouse_delta(50.0, 30.0)
        view, proj = camera.view_proj(aspect=1.0)
        pre = preprocess_gaussians_cpu(cloud, width=32, height=32, view=view, proj=proj)
        pixels = session.render(pre)
        assert len(pixels) == 32 * 32 * 3