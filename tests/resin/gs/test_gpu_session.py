"""Tests for reusable GPU forward-render sessions."""

import pytest

from resin.gs.gnomen import make_gnomen_cloud
from resin.gs.gpu_session import GpuForwardSession
from resin.gs.reference import preprocess_gaussians_cpu, render_gnomen_cpu


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