"""Offline single-frame render from a camera pose (no pygame window)."""

from pathlib import Path

import pytest

from resin.lib.gaussians.camera import FlyCamera
from resin.lib.gaussians.offline import render_frame, render_frame_to_png


def test_pose_roundtrip_json() -> None:
    cam = FlyCamera(x=1.0, y=-0.5, z=2.0, yaw=0.3, pitch=-0.1, fov_y_deg=50.0)
    text = cam.pose_json()
    restored = FlyCamera.from_pose_json(text)
    assert restored.x == pytest.approx(cam.x)
    assert restored.y == pytest.approx(cam.y)
    assert restored.z == pytest.approx(cam.z)
    assert restored.yaw == pytest.approx(cam.yaw)
    assert restored.pitch == pytest.approx(cam.pitch)
    assert restored.fov_y_deg == pytest.approx(cam.fov_y_deg)


def test_render_frame_default_pose() -> None:
    pixels, visible = render_frame(width=64, height=64, tiled=True)
    assert len(pixels) == 64 * 64 * 3
    assert visible > 0
    assert sum(pixels) > 0.05


def test_render_frame_from_pose_dict() -> None:
    pose = {
        "x": 0.2,
        "y": 0.1,
        "z": -0.5,
        "yaw": 0.15,
        "pitch": -0.05,
        "fov_y_deg": 60.0,
    }
    pixels, visible = render_frame(camera=pose, width=64, height=64)
    assert len(pixels) == 64 * 64 * 3
    assert visible >= 0  # may cull some from this pose
    # Center-ish energy or at least a valid buffer.
    assert all(p == p for p in pixels)  # no NaNs


def test_render_frame_to_png(tmp_path: Path) -> None:
    path = tmp_path / "frame.png"
    out, visible = render_frame_to_png(
        path,
        camera=FlyCamera(),
        width=32,
        height=32,
    )
    assert out == path
    assert path.is_file()
    assert path.stat().st_size > 50
    assert visible > 0
    # PNG signature
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
