"""Tests for fly camera helpers."""

import math

import pytest

from resin.gs.camera import FlyCamera
from resin.gs.gnomen import make_gnomen_cloud
from resin.gs.reference import preprocess_gaussians_cpu


def test_default_forward_looks_down_negative_z() -> None:
    camera = FlyCamera()
    forward = camera.forward()
    assert forward[0] == 0.0
    assert forward[1] == 0.0
    assert forward[2] == -1.0


def test_yaw_changes_forward_direction() -> None:
    camera = FlyCamera(yaw=math.pi / 2)
    forward = camera.forward()
    assert forward == pytest.approx((1.0, 0.0, 0.0))


def test_move_forward_updates_position(capsys: pytest.CaptureFixture[str]) -> None:
    camera = FlyCamera()
    camera.move(forward=1.0, right=0.0, up=0.0, dt=0.5)
    print(f"position: {camera.position()}")
    captured = capsys.readouterr()
    assert "position" in captured.out
    assert camera.z == -1.0


def test_camera_motion_changes_depths(capsys: pytest.CaptureFixture[str]) -> None:
    cloud = make_gnomen_cloud()
    default_pre = preprocess_gaussians_cpu(cloud, width=32, height=32)
    camera = FlyCamera(z=-1.0)
    view, proj = camera.view_proj(aspect=1.0)
    moved_pre = preprocess_gaussians_cpu(cloud, width=32, height=32, view=view, proj=proj)
    print(f"default depths: {default_pre['depths']}")
    print(f"moved depths: {moved_pre['depths']}")
    captured = capsys.readouterr()
    assert "default depths" in captured.out
    assert default_pre["depths"] != moved_pre["depths"]