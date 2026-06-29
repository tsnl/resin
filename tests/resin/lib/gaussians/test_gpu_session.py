"""Smoke tests for the interactive GPU forward session."""

from resin.lib.gaussians.camera import FlyCamera
from resin.lib.gaussians.gnomen import make_gnomen_cloud
from resin.lib.gaussians.gpu_session import GpuForwardSession
from resin.lib.gaussians.image_io import rgb_f32_to_rgb888_bytes
from resin.lib.gaussians.preprocess import preprocess_gaussians
from resin.lib.gaussians.reference import pad_preprocess_result


def test_pad_preprocess_to_fixed_count() -> None:
    cloud = make_gnomen_cloud()
    pre = preprocess_gaussians(cloud, width=32, height=32)
    padded = pad_preprocess_result(pre, cloud.count + 2)
    assert len(padded["depths"]) == cloud.count + 2
    assert padded["opacities"][-1] == 0.0


def test_gpu_session_renders_gnomen() -> None:
    width = height = 32
    cloud = make_gnomen_cloud()
    session = GpuForwardSession(
        width=width, height=height, fixed_count=cloud.count
    )
    camera = FlyCamera()
    view, proj = camera.view_proj(aspect=1.0)
    pre = preprocess_gaussians(
        cloud, width=width, height=height, view=view, proj=proj
    )
    pixels = session.render(pre)
    assert len(pixels) == width * height * 3
    assert sum(pixels) > 0.05
    rgb = rgb_f32_to_rgb888_bytes(pixels, width=width, height=height)
    assert len(rgb) == width * height * 3


def test_fly_camera_mouse_and_move() -> None:
    cam = FlyCamera()
    cam.apply_mouse_delta(100.0, 0.0)
    assert cam.yaw != 0.0
    before = cam.position()
    cam.move(forward=1.0, right=0.0, up=0.0, dt=0.1)
    after = cam.position()
    assert after != before


def test_fly_camera_yaw_unbounded() -> None:
    """Yaw must accumulate past ±360°; only pitch is limited."""
    cam = FlyCamera()
    for _ in range(5000):
        cam.apply_mouse_delta(100.0, 0.0)
    assert abs(cam.yaw) > 2.0 * 3.141592653589793
    # Vertical mouse hits pitch limit, not yaw.
    cam2 = FlyCamera()
    for _ in range(5000):
        cam2.apply_mouse_delta(0.0, 100.0)
    assert abs(cam2.pitch) < 1.58
    assert cam2.yaw == 0.0


def test_fly_camera_view_stable_at_high_pitch() -> None:
    cam = FlyCamera(pitch=1.5)
    view, _proj = cam.view_proj(aspect=1.0)
    for row in view[:3]:
        for v in row:
            assert v == v  # not NaN
