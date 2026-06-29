"""Offline single-frame rendering from a FlyCamera pose."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from resin.lib.gaussians.camera import FlyCamera
from resin.lib.gaussians.gnomen import GnomenCloud, make_gnomen_cloud
from resin.lib.gaussians.gpu_session import GpuForwardSession
from resin.lib.gaussians.image_io import save_rgb_f32_png
from resin.lib.gaussians.preprocess import preprocess_gaussians


def render_frame(
    *,
    camera: FlyCamera | Mapping[str, Any] | str | None = None,
    cloud: GnomenCloud | None = None,
    width: int = 1280,
    height: int = 720,
    tiled: bool = True,
    tile_size: int = 16,
) -> tuple[tuple[float, ...], int]:
    """Render one frame; return ``(pixels_rgb_f32, visible_gaussian_count)``."""
    if camera is None:
        cam = FlyCamera()
    elif isinstance(camera, FlyCamera):
        cam = camera
    elif isinstance(camera, str):
        cam = FlyCamera.from_pose_json(camera)
    else:
        cam = FlyCamera.from_pose(dict(camera))

    cloud = make_gnomen_cloud() if cloud is None else cloud
    view, proj = cam.view_proj(aspect=width / height)
    pre = preprocess_gaussians(
        cloud, width=width, height=height, view=view, proj=proj
    )
    session = GpuForwardSession(
        width=width,
        height=height,
        fixed_count=cloud.count,
        tiled=tiled,
        tile_size=tile_size,
    )
    pixels = session.render(pre)
    return pixels, len(pre["depths"])


def render_frame_to_png(
    path: str | Path,
    *,
    camera: FlyCamera | Mapping[str, Any] | str | None = None,
    cloud: GnomenCloud | None = None,
    width: int = 1280,
    height: int = 720,
    tiled: bool = True,
    tile_size: int = 16,
) -> tuple[Path, int]:
    """Render one frame and write an 8-bit RGB PNG. Returns ``(path, visible)``."""
    pixels, visible = render_frame(
        camera=camera,
        cloud=cloud,
        width=width,
        height=height,
        tiled=tiled,
        tile_size=tile_size,
    )
    out = save_rgb_f32_png(path, pixels, width=width, height=height)
    return out, visible
