# /// script
# requires-python = ">=3.14"
# dependencies = [
#   "resin",
#   "resin-rt-pybind",
#   "pygame-ce",
# ]
#
# [tool.uv.sources]
# resin = { path = "..", editable = true }
# resin-rt-pybind = { path = "../crates/resin-rt-pybind", editable = true }
# ///
"""Interactive pygame-ce viewport for flying around the gnomen gaussian cloud.

Renders at the display's **physical pixel** size (HiDPI / Retina aware): the
window is opened at logical points (default 1280×720), then GPU work uses
``surface.get_size()`` so a 2× display gets a 2560×1440 framebuffer.

Controls:
  W/S  — move forward / back (camera-relative)
  A/D  — strafe left / right
  Q/E  — move down / up (world Y)
  mouse — look (yaw / pitch); cursor hidden and re-centered every frame
  Tab  — toggle look (releases / shows cursor when off)
  C    — print camera pose JSON to stdout (for offline render)
  Esc  — quit
"""

from __future__ import annotations

import math
import os
import sys

# Ensure SDL does not disable HiDPI before pygame initializes (macOS Retina).
os.environ.setdefault("SDL_VIDEO_HIGHDPI_DISABLED", "0")

import pygame

from resin.lib.gaussians.camera import FlyCamera
from resin.lib.gaussians.gnomen import GnomenCloud, make_gnomen_cloud
from resin.lib.gaussians.gpu_session import GpuForwardSession
from resin.lib.gaussians.image_io import rgb_f32_to_rgb888_bytes
from resin.lib.gaussians.preprocess import preprocess_gaussians

# Window size in logical points (macOS “points”, Windows “DIPs”).
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

# key -> (forward, right, up) contributions while held
MOVE_KEYS: dict[int, tuple[float, float, float]] = {
    pygame.K_w: (1.0, 0.0, 0.0),
    pygame.K_s: (-1.0, 0.0, 0.0),
    pygame.K_a: (0.0, -1.0, 0.0),
    pygame.K_d: (0.0, 1.0, 0.0),
    pygame.K_q: (0.0, 0.0, -1.0),
    pygame.K_e: (0.0, 0.0, 1.0),
}


def _movement_from_keys(keys: pygame.key.ScancodeWrapper) -> tuple[float, float, float]:
    forward = right = up = 0.0
    for key, (f, r, u) in MOVE_KEYS.items():
        if keys[key]:
            forward += f
            right += r
            up += u
    return forward, right, up


def _window_flags() -> int:
    """Prefer a HiDPI-capable window when pygame/SDL exposes the flag."""
    return int(getattr(pygame, "WINDOW_ALLOW_HIGHDPI", 0))


def _logical_window_size() -> tuple[int, int]:
    """Size in points / DIPs (mouse coordinates live in this space)."""
    if hasattr(pygame.display, "get_window_size"):
        return pygame.display.get_window_size()
    screen = pygame.display.get_surface()
    if screen is not None:
        return screen.get_size()
    return (WINDOW_WIDTH, WINDOW_HEIGHT)


def _pixel_framebuffer_size(screen: pygame.Surface) -> tuple[int, int]:
    """Physical pixel size of the display surface (2× on Retina, etc.)."""
    return screen.get_size()


def _dpi_scale(screen: pygame.Surface) -> float:
    """Pixels per logical point (1.0 on standard DPI, ~2.0 on Retina)."""
    pw, ph = _pixel_framebuffer_size(screen)
    lw, lh = _logical_window_size()
    if lw <= 0 or lh <= 0:
        return 1.0
    return max(pw / lw, ph / lh)


def _draw_hud(
    surface: pygame.Surface,
    *,
    camera: FlyCamera,
    fps: float,
    visible: int,
    slots: int,
    look_enabled: bool,
    pixel_w: int,
    pixel_h: int,
    logical_w: int,
    logical_h: int,
    dpi_scale: float,
) -> None:
    font_px = max(14, int(round(16 * dpi_scale)))
    font = pygame.font.SysFont("menlo,monaco,consolas,courier new", font_px)
    x, y, z = camera.position()
    fx, fy, fz = camera.forward()
    rx, ry, rz = camera.right()
    yaw_deg = math.degrees(camera.yaw)
    pitch_deg = math.degrees(camera.pitch)
    aspect = pixel_w / pixel_h
    view, _proj = camera.view_proj(aspect=aspect)
    look_s = "on" if look_enabled else "off"
    lines = [
        "WASD move  QE down/up  mouse look  Tab free cursor  C pose  Esc quit",
        f"fps {fps:4.1f}  {pixel_w}x{pixel_h}px  "
        f"({logical_w}x{logical_h}pt @ {dpi_scale:.2f}x)  "
        f"visible {visible}/{slots}  look {look_s}",
        f"eye   ({x:+7.3f}, {y:+7.3f}, {z:+7.3f})",
        f"yaw   {yaw_deg:+7.2f} deg   pitch {pitch_deg:+7.2f} deg",
        f"fwd   ({fx:+7.3f}, {fy:+7.3f}, {fz:+7.3f})",
        f"right ({rx:+7.3f}, {ry:+7.3f}, {rz:+7.3f})",
        "view (rows = cam X / Y / Z / W):",
        f"  [{view[0][0]:+6.3f} {view[0][1]:+6.3f} {view[0][2]:+6.3f} | {view[0][3]:+7.3f}]",
        f"  [{view[1][0]:+6.3f} {view[1][1]:+6.3f} {view[1][2]:+6.3f} | {view[1][3]:+7.3f}]",
        f"  [{view[2][0]:+6.3f} {view[2][1]:+6.3f} {view[2][2]:+6.3f} | {view[2][3]:+7.3f}]",
        f"  [{view[3][0]:+6.3f} {view[3][1]:+6.3f} {view[3][2]:+6.3f} | {view[3][3]:+7.3f}]",
    ]
    pad = max(8, int(round(8 * dpi_scale)))
    y_pix = pad
    line_h = max(18, int(round(20 * dpi_scale)))
    for line in lines:
        text = font.render(line, True, (240, 240, 240))
        shadow = font.render(line, True, (0, 0, 0))
        _ = surface.blit(shadow, (pad + 1, y_pix + 1))
        _ = surface.blit(text, (pad, y_pix))
        y_pix += line_h


def _render_frame(
    *,
    camera: FlyCamera,
    cloud: GnomenCloud,
    session: GpuForwardSession,
    width: int,
    height: int,
) -> tuple[bytes, int]:
    view, proj = camera.view_proj(aspect=width / height)
    pre = preprocess_gaussians(
        cloud,
        width=width,
        height=height,
        view=view,
        proj=proj,
    )
    pixels = session.render(pre)
    rgb = rgb_f32_to_rgb888_bytes(pixels, width=width, height=height)
    return rgb, len(pre["depths"])


def main() -> None:
    _ = pygame.init()
    flags = _window_flags()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), flags)
    logical_w, logical_h = _logical_window_size()
    pixel_w, pixel_h = _pixel_framebuffer_size(screen)
    dpi = _dpi_scale(screen)

    _ = pygame.display.set_caption(
        f"Resin 3DGS Viewer — {pixel_w}x{pixel_h}px ({logical_w}x{logical_h}pt @ {dpi:.2f}x)"
    )
    clock = pygame.time.Clock()

    # Mouse positions are in **logical** coordinates; GPU uses **pixels**.
    pygame.event.set_grab(False)
    if hasattr(pygame.mouse, "set_relative_mode"):
        _ = pygame.mouse.set_relative_mode(False)
    look_enabled = True
    center = (logical_w // 2, logical_h // 2)

    def _set_look(enabled: bool) -> None:
        _ = pygame.mouse.set_visible(not enabled)
        if enabled:
            pygame.mouse.set_pos(center)

    _set_look(True)

    camera = FlyCamera()
    # Keep look speed stable across DPI (deltas are in points, not pixels).
    camera.look_sensitivity = 0.0025
    cloud = make_gnomen_cloud()
    session = GpuForwardSession(
        width=pixel_w,
        height=pixel_h,
        fixed_count=cloud.count,
    )

    print(
        f"window {logical_w}x{logical_h}pt, framebuffer {pixel_w}x{pixel_h}px "
        f"(dpi scale {dpi:.2f}x). controls: WASDQE, mouse look, Tab, C=pose, Esc",
        file=sys.stderr,
    )

    rgb, visible = _render_frame(
        camera=camera,
        cloud=cloud,
        session=session,
        width=pixel_w,
        height=pixel_h,
    )
    frame = pygame.image.frombuffer(rgb, (pixel_w, pixel_h), "RGB").convert(screen)
    needs_render = False
    running = True

    while running:
        dt = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_TAB:
                    look_enabled = not look_enabled
                    _set_look(look_enabled)
                elif event.key == pygame.K_c:
                    # One JSON object per line on stdout for offline render / paste.
                    print(camera.pose_json(), flush=True)
                    print(
                        f"# pose also on stderr: {camera.pose_json()}",
                        file=sys.stderr,
                        flush=True,
                    )

        # Re-query in case the OS changes scaling (rare) or window moves displays.
        logical_w, logical_h = _logical_window_size()
        center = (logical_w // 2, logical_h // 2)

        if look_enabled and pygame.mouse.get_focused():
            mx, my = pygame.mouse.get_pos()
            dx = mx - center[0]
            dy = my - center[1]
            if dx != 0 or dy != 0:
                # get_pos is in points; sensitivity is tuned for points.
                camera.apply_mouse_delta(float(dx), float(dy))
                needs_render = True
                pygame.mouse.set_pos(center)

        forward, right, up = _movement_from_keys(pygame.key.get_pressed())
        if forward != 0.0 or right != 0.0 or up != 0.0:
            camera.move(forward=forward, right=right, up=up, dt=dt)
            needs_render = True

        if needs_render:
            rgb, visible = _render_frame(
                camera=camera,
                cloud=cloud,
                session=session,
                width=pixel_w,
                height=pixel_h,
            )
            frame = pygame.image.frombuffer(rgb, (pixel_w, pixel_h), "RGB").convert(
                screen
            )
            needs_render = False

        _ = screen.blit(frame, (0, 0))
        _draw_hud(
            screen,
            camera=camera,
            fps=clock.get_fps(),
            visible=visible,
            slots=cloud.count,
            look_enabled=look_enabled,
            pixel_w=pixel_w,
            pixel_h=pixel_h,
            logical_w=logical_w,
            logical_h=logical_h,
            dpi_scale=dpi,
        )
        pygame.display.flip()

    pygame.quit()
    print("demo_3dgs_viewer: ok", file=sys.stderr)


if __name__ == "__main__":
    main()
