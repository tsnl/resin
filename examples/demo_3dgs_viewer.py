"""Interactive PyGame viewport for flying around the gnomen gaussian cloud."""

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

import sys

import pygame

from resin.lib.gs.camera import FlyCamera
from resin.lib.gs.gnomen import GnomenCloud, make_gnomen_cloud
from resin.lib.gs.gpu_session import GpuForwardSession
from resin.lib.gs.image_io import rgb_f32_to_rgb888_bytes
from resin.lib.gs.reference import preprocess_gaussians_cpu

RENDER_WIDTH = 320
RENDER_HEIGHT = 180
DISPLAY_SCALE = 3
MOVE_KEYS = {
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


def _draw_help(surface: pygame.Surface, *, fps: float, gaussian_count: int) -> None:
    font = pygame.font.SysFont("menlo,monaco,consolas,courier new", 14)
    lines = [
        "WASD move  QE up/down  mouse look  Esc quit",
        f"fps {fps:4.1f}  visible gaussians {gaussian_count}",
    ]
    y = 8
    for line in lines:
        text = font.render(line, True, (240, 240, 240))
        shadow = font.render(line, True, (0, 0, 0))
        _ = surface.blit(shadow, (9, y + 1))
        _ = surface.blit(text, (8, y))
        y += 18


def _render_frame(
    *,
    camera: FlyCamera,
    cloud: GnomenCloud,
    session: GpuForwardSession,
    display: pygame.Surface,
) -> tuple[pygame.Surface, int]:
    view, proj = camera.view_proj(aspect=RENDER_WIDTH / RENDER_HEIGHT)
    pre = preprocess_gaussians_cpu(
        cloud,
        width=RENDER_WIDTH,
        height=RENDER_HEIGHT,
        view=view,
        proj=proj,
    )
    pixels = session.render(pre)
    rgb = rgb_f32_to_rgb888_bytes(pixels, width=RENDER_WIDTH, height=RENDER_HEIGHT)
    frame = pygame.image.frombuffer(rgb, (RENDER_WIDTH, RENDER_HEIGHT), "RGB")
    return frame.convert(display), len(pre["depths"])


def main() -> None:
    _ = pygame.init()
    display_size = (RENDER_WIDTH * DISPLAY_SCALE, RENDER_HEIGHT * DISPLAY_SCALE)
    screen = pygame.display.set_mode(display_size)
    _ = pygame.display.set_caption("Resin 3DGS Gnomen Viewer")
    clock = pygame.time.Clock()

    pygame.event.set_grab(True)
    _ = pygame.mouse.set_visible(False)

    camera = FlyCamera()
    cloud = make_gnomen_cloud()
    session = GpuForwardSession(
        width=RENDER_WIDTH,
        height=RENDER_HEIGHT,
        fixed_count=cloud.count,
    )

    print("controls: WASDQE move, mouse look, Esc quit", file=sys.stderr)

    frame, gaussian_count = _render_frame(
        camera=camera,
        cloud=cloud,
        session=session,
        display=screen,
    )
    needs_render = False
    running = True

    while running:
        dt = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            elif event.type == pygame.MOUSEMOTION:
                camera.apply_mouse_delta(event.rel[0], event.rel[1])
                needs_render = True

        forward, right, up = _movement_from_keys(pygame.key.get_pressed())
        if forward != 0.0 or right != 0.0 or up != 0.0:
            camera.move(forward=forward, right=right, up=up, dt=dt)
            needs_render = True

        if needs_render:
            frame, gaussian_count = _render_frame(
                camera=camera,
                cloud=cloud,
                session=session,
                display=screen,
            )
            needs_render = False

        scaled = pygame.transform.scale(frame, display_size)
        _ = screen.blit(scaled, (0, 0))
        _draw_help(screen, fps=clock.get_fps(), gaussian_count=gaussian_count)
        pygame.display.flip()

    pygame.quit()
    print("demo_3dgs_viewer: ok", file=sys.stderr)


if __name__ == "__main__":
    main()