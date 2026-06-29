"""Fly camera helpers for interactive 3DGS viewing."""

import math
from dataclasses import dataclass

from resin.lib.gs.linalg import look_at_view_proj


def _normalize3(v: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = v
    length = math.sqrt(x * x + y * y + z * z)
    if length < 1e-12:
        return (0.0, 0.0, 0.0)
    inv = 1.0 / length
    return (x * inv, y * inv, z * inv)


@dataclass
class FlyCamera:
    """FPS-style camera with yaw/pitch look and WASDQE-style movement."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    move_speed: float = 2.0
    look_sensitivity: float = 0.0025
    fov_y_deg: float = 60.0

    def position(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def forward(self) -> tuple[float, float, float]:
        cp = math.cos(self.pitch)
        return (
            math.sin(self.yaw) * cp,
            math.sin(self.pitch),
            -math.cos(self.yaw) * cp,
        )

    def right(self) -> tuple[float, float, float]:
        forward = self.forward()
        return _normalize3((forward[2], 0.0, -forward[0]))

    def view_proj(self, *, aspect: float) -> tuple[tuple[tuple[float, ...], ...], tuple[tuple[float, ...], ...]]:
        eye = self.position()
        forward = self.forward()
        center = (eye[0] + forward[0], eye[1] + forward[1], eye[2] + forward[2])
        return look_at_view_proj(
            eye=eye,
            center=center,
            aspect=aspect,
            fov_y_deg=self.fov_y_deg,
        )

    def apply_mouse_delta(self, dx: float, dy: float) -> None:
        self.yaw += dx * self.look_sensitivity
        limit = math.pi * 0.49
        self.pitch = max(-limit, min(limit, self.pitch - dy * self.look_sensitivity))

    def move(self, *, forward: float, right: float, up: float, dt: float) -> None:
        step = self.move_speed * dt
        f = self.forward()
        r = self.right()
        self.x += f[0] * forward * step + r[0] * right * step
        self.y += up * step
        self.z += f[2] * forward * step + r[2] * right * step