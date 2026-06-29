"""Fly camera helpers for interactive 3DGS viewing."""

import json
import math
from dataclasses import dataclass
from typing import Any

from resin.lib.gaussians.linalg import perspective_proj

# Pitch only — yaw is unbounded so you can spin in place freely.
_PITCH_LIMIT = math.pi * 0.5 - 1e-3


@dataclass
class FlyCamera:
    """FPS-style camera with yaw/pitch look and WASDQE-style movement.

    The view matrix is built directly from yaw/pitch (not via look-at), so
    yaw never hits a gimbal / normalize singularity that feels like clamping.
    Only pitch is limited (to avoid flipping over the poles).
    """

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
        """Look direction. At yaw=pitch=0 this is (0, 0, -1) (toward -Z)."""
        cp = math.cos(self.pitch)
        return (
            math.sin(self.yaw) * cp,
            math.sin(self.pitch),
            -math.cos(self.yaw) * cp,
        )

    def right(self) -> tuple[float, float, float]:
        """World-horizontal right (independent of pitch — no zero-length basis)."""
        return (math.cos(self.yaw), 0.0, math.sin(self.yaw))

    def up(self) -> tuple[float, float, float]:
        """Camera up = right × forward (matches row layout of the view matrix)."""
        rx, ry, rz = self.right()
        fx, fy, fz = self.forward()
        return (
            ry * fz - rz * fy,
            rz * fx - rx * fz,
            rx * fy - ry * fx,
        )

    def view_matrix(self) -> tuple[tuple[float, ...], ...]:
        """World-to-camera matrix; rows are camera X (right), Y (up), Z (forward)."""
        ex, ey, ez = self.position()
        rx, ry, rz = self.right()
        ux, uy, uz = self.up()
        fx, fy, fz = self.forward()
        return (
            (rx, ry, rz, -(rx * ex + ry * ey + rz * ez)),
            (ux, uy, uz, -(ux * ex + uy * ey + uz * ez)),
            (fx, fy, fz, -(fx * ex + fy * ey + fz * ez)),
            (0.0, 0.0, 0.0, 1.0),
        )

    def view_proj(
        self, *, aspect: float
    ) -> tuple[tuple[tuple[float, ...], ...], tuple[tuple[float, ...], ...]]:
        return self.view_matrix(), perspective_proj(
            fov_y_deg=self.fov_y_deg, aspect=aspect
        )

    def apply_mouse_delta(self, dx: float, dy: float) -> None:
        # Mouse right → increase yaw (turn right). Yaw is *not* wrapped or clamped.
        self.yaw += dx * self.look_sensitivity
        self.pitch = max(
            -_PITCH_LIMIT,
            min(_PITCH_LIMIT, self.pitch - dy * self.look_sensitivity),
        )

    def move(self, *, forward: float, right: float, up: float, dt: float) -> None:
        """Move along look direction, horizontal right, and world up (Q/E)."""
        step = self.move_speed * dt
        f = self.forward()
        r = self.right()
        self.x += f[0] * forward * step + r[0] * right * step
        self.y += f[1] * forward * step + up * step
        self.z += f[2] * forward * step + r[2] * right * step

    def pose_dict(self) -> dict[str, float]:
        """Pose fields for offline render / copy-paste (radians for yaw/pitch)."""
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "yaw": self.yaw,
            "pitch": self.pitch,
            "fov_y_deg": self.fov_y_deg,
        }

    def pose_json(self) -> str:
        """Single-line JSON pose (stdout-friendly)."""
        return json.dumps(self.pose_dict(), separators=(",", ":"))

    @classmethod
    def from_pose(cls, pose: dict[str, Any]) -> "FlyCamera":
        """Restore from ``pose_dict`` / JSON (ignores unknown keys)."""
        return cls(
            x=float(pose.get("x", 0.0)),
            y=float(pose.get("y", 0.0)),
            z=float(pose.get("z", 0.0)),
            yaw=float(pose.get("yaw", 0.0)),
            pitch=float(pose.get("pitch", 0.0)),
            fov_y_deg=float(pose.get("fov_y_deg", 60.0)),
        )

    @classmethod
    def from_pose_json(cls, text: str) -> "FlyCamera":
        data = json.loads(text)
        if not isinstance(data, dict):
            raise TypeError("pose JSON must be an object")
        return cls.from_pose(data)
