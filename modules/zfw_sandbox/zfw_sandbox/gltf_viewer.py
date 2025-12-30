"""
Interactive GLTF model viewer with free camera and model/environment selection.
"""

from pathlib import Path
import math

import numpy as np

import zfw
from zfw.loader import load_gltf, GltfScene


LOG = zfw.logger(__name__)


# Available models
MODELS = [
    ("Avocado", "Avocado/glTF-Binary/Avocado.glb"),
    ("Damaged Helmet", "DamagedHelmet/glTF-Binary/DamagedHelmet.glb"),
    ("Flight Helmet", "FlightHelmet/glTF/FlightHelmet.gltf"),
    ("Water Bottle", "WaterBottle/glTF-Binary/WaterBottle.glb"),
    ("Sci-Fi Helmet", "SciFiHelmet/glTF/SciFiHelmet.gltf"),
    ("Lantern", "Lantern/glTF-Binary/Lantern.glb"),
    ("Antique Camera", "AntiqueCamera/glTF-Binary/AntiqueCamera.glb"),
    ("Boom Box", "BoomBox/glTF-Binary/BoomBox.glb"),
    ("Corset", "Corset/glTF-Binary/Corset.glb"),
    ("Duck", "Duck/glTF-Binary/Duck.glb"),
]

# Available environments
ENVIRONMENTS = [
    ("Neutral", "neutral.jpg"),
    ("Chromatic", "chromatic.jpg"),
    ("Directional", "directional.jpg"),
    ("Footprint Court", "footprint_court.jpg"),
    ("Pisa", "pisa.jpg"),
    ("Papermill", "papermill.jpg"),
    ("Helipad", "helipad.jpg"),
    ("Field", "field.jpg"),
    ("Doge 2", "doge2.jpg"),
    ("Ennis", "ennis.jpg"),
]

# Theme for the viewer HUD
_VIEWER_THEME: zfw.GuiTheme = {
    ".central": {
        "background": (0, 0, 0, 0),  # Transparent to show 3D behind
    },
    ".hud-bar": {
        "background": (20, 20, 20, 180),  # Semi-transparent dark bar
        "padding-x": 16,
        "padding-y": 8,
    },
    ".hud-label": {
        "background": (0, 0, 0, 0),
        "padding-x": 0,
        "padding-y": 0,
    },
}


class FreeCameraController:
    """Free-look camera controller with WASD + mouse."""

    def __init__(
        self,
        *,
        position: tuple[float, float, float] = (0.0, 0.0, 3.0),
        yaw: float = 0.0,  # radians, looking along -Z when 0
        pitch: float = 0.0,  # radians, looking forward when 0
        move_speed: float = 2.0,
        look_speed: float = 0.002,
    ):
        self.position = np.array(position, dtype=np.float32)
        self.yaw = yaw
        self.pitch = pitch
        self.move_speed = move_speed
        self.look_speed = look_speed

        # Movement state
        self.forward_pressed = False
        self.backward_pressed = False
        self.left_pressed = False
        self.right_pressed = False
        self.up_pressed = False
        self.down_pressed = False

    def handle_key(
        self,
        key: zfw.Key | None,
        action: zfw.ButtonAction,
    ) -> bool:
        """Handle key press/release. Returns True if key was consumed."""
        if key is None:
            return False

        pressed = action == "press"
        released = action == "release"

        if key == "w":
            if pressed:
                self.forward_pressed = True
            elif released:
                self.forward_pressed = False
            return True
        elif key == "s":
            if pressed:
                self.backward_pressed = True
            elif released:
                self.backward_pressed = False
            return True
        elif key == "a":
            if pressed:
                self.left_pressed = True
            elif released:
                self.left_pressed = False
            return True
        elif key == "d":
            if pressed:
                self.right_pressed = True
            elif released:
                self.right_pressed = False
            return True
        elif key == "space":
            if pressed:
                self.up_pressed = True
            elif released:
                self.up_pressed = False
            return True
        elif key == "left-shift":
            if pressed:
                self.down_pressed = True
            elif released:
                self.down_pressed = False
            return True

        return False

    def handle_mouse_move(self, dx: float, dy: float) -> None:
        """Handle mouse movement for looking around."""
        self.yaw -= dx * self.look_speed
        self.pitch -= dy * self.look_speed

        # Clamp pitch to avoid gimbal lock
        max_pitch = math.pi / 2 - 0.01
        self.pitch = max(-max_pitch, min(max_pitch, self.pitch))

    def update(self, dt: float) -> None:
        """Update camera position based on movement keys."""
        # Compute forward and right vectors
        forward = np.array(
            [
                math.cos(self.pitch) * math.sin(self.yaw),
                -math.sin(self.pitch),
                -math.cos(self.pitch) * math.cos(self.yaw),
            ],
            dtype=np.float32,
        )
        right = np.array(
            [math.cos(self.yaw), 0.0, math.sin(self.yaw)],
            dtype=np.float32,
        )
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        # Compute movement direction
        move_dir = np.zeros(3, dtype=np.float32)
        if self.forward_pressed:
            move_dir += forward
        if self.backward_pressed:
            move_dir -= forward
        if self.right_pressed:
            move_dir += right
        if self.left_pressed:
            move_dir -= right
        if self.up_pressed:
            move_dir += up
        if self.down_pressed:
            move_dir -= up

        # Normalize and apply movement
        length = np.linalg.norm(move_dir)
        if length > 0:
            move_dir /= length
            self.position += move_dir * self.move_speed * dt

    def get_transform(self) -> np.ndarray:
        """Get the camera's 4x4 world transform matrix (row-major)."""
        # Compute rotation matrix from yaw and pitch
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)

        # Camera axes (forward is -Z in camera space)
        right = np.array([cy, 0.0, sy])
        up = np.array([sy * sp, cp, -cy * sp])
        forward = np.array([sy * cp, -sp, -cy * cp])

        # Build transform matrix (row-major)
        transform = np.eye(4, dtype=np.float32)
        transform[0, :3] = right
        transform[1, :3] = up
        transform[2, :3] = -forward  # Camera looks along -Z
        transform[:3, 3] = self.position

        return transform


class GltfViewerWidget(zfw.GuiWidget):
    """Main GLTF viewer widget with HUD overlay."""

    def __init__(
        self,
        *,
        gui_window: zfw.GuiWindow,
        models_path: Path,
        environments_path: Path,
    ):
        # Transparent central style to see 3D behind
        super().__init__(
            gui_window=gui_window,
            grid_rows=(40, -1, 40),
            grid_cols=(-1,),
            style_classes=["central"],
            theme=_VIEWER_THEME,
        )

        self._gui_window = gui_window
        self._models_path = models_path
        self._environments_path = environments_path

        # State
        self._current_model_index = 0
        self._current_env_index = 0
        self._camera = FreeCameraController(position=(0.0, 0.0, 3.0))
        self._mouse_captured = False
        self._last_mouse_x = 0.0
        self._last_mouse_y = 0.0

        # 3D resources
        self._scene: GltfScene | None = None
        self._environment_map: zfw.GpuImage | None = None

        # HUD widgets
        self._top_bar = zfw.GuiWidget(
            parent_widget=self,
            row=0,
            col=0,
            grid_rows=(-1,),
            grid_cols=(200, -1, 200),
            style_classes=["hud-bar"],
        )

        self._model_label = zfw.GuiWidget(
            parent_widget=self._top_bar,
            row=0,
            col=0,
            style_classes=["hud-label"],
            text=f"Model: {MODELS[self._current_model_index][0]}",
        )

        self._env_label = zfw.GuiWidget(
            parent_widget=self._top_bar,
            row=0,
            col=2,
            style_classes=["hud-label"],
            text=f"Env: {ENVIRONMENTS[self._current_env_index][0]}",
        )

        self._bottom_bar = zfw.GuiWidget(
            parent_widget=self,
            row=2,
            col=0,
            style_classes=["hud-bar"],
        )

        self._help_label = zfw.GuiWidget(
            parent_widget=self._bottom_bar,
            row=0,
            col=0,
            style_classes=["hud-label"],
            text="Click to look | WASD+Space/Shift to move | 1-9 change model | F1-F9 change env | ESC to release mouse",
        )

        # Setup key event handler
        gui_window.set_key_event_callback(self._on_key_event)

        # Load initial model and environment
        self._load_model()
        self._load_environment()

    def _load_model(self) -> None:
        """Load the currently selected model."""
        # Dispose old scene
        if self._scene is not None:
            self._scene.dispose()
            self._scene = None

        # Load new scene
        model_name, model_file = MODELS[self._current_model_index]
        model_path = self._models_path / model_file

        if not model_path.exists():
            LOG.warning(f"Model not found: {model_path}")
            return

        LOG.info(f"Loading model: {model_name}")
        scenes = load_gltf(
            self._gui_window.draw_3d_renderer,
            model_path,
            transform_coordinate_system=False,
        )
        if scenes:
            self._scene = scenes[0]

        # Update label
        self._model_label._text = f"Model: {model_name}"

    def _load_environment(self) -> None:
        """Load the currently selected environment map."""
        # Dispose old environment
        if self._environment_map is not None:
            self._environment_map.dispose()
            self._environment_map = None

        # Load new environment
        env_name, env_file = ENVIRONMENTS[self._current_env_index]
        env_path = self._environments_path / env_file

        if not env_path.exists():
            LOG.warning(f"Environment not found: {env_path}")
            return

        LOG.info(f"Loading environment: {env_name}")
        env_pixels = zfw.load_rgba_image(env_path)
        env_height, env_width = env_pixels.shape[:2]

        # Convert from float32 linear to uint8 for GPU upload
        env_pixels_u8 = (np.clip(env_pixels, 0.0, 1.0) * 255).astype(np.uint8)

        self._environment_map = zfw.GpuImage(
            device=self._gui_window.gpu_device,
            usages=["texture-binding", "transfer-dst"],
            meta=zfw.GpuImageMeta(shape=(env_height, env_width, 4), dtype="u1"),
        )

        # Upload pixels
        self._environment_map.write(data=env_pixels_u8)

        # Update GUI
        self._gui_window.set_environment_map(self._environment_map)

        # Update label
        self._env_label._text = f"Env: {env_name}"

    def _on_key_event(
        self,
        key: zfw.Key | None,
        scancode: int,
        action: zfw.ButtonAction,
        mods: list[zfw.KeyModifier],
    ) -> None:
        # Handle camera movement
        if self._camera.handle_key(key, action):
            return

        if action != "press":
            return

        # Handle model switching (1-9)
        if key in ("1", "2", "3", "4", "5", "6", "7", "8", "9"):
            idx = int(key) - 1
            if idx < len(MODELS):
                self._current_model_index = idx
                self._load_model()
            return

        # Handle environment switching (F1-F9)
        if key is not None and key.startswith("f") and len(key) <= 2:
            try:
                idx = int(key[1:]) - 1
                if 0 <= idx < len(ENVIRONMENTS):
                    self._current_env_index = idx
                    self._load_environment()
            except ValueError:
                pass
            return

        # Handle escape to release mouse
        if key == "escape":
            if self._mouse_captured:
                self._mouse_captured = False
                self._gui_window.set_cursor_mode("cursor")
            return

    def _receive_mouse_button_action(
        self,
        button: zfw.MouseButton,
        action: zfw.ButtonAction,
        click_handled: bool,
    ) -> bool:
        # Capture mouse on click
        if button == "left" and action == "press" and not self._mouse_captured:
            self._mouse_captured = True
            self._gui_window.set_cursor_mode("joystick")
            return True

        return super()._receive_mouse_button_action(button, action, click_handled)

    def _receive_mouse_position_change(
        self, mouse_x_dip: int, mouse_y_dip: int
    ) -> None:
        # Handle mouse look when captured
        if self._mouse_captured:
            dx = mouse_x_dip - self._last_mouse_x
            dy = mouse_y_dip - self._last_mouse_y
            self._camera.handle_mouse_move(dx, dy)

        self._last_mouse_x = mouse_x_dip
        self._last_mouse_y = mouse_y_dip

        super()._receive_mouse_position_change(mouse_x_dip, mouse_y_dip)

    def _update_self(self, dt: float) -> None:
        """Update camera and 3D rendering."""
        # Update camera
        self._camera.update(dt)

        # Clear and add meshes
        self._gui_window.clear_3d_meshes()
        if self._scene is not None:
            for (geometry, material), transforms in self._scene.meshes.items():
                self._gui_window.add_3d_mesh(geometry, material, transforms)

        # Set camera
        self._gui_window.set_3d_camera(
            transform=self._camera.get_transform(),
            intrinsics=zfw.Draw3dCameraIntrinsics(
                vertical_fov=math.radians(60),
                clip_near=0.01,
                clip_far=100.0,
                ibl_samples=16,
            ),
        )

    def _on_dispose(self) -> None:
        if self._scene is not None:
            self._scene.dispose()
        if self._environment_map is not None:
            self._environment_map.dispose()
        super()._on_dispose()


_VIEWER_THEME: zfw.GuiTheme = {
    "central": {
        "default": {
            "bg_color": (0.0, 0.0, 0.0, 0.0),  # Transparent
        },
    },
    "hud-bar": {
        "default": {
            "bg_color": (0.0, 0.0, 0.0, 0.5),  # Semi-transparent black
        },
    },
    "hud-label": {
        "default": {
            "bg_color": (0.0, 0.0, 0.0, 0.0),
            "fg_color": (1.0, 1.0, 1.0, 1.0),
            "font": "monospaced",
            "font_size_dip": 14,
            "padding": (5, 10, 5, 10),
            "text_horizontal_alignment": "left",
            "text_vertical_alignment": "middle",
        },
    },
}
