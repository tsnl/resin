"""
Interactive GLTF model viewer with free camera and model/environment selection.
"""

from pathlib import Path
import math

import numpy as np

import zfw


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
        "default": {
            "background": (0, 0, 0, 0),  # Transparent to show 3D behind
        },
    },
    ".hud-bar": {
        "default": {
            "background": (20, 20, 20, 180),  # Semi-transparent dark bar
            "padding-x": 16,
            "padding-y": 8,
        },
    },
    ".hud-label": {
        "default": {
            "background": (0, 0, 0, 0),
            "padding-x": 0,
            "padding-y": 0,
        },
    },
}


class FreeCameraController:
    """Free-look camera controller with WASD + mouse.

    Camera coordinate system (local): +X right, +Y forward, +Z up
    World coordinate system: +X right, +Y forward, +Z up
    """

    def __init__(
        self,
        *,
        position: tuple[float, float, float] = (0.0, -3.0, 0.0),
        yaw: float = 0.0,  # radians, rotation around Z axis (0 = looking along +Y)
        pitch: float = 0.0,  # radians, rotation around X axis (0 = horizontal, + = up, - = down)
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
        # Yaw: rotate around world Z axis (left/right look)
        self.yaw += dx * self.look_speed

        # Pitch: rotate around camera's local X axis (up/down look)
        # dy is positive when mouse moves down, so we subtract to make it intuitive
        self.pitch -= dy * self.look_speed

        # Clamp pitch to prevent flipping over
        self.pitch = np.clip(self.pitch, -math.pi / 2 + 0.01, math.pi / 2 - 0.01)

    def update(self, dt: float) -> None:
        """Update camera position based on movement keys."""
        # Movement directions in world space (horizontal XY plane)
        cos_yaw = math.cos(self.yaw)
        sin_yaw = math.sin(self.yaw)

        # Forward/backward in XY plane (ignoring pitch)
        forward = np.array([sin_yaw, cos_yaw, 0.0], dtype=np.float32)

        # Right is perpendicular to forward in XY plane
        right = np.array([cos_yaw, -sin_yaw, 0.0], dtype=np.float32)

        # Up is always world +Z
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        # Accumulate movement
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

        # Normalize and apply
        length = np.linalg.norm(move_dir)
        if length > 0:
            move_dir /= length
            self.position += move_dir * self.move_speed * dt

    def get_transform(self) -> np.ndarray:
        """Get the camera's 4x4 world-to-camera transform matrix.

        Returns a matrix where columns represent camera's local axes in world space:
        - Column 0: camera's +X (right) in world space
        - Column 1: camera's +Y (forward) in world space
        - Column 2: camera's +Z (up) in world space
        - Column 3: camera position in world space
        """
        cos_yaw = math.cos(self.yaw)
        sin_yaw = math.sin(self.yaw)
        cos_pitch = math.cos(self.pitch)
        sin_pitch = math.sin(self.pitch)

        # Camera's local X axis (right) in world space
        # Yaw rotates this in the XY plane
        right = np.array([cos_yaw, -sin_yaw, 0.0], dtype=np.float32)

        # Camera's local Y axis (forward/look direction) in world space
        # Combines yaw (horizontal rotation) and pitch (vertical tilt)
        forward = np.array(
            [sin_yaw * cos_pitch, cos_yaw * cos_pitch, sin_pitch],
            dtype=np.float32,
        )

        # Camera's local Z axis (up) in world space
        # Perpendicular to both right and forward (right-handed: X × Y = Z)
        up = np.cross(right, forward)

        # Build the transform matrix (camera-to-world)
        transform = np.eye(4, dtype=np.float32)
        transform[:3, 0] = right
        transform[:3, 1] = forward
        transform[:3, 2] = up
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
        self._camera = FreeCameraController(position=(0.0, -3.0, 0.0))
        self._mouse_captured = False
        self._last_mouse_x = 0.0
        self._last_mouse_y = 0.0

        # 3D resources
        self._resource_meshes: (
            dict[
                tuple[zfw.GeometryResource, zfw.MaterialResource],
                np.ndarray,
            ]
            | None
        ) = None
        self._meshes: (
            dict[
                tuple[zfw.Draw3dGeometry, zfw.Draw3dMaterial],
                np.ndarray,
            ]
            | None
        ) = None

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
            text="Click to look | WASD+Space/Shift to move | 1-9 change model | ESC to release mouse",
        )

        # Setup key event handler
        gui_window.set_key_event_callback(self._on_key_event)

        # Load initial model and environment
        self._load_model()

    def _load_model(self) -> None:
        """Load the currently selected model."""
        # Clear old meshes
        self._resource_meshes = None
        self._meshes = None

        # Load new scene
        model_name, model_file = MODELS[self._current_model_index]
        model_path = self._models_path / model_file

        if not model_path.exists():
            LOG.warning(f"Model not found: {model_path}")
            return

        LOG.info(f"Loading model: {model_name}")
        self._resource_meshes = zfw.load_gltf(model_path)

        # Convert resource types to Draw3d objects using renderer cache
        self._meshes = {}
        renderer = self._gui_window.draw_3d_renderer
        for (geom_res, mat_res), transforms in self._resource_meshes.items():
            geometry = renderer.get_geometry(geom_res)
            material = renderer.get_material(mat_res)
            self._meshes[(geometry, material)] = transforms

        # Update label
        self._model_label._text = f"Model: {model_name}"

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
        if self._meshes is not None:
            for (geometry, material), transforms in self._meshes.items():
                self._gui_window.add_3d_mesh(geometry, material, transforms)

        # Set camera
        aspect_ratio = self._gui_window.width_dip / self._gui_window.height_dip
        self._gui_window.set_3d_camera(
            transform=self._camera.get_transform(),
            intrinsics=zfw.Draw3dCamera(
                transform=self._camera.get_transform(),
                fov_y_rad=math.radians(60),
                aspect_ratio=aspect_ratio,
                max_distance=100.0,
            ),
        )


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
            "font_size": "regular",
            "padding": (5, 10, 5, 10),
            "text_horizontal_alignment": "left",
            "text_vertical_alignment": "middle",
        },
    },
}
