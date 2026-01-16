"""
Interactive GLTF model viewer with free camera and model/environment selection.

Uses immediate-mode GUI for the settings panel overlay.
"""

import logging
import math
from pathlib import Path

import numpy as np
import numpy.typing as npt
import wgpu

import resin
from resin import gui


LOG = logging.getLogger(__name__)


# Available models
MODELS = [
    ("Avocado", "Avocado/glTF-Binary/Avocado.glb"),
    ("Damaged Helmet", "DamagedHelmet/glTF/DamagedHelmet.gltf"),
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
    ("Neutral", "neutral.hdr"),
    ("Chromatic", "chromatic.hdr"),
    ("Directional", "directional.hdr"),
    ("Footprint Court", "footprint_court.hdr"),
    ("Pisa", "pisa.hdr"),
    ("Papermill", "papermill.hdr"),
    ("Helipad", "helipad.hdr"),
    ("Field", "field.hdr"),
    ("Doge 2", "doge2.hdr"),
    ("Ennis", "ennis.hdr"),
]


class FreeCameraController:
    """Free-look camera controller with WASD + mouse.

    Camera coordinate system (local): +X right, +Y forward, +Z up
    World coordinate system: +X right, +Y forward, +Z up
    """

    def __init__(
        self,
        *,
        position: tuple[float, float, float] = (0.0, -3.0, 0.0),
        yaw: float = 0.0,
        pitch: float = 0.0,
        move_speed: float = 2.0,
        look_speed: float = 0.002,
    ):
        self.position = np.array(position, dtype=np.float32)
        self.yaw = yaw
        self.pitch = pitch
        self.move_speed = move_speed
        self.look_speed = look_speed
        self.moved_this_frame = False

    def update(self, input_state: gui.InputState, dt: float) -> None:
        """Update camera based on input state."""
        # Movement directions in world space
        cos_yaw = math.cos(self.yaw)
        sin_yaw = math.sin(self.yaw)

        forward = np.array([sin_yaw, cos_yaw, 0.0], dtype=np.float32)
        right = np.array([cos_yaw, -sin_yaw, 0.0], dtype=np.float32)
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        # Accumulate movement
        move_dir = np.zeros(3, dtype=np.float32)
        if "w" in input_state.keys_down:
            move_dir += forward
        if "s" in input_state.keys_down:
            move_dir -= forward
        if "d" in input_state.keys_down:
            move_dir += right
        if "a" in input_state.keys_down:
            move_dir -= right
        if "space" in input_state.keys_down:
            move_dir += up
        if "left-shift" in input_state.keys_down:
            move_dir -= up

        # Normalize and apply
        length = float(np.linalg.norm(move_dir))
        if length > 0:
            move_dir /= length
            self.position += move_dir * self.move_speed * dt
            self.moved_this_frame = True

    def handle_mouse_look(self, dx: float, dy: float) -> None:
        """Handle mouse movement for looking around."""
        if dx != 0 or dy != 0:
            self.moved_this_frame = True

        self.yaw += dx * self.look_speed
        self.pitch -= dy * self.look_speed
        self.pitch = float(np.clip(self.pitch, -math.pi / 2 + 0.01, math.pi / 2 - 0.01))

    def get_transform(self) -> npt.NDArray[np.float32]:
        """Get camera-to-world transform matrix."""
        cos_yaw = math.cos(self.yaw)
        sin_yaw = math.sin(self.yaw)
        cos_pitch = math.cos(self.pitch)
        sin_pitch = math.sin(self.pitch)

        right = np.array([cos_yaw, -sin_yaw, 0.0], dtype=np.float32)
        forward = np.array(
            [sin_yaw * cos_pitch, cos_yaw * cos_pitch, sin_pitch],
            dtype=np.float32,
        )
        up = np.cross(right, forward)

        transform = np.eye(4, dtype=np.float32)
        transform[:3, 0] = right
        transform[:3, 1] = forward
        transform[:3, 2] = up
        transform[:3, 3] = self.position

        return transform


# Semi-transparent style for HUD overlay
_OVERLAY_STYLE = gui.GuiStyle(
    bg_color=(0.1, 0.1, 0.1, 0.85),
    bg_color_hover=(0.2, 0.2, 0.2, 0.9),
    bg_color_active=(0.08, 0.08, 0.08, 0.9),
    input_bg_color=(0.05, 0.05, 0.05, 0.9),
    combo_dropdown_bg=(0.12, 0.12, 0.12, 0.95),
    window_padding=12,
    item_spacing=6,
    label_width=80,
)


class GltfViewer:
    """GLTF model viewer with immediate-mode GUI overlay."""

    def __init__(
        self,
        *,
        window: resin.Window,
        device: wgpu.GPUDevice,
        models_path: Path,
        environments_path: Path,
    ):
        self._window = window
        self._device = device
        self._queue = device.queue
        self._models_path = models_path
        self._environments_path = environments_path

        # Renderers (created in _create_renderers)
        self._framebuffer_size = (window.width_px, window.height_px)
        self._draw_2d_renderer: resin.Draw2dRenderer | None = None
        self._draw_2d_canvas = resin.Draw2dExtCanvas(device=device, queue=self._queue)
        self._draw_3d_renderer: resin.Draw3dRenderer | None = None
        self._blit_renderer = resin.BlitRenderer(device=device)
        self._create_renderers()

        # GUI input state
        self._input = gui.InputState()
        self._setup_input_callbacks()

        # Camera
        self._camera = FreeCameraController(position=(0.0, -5.0, 1.0))
        self._last_mouse_x = 0.0
        self._last_mouse_y = 0.0

        # Viewer state
        self._current_model = 1
        self._current_env = 0
        self._exposure = 1.0
        self._show_settings = True

        # 3D resources
        self._meshes: (
            dict[
                tuple[resin.Draw3dGeometry, resin.Draw3dMaterial],
                npt.NDArray[np.float32],
            ]
            | None
        ) = None
        self._environment_texture: resin.Draw3dTexture | None = None

        # Load initial content
        self._load_model()
        self._load_environment()

    def _create_renderers(self) -> None:
        """Create or recreate renderers at current framebuffer size."""
        size = self._framebuffer_size

        # Dispose old renderers if they exist
        if self._draw_3d_renderer is not None:
            self._draw_3d_renderer.dispose()

        self._draw_2d_renderer = resin.Draw2dRenderer(
            self._device, self._queue, size, target_format="rgba8unorm-srgb"
        )
        self._draw_3d_renderer = resin.Draw3dRenderer(
            device=self._device, queue=self._queue, target_size_wh_px=size
        )

    def _setup_input_callbacks(self) -> None:
        """Wire up window callbacks to input state."""
        self._window.set_mouse_button_callback(self._on_mouse_button)
        self._window.set_cursor_pos_callback(self._on_cursor_pos)
        self._window.set_key_callback(self._input.on_key)
        self._window.set_char_callback(self._input.on_char)
        self._window.set_scroll_callback(self._input.on_scroll)
        self._window.set_framebuffer_size_callback(self._on_framebuffer_resize)

    def _on_mouse_button(
        self,
        button: resin.MouseButton,
        action: resin.ButtonAction,
        mods: list[resin.KeyModifier],
    ) -> None:
        """Handle mouse button events."""
        self._input.on_mouse_button(button, action, mods)

    def _on_cursor_pos(self, x: float, y: float) -> None:
        """Handle cursor position with click-and-drag camera look."""
        dx = x - self._last_mouse_x
        dy = y - self._last_mouse_y
        self._last_mouse_x = x
        self._last_mouse_y = y

        self._input.on_cursor_pos(x, y)

        # Camera look when left mouse button is held
        if self._input.mouse_down.get("left", False):
            self._camera.handle_mouse_look(dx, dy)

    def _on_framebuffer_resize(self, width: int, height: int) -> None:
        """Handle framebuffer resize by recreating renderers."""
        if width == 0 or height == 0:
            return

        new_size = (width, height)
        if new_size == self._framebuffer_size:
            return

        LOG.info(f"Resizing to {width}x{height}")
        self._framebuffer_size = new_size

        # Recreate renderers at new size
        self._create_renderers()

        # Reload resources (they're tied to the renderer)
        self._load_model()
        self._load_environment()

    def _load_model(self) -> None:
        """Load the currently selected model."""
        assert self._draw_3d_renderer is not None
        self._meshes = None

        model_name, model_file = MODELS[self._current_model]
        model_path = self._models_path / model_file

        if not model_path.exists():
            LOG.warning(f"Model not found: {model_path}")
            return

        LOG.info(f"Loading model: {model_name}")
        resource_meshes = resin.load_gltf(model_path)

        self._meshes = {}
        for (geom_res, mat_res), transforms in resource_meshes.items():
            geometry = resin.Draw3dGeometry.from_resource(
                geom_res, self._draw_3d_renderer
            )
            material = resin.Draw3dMaterial.from_resource(
                mat_res, self._draw_3d_renderer
            )
            self._meshes[(geometry, material)] = transforms

    def _load_environment(self) -> None:
        """Load the currently selected environment."""
        assert self._draw_3d_renderer is not None
        env_name, env_file = ENVIRONMENTS[self._current_env]
        env_path = self._environments_path / env_file

        if not env_path.exists():
            LOG.warning(f"Environment not found: {env_path}")
            self._environment_texture = None
            return

        LOG.info(f"Loading environment: {env_name}")
        env_data = resin.load_image(image_path=env_path)
        self._environment_texture = resin.Draw3dTexture(
            self._draw_3d_renderer, data=env_data, usage="environment"
        )

        # Reset renderer accumulator
        self._draw_3d_renderer.reset(
            geometry_heap=False,
            material_heap=False,
            texture_heap=False,
            per_frame_state=True,
        )

    def run(self, dt: float = 1 / 60) -> None:
        """Run one frame of the viewer."""
        self._input.begin_frame()
        self._window.poll_events()
        self._handle_input()
        self._update(dt)
        self._render()

    def _handle_input(self) -> None:
        """Handle input events."""
        # Tab toggles settings panel
        if "tab" in self._input.keys_pressed:
            self._show_settings = not self._show_settings

    def _update(self, dt: float) -> None:
        """Update viewer state and GUI."""
        assert self._draw_3d_renderer is not None

        # Update camera
        self._camera.update(self._input, dt)

        if self._camera.moved_this_frame:
            self._draw_3d_renderer.reset(
                geometry_heap=False,
                material_heap=False,
                texture_heap=False,
                per_frame_state=True,
            )
            self._camera.moved_this_frame = False

        # Build GUI
        self._gui_primitives: list[resin.Draw2dExtBasePrimitive] = []

        if self._show_settings:
            with gui.window(
                width=self._window.width_dip,
                height=self._window.height_dip,
                input_state=self._input,
                style=_OVERLAY_STYLE,
            ) as g:
                g.label("GLTF Viewer")
                g.separator()

                # Model selection
                new_model = g.combo(
                    "Model", self._current_model, [m[0] for m in MODELS]
                )
                if new_model != self._current_model:
                    self._current_model = new_model
                    self._load_model()

                # Environment selection
                new_env = g.combo(
                    "Env", self._current_env, [e[0] for e in ENVIRONMENTS]
                )
                if new_env != self._current_env:
                    self._current_env = new_env
                    self._load_environment()

                g.space(8)

                # Camera controls
                self._exposure = g.slider_float("Exposure", self._exposure, 0.1, 5.0)
                self._camera.move_speed = g.slider_float(
                    "Move Speed", self._camera.move_speed, 0.5, 10.0
                )

                g.space(16)
                g.separator()
                g.space(4)

                # Help text
                g.label("Controls:")
                g.label("  WASD + Space/Shift - Move")
                g.label("  LMB + Drag - Look around")
                g.label("  TAB - Toggle this panel")

            self._gui_primitives = g.primitives

    def _render(self) -> None:
        """Render 3D scene and GUI overlay."""
        assert self._draw_2d_renderer is not None
        assert self._draw_3d_renderer is not None

        current_texture = self._window.canvas_context.get_current_texture()
        if current_texture is None:
            return

        command_encoder = self._device.create_command_encoder()

        # 3D rendering
        aspect_ratio = self._window.width_dip / self._window.height_dip
        camera = resin.Draw3dCamera(
            transform=self._camera.get_transform(),
            fov_y_rad=math.radians(60),
            aspect_ratio=aspect_ratio,
            max_distance=100.0,
        )

        scene = resin.Draw3dScene(
            camera=camera,
            meshes=self._meshes or {},
            environment_map=self._environment_texture,
        )

        self._draw_3d_renderer.record(
            scene=scene,
            command_encoder=command_encoder,
        )

        # Composite: 3D background + 2D GUI overlay using Draw2dRenderer's alpha blending
        scale = self._window.content_scale[0]
        size = (self._window.width_px, self._window.height_px)

        # Full-screen quad with 3D output as texture (background layer)
        background_quad = resin.Draw2dQuad(
            dst_xy_px=(0, 0),
            dst_wh_px=size,
            fill_texture=self._draw_3d_renderer.get_output_image(),
        )

        # GUI overlay quads
        gui_quads = self._draw_2d_canvas.quads(
            primitives=self._gui_primitives, scale=scale
        )

        # Render 3D background first, then GUI on top (alpha blended)
        self._draw_2d_renderer.record(
            quads=[background_quad] + gui_quads,
            command_encoder=command_encoder,
        )

        # Present final composited image to screen
        self._blit_renderer.record(
            input_texture=self._draw_2d_renderer.get_output_image(),
            output_texture=current_texture,
            command_encoder=command_encoder,
        )

        self._queue.submit([command_encoder.finish()])
        self._window.canvas_context.present()

    def dispose(self) -> None:
        """Clean up resources."""
        self._draw_2d_canvas.dispose()
        if self._draw_3d_renderer is not None:
            self._draw_3d_renderer.dispose()
