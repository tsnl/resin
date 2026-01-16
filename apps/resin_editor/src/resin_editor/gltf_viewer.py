"""
Interactive GLTF model viewer with free camera and model/environment selection.

Uses immediate-mode GUI with GuiWindow for the settings panel and 3D viewport.
"""

import logging
import math
from pathlib import Path

import numpy as np
import numpy.typing as npt
import wgpu

import resin
from resin import gui, trace


LOG = logging.getLogger(__name__)


# Available models
MODELS = [
    ("Sponza", "Sponza/glTF/Sponza.gltf"),
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


# Style for settings panel
_PANEL_STYLE = gui.GuiStyle(
    window_bg_color=None,  # No background for horizontal root
    bg_color=(0.15, 0.15, 0.15, 0.95),
    bg_color_hover=(0.25, 0.25, 0.25, 0.95),
    bg_color_active=(0.1, 0.1, 0.1, 0.95),
    input_bg_color=(0.05, 0.05, 0.05, 0.9),
    combo_dropdown_bg=(0.12, 0.12, 0.12, 0.95),
    window_padding=0,
    item_spacing=0,
    label_width=80,
)


class GltfViewer:
    """GLTF model viewer with immediate-mode GUI and 3D viewport."""

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

        # GUI window (handles 2D rendering and input)
        self._gui_window = gui.GuiWindow(
            window, device, self._queue, style=_PANEL_STYLE
        )

        # 3D renderer (created in _create_3d_renderer)
        self._framebuffer_size = (window.width_px, window.height_px)
        self._draw_3d_renderer: resin.Draw3dRenderer | None = None
        self._create_3d_renderer()

        # Camera
        self._camera = FreeCameraController(position=(0.0, -5.0, 1.0))
        self._last_mouse_x = 0.0
        self._last_mouse_y = 0.0

        # Viewer state
        self._current_model = 0  # Default to Sponza for benchmarking
        self._current_env = 0
        self._exposure = 1.0
        self._show_settings = True
        self._panel_width = 220

        # Render settings
        self._samples_per_pixel = 1
        self._max_bounces = 3

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

        # Set up resize callback
        self._window.set_framebuffer_size_callback(self._on_framebuffer_resize)

    def _create_3d_renderer(self) -> None:
        """Create 3D renderer at current framebuffer size."""
        self._draw_3d_renderer = resin.Draw3dRenderer(
            device=self._device,
            queue=self._queue,
            target_size_wh_px=self._framebuffer_size,
        )

    def _on_framebuffer_resize(self, width: int, height: int) -> None:
        """Handle framebuffer resize."""
        if width == 0 or height == 0:
            return

        new_size = (width, height)
        if new_size == self._framebuffer_size:
            return

        LOG.info(f"Resizing to {width}x{height}")
        self._framebuffer_size = new_size

        # Resize 3D renderer (preserves geometry, materials, textures)
        if self._draw_3d_renderer is not None:
            self._draw_3d_renderer.resize(new_size)

    def _load_model(self) -> None:
        """Load the currently selected model."""
        with trace.span("GltfViewer/load_model", "viewer"):
            self._load_model_impl()

    def _load_model_impl(self) -> None:
        """Load the currently selected model implementation."""
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
        with trace.span("GltfViewer/load_environment", "viewer"):
            self._load_environment_impl()

    def _load_environment_impl(self) -> None:
        """Load the currently selected environment implementation."""
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
        assert self._draw_3d_renderer is not None

        # GUI frame with horizontal layout
        # Note: viewport() automatically resizes the 3D renderer to match
        with self._gui_window.frame() as g:
            # Left panel: settings (vertical layout with fixed width)
            if self._show_settings:
                with g.vertical(width=self._panel_width):
                    self._draw_settings_panel(g)

            # Main viewport: 3D scene (fills remaining space)
            viewport_input = g.viewport("main", self._draw_3d_renderer)

            # Handle viewport input
            self._handle_viewport_input(viewport_input, dt)

        # Handle global input (tab to toggle panel)
        if "tab" in self._gui_window.input.keys_pressed:
            self._show_settings = not self._show_settings

        # Render 3D scene
        self._render_3d()

        # Present everything
        self._present()

    def _draw_settings_panel(self, g: gui.Gui) -> None:
        """Draw the settings panel widgets."""
        # Panel background
        g._draw_rect(
            0,
            0,
            self._panel_width,
            self._window.height_dip,
            (0.08, 0.08, 0.08, 0.9),
        )

        # Add padding
        g.space(8)
        g.indent(8)

        g.label("GLTF Viewer")
        g.separator()

        # Model selection
        new_model = g.combo("Model", self._current_model, [m[0] for m in MODELS])
        if new_model != self._current_model:
            self._current_model = new_model
            self._load_model()

        # Environment selection
        new_env = g.combo("Env", self._current_env, [e[0] for e in ENVIRONMENTS])
        if new_env != self._current_env:
            self._current_env = new_env
            self._load_environment()

        g.space(8)

        # Camera controls
        self._exposure = g.slider_float("Exposure", self._exposure, 0.1, 5.0)
        self._camera.move_speed = g.slider_float(
            "Move Speed", self._camera.move_speed, 0.5, 10.0
        )

        g.space(8)
        g.separator()
        g.space(4)
        g.label("Render Settings:")

        # Samples per pixel
        new_spp = g.slider_int("SPP", self._samples_per_pixel, 1, 16)
        if new_spp != self._samples_per_pixel:
            self._samples_per_pixel = new_spp
            assert self._draw_3d_renderer is not None
            self._draw_3d_renderer.set_render_settings(samples_per_pixel=new_spp)

        # Max bounces
        new_bounces = g.slider_int("Bounces", self._max_bounces, 1, 8)
        if new_bounces != self._max_bounces:
            self._max_bounces = new_bounces
            assert self._draw_3d_renderer is not None
            self._draw_3d_renderer.set_render_settings(max_bounces=new_bounces)

        g.space(16)
        g.separator()
        g.space(4)

        # Help text
        g.label("Controls:")
        g.label("  WASD + Space/Shift - Move")
        g.label("  LMB + Drag - Look around")
        g.label("  TAB - Toggle this panel")

        g.unindent(8)

    def _handle_viewport_input(self, input_state: gui.InputState, dt: float) -> None:
        """Handle input for the 3D viewport."""
        assert self._draw_3d_renderer is not None

        # Camera movement (keyboard)
        self._camera.update(input_state, dt)

        # Camera look (mouse drag)
        if input_state.mouse_down.get("left", False):
            dx = input_state.mouse_x - self._last_mouse_x
            dy = input_state.mouse_y - self._last_mouse_y
            self._camera.handle_mouse_look(dx, dy)

        self._last_mouse_x = input_state.mouse_x
        self._last_mouse_y = input_state.mouse_y

        # Reset accumulator if camera moved
        if self._camera.moved_this_frame:
            self._draw_3d_renderer.reset(
                geometry_heap=False,
                material_heap=False,
                texture_heap=False,
                per_frame_state=True,
            )
            self._camera.moved_this_frame = False

    @trace.decorator("GltfViewer/render_3d", "viewer")
    def _render_3d(self) -> None:
        """Render the 3D scene."""
        assert self._draw_3d_renderer is not None

        command_encoder = self._device.create_command_encoder()

        # Use render target aspect ratio (not window) since viewport may be smaller
        target_w, target_h = self._draw_3d_renderer.target_size_wh_px
        aspect_ratio = target_w / target_h
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

        self._queue.submit([command_encoder.finish()])

    @trace.decorator("GltfViewer/present", "viewer")
    def _present(self) -> None:
        """Present the final image to screen."""
        self._gui_window.present()

    def dispose(self) -> None:
        """Clean up resources."""
        self._gui_window.dispose()
        if self._draw_3d_renderer is not None:
            self._draw_3d_renderer.dispose()
