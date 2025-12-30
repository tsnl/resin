import argparse
import logging
from pathlib import Path

import zfw


from .gltf_viewer import GltfViewerWidget
from .universal_paperclips import (
    UniversalPaperclipsMainMenuWidget,
)


class MainMenuWidget(zfw.GuiWidget):
    def __init__(self, gui_window: zfw.GuiWindow):
        super().__init__(
            gui_window=gui_window,
            grid_rows=(100, -1, -1),
            grid_cols=(-1, -1, -1),
            style_classes=["central"],
        )
        self._gui_window = gui_window

        self._title_widget = zfw.GuiWidget(
            parent_widget=self,
            row=0,
            col=0,
            col_span=3,
            style_classes=["h1"],
            text="Main Menu",
        )

        self._universal_paperclips_button = zfw.GuiWidget(
            parent_widget=self,
            row=1,
            col=0,
            style_classes=["button"],
            text="Universal Paperclips",
        )

        self._gltf_viewer_button = zfw.GuiWidget(
            parent_widget=self,
            row=1,
            col=1,
            style_classes=["button"],
            text="GLTF Viewer",
        )

        self._universal_paperclips_widget = None
        self._gltf_viewer_widget = None

        @self._universal_paperclips_button.click_event.subscribe()
        def universal_paperclips_button_click(button: zfw.MouseButton):
            if self._universal_paperclips_widget is None:
                self._universal_paperclips_widget = UniversalPaperclipsMainMenuWidget(
                    gui_window=gui_window,
                )
            gui_window.push_central_widget(self._universal_paperclips_widget)

        @self._gltf_viewer_button.click_event.subscribe()
        def gltf_viewer_button_click(button: zfw.MouseButton):
            if self._gltf_viewer_widget is None:
                # Get paths for models and environments
                workspace_root = Path(__file__).parent.parent.parent.parent
                models_path = workspace_root / "tests_data" / "glTF-Sample-Assets" / "Models"
                environments_path = workspace_root / "tests_data" / "glTF-Sample-Environments"
                self._gltf_viewer_widget = GltfViewerWidget(
                    gui_window=gui_window,
                    models_path=models_path,
                    environments_path=environments_path,
                )
            gui_window.push_central_widget(self._gltf_viewer_widget)

    def _on_dispose(self) -> None:
        super()._on_dispose()
        self._title_widget.dispose()
        self._universal_paperclips_button.dispose()
        self._gltf_viewer_button.dispose()
        if self._universal_paperclips_widget is not None:
            self._universal_paperclips_widget.dispose()
        if self._gltf_viewer_widget is not None:
            self._gltf_viewer_widget.dispose()


def main():
    ap = argparse.ArgumentParser(description="ZFW Sandbox CLI")
    ap.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode (Vulkan validation layers, verbose debug logging, etc)",
    )
    args = ap.parse_args()

    # Setup logging
    zfw.setup_logging(
        level=logging.DEBUG if args.debug else logging.INFO,
        console=True,
        file=Path("zfw.log"),
    )

    LOG = zfw.logger(__name__)
    LOG.info(f"Starting ZFW Sandbox: debug={args.debug}")

    # Create GPU context
    gpu_context = zfw.GpuContext(
        app_name="ZFW Sandbox",
        enable_debug_layer_support=args.debug,
        enable_present_support=True,
    )

    # Create window context (manages GLFW)
    window_context = zfw.WindowContext()

    # Create window
    window = zfw.Window(
        gpu_context=gpu_context,
        window_context=window_context,
        width_dip=1280,
        height_dip=720,
        title="ZFW Sandbox",
    )

    # Create GPU device
    physical_device = next(iter(gpu_context.enumerate_physical_devices()))
    gpu_device = zfw.GpuDevice(
        context=gpu_context,
        physical_device=physical_device,
        surface=window.gpu_surface,
    )

    # Create 2D render context
    draw_2d_context = zfw.Draw2dContext(
        gpu_context=gpu_context,
    )

    # Create 3D render context
    draw_3d_context = zfw.Draw3dContext()

    # Create GUI window
    gui_window = zfw.GuiWindow(
        window=window,
        gpu_context=gpu_context,
        gpu_device=gpu_device,
        draw_2d_context=draw_2d_context,
        draw_3d_context=draw_3d_context,
        swapchain_image_count=3,
    )

    # Create main menu widget
    start_widget = MainMenuWidget(gui_window=gui_window)
    gui_window.set_central_widget(start_widget)
    gui_window.show()

    # Main loop
    while not gui_window.should_close():
        gui_window.update()
        gui_window.render()

    # Cleanup
    start_widget.dispose()
    gui_window.dispose()
    draw_3d_context.dispose()
    draw_2d_context.dispose()
    gpu_device.dispose()
    window.dispose()
    window_context.dispose()
    gpu_context.dispose()


if __name__ == "__main__":
    main()
