import argparse
import logging
from pathlib import Path

import wgpu
import resin


from .gltf_viewer import GltfViewerWidget
from .universal_paperclips import (
    UniversalPaperclipsMainMenuWidget,
)


class MainMenuWidget(resin.GuiWidget):
    def __init__(self, gui_window: resin.GuiWindow):
        super().__init__(
            gui_window=gui_window,
            grid_rows=(100, -1, -1),
            grid_cols=(-1, -1, -1),
            style_classes=["central"],
        )
        self._gui_window = gui_window

        self._title_widget = resin.GuiWidget(
            parent_widget=self,
            row=0,
            col=0,
            col_span=3,
            style_classes=["h1"],
            text="Main Menu",
        )

        self._universal_paperclips_button = resin.GuiWidget(
            parent_widget=self,
            row=1,
            col=0,
            style_classes=["button"],
            text="Universal Paperclips",
        )

        self._gltf_viewer_button = resin.GuiWidget(
            parent_widget=self,
            row=1,
            col=1,
            style_classes=["button"],
            text="GLTF Viewer",
        )

        self._universal_paperclips_widget = None
        self._gltf_viewer_widget = None

        @self._universal_paperclips_button.click_event.subscribe()
        def universal_paperclips_button_click(button: resin.MouseButton):
            if self._universal_paperclips_widget is None:
                self._universal_paperclips_widget = UniversalPaperclipsMainMenuWidget(
                    gui_window=gui_window,
                )
            gui_window.push_central_widget(self._universal_paperclips_widget)

        @self._gltf_viewer_button.click_event.subscribe()
        def gltf_viewer_button_click(button: resin.MouseButton):
            if self._gltf_viewer_widget is None:
                # Get paths for models and environments
                workspace_root = Path(__file__).parent.parent.parent.parent.parent
                models_path = (
                    workspace_root / "tests" / "data" / "glTF-Sample-Assets" / "Models"
                )
                environments_path = (
                    workspace_root / "tests" / "data" / "glTF-Sample-Environments"
                )
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
    ap = argparse.ArgumentParser(description="Resin Sandbox CLI")
    ap.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode (Vulkan validation layers, verbose debug logging, etc)",
    )
    args = ap.parse_args()

    # Setup logging
    resin.setup_logging(
        level=logging.DEBUG if args.debug else logging.INFO,
        console=True,
        file=Path("resin.log"),
    )

    LOG = resin.logger(__name__)
    LOG.info(f"Starting Resin Sandbox: debug={args.debug}")

    # Create WebGPU device
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    device = resin.help_request_wgpu_device(adapter)

    # Create window context (manages GLFW)
    window_context = resin.WindowContext()

    # Create window with device
    window = resin.Window(
        device=device,
        window_context=window_context,
        width_dip=1280,
        height_dip=720,
        title="Resin Sandbox",
    )

    # Create GUI window
    gui_window = resin.GuiWindow(
        window=window,
        device=device,
    )

    # Create main menu widget
    start_widget = MainMenuWidget(gui_window=gui_window)
    gui_window.set_central_widget(start_widget)
    window.show()

    # Main loop
    while not window.should_close():
        gui_window.update()
        gui_window.render()
        resin.Window.poll_events()

    # Cleanup
    start_widget.dispose()
    gui_window.dispose()
    window.dispose()
    window_context.dispose()


if __name__ == "__main__":
    main()
