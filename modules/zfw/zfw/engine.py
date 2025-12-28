from pathlib import Path
import sys
import logging

from .basic import BaseResource, SupportsWrite, expect, logger, setup_logging
from .gpu import GpuContext, GpuDevice, GpuSwapChain, GpuCommandEncoder
from .draw_2d import Draw2dContext, Draw2dRenderer
from .gui import GuiWindow, GuiContext, GuiTheme


LOG = logger(__name__)


class Engine(BaseResource):
    _gpu_context: GpuContext
    _gui_context: GuiContext | None
    _render_context: Draw2dContext
    _window: GuiWindow | None
    _gpu_device: GpuDevice
    _renderer: Draw2dRenderer
    _rendered_frame_count: int

    def __init__(
        self,
        *,
        app_name: str,
        debug: bool,
        swapchain_image_count: int,
        enable_gui: bool = True,
        gui_theme: GuiTheme | None = None,
    ):
        super().__init__(parent_resource=None)

        # Setup logging:
        setup_logging(
            level=logging.DEBUG if debug else logging.INFO,
            console=True,
            file=Path("zfw.log"),
        )
        LOG.info(f"Starting ZFW Engine: {app_name=}, {debug=}, {enable_gui=}")

        # Create contexts:
        self._gpu_context = GpuContext(
            app_name=app_name,
            enable_debug_layer_support=debug,
            enable_present_support=enable_gui,
        )

        self._gui_context = (
            GuiContext(gpu_context=self._gpu_context) if enable_gui else None
        )

        self._render_context = Draw2dContext(
            gpu_context=self._gpu_context,
        )

        # Create window, GPU surface:
        self._window = (
            GuiWindow(
                gui_context=expect(self._gui_context),
                width_dip=1280,
                height_dip=720,
                title=app_name,
                theme=gui_theme,
            )
            if enable_gui
            else None
        )

        # Create GPU device (using surface if window exists):
        physical_device = next(iter(self._gpu_context.enumerate_physical_devices()))
        self._gpu_device = GpuDevice(
            context=self._gpu_context,
            physical_device=physical_device,
            surface=self._window._gpu_surface if self._window else None,
        )

        # Set device on window and create swapchain if window exists
        if self._window is not None:
            self._window.set_gpu_device(
                gpu_device=self._gpu_device,
                swapchain_image_count=swapchain_image_count,
            )

        # Create renderer:
        if self._window:
            scale_x, scale_y = self._window.content_scale
            scale = scale_x
        else:
            scale = 1.0

        self._renderer = Draw2dRenderer(
            context=self._render_context,
            device=self._gpu_device,
            scale=scale,
        )

        # State:
        self._rendered_frame_count = 0

    @property
    def gpu_context(self) -> GpuContext:
        return self._gpu_context

    @property
    def draw_2d_context(self) -> Draw2dContext:
        return self._render_context

    @property
    def gui_context(self) -> GuiContext:
        assert self._gui_context is not None
        return self._gui_context

    @property
    def window(self) -> GuiWindow:
        assert self._window is not None
        return self._window

    @property
    def gpu_device(self) -> GpuDevice:
        return self._gpu_device

    @property
    def gpu_swap_chain(self) -> GpuSwapChain | None:
        return self._window._gpu_swap_chain if self._window else None

    @property
    def renderer(self) -> Draw2dRenderer:
        return self._renderer

    def _on_dispose(self) -> None:
        #
        # Resources
        #

        if renderer := getattr(self, "_renderer", None):
            renderer.dispose()

        if window := getattr(self, "_window", None):
            window.dispose()

        if gpu_device := getattr(self, "_gpu_device", None):
            gpu_device.dispose()

        #
        # Contexts:
        #

        if render_context := getattr(self, "_render_context", None):
            render_context.dispose()

        if gui_context := getattr(self, "_gui_context", None):
            gui_context.dispose()

        if gpu_context := getattr(self, "_gpu_context", None):
            gpu_context.dispose()

    def print_gpu_debug_info(
        self,
        file: SupportsWrite[str] = sys.stdout,
    ):
        print("<gpu-debug-info>")
        self._gpu_context.print_debug_info(out=file)
        print()
        print("</gpu-debug-info>")

    def update(self):
        if not self._window:
            return

        # This specific update order is important, and is documented in the docstring
        # for `GuiWidget`.

        # Update style:
        self._window.update_style()

        # Update layout:
        self._window.update_layout()

        # Receive input events:
        GuiWindow.poll_events()

        # Ready to 'render()'.

    def render(self):
        """Context manager for rendering a frame with quads."""
        if self._window is None or self._window._gpu_swap_chain is None:
            return

        if self._rendered_frame_count == 0:
            self._window.show()

        with self._window._gpu_swap_chain.present() as target:
            self._renderer.clear()
            self._window.render(renderer=self._renderer)

            command_encoder = GpuCommandEncoder(
                device=self.gpu_device,
                queue_type="graphics",
            )
            self._renderer.draw(
                command_encoder=command_encoder,
                target=target.image,
            )
            command_encoder.transition_image_layout(
                image=target.image,
                layout=(
                    "present-src"
                    if self.gpu_device.present_support_enabled
                    else "transfer-src-optimal"
                ),
            )
            command_encoder.submit(
                fence=target.render_done_fence,
                wait_semaphores=[target.render_wait_semaphore],
                signal_semaphores=[target.render_done_semaphore],
            )

        self._rendered_frame_count += 1
