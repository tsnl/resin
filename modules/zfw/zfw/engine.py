import sys

from .basic import BaseResource, SupportsWrite
from .gpu import GpuContext, GpuDevice, GpuSwapChain
from .renderer import Renderer, RendererContext, Canvas
from .gui import GuiWindow, GuiContext


class Engine(BaseResource):
    _gpu_context: GpuContext
    _gui_context: GuiContext | None
    _render_context: RendererContext
    _window: GuiWindow | None
    _gpu_device: GpuDevice
    _gpu_swap_chain: GpuSwapChain | None
    _renderer: Renderer
    _rendered_frame_count: int
    _canvas: Canvas

    def __init__(
        self,
        *,
        app_name: str,
        debug: bool,
        swapchain_image_count: int,
        enable_gui: bool = True,
    ):
        super().__init__(parent_resource=None)

        # Create contexts:
        self._gpu_context = GpuContext(
            app_name=app_name,
            enable_debug_layer_support=debug,
            enable_present_support=enable_gui,
        )

        if enable_gui:
            self._gui_context = GuiContext(
                gpu_context=self._gpu_context,
            )
        else:
            self._gui_context = None

        self._render_context = RendererContext(
            gpu_context=self._gpu_context,
        )

        if enable_gui:
            assert self._gui_context is not None
            # Create window, GPU surface:
            self._window = GuiWindow(
                gui_context=self._gui_context,
                width=1280,
                height=720,
                title="Zero Sandbox",
            )

            # Create GPU device using the surface:
            physical_device = next(iter(self._gpu_context.enumerate_physical_devices()))
            self._gpu_device = GpuDevice(
                context=self._gpu_context,
                physical_device=physical_device,
                surface=self._window.gpu_surface,
            )

            # Create swap chain:
            self._gpu_swap_chain = GpuSwapChain(
                device=self._gpu_device,
                surface=self._window.gpu_surface,
                image_count=swapchain_image_count,
            )

            # Create renderer:
            scale_x, scale_y = self._window.content_scale
            scale = scale_x

            self._renderer = Renderer(
                context=self._render_context,
                gpu_device=self._gpu_device,
                scale=scale,
            )
        else:
            self._window = None
            self._gpu_swap_chain = None

            physical_device = next(iter(self._gpu_context.enumerate_physical_devices()))
            self._gpu_device = GpuDevice(
                context=self._gpu_context,
                physical_device=physical_device,
                surface=None,
            )

            self._renderer = Renderer(
                context=self._render_context,
                gpu_device=self._gpu_device,
                scale=1.0,
            )

        self._canvas = Canvas(renderer=self._renderer)

        # State:
        self._rendered_frame_count = 0

    @property
    def gpu_context(self) -> GpuContext:
        return self._gpu_context

    @property
    def gui_context(self) -> GuiContext:
        assert self._gui_context is not None
        return self._gui_context

    @property
    def render_context(self) -> RendererContext:
        return self._render_context

    @property
    def window(self) -> GuiWindow:
        assert self._window is not None
        return self._window

    @property
    def gpu_device(self) -> GpuDevice:
        return self._gpu_device

    @property
    def gpu_swap_chain(self) -> GpuSwapChain | None:
        return self._gpu_swap_chain

    @property
    def renderer(self) -> Renderer:
        return self._renderer

    def _on_dispose_resource(self) -> None:
        if self._gpu_swap_chain:
            self._gpu_swap_chain.dispose_resource()
        self._gpu_device.dispose_resource()

        if self._window:
            self._window.dispose_resource()

        self._render_context.dispose_resource()
        if self._gui_context:
            self._gui_context.dispose_resource()
        self._gpu_context.dispose_resource()

    def print_gpu_debug_info(
        self,
        file: SupportsWrite[str] = sys.stdout,
    ):
        print("<gpu-debug-info>")
        self._gpu_context.print_debug_info(out=file)
        print()
        print("</gpu-debug-info>")

    def update(self):
        if self._window:
            GuiWindow.poll_events()

    def render(self):
        """Context manager for rendering a frame with quads."""
        if self._window is None or self._gpu_swap_chain is None:
            return

        if self._rendered_frame_count == 0:
            self._window.show()

        with self._gpu_swap_chain.present() as target:
            self._canvas.clear()
            self._window.render(canvas=self._canvas)

            self._renderer.draw(
                canvas=self._canvas,
                target=target.image,
                wait_semaphores=[target.render_wait_semaphore],
                done_semaphores=[target.render_done_semaphore],
                fence=target.render_done_fence,
            )

        self._rendered_frame_count += 1
