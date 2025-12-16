import sys

from .basic import BaseResource, SupportsWrite
from .gpu import GpuContext, GpuDevice, GpuSwapChain
from .renderer import Renderer, RendererContext, Canvas
from .window import Window, WindowContext


class Engine(BaseResource):
    _gpu_context: GpuContext
    _window_context: WindowContext
    _render_context: RendererContext
    _window: Window
    _gpu_device: GpuDevice
    _gpu_swap_chain: GpuSwapChain
    _renderer: Renderer
    _rendered_frame_count: int

    def __init__(
        self,
        *,
        app_name: str,
        debug: bool,
        swapchain_image_count: int,
    ):
        super().__init__(parent_resource=None)

        # Create contexts:
        self._gpu_context = GpuContext(
            app_name=app_name,
            enable_debug_layer_support=debug,
            enable_present_support=True,
        )
        self._window_context = WindowContext(
            gpu_context=self._gpu_context,
        )
        self._render_context = RendererContext(
            gpu_context=self._gpu_context,
        )

        # Create window, GPU surface:
        self._window = Window(
            window_context=self._window_context,
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

        # State:
        self._rendered_frame_count = 0

    @property
    def gpu_context(self) -> GpuContext:
        return self._gpu_context

    @property
    def window_context(self) -> WindowContext:
        return self._window_context

    @property
    def render_context(self) -> RendererContext:
        return self._render_context

    @property
    def window(self) -> Window:
        return self._window

    @property
    def gpu_device(self) -> GpuDevice:
        return self._gpu_device

    @property
    def gpu_swap_chain(self) -> GpuSwapChain:
        return self._gpu_swap_chain

    @property
    def renderer(self) -> Renderer:
        return self._renderer

    def _on_dispose_resource(self) -> None:
        self._gpu_swap_chain.dispose_resource()
        self._gpu_device.dispose_resource()

        self._window.dispose_resource()

        self._render_context.dispose_resource()
        self._window_context.dispose_resource()
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
        Window.poll_events()

    def render(self, canvas: Canvas):
        """Context manager for rendering a frame with quads."""
        if self._rendered_frame_count == 0:
            self._window.show()

        with self._gpu_swap_chain.present() as target:
            self._renderer.draw(
                canvas=canvas,
                target=target.image,
                wait_semaphores=[target.render_wait_semaphore],
                done_semaphores=[target.render_done_semaphore],
                fence=target.render_done_fence,
            )

        self._rendered_frame_count += 1
