import sys
from contextlib import contextmanager
from typing import TYPE_CHECKING

from .basic import BaseResource, SupportsWrite
from .gpu import GpuContext, GpuDevice, GpuSwapChain
from .renderer import Renderer, RendererCanvas, RendererContext
from .window import Window, WindowContext

if TYPE_CHECKING:
    from .renderer import RendererCanvas


class Engine(BaseResource):
    _gpu_context: GpuContext
    _window_context: WindowContext
    _render_context: RendererContext
    _window: Window
    _gpu_device: GpuDevice
    _gpu_swap_chain: GpuSwapChain
    _renderer: Renderer
    _render_canvas: RendererCanvas
    _rendered_frame_count: int

    def __init__(self, *, app_name: str, debug: bool, swapchain_image_count: int):
        super().__init__(parent=None)

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
            context=self._window_context,
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
        self._renderer = Renderer(
            context=self._render_context,
            gpu_device=self._gpu_device,
        )
        self._render_canvas = RendererCanvas(renderer=self._renderer)

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

    def _on_dispose(self) -> None:
        self._render_canvas.dispose()

        self._gpu_swap_chain.dispose()
        self._gpu_device.dispose()

        self._window.dispose()

        self._render_context.dispose()
        self._window_context.dispose()
        self._gpu_context.dispose()

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

    @contextmanager
    def render(self):
        self._render_canvas.reset()

        yield self._render_canvas

        if self._rendered_frame_count == 0:
            self._window.show()

        with self._gpu_swap_chain.present() as target:
            self._renderer.show(
                canvas=self._render_canvas,
                target=target.image,
                wait_semaphores=[target.render_wait_semaphore],
                done_semaphores=[target.render_done_semaphore],
                fence=target.render_done_fence,
            )

        self._rendered_frame_count += 1
