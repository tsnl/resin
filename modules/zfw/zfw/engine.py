import sys
from contextlib import contextmanager

import numpy as np

from .basic import BaseResource, SupportsWrite
from .gpu import GpuContext, GpuDevice, GpuSwapChain
from .renderer import Renderer, RendererContext, RendererQuadArray
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
    _quad_buffer: RendererQuadArray

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

        # Initialize quad buffer for drawing
        self._quad_buffer = RendererQuadArray((0,))

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
        """Context manager for rendering a frame with quads."""
        # Reset quad buffer for the frame
        self._quad_buffer = RendererQuadArray((0,))

        yield self

        if self._rendered_frame_count == 0:
            self._window.show()

        with self._gpu_swap_chain.present() as target:
            self._renderer.draw(
                quads=self._quad_buffer,
                target=target.image,
                wait_semaphores=[target.render_wait_semaphore],
                done_semaphores=[target.render_done_semaphore],
                fence=target.render_done_fence,
            )

        self._rendered_frame_count += 1

    def add_quad(
        self,
        *,
        dst_xy: tuple[int, int],
        dst_wh: tuple[int, int],
        color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        border_thickness_px: tuple[int, int, int, int] = (0, 0, 0, 0),
        border_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    ):
        """Add a quad to the render buffer."""
        # Use default white image
        image = self.renderer.default_white_image

        # Create a new quad entry
        quad = RendererQuadArray((1,))

        # Compute destination coordinates
        dst_x0_px, dst_y0_px = dst_xy
        dst_w_px, dst_h_px = dst_wh
        dst_x1_px = dst_x0_px + dst_w_px
        dst_y1_px = dst_y0_px + dst_h_px

        quad[0]["dst_px"] = (
            (dst_x0_px, dst_y0_px),  # TL
            (dst_x1_px, dst_y0_px),  # TR
            (dst_x1_px, dst_y1_px),  # BR
            (dst_x0_px, dst_y1_px),  # BL
        )

        # Get UV coordinates from image
        quad[0]["src_uv"] = (
            (0.0, 0.0),  # TL
            (1.0, 0.0),  # TR
            (1.0, 1.0),  # BR
            (0.0, 1.0),  # BL
        )

        quad[0]["color"] = color
        quad[0]["border_color"] = border_color
        quad[0]["border_thickness_px"] = border_thickness_px
        quad[0]["height"] = len(self._quad_buffer)
        quad[0]["image_id"] = image.index
        quad[0]["flags"] = 1  # Linear

        # Append to quad buffer
        self._quad_buffer = np.concatenate([self._quad_buffer, quad]).view(
            RendererQuadArray
        )
