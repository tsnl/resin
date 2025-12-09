import sys

import zfw_core
from zfw_core.gpu import GpuDevice, GpuSwapChain
from zfw_core.renderer import Renderer, RendererCanvas
from zfw_core.window import Window


class ZfwEngine(zfw_core.BaseResource):
    def __init__(self, *, app_name: str, debug: bool, swapchain_image_count: int):
        super().__init__(parent=None)

        # Create contexts:
        self.gpu_context = zfw_core.GpuContext(
            parent=self,
            app_name=app_name,
            enable_debug_layer_support=debug,
            enable_present_support=True,
        )
        self.window_context = zfw_core.WindowContext(
            parent=self,
            gpu_context=self.gpu_context,
        )
        self.render_context = zfw_core.RendererContext(
            parent=self,
            gpu_context=self.gpu_context,
        )

        # Create window, GPU surface:
        self.window = Window(
            context=self.window_context,
            width=1280,
            height=720,
            title="Zero Sandbox",
        )

        # Create GPU device using the surface:
        physical_device = next(iter(self.gpu_context.enumerate_physical_devices()))
        self.gpu_device = GpuDevice(
            context=self.gpu_context,
            physical_device=physical_device,
            surface=self.window.gpu_surface,
        )

        # Create swap chain:
        self.gpu_swap_chain = GpuSwapChain(
            device=self.gpu_device,
            surface=self.window.gpu_surface,
            image_count=swapchain_image_count,
        )

        # Create renderer:
        self.renderer = Renderer(
            context=self.render_context,
            gpu_device=self.gpu_device,
        )
        self.render_canvas = RendererCanvas(renderer=self.renderer)

        # State:
        self.rendered_frame_count = 0

        # Debug: scan for resource cycles
        if debug:
            n = self.detect_resource_cycles(verbose=True)
            print(f"ZfwEngine: no resource cycles detected: {n=}", file=sys.stderr)

    def print_gpu_debug_info(
        self,
        file: zfw_core.SupportsWrite[str] = sys.stdout,
    ):
        print("<gpu-debug-info>")
        self.gpu_context.print_debug_info(out=file)
        print()
        print("</gpu-debug-info>")

    def update(self):
        Window.poll_events()

    def render(self):
        if self.rendered_frame_count == 0:
            self.window.show()

        with self.gpu_swap_chain.present() as target:
            self.renderer.show(
                canvas=self.render_canvas,
                target=target.image,
                wait_semaphores=[target.render_wait_semaphore],
                done_semaphores=[target.render_done_semaphore],
                done_fence=target.render_done_fence,
            )

        self.rendered_frame_count += 1
