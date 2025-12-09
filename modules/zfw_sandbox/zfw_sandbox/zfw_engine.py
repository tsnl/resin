import sys

import zfw_core
from zfw_core.gpu import GpuDevice, GpuSwapchain
from zfw_core.renderer import Renderer


class ZfwEngine:
    def __init__(self, *, app_name: str, debug: bool, swapchain_image_count: int):
        self.gpu_context = zfw_core.GpuContext(
            app_name=app_name,
            enable_debug_layer_support=debug,
            enable_present_support=True,
        )
        self.window_context = zfw_core.WindowContext(gpu_context=self.gpu_context)
        self.render_context = zfw_core.RendererContext(gpu_context=self.gpu_context)

        # Create window, GPU surface:
        self.window = self.window_context.create_window(
            width=1280,
            height=720,
            title="Zero Sandbox",
        )
        self.surface = self.window.create_surface()

        # Create GPU device using the surface:
        physical_device = next(iter(self.gpu_context.enumerate_physical_devices()))
        self.device = GpuDevice(
            context=self.gpu_context,
            physical_device=physical_device,
            surface=self.surface,
        )

        # Create swapchain:
        self.swapchain = GpuSwapchain(
            device=self.device,
            surface=self.surface,
            image_count=swapchain_image_count,
        )

        # Create renderer:
        self.renderer = Renderer(context=self.render_context, gpu_device=self.device)

    def print_gpu_debug_info(
        self,
        file: zfw_core.SupportsWrite[str] = sys.stdout,
    ):
        print("<gpu-debug-info>")
        self.gpu_context.print_debug_info(out=file)
        print()
        print("</gpu-debug-info>")
