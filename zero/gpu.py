from typing import Any, TypeAlias, cast

import glfw
from vulkan import (
    vkCreateInstance,
    VkInstanceCreateInfo,
    VkApplicationInfo,
    vkEnumeratePhysicalDevices,
)

from .core import ensure_glfw_init


VkInstance: TypeAlias = Any
VkPhysicalDevice: TypeAlias = Any
VkDevice: TypeAlias = Any


class GpuContext:
    def __init__(
        self,
        app_name: str = "Unnamed Zero App",
        enable_debug_layer_support: bool = True,
        enable_present_support: bool = True,
    ) -> None:
        super().__init__()

        ensure_glfw_init()

        self._vk_instance = GpuContext._create_instance(
            app_name,
            enable_debug_layer_support,
            enable_present_support,
        )

    @staticmethod
    def _create_instance(
        app_name: str,
        enable_debug_layers: bool,
        enable_present_support: bool,
    ) -> VkInstance:
        layers = []
        extensions = []

        if enable_debug_layers:
            layers.append("VK_LAYER_KHRONOS_validation")
            extensions.append("VK_EXT_debug_utils")
            extensions.append("VK_EXT_debug_report")

        if enable_present_support:
            extensions += glfw.get_required_instance_extensions()

        return vkCreateInstance(
            pCreateInfo=VkInstanceCreateInfo(
                pApplicationInfo=VkApplicationInfo(
                    pApplicationName=app_name,
                    pEngineName="zero",
                ),
                enabledLayerCount=len(layers),
                ppEnabledLayerNames=layers,
                enabledExtensionCount=len(extensions),
                ppEnabledExtensionNames=extensions,
            ),
            pAllocator=None,
        )

    @staticmethod
    def _enumerate_physical_devices(vk_instance: VkInstance) -> list[VkPhysicalDevice]:
        return cast(
            list[VkPhysicalDevice],
            vkEnumeratePhysicalDevices(vk_instance),
        )
