__all__ = [
    # GpuContext
    "GpuContext",
    # GpuPhysicalDevice
    "GpuPhysicalDevice",
    # GpuDevice
    "GpuDevice",
]

from dataclasses import dataclass
from enum import IntEnum
from typing import TypeAlias, Literal

import glfw

from .core import ensure_glfw_init
from .typed_vulkan import (
    # Common:
    VkDeviceSize,
    VkSampleCountFlags,
    # Instance:
    VkInstance,
    vkCreateInstance,
    VkInstanceCreateInfo,
    VkApplicationInfo,
    vkEnumeratePhysicalDevices,
    # Physical devices:
    VkPhysicalDevice,
    vkGetPhysicalDeviceProperties,
    VkPhysicalDeviceProperties,
    VkPhysicalDeviceLimits,
    VK_PHYSICAL_DEVICE_TYPE_OTHER,
    VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU,
    VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU,
    VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU,
    VK_PHYSICAL_DEVICE_TYPE_CPU,
    # Physical device queues:
    vkGetPhysicalDeviceQueueFamilyProperties,
    VK_QUEUE_COMPUTE_BIT,
    VK_QUEUE_TRANSFER_BIT,
    VK_QUEUE_GRAPHICS_BIT,
    # Devices:
    VkDevice,
    vkCreateDevice,
    VkDeviceCreateInfo,
)


#
# GpuContext
#


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
        self._enable_debug_layer_support = enable_debug_layer_support
        self._enable_present_support = enable_present_support

    @property
    def vk_instance(self) -> VkInstance:
        return self._vk_instance

    @property
    def enable_debug_layer_support(self) -> bool:
        return self._enable_debug_layer_support

    @property
    def enable_present_support(self) -> bool:
        return self._enable_present_support

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

    def enumerate_physical_devices(self) -> list[GpuPhysicalDevice]:
        return [
            GpuPhysicalDevice(self, vk_physical_device)
            for vk_physical_device in vkEnumeratePhysicalDevices(self.vk_instance)
        ]

    def create_device(self, physical_device: GpuPhysicalDevice) -> GpuDevice:
        return GpuDevice(physical_device)


class GpuContextResource:
    def __init__(self, context: GpuContext) -> None:
        super().__init__()
        self._context = context

    @property
    def context(self) -> GpuContext:
        return self._context


#
# GpuPhysicalDevice
#

GpuPhysicalDeviceType: TypeAlias = Literal[
    "Other",
    "IntegratedGpu",
    "DiscreteGpu",
    "VirtualGpu",
    "Cpu",
]


class GpuPhysicalDevice(GpuContextResource):
    vk_handle: VkPhysicalDevice
    properties: VkPhysicalDeviceProperties

    def __init__(
        self,
        context: GpuContext,
        vk_physical_device: VkPhysicalDevice,
    ) -> None:
        super().__init__(context)
        self.vk_handle = vk_physical_device
        self.properties = vkGetPhysicalDeviceProperties(vk_physical_device)

    def get_queue_families(self) -> list["GpuPhysicalDeviceQueueFamily"]:
        qfi_props = vkGetPhysicalDeviceQueueFamilyProperties(self.vk_handle)

        res = []
        for index, props in enumerate(qfi_props):
            assert props.queueCount > 0

            supports_graphics = bool(props.queueFlags & VK_QUEUE_GRAPHICS_BIT)
            supports_compute = bool(props.queueFlags & VK_QUEUE_COMPUTE_BIT)
            supports_transfer = bool(props.queueFlags & VK_QUEUE_TRANSFER_BIT)
            supports_present = False
            if self.context.enable_present_support:
                supports_present = bool(
                    glfw.get_physical_device_presentation_support(
                        self.context.vk_instance,
                        self.vk_handle,
                        index,
                    )
                )

            qfi = GpuPhysicalDeviceQueueFamily(
                index=index,
                queue_count=props.queueCount,
                supports_graphics=supports_graphics,
                supports_compute=supports_compute,
                supports_transfer=supports_transfer,
                supports_present=supports_present,
            )
            res.append(qfi)

        return res

    def spell_device_type(self) -> str:
        return {
            int(VK_PHYSICAL_DEVICE_TYPE_OTHER): "Other",
            int(VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU): "IntegratedGpu",
            int(VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU): "DiscreteGpu",
            int(VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU): "VirtualGpu",
            int(VK_PHYSICAL_DEVICE_TYPE_CPU): "Cpu",
        }[self.properties.deviceType]


@dataclass
class GpuPhysicalDeviceQueueFamily:
    index: int
    queue_count: int
    supports_graphics: bool
    supports_compute: bool
    supports_transfer: bool
    supports_present: bool


@dataclass
class GpuPhysicalDeviceQueueFamilyIndices:
    graphics: int | None = None
    compute: int | None = None
    transfer: int | None = None
    present: int | None = None

    def is_complete(self, require_present_support: bool) -> bool:
        return (
            self.graphics is not None
            and self.compute is not None
            and self.transfer is not None
            and (not require_present_support or self.present is not None)
        )

    def is_any_queue_family_shared(self) -> bool:
        all_indices = [
            index
            for index in [
                self.graphics,
                self.compute,
                self.transfer,
                self.present,
            ]
            if index is not None
        ]
        return len(set(all_indices)) < len(all_indices)

    @staticmethod
    def find(
        physical_device: "GpuPhysicalDevice",
        require_present_support: bool,
    ) -> "GpuPhysicalDeviceQueueFamilyIndices | None":
        # First, try to find exclusive queue families:
        qfi = GpuPhysicalDeviceQueueFamilyIndices._find_with_exclusivity_constraint(
            physical_device=physical_device,
            require_present_support=require_present_support,
            require_exclusive_queues=True,
        )
        if qfi is not None:
            return qfi

        # Fallback: allow shared queue families:
        qfi = GpuPhysicalDeviceQueueFamilyIndices._find_with_exclusivity_constraint(
            physical_device=physical_device,
            require_present_support=require_present_support,
            require_exclusive_queues=False,
        )
        if qfi is not None:
            return qfi

        # Failed
        return None

    @staticmethod
    def _find_with_exclusivity_constraint(
        physical_device: "GpuPhysicalDevice",
        require_present_support: bool,
        require_exclusive_queues: bool,
    ) -> "GpuPhysicalDeviceQueueFamilyIndices | None":
        res = GpuPhysicalDeviceQueueFamilyIndices()

        for queue_family in physical_device.get_queue_families():
            if res.graphics is not None and queue_family.supports_graphics:
                res.graphics = queue_family.index
                if require_exclusive_queues:
                    continue

            if res.compute is not None and queue_family.supports_compute:
                res.compute = queue_family.index
                if require_exclusive_queues:
                    continue

            if res.transfer is not None and queue_family.supports_transfer:
                res.transfer = queue_family.index
                if require_exclusive_queues:
                    continue

            if require_present_support:
                if res.present is not None and queue_family.supports_present:
                    res.present = queue_family.index
                    if require_exclusive_queues:
                        continue

            if res.is_complete(require_present_support):
                break

        if res.is_complete(require_present_support):
            return res
        else:
            return None


#
# GpuDevice
#


class GpuDevice(GpuContextResource):
    def __init__(self, physical_device: GpuPhysicalDevice) -> None:
        super().__init__(physical_device.context)

        extensions = []

        if self.context.enable_present_support:
            extensions.append("VK_KHR_swapchain")

        self.vk_device: VkDevice = vkCreateDevice(
            physical_device.vk_handle,
            VkDeviceCreateInfo(
                enabledExtensionCount=len(extensions),
                ppEnabledExtensionNames=extensions,
            ),
        )
