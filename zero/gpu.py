__all__ = [
    # GpuContext
    "GpuContext",
    # GpuPhysicalDevice
    "GpuPhysicalDevice",
    # GpuDevice
    "GpuDevice",
]

from dataclasses import dataclass
from collections import defaultdict
from typing import TypeAlias, Literal
import sys

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
    VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR,
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
    VkDeviceQueueCreateInfo,
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
        enable_portability_subset_override: bool | None = None,
    ) -> None:
        super().__init__()

        ensure_glfw_init()

        enable_portability_subset = (
            enable_portability_subset_override
            if enable_portability_subset_override is not None
            else (sys.platform == "darwin")  # -> inferred
        )

        self._vk_instance = GpuContext._create_instance(
            app_name,
            enable_debug_layer_support,
            enable_present_support,
            enable_portability_subset,
        )
        self._enable_debug_layer_support = enable_debug_layer_support
        self._enable_present_support = enable_present_support
        self._enable_portability_subset = enable_portability_subset

    @property
    def vk_instance(self) -> VkInstance:
        return self._vk_instance

    @property
    def enable_debug_layer_support(self) -> bool:
        return self._enable_debug_layer_support

    @property
    def enable_present_support(self) -> bool:
        return self._enable_present_support

    @property
    def enable_portability_subset(self) -> bool | None:
        return self._enable_portability_subset

    @staticmethod
    def _create_instance(
        app_name: str,
        enable_debug_layers: bool,
        enable_present_support: bool,
        enable_portability_subset: bool,
    ) -> VkInstance:
        layers = []
        extensions = []
        flags = 0

        if enable_debug_layers:
            layers.append("VK_LAYER_KHRONOS_validation")
            extensions.append("VK_EXT_debug_utils")
            extensions.append("VK_EXT_debug_report")

        if enable_present_support:
            extensions += glfw.get_required_instance_extensions()

        if enable_portability_subset:
            extensions.append("VK_KHR_portability_enumeration")
            extensions.append("VK_KHR_get_physical_device_properties2")
            flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR

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
                flags=flags,
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

    def compute_queue_create_info_list(self) -> list[VkDeviceQueueCreateInfo]:
        dd = defaultdict(lambda: 0)
        if self.graphics is not None:
            dd[self.graphics] += 1
        if self.compute is not None:
            dd[self.compute] += 1
        if self.transfer is not None:
            dd[self.transfer] += 1
        if self.present is not None:
            dd[self.present] += 1
        return [
            VkDeviceQueueCreateInfo(
                queueFamilyIndex=index,
                queueCount=count,
                pQueuePriorities=[1.0] * count,
            )
            for index, count in dd.items()
        ]

    @staticmethod
    def find(
        physical_device: "GpuPhysicalDevice",
        present_support_enabled: bool,
    ) -> "GpuPhysicalDeviceQueueFamilyIndices":
        # First, try to find exclusive queue families:
        qfi_exclusive = (
            GpuPhysicalDeviceQueueFamilyIndices._find_with_exclusivity_constraint(
                physical_device=physical_device,
                require_present_support=present_support_enabled,
                require_exclusive_queues=True,
            )
        )
        if qfi_exclusive.is_complete(present_support_enabled):
            return qfi_exclusive

        # Fallback: allow shared queue families:
        qfi_shared = (
            GpuPhysicalDeviceQueueFamilyIndices._find_with_exclusivity_constraint(
                physical_device=physical_device,
                require_present_support=present_support_enabled,
                require_exclusive_queues=False,
            )
        )
        if qfi_shared.is_complete(present_support_enabled):
            return qfi_shared

        # Failed
        raise RuntimeError(
            f"Failed to find suitable queue families:\n- {qfi_exclusive=}\n- {qfi_shared=}"
        )

    @staticmethod
    def _find_with_exclusivity_constraint(
        physical_device: "GpuPhysicalDevice",
        require_present_support: bool,
        require_exclusive_queues: bool,
    ) -> "GpuPhysicalDeviceQueueFamilyIndices":
        res = GpuPhysicalDeviceQueueFamilyIndices()

        for queue_family in physical_device.get_queue_families():
            queue_family_use_count = 0
            queue_family_max_use_count = (
                1 if require_exclusive_queues else queue_family.queue_count
            )

            def reserve_queue_and_check_can_continue() -> bool:
                nonlocal queue_family_use_count
                queue_family_use_count += 1
                return queue_family_use_count == queue_family_max_use_count

            if res.graphics is None and queue_family.supports_graphics:
                res.graphics = queue_family.index
                if reserve_queue_and_check_can_continue():
                    continue

            if res.compute is None and queue_family.supports_compute:
                res.compute = queue_family.index
                if reserve_queue_and_check_can_continue():
                    continue

            if res.transfer is None and queue_family.supports_transfer:
                res.transfer = queue_family.index
                if reserve_queue_and_check_can_continue():
                    continue

            if require_present_support:
                if res.present is None and queue_family.supports_present:
                    res.present = queue_family.index
                    if reserve_queue_and_check_can_continue():
                        continue

            if res.is_complete(require_present_support):
                break

        return res


#
# GpuDevice
#


class GpuDevice(GpuContextResource):
    def __init__(self, physical_device: GpuPhysicalDevice) -> None:
        super().__init__(physical_device.context)

        extensions = []
        if self.context.enable_present_support:
            extensions.append("VK_KHR_swapchain")
        if self.context.enable_portability_subset:
            extensions.append("VK_KHR_portability_subset")

        queue_create_info_list = GpuPhysicalDeviceQueueFamilyIndices.find(
            physical_device,
            present_support_enabled=self.context.enable_present_support,
        ).compute_queue_create_info_list()

        self.vk_device: VkDevice = vkCreateDevice(
            physical_device.vk_handle,
            VkDeviceCreateInfo(
                enabledExtensionCount=len(extensions),
                ppEnabledExtensionNames=extensions,
                queueCreateInfoCount=len(queue_create_info_list),
                pQueueCreateInfos=queue_create_info_list,
            ),
            pAllocator=None,
        )
