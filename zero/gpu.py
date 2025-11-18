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
from typing import TypeAlias, Literal, final
import sys
from abc import abstractmethod, ABC

import glfw

from .core import ensure_glfw_init
from .typed_vulkan import (
    # Basic:
    vkGetInstanceProcAddr,
    ffi,
    # Common:
    VkDeviceSize,
    VkSampleCountFlags,
    # Instance:
    VkInstance,
    vkCreateInstance,
    vkDestroyInstance,
    VkInstanceCreateInfo,
    VkApplicationInfo,
    vkEnumeratePhysicalDevices,
    VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR,
    # Surface:
    VkSurfaceKHR,
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
    # Queues:
    vkGetPhysicalDeviceQueueFamilyProperties,
    VkDeviceQueueCreateInfo,
    VK_QUEUE_COMPUTE_BIT,
    VK_QUEUE_TRANSFER_BIT,
    VK_QUEUE_GRAPHICS_BIT,
    # Devices:
    VkDevice,
    vkCreateDevice,
    vkDestroyDevice,
    VkDeviceCreateInfo,
)


#
# BaseGpuResource
#


class GpuResource(ABC):
    def __init__(self, parent: GpuResource | None, context: GpuContext) -> None:
        super().__init__()

        self._parent: GpuResource | None = parent
        self._context: GpuContext = context
        self._children: set[GpuResource] = set()
        self._is_disposed: bool = False

        if self._parent:
            self._parent._notify_child_constructed(self)

    def _notify_child_constructed(self, child: "GpuResource") -> None:
        self._children.add(child)

    def _notify_child_disposed(self, child: "GpuResource") -> None:
        assert child._is_disposed
        self._children.remove(child)

    @property
    def context(self) -> GpuContext:
        return self._context

    def dispose(self) -> None:
        # If already disposed, no-op.
        if self._is_disposed:
            return

        # Dispose children.
        # Each child's `dispose()` method will mutate 'self._children`, so we need to
        # iterate over a copy of the set. We also expect all children to be disposed by
        # the end.
        for child in set(self._children):
            child.dispose()
        assert not self._children, "Expected all children to be disposed."

        # Dispose self, then notify parent.
        self._on_dispose()
        self._is_disposed = True
        if self._parent:
            self._parent._notify_child_disposed(self)

    @abstractmethod
    def _on_dispose(self) -> None:
        pass

    def __del__(self) -> None:
        self.dispose()


#
# GpuContext
#


class GpuContext(GpuResource):
    def __init__(
        self,
        app_name: str = "Unnamed Zero App",
        enable_debug_layer_support: bool = True,
        enable_present_support: bool = True,
        enable_portability_subset_override: bool | None = None,
    ) -> None:
        super().__init__(parent=None, context=self)

        ensure_glfw_init()

        enable_portability_subset = (
            enable_portability_subset_override
            if enable_portability_subset_override is not None
            else (sys.platform == "darwin")  # -> inferred
        )

        self._enable_debug_layer_support = enable_debug_layer_support
        self._enable_present_support = enable_present_support
        self._enable_portability_subset = enable_portability_subset

        self._vk_instance = GpuContext._help_create_instance(
            app_name,
            enable_debug_layer_support,
            enable_present_support,
            enable_portability_subset,
        )
        self._vk_extra_proc_tab = {}

        if enable_present_support:
            self._vk_extra_proc_tab["vkGetPhysicalDeviceSurfaceSupportKHR"] = (
                vkGetInstanceProcAddr(
                    self._vk_instance,
                    "vkGetPhysicalDeviceSurfaceSupportKHR",
                )
            )

    @staticmethod
    def _help_create_instance(
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

    def _on_dispose(self) -> None:
        if hasattr(self, "_vk_instance"):
            vkDestroyInstance(self._vk_instance, pAllocator=None)

    def get_physical_device_surface_support(
        self,
        physical_device: VkPhysicalDevice,
        queue_family_index: int,
        surface: VkSurfaceKHR,
    ) -> bool:
        proc = self._vk_extra_proc_tab.get("vkGetPhysicalDeviceSurfaceSupportKHR", None)
        if proc is None:
            assert not self.enable_present_support
            return False
        return proc(
            physical_device,
            queue_family_index,
            surface,
        )

    @property
    def enable_debug_layer_support(self) -> bool:
        return self._enable_debug_layer_support

    @property
    def enable_present_support(self) -> bool:
        return self._enable_present_support

    @property
    def enable_portability_subset(self) -> bool | None:
        return self._enable_portability_subset

    def enumerate_physical_devices(self) -> list[GpuPhysicalDevice]:
        return [
            GpuPhysicalDevice(self, vk_physical_device)
            for vk_physical_device in vkEnumeratePhysicalDevices(self._vk_instance)
        ]

    def create_device(
        self,
        physical_device: GpuPhysicalDevice,
        surface: VkSurfaceKHR | None,
    ) -> GpuDevice:
        return GpuDevice(physical_device, surface)


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


class GpuPhysicalDevice(GpuResource):
    def __init__(
        self,
        context: GpuContext,
        vk_physical_device: VkPhysicalDevice,
    ) -> None:
        super().__init__(parent=context, context=context)
        self._vk_physical_device = vk_physical_device
        self._vk_properties = vkGetPhysicalDeviceProperties(vk_physical_device)

    def _on_dispose(self) -> None:
        pass  # no private resources to free

    def get_queue_families(
        self,
        surface: VkSurfaceKHR | None,
    ) -> list["GpuPhysicalDeviceQueueFamily"]:
        if surface is not None:
            assert self.context.enable_present_support

        qfi_props = vkGetPhysicalDeviceQueueFamilyProperties(self._vk_physical_device)

        res = []
        for index, props in enumerate(qfi_props):
            assert props.queueCount > 0
            supports_graphics = bool(props.queueFlags & VK_QUEUE_GRAPHICS_BIT)
            supports_compute = bool(props.queueFlags & VK_QUEUE_COMPUTE_BIT)
            supports_transfer = bool(props.queueFlags & VK_QUEUE_TRANSFER_BIT)

            supports_present = False
            if surface is not None:
                supports_present = self.context.get_physical_device_surface_support(
                    self._vk_physical_device,
                    index,
                    surface,
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
        }[self._vk_properties.deviceType]


@dataclass
class GpuPhysicalDeviceQueueFamily:
    index: int
    queue_count: int
    supports_graphics: bool
    supports_compute: bool
    supports_transfer: bool
    supports_present: bool


#
# QueueFamilyIndices
#


@dataclass
class GpuQueueFamilyIndices:
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
        surface: VkSurfaceKHR | None = None,
    ) -> "GpuQueueFamilyIndices":
        # First, try to find exclusive queue families:
        qfi_exclusive = GpuQueueFamilyIndices._find_with_exclusivity_constraint(
            physical_device=physical_device,
            surface=surface,
            require_exclusive_queues=True,
        )
        if qfi_exclusive.is_complete(require_present_support=surface is not None):
            return qfi_exclusive

        # Fallback: allow shared queue families:
        qfi_shared = GpuQueueFamilyIndices._find_with_exclusivity_constraint(
            physical_device=physical_device,
            surface=surface,
            require_exclusive_queues=False,
        )
        if qfi_shared.is_complete(require_present_support=surface is not None):
            return qfi_shared

        # Failed
        raise RuntimeError(
            f"Failed to find suitable queue families:\n- {qfi_exclusive=}\n- {qfi_shared=}"
        )

    @staticmethod
    def _find_with_exclusivity_constraint(
        physical_device: "GpuPhysicalDevice",
        surface: VkSurfaceKHR | None,
        require_exclusive_queues: bool,
    ) -> "GpuQueueFamilyIndices":
        res = GpuQueueFamilyIndices()

        for queue_family in physical_device.get_queue_families(surface=surface):
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

            if surface is not None:
                if res.present is None and queue_family.supports_present:
                    res.present = queue_family.index
                    if reserve_queue_and_check_can_continue():
                        continue

            if res.is_complete(require_present_support=surface is not None):
                break

        return res


#
# GpuDevice
#


class GpuDevice(GpuResource):
    def __init__(
        self,
        physical_device: GpuPhysicalDevice,
        surface: VkSurfaceKHR | None,
    ) -> None:
        super().__init__(parent=physical_device, context=physical_device.context)

        self._qfis, self._vk_device = GpuDevice._help_create_vk_device(
            context=self.context,
            physical_device=physical_device,
            surface=surface,
        )

    @staticmethod
    def _help_create_vk_device(
        context: GpuContext,
        physical_device: GpuPhysicalDevice,
        surface: VkSurfaceKHR | None,
    ) -> tuple[GpuQueueFamilyIndices, VkDevice]:
        extensions = []
        if context.enable_present_support:
            extensions.append("VK_KHR_swapchain")
        if context.enable_portability_subset:
            extensions.append("VK_KHR_portability_subset")

        qfis = GpuQueueFamilyIndices.find(
            physical_device,
            surface=surface,
        )
        queue_create_info_list = qfis.compute_queue_create_info_list()

        return qfis, vkCreateDevice(
            physical_device._vk_physical_device,
            VkDeviceCreateInfo(
                enabledExtensionCount=len(extensions),
                ppEnabledExtensionNames=extensions,
                queueCreateInfoCount=len(queue_create_info_list),
                pQueueCreateInfos=queue_create_info_list,
            ),
            pAllocator=None,
        )

    def _on_dispose(self) -> None:
        if hasattr(self, "_vk_device"):
            vkDestroyDevice(device=self._vk_device, pAllocator=None)

    def create_texture(self) -> "GpuTexture":
        # TODO: implement the rest of this
        return GpuTexture(self)


#
# GpuTexture
#


@final
class GpuTexture(GpuResource):
    def __init__(
        self,
        device: GpuDevice,
    ) -> None:
        super().__init__(parent=device, context=device.context)

    def _on_dispose(self) -> None:
        pass
