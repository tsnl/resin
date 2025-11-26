"""
GPU abstraction layer.

Required Vulkan version:
-   Vulkan 1.3 for VK_KHR_dynamic_rendering
    https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_dynamic_rendering.html
"""

__all__ = [
    # GpuContext
    "GpuContext",
    # GpuPhysicalDevice
    "GpuPhysicalDevice",
    # GpuDevice
    "GpuDevice",
    # GpuImage
    "GpuImage",
    "GpuImageUsage",
    "GpuImageMeta",
]

from dataclasses import dataclass, field
from collections import defaultdict
from contextlib import contextmanager
from typing import TypeAlias, Literal, Optional
import sys
import os

import torch

from .core import BaseContext, BaseContextResource
from .excepts import PlatformSupportError, LogicError
from .typed_vulkan import (
    # Basic
    vkGetInstanceProcAddr,
    ffi,
    # Common
    VkDeviceSize,
    VkSampleCountFlags,
    VkExtent3D,
    VK_API_VERSION_1_3,
    VK_API_VERSION_1_4,
    vk_decompose_api_version,
    vk_api_version_str,
    # VkInstance
    VkInstance,
    vkCreateInstance,
    vkDestroyInstance,
    VkInstanceCreateInfo,
    VkApplicationInfo,
    vkEnumeratePhysicalDevices,
    VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR,
    # VkSurfaceKHR
    VkSurfaceKHR,
    # VkPhysicalDevice
    VkPhysicalDevice,
    vkGetPhysicalDeviceProperties,
    VkPhysicalDeviceProperties,
    VkPhysicalDeviceLimits,
    VK_PHYSICAL_DEVICE_TYPE_OTHER,
    VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU,
    VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU,
    VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU,
    VK_PHYSICAL_DEVICE_TYPE_CPU,
    # Physical device memory
    VkPhysicalDeviceMemoryProperties,
    vkGetPhysicalDeviceMemoryProperties,
    # Physical device queues
    vkGetPhysicalDeviceQueueFamilyProperties,
    VkDeviceQueueCreateInfo,
    VK_QUEUE_COMPUTE_BIT,
    VK_QUEUE_TRANSFER_BIT,
    VK_QUEUE_GRAPHICS_BIT,
    # VkDevice
    VkDevice,
    vkCreateDevice,
    vkDestroyDevice,
    VkDeviceCreateInfo,
    # VkDeviceMemory
    VkDeviceMemory,
    VkMemoryRequirements,
    VkMemoryAllocateInfo,
    vkAllocateMemory,
    vkFreeMemory,
    vkMapMemory,
    vkUnmapMemory,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
    VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT,
    # VkImage
    ## VkImageCreateFlags
    ## VkImageCreateFlagBits
    VK_IMAGE_CREATE_SPARSE_BINDING_BIT,
    VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT,
    VK_IMAGE_CREATE_SPARSE_ALIASED_BIT,
    VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT,
    VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT,
    ## VkImageType
    VK_IMAGE_TYPE_1D,
    VK_IMAGE_TYPE_2D,
    VK_IMAGE_TYPE_3D,
    ## VkFormat
    VK_FORMAT_R32_SFLOAT,
    VK_FORMAT_R8G8B8A8_UNORM,
    VK_FORMAT_R32G32B32A32_SFLOAT,
    VK_FORMAT_D32_SFLOAT,
    ## VkSampleCountFlags
    ## VkSampleCountFlagBits
    VK_SAMPLE_COUNT_1_BIT,
    VK_SAMPLE_COUNT_2_BIT,
    VK_SAMPLE_COUNT_4_BIT,
    VK_SAMPLE_COUNT_8_BIT,
    VK_SAMPLE_COUNT_16_BIT,
    VK_SAMPLE_COUNT_32_BIT,
    VK_SAMPLE_COUNT_64_BIT,
    ## VkImageTiling
    VK_IMAGE_TILING_OPTIMAL,
    VK_IMAGE_TILING_LINEAR,
    ## VkImageUsageFlags
    ## VkImageUsageFlagBits
    VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
    VK_IMAGE_USAGE_TRANSFER_DST_BIT,
    VK_IMAGE_USAGE_SAMPLED_BIT,
    VK_IMAGE_USAGE_STORAGE_BIT,
    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
    ## VkSharingMode
    VK_SHARING_MODE_EXCLUSIVE,
    VK_SHARING_MODE_CONCURRENT,
    ## VkImageLayout
    VK_IMAGE_LAYOUT_UNDEFINED,
    VK_IMAGE_LAYOUT_GENERAL,
    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
    VK_IMAGE_LAYOUT_PREINITIALIZED,
    VkImage,
    vkCreateImage,
    VkImageCreateInfo,
    vkDestroyImage,
    vkGetImageMemoryRequirements,
    vkBindImageMemory,
    # VkImageView
    VkImageView,
    vkCreateImageView,
    VkImageViewCreateInfo,
    vkDestroyImageView,
    VkComponentMapping,
    VkImageSubresourceRange,
    VK_COMPONENT_SWIZZLE_IDENTITY,
    VK_IMAGE_ASPECT_DEPTH_BIT,
    VK_IMAGE_ASPECT_COLOR_BIT,
    # VkSampler
    VkSampler,
    vkCreateSampler,
    VkSamplerCreateInfo,
    vkDestroySampler,
    # VkRenderingAttachmentInfo
    VkRenderingAttachmentInfo,
    VkResolveModeFlagBits,
    VkAttachmentLoadOp,
    VkAttachmentStoreOp,
    VkClearValue,
    VkClearColorValue,
    VkClearDepthStencilValue,
    # VkBuffer
    VkBuffer,
    vkCreateBuffer,
    VkBufferCreateInfo,
    vkDestroyBuffer,
    vkGetBufferMemoryRequirements,
    vkBindBufferMemory,
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    # VkBufferView
    vkCreateBufferView,
    VkBufferViewCreateInfo,
    vkDestroyBufferView,
    # VkCommandPool
    VkCommandPool,
    vkCreateCommandPool,
    VkCommandPoolCreateInfo,
    vkDestroyCommandPool,
    VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
    # VkCommandBuffer
    VkCommandBuffer,
    vkAllocateCommandBuffers,
    VkCommandBufferAllocateInfo,
    vkFreeCommandBuffers,
    vkResetCommandBuffer,
    vkBeginCommandBuffer,
    VkCommandBufferBeginInfo,
    vkEndCommandBuffer,
    VK_COMMAND_BUFFER_LEVEL_PRIMARY,
    VK_COMMAND_BUFFER_LEVEL_SECONDARY,
    vkCmdCopyBuffer,
    VkBufferCopy,
    vkCmdCopyBufferToImage,
    VkBufferImageCopy,
    VkImageSubresourceLayers,
    VkOffset3D,
    # Sync & Queues
    VkSemaphore,
    VkSemaphoreCreateInfo,
    vkCreateSemaphore,
    vkDestroySemaphore,
    VkFence,
    VkFenceCreateInfo,
    vkCreateFence,
    vkDestroyFence,
    VkQueue,
    vkGetDeviceQueue,
    VkSubmitInfo,
    vkQueueSubmit,
    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
    vkCmdCopyImageToBuffer,
)


#
# GpuContext
#


class GpuContext(BaseContext["GpuContext"]):
    def __init__(
        self,
        app_name: str = "Zero App",
        enable_debug_layer_support: bool = True,
        enable_present_support: bool = True,
        enable_portability_subset_override: bool | None = None,
    ) -> None:
        super().__init__()

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
            extensions.append("VK_KHR_surface")
            extensions += GpuContext._help_compute_required_platform_instance_extensions_for_present_support()

        if enable_portability_subset:
            extensions.append("VK_KHR_portability_enumeration")
            extensions.append("VK_KHR_get_physical_device_properties2")
            flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR

        return vkCreateInstance(
            pCreateInfo=VkInstanceCreateInfo(
                pApplicationInfo=VkApplicationInfo(
                    pApplicationName=app_name,
                    pEngineName="zero",
                    apiVersion=VK_API_VERSION_1_4,  # newest supported version
                ),
                enabledLayerCount=len(layers),
                ppEnabledLayerNames=layers,
                enabledExtensionCount=len(extensions),
                ppEnabledExtensionNames=extensions,
                flags=flags,
            ),
            pAllocator=None,
        )

    @staticmethod
    def _help_compute_required_platform_instance_extensions_for_present_support() -> (
        list[str]
    ):
        """
        glfw.get_required_instance_extensions(), but no need to init GLFW.
        """

        # See: GLFW's recognized extensions
        # https://github.com/glfw/glfw/blob/162896e5b9a40dc382c5c438cd12c90a5ff86ddd/src/vulkan.c#L129

        match sys.platform:
            case "win32":
                return ["VK_KHR_win32_surface"]
            case "linux":
                return GpuContext._help_compute_required_platform_instance_extensions_for_present_support_on_linux()
            case "darwin":
                return ["VK_EXT_metal_surface"]
            case _:
                raise PlatformSupportError(
                    "Unrecognized platform when computing required Vulkan instance "
                    f"extensions for Present (VkSurfaceKHR) support: {sys.platform!r}"
                )

    @staticmethod
    def _help_compute_required_platform_instance_extensions_for_present_support_on_linux() -> (
        list[str]
    ):
        """
        glfw.get_required_instance_extensions(), but no need to init GLFW: Linux only.
        Wayland or X11?
        """

        # See: GLFW's code for detecting X11 or Wayland (or other)
        # https://github.com/glfw/glfw/blob/162896e5b9a40dc382c5c438cd12c90a5ff86ddd/src/platform.c#L89
        #
        # In addition to checking XDG_SESSION_TYPE, we further examine other environment
        # variables to determine whether the session is valid.

        session = os.environ.get("XDG_SESSION_TYPE")
        if session == "wayland" and os.environ.get("WAYLAND_DISPLAY"):
            return ["VK_KHR_wayland_surface"]
        elif session == "x11" and os.environ.get("DISPLAY"):
            return ["VK_KHR_xcb_surface"]
        else:
            raise PlatformSupportError(
                "Unrecognized display backend when computing required Vulkan instance "
                "extensions for Present (VkSurfaceKHR) support: failed to detect "
                "Wayland or X11 on a Linux host: are you running a window server?"
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
            GpuPhysicalDevice(
                context=self,
                vk_physical_device=vk_physical_device,
                vk_physical_device_properties=vkGetPhysicalDeviceProperties(
                    vk_physical_device
                ),
                vk_physical_device_memory_properties=vkGetPhysicalDeviceMemoryProperties(
                    vk_physical_device
                ),
            )
            for vk_physical_device in vkEnumeratePhysicalDevices(self._vk_instance)
        ]

    def create_device(
        self,
        *,
        physical_device: GpuPhysicalDevice,
        surface: VkSurfaceKHR | None,
    ) -> GpuDevice:
        # Ensure Vulkan 1.3 support
        physical_device.check_vulkan_1_3_support()

        # Compute queue family indices
        qfis = GpuQueueFamilyIndices.find(
            physical_device,
            surface=surface,
        )
        queue_create_info_list = qfis.compute_queue_create_info_list()

        # Compute extensions
        extensions = []
        extensions.append("VK_KHR_dynamic_rendering")
        if self.enable_present_support:
            extensions.append("VK_KHR_swapchain")
        if self.enable_portability_subset:
            extensions.append("VK_KHR_portability_subset")

        # Create device:
        vk_device = vkCreateDevice(
            physical_device.vk_physical_device,
            VkDeviceCreateInfo(
                enabledExtensionCount=len(extensions),
                ppEnabledExtensionNames=extensions,
                queueCreateInfoCount=len(queue_create_info_list),
                pQueueCreateInfos=queue_create_info_list,
            ),
            pAllocator=None,
        )

        # Create command pools:
        vk_command_pools: dict[int, VkCommandPool] = {}
        for qfi_index in {idx for _, idx in qfis}:
            vk_command_pools[qfi_index] = vkCreateCommandPool(
                device=vk_device,
                pCreateInfo=VkCommandPoolCreateInfo(
                    flags=VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
                    queueFamilyIndex=qfi_index,
                ),
                pAllocator=None,
            )

        # Return final GpuDevice:
        return GpuDevice(
            context=self,
            physical_device=physical_device,
            qfis=qfis,
            vk_device=vk_device,
            vk_command_pools=vk_command_pools,
        )


GpuResource: TypeAlias = BaseContextResource[GpuContext]


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
    vk_physical_device: VkPhysicalDevice
    vk_properties: VkPhysicalDeviceProperties
    vk_memory_properties: VkPhysicalDeviceMemoryProperties

    def __init__(
        self,
        *,
        context: GpuContext,
        vk_physical_device: VkPhysicalDevice,
        vk_physical_device_properties: VkPhysicalDeviceProperties,
        vk_physical_device_memory_properties: VkPhysicalDeviceMemoryProperties,
    ) -> None:
        super().__init__(parent=context)
        self.vk_physical_device = vk_physical_device
        self.vk_properties = vk_physical_device_properties
        self.vk_memory_properties = vk_physical_device_memory_properties

    def _on_dispose(self) -> None:
        pass  # no private resources to free

    @property
    def name(self) -> str:
        return self.vk_properties.deviceName

    @property
    def vk_api_version(self) -> int:
        return self.vk_properties.apiVersion

    def check_vulkan_1_3_support(self):
        if self.vk_properties.apiVersion >= VK_API_VERSION_1_3:
            return
        raise PlatformSupportError(
            f"Physical device {self.name!r} does not support Vulkan 1.3.\n"
            f"- provided: {vk_api_version_str(self.vk_api_version)}\n"
            f"- required: {vk_api_version_str(VK_API_VERSION_1_3)}"
        )

    def spell_device_type(self) -> str:
        return {
            int(VK_PHYSICAL_DEVICE_TYPE_OTHER): "Other",
            int(VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU): "IntegratedGpu",
            int(VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU): "DiscreteGpu",
            int(VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU): "VirtualGpu",
            int(VK_PHYSICAL_DEVICE_TYPE_CPU): "Cpu",
        }[self.vk_properties.deviceType]

    def get_queue_families(
        self,
        surface: VkSurfaceKHR | None,
    ) -> list["GpuPhysicalDeviceQueueFamily"]:
        if surface is not None:
            assert self.context.enable_present_support

        qfi_props = vkGetPhysicalDeviceQueueFamilyProperties(self.vk_physical_device)

        res = []
        for index, props in enumerate(qfi_props):
            assert props.queueCount > 0
            supports_graphics = bool(props.queueFlags & VK_QUEUE_GRAPHICS_BIT)
            supports_compute = bool(props.queueFlags & VK_QUEUE_COMPUTE_BIT)
            supports_transfer = bool(props.queueFlags & VK_QUEUE_TRANSFER_BIT)

            supports_present = False
            if surface is not None:
                supports_present = self.context.get_physical_device_surface_support(
                    self.vk_physical_device,
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


GpuQueueType: TypeAlias = Literal[
    "graphics",
    "compute",
    "transfer",
    "present",
]


@dataclass
class GpuQueueFamilyIndices:
    type_to_qfi_map: dict[GpuQueueType, int] = field(default_factory=dict)

    def __getitem__(self, key: GpuQueueType) -> int:
        return self.type_to_qfi_map[key]

    def __setitem__(self, key: GpuQueueType, value: int) -> None:
        self.type_to_qfi_map[key] = value

    def __contains__(self, key: GpuQueueType) -> bool:
        return key in self.type_to_qfi_map

    def __iter__(self):
        yield from self.type_to_qfi_map.items()

    def is_complete(self, require_present_support: bool) -> bool:
        return (
            "graphics" in self.type_to_qfi_map
            and "compute" in self.type_to_qfi_map
            and "transfer" in self.type_to_qfi_map
            and (not require_present_support or "present" in self.type_to_qfi_map)
        )

    def is_any_queue_family_shared(self) -> bool:
        all_indices = list(self.type_to_qfi_map.values())
        all_unique_indices = set(all_indices)
        return len(all_unique_indices) < len(all_indices)

    def compute_queue_create_info_list(self) -> list[VkDeviceQueueCreateInfo]:
        dd = defaultdict(lambda: 0)
        for _, index in self.type_to_qfi_map.items():
            dd[index] += 1
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
        raise PlatformSupportError(
            "\n".join(
                [
                    f"Physical device {physical_device.name} does not meet the minimum "
                    "engine requirements: failed to find suitable queue families:",
                    f"- {qfi_exclusive=}",
                    f"- {qfi_shared=}",
                ]
            )
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

            if "graphics" not in res and queue_family.supports_graphics:
                res["graphics"] = queue_family.index
                if reserve_queue_and_check_can_continue():
                    continue

            if "compute" not in res and queue_family.supports_compute:
                res["compute"] = queue_family.index
                if reserve_queue_and_check_can_continue():
                    continue

            if "transfer" not in res and queue_family.supports_transfer:
                res["transfer"] = queue_family.index
                if reserve_queue_and_check_can_continue():
                    continue

            if surface is not None:
                if "present" not in res and queue_family.supports_present:
                    res["present"] = queue_family.index
                    if reserve_queue_and_check_can_continue():
                        continue

            if res.is_complete(require_present_support=surface is not None):
                break

        return res


#
# GpuDevice
#


class GpuDevice(GpuResource):
    physical_device: GpuPhysicalDevice
    qfis: GpuQueueFamilyIndices
    vk_device: VkDevice
    vk_command_pools: dict[int, VkCommandPool]
    vk_queues: dict[int, VkQueue]

    def __init__(
        self,
        *,
        context: GpuContext,
        physical_device: GpuPhysicalDevice,
        qfis: GpuQueueFamilyIndices,
        vk_device: VkDevice,
        vk_command_pools: dict[int, VkCommandPool],
    ) -> None:
        super().__init__(parent=context)
        self.physical_device = physical_device
        self.qfis = qfis
        self.vk_device = vk_device
        self.vk_command_pools = vk_command_pools
        # Retrieve one queue per queue family index (queueIndex = 0)
        self.vk_queues = {
            qfi_index: vkGetDeviceQueue(
                self.vk_device,
                queueFamilyIndex=qfi_index,
                queueIndex=0,
            )
            for qfi_index in {idx for _, idx in qfis}
        }

    def _on_dispose(self) -> None:
        # Destroy command pools:
        for _, vk_command_pool in self.vk_command_pools.items():
            vkDestroyCommandPool(
                device=self.vk_device,
                commandPool=vk_command_pool,
                pAllocator=None,
            )

        # Destroy device:
        vkDestroyDevice(device=self.vk_device, pAllocator=None)

    def create_image(
        self,
        *,
        usages: list[GpuImageUsage],
        meta: GpuImageMeta,
    ) -> "GpuImage":
        # Compute the VkImageUsageFlags for image creation:
        # Always include transfer src/dst for copy operations
        vk_usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
        for usage in usages:
            vk_usage |= {
                "texture-binding": VK_IMAGE_USAGE_SAMPLED_BIT,
                "storage-binding": VK_IMAGE_USAGE_STORAGE_BIT,
                "color-attachment": VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                "depth-attachment": (VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT),
            }[usage]

        # Compute the VkImageAspectFlags for image view creation:
        vk_image_aspect = 0
        for usage in usages:
            vk_image_aspect |= {
                "texture-binding": VK_IMAGE_ASPECT_COLOR_BIT,
                "storage-binding": VK_IMAGE_ASPECT_COLOR_BIT,
                "color-attachment": VK_IMAGE_ASPECT_COLOR_BIT,
                "depth-attachment": VK_IMAGE_ASPECT_DEPTH_BIT,
            }[usage]

        # Determine which queue families will access the image:
        queue_family_indices = list({idx for _, idx in self.qfis})

        # Create the VkImage:
        vk_image = vkCreateImage(
            device=self.vk_device,
            pCreateInfo=VkImageCreateInfo(
                flags=0,
                imageType=VK_IMAGE_TYPE_2D,
                format=meta.infer_vk_format(usages),
                extent=VkExtent3D(width=meta.shape[1], height=meta.shape[0], depth=1),
                mipLevels=1,
                arrayLayers=1,
                samples=VK_SAMPLE_COUNT_1_BIT,
                tiling=VK_IMAGE_TILING_OPTIMAL,
                usage=vk_usage,
                sharingMode=VK_SHARING_MODE_EXCLUSIVE,
                queueFamilyIndexCount=len(queue_family_indices),
                pQueueFamilyIndices=queue_family_indices,
                initialLayout=VK_IMAGE_LAYOUT_UNDEFINED,
            ),
            pAllocator=None,
        )

        # Allocate and bind memory for the image
        memory_requirements = vkGetImageMemoryRequirements(
            device=self.vk_device,
            image=vk_image,
        )
        memory = self._allocate_memory(
            memory_requirements=memory_requirements,
            device_local=True,
        )
        vkBindImageMemory(
            device=self.vk_device,
            image=vk_image,
            memory=memory.vk_device_memory,
            memoryOffset=VkDeviceSize(0),
        )

        # Create the default VkImageView:
        vk_image_view = vkCreateImageView(
            device=self.vk_device,
            pCreateInfo=VkImageViewCreateInfo(
                flags=0,
                image=vk_image,
                viewType=VK_IMAGE_TYPE_2D,
                format=meta.infer_vk_format(usages),
                components=VkComponentMapping(
                    r=VK_COMPONENT_SWIZZLE_IDENTITY,
                    g=VK_COMPONENT_SWIZZLE_IDENTITY,
                    b=VK_COMPONENT_SWIZZLE_IDENTITY,
                    a=VK_COMPONENT_SWIZZLE_IDENTITY,
                ),
                subresourceRange=VkImageSubresourceRange(
                    aspectMask=vk_image_aspect,
                    baseMipLevel=0,
                    levelCount=1,
                    baseArrayLayer=0,
                    layerCount=1,
                ),
            ),
            pAllocator=None,
        )

        # Make the image:
        image = GpuImage(
            device=self,
            vk_image=vk_image,
            vk_image_view=vk_image_view,
            memory=memory,
            meta=meta,
            aspect_mask=vk_image_aspect,
            usages=usages,
        )

        # Done:
        return image

    def create_buffer(
        self,
        *,
        usages: list[GpuBufferUsage],
        meta: GpuBufferMeta,
    ):
        # Compute VkBufferUsageFlags
        vk_usage = 0
        for usage in usages:
            vk_usage |= {
                "copy-src": VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                "copy-dst": VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                "uniform": VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                "storage": VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            }.get(usage, 0)

        # Compute whether 'device_local' is required to be True or False
        device_local = None
        for usage in usages:
            required_device_local_value = {
                "staging": False,
                "uniform": True,
                "storage": True,
            }.get(usage)

            # If no constraint is imposed, continue
            if required_device_local_value is None:
                continue

            # Check for conflict:
            if device_local is not None and required_device_local_value != device_local:
                raise LogicError(
                    "\n".join(
                        [
                            f"Inconsistent buffer usages supplied: {usages=}",
                            (
                                "Some usages require the memory to be "
                                "device-local, while others require the memory to "
                                "be host-local."
                            ),
                        ]
                    )
                )

            # Apply the constraint
            device_local = required_device_local_value

        # Check if the 'device_local' bool was inferred successfully.
        if device_local is None:
            raise LogicError(
                "\n".join(
                    [
                        f"Insufficient buffer usages supplied: {usages=}",
                        (
                            "Could not determine whether to allocate the buffer on the "
                            "device or the host."
                        ),
                    ]
                )
            )

        # Determine which queue families will access the buffer:
        queue_family_indices = list({idx for _, idx in self.qfis})

        # Create the VkBuffer
        vk_buffer = vkCreateBuffer(
            device=self.vk_device,
            pCreateInfo=VkBufferCreateInfo(
                flags=0,
                size=meta.size,
                usage=vk_usage,
                sharingMode=VK_SHARING_MODE_EXCLUSIVE,
                queueFamilyIndexCount=len(queue_family_indices),
                pQueueFamilyIndices=queue_family_indices,
            ),
            pAllocator=None,
        )

        # Allocate and bind memory for the buffer:
        memory_requirements = vkGetBufferMemoryRequirements(
            device=self.vk_device,
            buffer=vk_buffer,
        )
        memory = self._allocate_memory(
            memory_requirements=memory_requirements,
            device_local=device_local,
        )
        vkBindBufferMemory(
            device=self.vk_device,
            buffer=vk_buffer,
            memory=memory.vk_device_memory,
            memoryOffset=0,
        )

        # Make the GpuBuffer:
        buffer = GpuBuffer(
            device=self,
            vk_buffer=vk_buffer,
            memory=memory,
            meta=meta,
            usages=usages,
        )

        # Done:
        return buffer

    @contextmanager
    def command(self, *, queue_type: GpuQueueType):
        queue_family_index = self.qfis[queue_type]
        vk_command_pool = self.vk_command_pools[queue_family_index]

        vk_command_buffer = vkAllocateCommandBuffers(
            device=self.vk_device,
            pAllocateInfo=VkCommandBufferAllocateInfo(
                commandPool=vk_command_pool,
                level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                commandBufferCount=1,
            ),
        )[0]

        vkBeginCommandBuffer(
            commandBuffer=vk_command_buffer,
            pBeginInfo=VkCommandBufferBeginInfo(
                flags=0,
                pInheritanceInfo=None,
            ),
        )
        yield GpuCommandEncoder(
            device=self,
            queue_family_index=queue_family_index,
            vk_command_buffer=vk_command_buffer,
        )
        vkEndCommandBuffer(commandBuffer=vk_command_buffer)
        # Auto-submit and free
        self.submit(
            queue_type=queue_type,
            command_buffers=[vk_command_buffer],
        )
        vkFreeCommandBuffers(
            device=self.vk_device,
            commandPool=vk_command_pool,
            commandBufferCount=1,
            pCommandBuffers=[vk_command_buffer],
        )

    def submit(
        self,
        *,
        queue_type: GpuQueueType,
        command_buffers: list[VkCommandBuffer],
        wait_semaphores: Optional[list["GpuSemaphore"]] = None,
        signal_semaphores: Optional[list["GpuSemaphore"]] = None,
        fence: Optional["GpuFence"] = None,
    ) -> None:
        vk_queue = self.vk_queues[self.qfis[queue_type]]
        wait_sems = [s.vk_semaphore for s in (wait_semaphores or [])]
        signal_sems = [s.vk_semaphore for s in (signal_semaphores or [])]
        submit_info = VkSubmitInfo(
            waitSemaphoreCount=len(wait_sems),
            pWaitSemaphores=wait_sems or None,
            pWaitDstStageMask=[VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT] * len(wait_sems)
            if wait_sems
            else None,
            commandBufferCount=len(command_buffers),
            pCommandBuffers=command_buffers,
            signalSemaphoreCount=len(signal_sems),
            pSignalSemaphores=signal_sems or None,
        )
        vkQueueSubmit(
            vk_queue,
            submitCount=1,
            pSubmits=[submit_info],
            fence=(fence.vk_fence if fence else None),
        )

    def _allocate_memory(
        self,
        *,
        memory_requirements: VkMemoryRequirements,
        device_local: bool,
    ) -> GpuMemory:
        vk_device_memory = vkAllocateMemory(
            device=self.vk_device,
            pAllocateInfo=VkMemoryAllocateInfo(
                allocationSize=memory_requirements.size,
                memoryTypeIndex=self._find_memory_type(
                    memory_requirements.memoryTypeBits,
                    required_properties=(
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
                        if device_local
                        else (
                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                            | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
                        )
                    ),
                ),
            ),
            pAllocator=None,
        )
        return GpuMemory(
            device=self,
            vk_device_memory=vk_device_memory,
            size=memory_requirements.size,
            device_local=device_local,
        )

    def _find_memory_type(
        self,
        type_filter: int,
        required_properties: int,
    ) -> int:
        mem_props = self.physical_device.vk_memory_properties
        for i in range(mem_props.memoryTypeCount):
            if (type_filter & (1 << i)) == 0:
                continue

            matched = mem_props.memoryTypes[i].propertyFlags & required_properties
            if matched != required_properties:
                continue

            return i

        raise LogicError("Failed to find suitable memory type")


#
# GpuMemory
#


class GpuMemory(GpuResource):
    device: GpuDevice
    vk_device_memory: VkDeviceMemory
    size: int
    device_local: bool
    _mapped_view_counter: int
    _mapped_view: memoryview | None

    def __init__(
        self,
        *,
        device: GpuDevice,
        vk_device_memory: VkDeviceMemory,
        size: int,
        device_local: bool,
    ):
        super().__init__(parent=device)
        self.device = device
        self.vk_device_memory = vk_device_memory
        self.size = size
        self.device_local = device_local
        self._mapped_view_use_count = 0
        self._mapped_view = None

    def _on_dispose(self) -> None:
        vkFreeMemory(self.device.vk_device, self.vk_device_memory, pAllocator=None)

    @contextmanager
    def map(self):
        if self.device_local:
            raise LogicError("Cannot memory-map a device-local GpuMemory.")

        self._mapped_view_use_count += 1
        if self._mapped_view is None:
            self._mapped_view = vkMapMemory(
                device=self.device.vk_device,
                memory=self.vk_device_memory,
                offset=0,
                size=self.size,
                flags=0,
            )

        assert self._mapped_view is not None
        yield self._mapped_view

        self._mapped_view_use_count -= 1
        if self._mapped_view_use_count == 0:
            vkUnmapMemory(
                device=self.device.vk_device,
                memory=self.vk_device_memory,
            )
            self._mapped_view = None


#
# GpuImage
#


@dataclass
class GpuImageMeta:
    shape: tuple[int, int, int]  # (height, width, channels)
    dtype: torch.dtype

    @staticmethod
    def from_tensor(tensor: torch.Tensor) -> "GpuImageMeta":
        if tensor.ndim != 3:
            raise LogicError(
                f"GpuTextureSpec can only be created from 3D tensors, got tensor with "
                f"{tensor.ndim} dimensions instead"
            )
        return GpuImageMeta(
            shape=(tensor.shape[0], tensor.shape[1], tensor.shape[2]),
            dtype=tensor.dtype,
        )

    def infer_vk_format(
        self,
        usages: list[GpuImageUsage],
    ) -> int:
        dtype = self.dtype
        depth = self.shape[2]
        is_depth_attachment = "depth-attachment" in usages

        if is_depth_attachment:
            match (dtype, depth):
                case (torch.float32, 1):
                    return VK_FORMAT_D32_SFLOAT
                case _:
                    raise LogicError(
                        f"Invalid depth attachment image meta: {dtype=}, {depth=}"
                    )
        else:
            match (dtype, depth):
                case (torch.uint8, 4):
                    return VK_FORMAT_R8G8B8A8_UNORM
                case (torch.float32, 1):
                    return VK_FORMAT_R32_SFLOAT
                case (torch.float32, 4):
                    return VK_FORMAT_R32G32B32A32_SFLOAT
                case _:
                    raise LogicError(
                        f"Invalid image meta: "
                        f"{dtype=}, {depth=}, {is_depth_attachment=}"
                    )

    def into_buffer_meta(self) -> GpuBufferMeta:
        return GpuBufferMeta(
            element_count=(self.shape[0] * self.shape[1] * self.shape[2]),
            element_dtype=self.dtype,
        )


GpuImageUsage: TypeAlias = Literal[
    "texture-binding",
    "storage-binding",
    "color-attachment",
    "depth-attachment",
]


class GpuImage(GpuResource):
    device: GpuDevice
    vk_image: VkImage
    vk_image_view: VkImageView
    memory: GpuMemory
    meta: GpuImageMeta
    aspect_mask: int
    usages: list[GpuImageUsage]

    def __init__(
        self,
        *,
        device: GpuDevice,
        vk_image: VkImage,
        vk_image_view: VkImageView,
        memory: GpuMemory,
        meta: GpuImageMeta,
        aspect_mask: int,
        usages: list[GpuImageUsage],
    ) -> None:
        super().__init__(parent=device)
        self.device = device
        self.vk_image = vk_image
        self.vk_image_view = vk_image_view
        self.memory = memory
        self.meta = meta
        self.aspect_mask = aspect_mask
        self.usages = usages

    def _on_dispose(self) -> None:
        vkDestroyImage(self.device.vk_device, self.vk_image, pAllocator=None)


# Synchronization wrappers


class GpuSemaphore(GpuResource):
    device: GpuDevice
    vk_semaphore: VkSemaphore

    def __init__(self, *, device: GpuDevice) -> None:
        super().__init__(parent=device)
        self.device = device
        self.vk_semaphore = vkCreateSemaphore(
            device=device.vk_device,
            pCreateInfo=VkSemaphoreCreateInfo(flags=0),
            pAllocator=None,
        )

    def _on_dispose(self) -> None:
        vkDestroySemaphore(self.device.vk_device, self.vk_semaphore, pAllocator=None)


class GpuFence(GpuResource):
    device: GpuDevice
    vk_fence: VkFence

    def __init__(self, *, device: GpuDevice) -> None:
        super().__init__(parent=device)
        self.device = device
        self.vk_fence = vkCreateFence(
            device=device.vk_device,
            pCreateInfo=VkFenceCreateInfo(flags=0),
            pAllocator=None,
        )

    def _on_dispose(self) -> None:
        vkDestroyFence(self.device.vk_device, self.vk_fence, pAllocator=None)


#
# GpuBuffer
#


@dataclass
class GpuBufferMeta:
    element_count: int
    element_dtype: torch.dtype

    @property
    def size(self) -> int:
        return self.element_count * self.element_size

    @property
    def element_size(self) -> int:
        return {torch.uint8: 1, torch.float32: 4}[self.element_dtype]

    @staticmethod
    def from_tensor(tensor: torch.Tensor) -> GpuBufferMeta:
        return GpuBufferMeta(
            element_count=tensor.numel(),
            element_dtype=tensor.dtype,
        )


GpuBufferUsage: TypeAlias = Literal[
    "staging",
    "copy-src",
    "copy-dst",
    "uniform",
    "storage",
]


class GpuBuffer(GpuResource):
    device: GpuDevice
    vk_buffer: VkBuffer
    memory: GpuMemory
    meta: GpuBufferMeta
    usages: list[GpuBufferUsage]

    def __init__(
        self,
        device: GpuDevice,
        vk_buffer: VkBuffer,
        memory: GpuMemory,
        meta: GpuBufferMeta,
        usages: list[GpuBufferUsage],
    ):
        super().__init__(parent=device)
        self.device = device
        self.vk_buffer = vk_buffer
        self.memory = memory
        self.meta = meta
        self.usages = usages

    def _on_dispose(self):
        vkDestroyBuffer(self.device.vk_device, self.vk_buffer, pAllocator=None)


#
# GpuCommandBuffer
#


GpuCommandBufferLevel: TypeAlias = Literal["primary", "secondary"]


class GpuCommandEncoder(GpuResource):
    device: GpuDevice
    vk_command_buffer: VkCommandBuffer
    queue_family_index: int

    def __init__(
        self,
        *,
        device: GpuDevice,
        queue_family_index: int,
        vk_command_buffer: VkCommandBuffer,
    ) -> None:
        super().__init__(parent=device)
        self.device = device
        self.vk_command_buffer = vk_command_buffer
        self.queue_family_index = queue_family_index

    def _on_dispose(self) -> None:
        vkFreeCommandBuffers(
            device=self.device.vk_device,
            commandPool=self.device.vk_command_pools[self.queue_family_index],
            commandBufferCount=1,
            pCommandBuffers=[self.vk_command_buffer],
        )

    def copy_buffer_to_buffer(
        self,
        *,
        src: GpuBuffer,
        dst: GpuBuffer,
        size: int,
        src_offset: int = 0,
        dst_offset: int = 0,
    ) -> None:
        vkCmdCopyBuffer(
            commandBuffer=self.vk_command_buffer,
            srcBuffer=src.vk_buffer,
            dstBuffer=dst.vk_buffer,
            regionCount=1,
            pRegions=[
                VkBufferCopy(
                    srcOffset=src_offset,
                    dstOffset=dst_offset,
                    size=size,
                )
            ],
        )

    def copy_buffer_to_image(
        self,
        *,
        src: GpuBuffer,
        dst: GpuImage,
    ) -> None:
        vkCmdCopyBufferToImage(
            commandBuffer=self.vk_command_buffer,
            srcBuffer=src.vk_buffer,
            dstImage=dst.vk_image,
            dstImageLayout=VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            regionCount=1,
            pRegions=[
                VkBufferImageCopy(
                    bufferOffset=0,
                    bufferRowLength=0,
                    bufferImageHeight=0,
                    imageSubresource=VkImageSubresourceLayers(
                        aspectMask=dst.aspect_mask,
                        mipLevel=0,
                        baseArrayLayer=0,
                        layerCount=1,
                    ),
                    imageOffset=VkOffset3D(x=0, y=0, z=0),
                    imageExtent=VkExtent3D(
                        width=dst.meta.shape[1],
                        height=dst.meta.shape[0],
                        depth=1,
                    ),
                )
            ],
        )

    # Convenience methods

    def _write_memory_with_mmap(self, *, buffer: GpuBuffer, data: torch.Tensor) -> None:
        if buffer.memory.device_local:
            raise LogicError("Cannot mmap device-local memory; use a staging buffer.")
        with buffer.memory.map() as mv:
            mv[:] = memoryview(data.numpy())

    def _read_memory_with_mmap(self, *, buffer: GpuBuffer) -> torch.Tensor:
        if buffer.memory.device_local:
            raise LogicError("Cannot mmap device-local memory; use a staging buffer.")
        with buffer.memory.map() as mv:
            return torch.frombuffer(
                mv[: buffer.meta.size],
                dtype=buffer.meta.element_dtype,
            )

    def write_buffer(
        self,
        *,
        dst: GpuBuffer,
        data: torch.Tensor,
        staging_buffer: GpuBuffer | None = None,
    ) -> None:
        if staging_buffer is not None:
            if "staging" not in staging_buffer.usages:
                raise LogicError(
                    "Provided staging_buffer does not have 'staging' usage"
                )
            if "staging" in dst.usages:
                raise LogicError("Destination buffer must not have 'staging' usage")
            self._write_memory_with_mmap(buffer=staging_buffer, data=data)
            self.copy_buffer_to_buffer(src=staging_buffer, dst=dst, size=len(data))
        else:
            if dst.memory.device_local:
                raise LogicError(
                    "Need a staging buffer to write to device-local memory"
                )
            self._write_memory_with_mmap(buffer=dst, data=data)

    def read_buffer(
        self,
        *,
        src: GpuBuffer,
        staging_buffer: GpuBuffer | None = None,
    ) -> torch.Tensor | None:
        if staging_buffer is not None:
            if "staging" not in staging_buffer.usages:
                raise LogicError(
                    "Provided staging_buffer does not have 'staging' usage"
                )
            if "staging" in src.usages:
                raise LogicError(
                    "Source buffer must not have 'staging' usage when using a staging buffer"
                )
            self.copy_buffer_to_buffer(src=src, dst=staging_buffer, size=src.meta.size)
            # Defer mapping until after submission (returns None)
            return None
        else:
            if src.memory.device_local:
                raise LogicError(
                    "Need a staging buffer to read from device-local memory"
                )
            raw = self._read_memory_with_mmap(buffer=src)
            arr = torch.frombuffer(raw, dtype=src.meta.element_dtype)
            return arr.clone()

    def write_image(
        self,
        *,
        dst: GpuImage,
        data: torch.Tensor,
        staging_buffer: GpuBuffer | None = None,
    ) -> None:
        if staging_buffer is not None:
            if "staging" not in staging_buffer.usages:
                raise LogicError(
                    "Provided staging_buffer does not have 'staging' usage"
                )
            # Images are always device-local; ensure destination image does not erroneously claim staging usage
            if "depth-attachment" in dst.usages and data.shape[2] != 1:
                raise LogicError(
                    "Depth attachment image write expects single channel data"
                )
            self._write_memory_with_mmap(buffer=staging_buffer, data=data)
            self.copy_buffer_to_image(src=staging_buffer, dst=dst)
        else:
            raise LogicError(
                "Image writes require a staging buffer (no direct mapping)."
            )

    def read_image(
        self,
        *,
        src: GpuImage,
        staging_buffer: GpuBuffer,
    ) -> None:
        if "staging" not in staging_buffer.usages:
            raise LogicError("Provided staging_buffer does not have 'staging' usage")
        # Copy image -> buffer
        vkCmdCopyImageToBuffer(
            commandBuffer=self.vk_command_buffer,
            srcImage=src.vk_image,
            srcImageLayout=VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            dstBuffer=staging_buffer.vk_buffer,
            regionCount=1,
            pRegions=[
                VkBufferImageCopy(
                    bufferOffset=0,
                    bufferRowLength=0,
                    bufferImageHeight=0,
                    imageSubresource=VkImageSubresourceLayers(
                        aspectMask=src.aspect_mask,
                        mipLevel=0,
                        baseArrayLayer=0,
                        layerCount=1,
                    ),
                    imageOffset=VkOffset3D(x=0, y=0, z=0),
                    imageExtent=VkExtent3D(
                        width=src.meta.shape[1],
                        height=src.meta.shape[0],
                        depth=1,
                    ),
                )
            ],
        )
        # Defer mapping until after submission

    # Finalization helpers (called after command buffer submission)
    def finalize_read_buffer(self, staging_buffer: GpuBuffer) -> torch.Tensor:
        if "staging" not in staging_buffer.usages:
            raise LogicError("Provided staging_buffer does not have 'staging' usage")
        arr = self._read_memory_with_mmap(buffer=staging_buffer)
        return arr.clone()

    def finalize_read_image(
        self, staging_buffer: GpuBuffer, image: GpuImage
    ) -> torch.Tensor:
        if "staging" not in staging_buffer.usages:
            raise LogicError("Provided staging_buffer does not have 'staging' usage")
        raw = self._read_memory_with_mmap(buffer=staging_buffer)
        meta = image.meta
        numel = meta.shape[0] * meta.shape[1] * meta.shape[2]
        arr = torch.frombuffer(raw, dtype=meta.dtype, count=numel).clone()
        return arr.view(*meta.shape)
