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

from dataclasses import dataclass
from collections import defaultdict
from typing import TypeAlias, Literal
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
    # VkMemory
    VkMemoryRequirements,
    VkMemoryAllocateInfo,
    vkAllocateMemory,
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

        # Done:
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
        return GpuDevice(
            context=self,
            physical_device=physical_device,
            qfis=qfis,
            vk_device=vk_device,
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
            for index in [self.graphics, self.compute, self.transfer, self.present]
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

    def __iter__(self):
        if self.graphics is not None:
            yield self.graphics
        if self.compute is not None:
            yield self.compute
        if self.transfer is not None:
            yield self.transfer
        if self.present is not None:
            yield self.present


#
# GpuDevice
#


class GpuDevice(GpuResource):
    physical_device: GpuPhysicalDevice
    qfis: GpuQueueFamilyIndices
    vk_device: VkDevice

    def __init__(
        self,
        *,
        context: GpuContext,
        physical_device: GpuPhysicalDevice,
        qfis: GpuQueueFamilyIndices,
        vk_device: VkDevice,
    ) -> None:
        super().__init__(parent=context)
        self.physical_device = physical_device
        self.qfis = qfis
        self.vk_device = vk_device

    def _on_dispose(self) -> None:
        vkDestroyDevice(device=self.vk_device, pAllocator=None)

    def create_image(
        self,
        *,
        usages: tuple[GpuImageUsage, ...],
        init: torch.Tensor | None = None,
        meta: GpuImageMeta | None = None,
    ) -> "GpuImage":
        # Resolve 'shape'
        if meta is not None and init is not None:
            raise LogicError("GpuImage(): cannot provide both init and spec")
        elif meta is not None:
            resolved_meta: GpuImageMeta = meta
        elif init is not None:
            resolved_meta = GpuImageMeta.from_tensor(init)
        else:
            raise LogicError("GpuImage(): either init or spec must be provided")

        # Compute the VkImageUsageFlags for image creation:
        vk_usage = 0
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
        queue_family_indices = list(self.qfis)

        # Create the VkImage:
        vk_image = vkCreateImage(
            device=self.vk_device,
            pCreateInfo=VkImageCreateInfo(
                flags=0,
                imageType=VK_IMAGE_TYPE_2D,
                format=resolved_meta.infer_vk_format(usages),
                extent=VkExtent3D(
                    width=resolved_meta.shape[1],
                    height=resolved_meta.shape[0],
                    depth=1,
                ),
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
        memory = vkAllocateMemory(
            device=self.vk_device,
            pAllocateInfo=VkMemoryAllocateInfo(
                allocationSize=memory_requirements.size,
                memoryTypeIndex=self._find_memory_type(
                    memory_requirements.memoryTypeBits,
                    required_properties=VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                ),
            ),
            pAllocator=None,
        )
        vkBindImageMemory(
            device=self.vk_device,
            image=vk_image,
            memory=memory,
            memoryOffset=0,
        )

        # Create the default VkImageView:
        vk_image_view = vkCreateImageView(
            device=self.vk_device,
            pCreateInfo=VkImageViewCreateInfo(
                flags=0,
                image=vk_image,
                viewType=VK_IMAGE_TYPE_2D,
                format=resolved_meta.infer_vk_format(usages),
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

        # TODO: upload 'init' data to the image

        # Done:
        return GpuImage(device=self, vk_image=vk_image, vk_image_view=vk_image_view)

    def _find_memory_type(
        self,
        type_filter: int,
        required_properties: int,
    ) -> int:
        mem_props = self.physical_device.vk_memory_properties
        for i in range(mem_props.memoryTypeCount):
            if (type_filter & (1 << i)) == 0:
                print("Rejecting memory type", i)
                continue

            matched = mem_props.memoryTypes[i].propertyFlags & required_properties
            if matched != required_properties:
                print("Rejecting memory type", i, "due to properties")
                continue

            return i

        raise LogicError("Failed to find suitable memory type")


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
        usages: tuple[GpuImageUsage, ...],
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
                        f"Invalid depth attachment spec: {dtype=}, {depth=}"
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
                        f"Invalid image spec: "
                        f"{dtype=}, {depth=}, {is_depth_attachment=}"
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

    def __init__(
        self,
        *,
        device: GpuDevice,
        vk_image: VkImage,
        vk_image_view: VkImageView,
    ) -> None:
        super().__init__(parent=device)
        self.device = device
        self.vk_image = vk_image
        self.vk_image_view = vk_image_view

    def _on_dispose(self) -> None:
        vkDestroyImage(self.device.vk_device, self.vk_image, pAllocator=None)


#
# GpuCommandBuffer
#


class GpuCommandBuffer:
    pass
