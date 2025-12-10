"""
GPU abstraction layer.

Required Vulkan version:
-   Vulkan 1.3 for VK_KHR_dynamic_rendering
    https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_dynamic_rendering.html
"""

__all__ = [
    "GpuContext",
    "GpuPhysicalDevice",
    "GpuDevice",
    "GpuShader",
    "GpuPipeline",
    "GpuPipelineLayout",
    "GpuImage",
    "GpuImageUsage",
    "GpuImageMeta",
    "GpuSurface",
    "GpuSwapChain",
    "GpuSampler",
    "GpuDescriptorSetLayout",
    "GpuDescriptorSet",
]

import json
import os
import sys
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Literal, TypeAlias

import numpy as np
import numpy.typing as npt

from .basic import BaseResource
from .excepts import LogicError, PlatformSupportError
from .typed_vulkan import (
    VK_ACCESS_SHADER_READ_BIT,
    VK_ACCESS_TRANSFER_READ_BIT,
    VK_ACCESS_TRANSFER_WRITE_BIT,
    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
    VK_API_VERSION_1_3,
    VK_API_VERSION_1_4,
    VK_ATTACHMENT_LOAD_OP_CLEAR,
    VK_ATTACHMENT_LOAD_OP_LOAD,
    VK_ATTACHMENT_STORE_OP_STORE,
    VK_BLEND_FACTOR_ZERO,
    VK_BLEND_FACTOR_ONE,
    VK_BLEND_FACTOR_SRC_ALPHA,
    VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
    VK_BLEND_OP_ADD,
    VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
    VK_COLOR_COMPONENT_A_BIT,
    VK_COLOR_COMPONENT_B_BIT,
    VK_COLOR_COMPONENT_G_BIT,
    VK_COLOR_COMPONENT_R_BIT,
    VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,
    VK_COMMAND_BUFFER_LEVEL_PRIMARY,
    VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
    VK_COMPONENT_SWIZZLE_IDENTITY,
    VK_COMPARE_OP_ALWAYS,
    VK_COMPARE_OP_LESS,
    VK_COMPARE_OP_LESS_OR_EQUAL,
    VK_COMPARE_OP_GREATER,
    VK_COMPARE_OP_GREATER_OR_EQUAL,
    VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
    VK_CULL_MODE_NONE,
    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
    VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
    VK_DESCRIPTOR_TYPE_SAMPLER,
    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
    VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
    VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
    VK_FENCE_CREATE_SIGNALED_BIT,
    VK_FILTER_LINEAR,
    VK_FILTER_NEAREST,
    VK_FORMAT_R8G8B8A8_SRGB,
    VK_FORMAT_B8G8R8A8_SRGB,
    VK_FORMAT_D32_SFLOAT,
    VK_FORMAT_R8G8B8A8_UNORM,
    VK_FORMAT_R32_SFLOAT,
    VK_FORMAT_R32G32B32A32_SFLOAT,
    VK_FORMAT_UNDEFINED,
    VK_FRONT_FACE_COUNTER_CLOCKWISE,
    VK_IMAGE_ASPECT_COLOR_BIT,
    VK_IMAGE_ASPECT_DEPTH_BIT,
    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    VK_IMAGE_LAYOUT_GENERAL,
    VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
    VK_IMAGE_LAYOUT_UNDEFINED,
    VK_IMAGE_TILING_OPTIMAL,
    VK_IMAGE_TYPE_2D,
    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
    VK_IMAGE_USAGE_SAMPLED_BIT,
    VK_IMAGE_USAGE_STORAGE_BIT,
    VK_IMAGE_USAGE_TRANSFER_DST_BIT,
    VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
    VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR,
    VK_LOGIC_OP_COPY,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
    VK_PHYSICAL_DEVICE_TYPE_CPU,
    VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU,
    VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU,
    VK_PHYSICAL_DEVICE_TYPE_OTHER,
    VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU,
    VK_PIPELINE_BIND_POINT_GRAPHICS,
    VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
    VK_PIPELINE_STAGE_TRANSFER_BIT,
    VK_POLYGON_MODE_FILL,
    VK_PRESENT_MODE_FIFO_KHR,
    VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
    VK_QUEUE_COMPUTE_BIT,
    VK_QUEUE_FAMILY_IGNORED,
    VK_QUEUE_GRAPHICS_BIT,
    VK_QUEUE_TRANSFER_BIT,
    VK_SAMPLE_COUNT_1_BIT,
    VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
    VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
    VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT,
    VK_SAMPLER_ADDRESS_MODE_REPEAT,
    VK_SAMPLER_MIPMAP_MODE_LINEAR,
    VK_SHADER_STAGE_COMPUTE_BIT,
    VK_SHADER_STAGE_FRAGMENT_BIT,
    VK_SHADER_STAGE_VERTEX_BIT,
    VK_SHARING_MODE_EXCLUSIVE,
    VK_STENCIL_OP_KEEP,
    VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
    VkApplicationInfo,
    VkBuffer,
    VkBufferCopy,
    VkBufferCreateInfo,
    VkBufferImageCopy,
    VkClearColorValue,
    VkClearDepthStencilValue,
    VkClearValue,
    VkCommandBuffer,
    VkCommandBufferAllocateInfo,
    VkCommandBufferBeginInfo,
    VkCommandPool,
    VkCommandPoolCreateInfo,
    VkComponentMapping,
    VkDescriptorBufferInfo,
    VkDescriptorImageInfo,
    VkDescriptorPool,
    VkDescriptorPoolCreateInfo,
    VkDescriptorPoolSize,
    VkDescriptorSet,
    VkDescriptorSetAllocateInfo,
    VkDescriptorSetLayout,
    VkDescriptorSetLayoutBinding,
    VkDescriptorSetLayoutCreateInfo,
    VkDescriptorType,
    VkDevice,
    VkDeviceCreateInfo,
    VkDeviceMemory,
    VkDeviceQueueCreateInfo,
    VkDeviceSize,
    VkExtent2D,
    VkExtent3D,
    VkFence,
    VkFenceCreateInfo,
    VkFormat,
    VkColorSpaceKHR,
    VkGraphicsPipelineCreateInfo,
    VkImage,
    VkImageCopy,
    VkImageCreateInfo,
    VkImageLayout,
    VkImageMemoryBarrier,
    VkImageSubresourceLayers,
    VkImageSubresourceRange,
    VkImageView,
    VkImageViewCreateInfo,
    VkInstance,
    VkInstanceCreateInfo,
    VkMemoryAllocateInfo,
    VkMemoryRequirements,
    VkOffset2D,
    VkOffset3D,
    VkPhysicalDevice,
    VkPhysicalDeviceDynamicRenderingFeatures,
    VkPhysicalDeviceMemoryProperties,
    VkPhysicalDeviceProperties,
    VkPipeline,
    VkPipelineColorBlendAttachmentState,
    VkPipelineColorBlendStateCreateInfo,
    VkPipelineDepthStencilStateCreateInfo,
    VkPipelineInputAssemblyStateCreateInfo,
    VkPipelineLayout,
    VkPipelineLayoutCreateInfo,
    VkPipelineMultisampleStateCreateInfo,
    VkPipelineRasterizationStateCreateInfo,
    VkPipelineRenderingCreateInfo,
    VkPipelineShaderStageCreateInfo,
    VkPipelineVertexInputStateCreateInfo,
    VkPipelineViewportStateCreateInfo,
    VkPresentInfoKHR,
    VkQueue,
    VkRect2D,
    VkRenderingAttachmentInfo,
    VkRenderingInfo,
    VkResolveModeFlagBits,
    VkSampler,
    VkSamplerCreateInfo,
    VkSemaphore,
    VkSemaphoreCreateInfo,
    VkShaderModule,
    VkShaderModuleCreateInfo,
    VkStencilOpState,
    VkSubmitInfo,
    VkSurfaceFormatKHR,
    VkSurfaceKHR,
    VkSwapchainCreateInfoKHR,
    VkSwapchainKHR,
    VkViewport,
    VkWriteDescriptorSet,
    raw_ffi,
    vk_api_version_str,
    vkAllocateCommandBuffers,
    vkAllocateDescriptorSets,
    vkAllocateMemory,
    vkBeginCommandBuffer,
    vkBindBufferMemory,
    vkBindImageMemory,
    vkCmdBeginRendering,
    vkCmdBindDescriptorSets,
    vkCmdBindPipeline,
    vkCmdCopyBuffer,
    vkCmdCopyBufferToImage,
    vkCmdCopyImage,
    vkCmdCopyImageToBuffer,
    vkCmdDraw,
    vkCmdEndRendering,
    vkCmdPipelineBarrier,
    vkCreateBuffer,
    vkCreateCommandPool,
    vkCreateDescriptorPool,
    vkCreateDescriptorSetLayout,
    vkCreateDevice,
    vkCreateFence,
    vkCreateGraphicsPipelines,
    vkCreateImage,
    vkCreateImageView,
    vkCreateInstance,
    vkCreatePipelineLayout,
    vkCreateSampler,
    vkCreateSemaphore,
    vkCreateShaderModule,
    vkDestroyBuffer,
    vkDestroyCommandPool,
    vkDestroyDescriptorPool,
    vkDestroyDescriptorSetLayout,
    vkDestroyDevice,
    vkDestroyFence,
    vkDestroyImage,
    vkDestroyImageView,
    vkDestroyInstance,
    vkDestroyPipeline,
    vkDestroyPipelineLayout,
    vkDestroySampler,
    vkDestroySemaphore,
    vkDestroyShaderModule,
    vkDeviceWaitIdle,
    vkEndCommandBuffer,
    vkEnumeratePhysicalDevices,
    vkFreeCommandBuffers,
    vkFreeMemory,
    vkGetBufferMemoryRequirements,
    vkGetDeviceQueue,
    vkGetImageMemoryRequirements,
    vkGetInstanceProcAddr,
    vkGetPhysicalDeviceMemoryProperties,
    vkGetPhysicalDeviceProperties,
    vkGetPhysicalDeviceQueueFamilyProperties,
    vkMapMemory,
    vkQueueSubmit,
    vkResetFences,
    vkUnmapMemory,
    vkUpdateDescriptorSets,
    vkWaitForFences,
)

if TYPE_CHECKING:
    from _typeshed import SupportsWrite

#
# GpuContext
#


class GpuContext(BaseResource):
    enable_debug_layer_support: bool
    enable_present_support: bool
    enable_portability_subset: bool
    vk_instance: VkInstance
    vk_extra_proc_tab: dict[str, Callable]

    def __init__(
        self,
        *,
        parent: BaseResource | None = None,
        app_name: str = "Zfw App",
        enable_debug_layer_support: bool = True,
        enable_present_support: bool = True,
        enable_portability_subset_override: bool | None = None,
    ) -> None:
        super().__init__(parent=parent)

        enable_portability_subset = (
            enable_portability_subset_override
            if enable_portability_subset_override is not None
            else (sys.platform == "darwin")  # -> inferred
        )

        self.enable_debug_layer_support = enable_debug_layer_support
        self.enable_present_support = enable_present_support
        self.enable_portability_subset = enable_portability_subset

        self.vk_instance = GpuContext._help_create_instance(
            app_name,
            enable_debug_layer_support,
            enable_present_support,
            enable_portability_subset,
        )
        self.vk_extra_proc_tab = {}

        if enable_present_support:
            self._load_extra_proc("vkGetPhysicalDeviceSurfaceSupportKHR")
            self._load_extra_proc("vkGetPhysicalDeviceSurfaceFormatsKHR")
            self._load_extra_proc("vkDestroySurfaceKHR")
            self._load_extra_proc("vkCreateSwapchainKHR")
            self._load_extra_proc("vkDestroySwapchainKHR")
            self._load_extra_proc("vkGetSwapchainImagesKHR")
            self._load_extra_proc("vkAcquireNextImageKHR")
            self._load_extra_proc("vkQueuePresentKHR")

    def _load_extra_proc(self, name: str):
        proc = vkGetInstanceProcAddr(self.vk_instance, name)
        if proc is None:
            raise RuntimeError(f"Failed to load Vulkan instance procedure: {name!r}")
        self.vk_extra_proc_tab[name] = proc
        return proc

    def ext_fn(self, name: str):
        return self.vk_extra_proc_tab[name]

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
                    pEngineName="zfw",
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

        This greatly streamlines initialization by letting the GpuContext be fully
        independent of the WindowContext.
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
            vkDestroyInstance(self.vk_instance, pAllocator=None)

    #
    # Vulkan extension functions:
    #

    def vkGetPhysicalDeviceSurfaceSupportKHR(
        self,
        physical_device: VkPhysicalDevice,
        queue_family_index: int,
        surface: VkSurfaceKHR,
    ) -> bool:
        return bool(
            self.ext_fn("vkGetPhysicalDeviceSurfaceSupportKHR")(
                physical_device, queue_family_index, surface
            )
        )

    def vkGetPhysicalDeviceSurfaceFormatsKHR(
        self,
        physical_device: VkPhysicalDevice,
        surface: VkSurfaceKHR,
    ) -> list[VkSurfaceFormatKHR]:
        return self.ext_fn("vkGetPhysicalDeviceSurfaceFormatsKHR")(
            physical_device, surface
        )

    def vkDestroySurfaceKHR(
        self,
        instance: VkInstance,
        surface: VkSurfaceKHR,
        pAllocator=None,
    ) -> None:
        self.ext_fn("vkDestroySurfaceKHR")(instance, surface, pAllocator)

    def vkCreateSwapchainKHR(
        self,
        device: VkDevice,
        pCreateInfo: VkSwapchainCreateInfoKHR,
        pAllocator=None,
    ) -> VkSwapchainKHR:
        return self.ext_fn("vkCreateSwapchainKHR")(device, pCreateInfo, pAllocator)

    def vkDestroySwapchainKHR(
        self,
        device: VkDevice,
        swapchain: VkSwapchainKHR,
        pAllocator=None,
    ) -> None:
        self.ext_fn("vkDestroySwapchainKHR")(device, swapchain, pAllocator)

    def vkGetSwapchainImagesKHR(
        self,
        device: VkDevice,
        swapchain: VkSwapchainKHR,
    ) -> list[VkImage]:
        return self.ext_fn("vkGetSwapchainImagesKHR")(device, swapchain)

    def vkAcquireNextImageKHR(
        self,
        device: VkDevice,
        swapchain: VkSwapchainKHR,
        timeout: int,
        semaphore: VkSemaphore | None,
        fence: VkFence | None,
    ) -> int:
        return self.ext_fn("vkAcquireNextImageKHR")(
            device, swapchain, timeout, semaphore, fence
        )

    def vkQueuePresentKHR(
        self,
        queue: VkQueue,
        pPresentInfo,
    ) -> int:
        return self.ext_fn("vkQueuePresentKHR")(queue, pPresentInfo)

    #
    # GpuContext methods:
    #

    def enumerate_physical_devices(self) -> list[GpuPhysicalDevice]:
        return [
            GpuPhysicalDevice(
                context=self,
                vk_physical_device=vk_physical_device,
                vk_properties=vkGetPhysicalDeviceProperties(vk_physical_device),
                vk_memory_properties=vkGetPhysicalDeviceMemoryProperties(
                    vk_physical_device
                ),
            )
            for vk_physical_device in vkEnumeratePhysicalDevices(self.vk_instance)
        ]

    def print_debug_info(self, out: SupportsWrite[str], indent: int = 4) -> None:
        json.dump(
            {
                "physical-devices": [
                    {
                        "name": physical_device.vk_properties.deviceName,
                        "vendor-id": f"0x{physical_device.vk_properties.vendorID:08x}",
                        "device-id": f"0x{physical_device.vk_properties.deviceID:08x}",
                        "api-version": f"0x{physical_device.vk_properties.apiVersion:08x}",
                        "device-type": physical_device.spell_device_type(),
                    }
                    for physical_device in self.enumerate_physical_devices()
                ]
            },
            out,
            indent=indent,
        )


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


class GpuPhysicalDevice(BaseResource):
    context: GpuContext
    vk_physical_device: VkPhysicalDevice
    vk_properties: VkPhysicalDeviceProperties
    vk_memory_properties: VkPhysicalDeviceMemoryProperties

    def __init__(
        self,
        *,
        context: GpuContext,
        vk_physical_device: VkPhysicalDevice,
        vk_properties: VkPhysicalDeviceProperties,
        vk_memory_properties: VkPhysicalDeviceMemoryProperties,
    ) -> None:
        super().__init__(parent=context)
        self.context = context
        self.vk_physical_device = vk_physical_device
        self.vk_properties = vk_properties
        self.vk_memory_properties = vk_memory_properties

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
        surface: GpuSurface | None,
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
                supports_present = self.context.vkGetPhysicalDeviceSurfaceSupportKHR(
                    self.vk_physical_device,
                    index,
                    surface.vk_surface,
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

    def get_surface_formats(
        self, surface: "GpuSurface"
    ) -> list[tuple[VkFormat, VkColorSpaceKHR]]:
        assert self.context.enable_present_support
        return [
            (it.format, it.colorSpace)
            for it in self.context.vkGetPhysicalDeviceSurfaceFormatsKHR(
                self.vk_physical_device,
                surface.vk_surface,
            )
        ]


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
        surface: GpuSurface | None = None,
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
        surface: GpuSurface | None,
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


class GpuDevice(BaseResource):
    context: GpuContext
    physical_device: GpuPhysicalDevice
    qfis: GpuQueueFamilyIndices
    vk_device: VkDevice
    vk_command_pools: dict[int, VkCommandPool]
    vk_queues: dict[int, VkQueue]
    descriptor_pool_config: dict[GpuDescriptorType, int] | None
    max_descriptor_pool_set_count: int
    vk_descriptor_pool: VkDescriptorPool
    present_support_enabled: bool

    def __init__(
        self,
        *,
        context: GpuContext,
        physical_device: GpuPhysicalDevice,
        surface: GpuSurface | None,
        descriptor_pool_config: dict[GpuDescriptorType, int] | None = None,
        max_descriptor_pool_set_count: int = 1024,
    ) -> None:
        super().__init__(parent=context)

        self.context = context

        physical_device.check_vulkan_1_3_support()
        self.physical_device = physical_device

        self.qfis = GpuQueueFamilyIndices.find(physical_device, surface=surface)
        self.vk_device = self._help_create_device(
            physical_device=physical_device,
            qfis=self.qfis,
            extensions=self._help_compute_extensions(context),
        )
        self.vk_command_pools = self._help_create_command_pools(
            vk_device=self.vk_device, qfis=self.qfis
        )
        self.vk_queues = self._help_get_queues(vk_device=self.vk_device, qfis=self.qfis)
        self.present_support_enabled = surface is not None
        self.descriptor_pool_config = self._help_compute_descriptor_pool_config(
            descriptor_pool_config
        )
        self.max_descriptor_pool_set_count = max_descriptor_pool_set_count
        self.vk_descriptor_pool = self._help_create_descriptor_pool(
            max_sets=self.max_descriptor_pool_set_count,
            pool_sizes=self.descriptor_pool_config,
        )

    def _help_create_descriptor_pool(
        self,
        max_sets: int,
        pool_sizes: dict[GpuDescriptorType, int],
    ) -> VkDescriptorPool:
        vk_pool_sizes = [
            VkDescriptorPoolSize(
                type=vk_descriptor_type(desc_type),
                descriptorCount=count,
            )
            for desc_type, count in pool_sizes.items()
        ]

        create_info = VkDescriptorPoolCreateInfo(
            flags=0,
            maxSets=max_sets,
            poolSizeCount=len(vk_pool_sizes),
            pPoolSizes=vk_pool_sizes,
        )

        return vkCreateDescriptorPool(
            device=self.vk_device,
            pCreateInfo=create_info,
            pAllocator=None,
        )

    @staticmethod
    def _help_compute_extensions(context: GpuContext) -> list[str]:
        """Compute the list of required device extensions."""
        extensions = [
            "VK_KHR_dynamic_rendering",  # Vulkan 1.3
            "VK_KHR_shader_draw_parameters",  # needed for Slang shaders
        ]
        if context.enable_present_support:
            extensions.append("VK_KHR_swapchain")
        if context.enable_portability_subset:
            extensions.append("VK_KHR_portability_subset")  # macOS
        return extensions

    @staticmethod
    def _help_create_device(
        *,
        physical_device: GpuPhysicalDevice,
        qfis: GpuQueueFamilyIndices,
        extensions: list[str],
    ) -> VkDevice:
        """Create a VkDevice with the specified configuration."""
        queue_create_info_list = qfis.compute_queue_create_info_list()
        dynamic_rendering_features = VkPhysicalDeviceDynamicRenderingFeatures(
            dynamicRendering=True,
        )
        return vkCreateDevice(
            physical_device.vk_physical_device,
            VkDeviceCreateInfo(
                pNext=dynamic_rendering_features,
                enabledExtensionCount=len(extensions),
                ppEnabledExtensionNames=extensions,
                queueCreateInfoCount=len(queue_create_info_list),
                pQueueCreateInfos=queue_create_info_list,
            ),
            pAllocator=None,
        )

    @staticmethod
    def _help_create_command_pools(
        *,
        vk_device: VkDevice,
        qfis: GpuQueueFamilyIndices,
    ) -> dict[int, VkCommandPool]:
        """Create command pools for each unique queue family index."""
        return {
            qfi_index: vkCreateCommandPool(
                device=vk_device,
                pCreateInfo=VkCommandPoolCreateInfo(
                    flags=VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
                    queueFamilyIndex=qfi_index,
                ),
                pAllocator=None,
            )
            for qfi_index in {idx for _, idx in qfis}
        }

    @staticmethod
    def _help_get_queues(
        *,
        vk_device: VkDevice,
        qfis: GpuQueueFamilyIndices,
    ) -> dict[int, VkQueue]:
        """Retrieve one queue per unique queue family index (queueIndex = 0)."""
        return {
            qfi_index: vkGetDeviceQueue(
                vk_device,
                queueFamilyIndex=qfi_index,
                queueIndex=0,
            )
            for qfi_index in {idx for _, idx in qfis}
        }

    @staticmethod
    def _help_compute_descriptor_pool_config(
        descriptor_pool_config: dict[GpuDescriptorType, int] | None,
    ) -> dict[GpuDescriptorType, int]:
        """Compute descriptor pool config with defaults."""
        defaults: dict[GpuDescriptorType, int] = {
            "combined-image-sampler": 1024,
            "storage-buffer": 1024,
            "uniform-buffer": 1024,
        }
        return defaults | (descriptor_pool_config or {})

    def _on_dispose(self) -> None:
        # Destroy default descriptor pool first:
        vkDestroyDescriptorPool(
            device=self.vk_device,
            descriptorPool=self.vk_descriptor_pool,
            pAllocator=None,
        )

        # Destroy command pools:
        for _, vk_command_pool in self.vk_command_pools.items():
            vkDestroyCommandPool(
                device=self.vk_device,
                commandPool=vk_command_pool,
                pAllocator=None,
            )

        # Destroy device:
        vkDestroyDevice(device=self.vk_device, pAllocator=None)

    def submit(
        self,
        *,
        queue_type: GpuQueueType,
        command_buffers: list[VkCommandBuffer],
        wait_semaphores: list["GpuSemaphore"] | None = None,
        signal_semaphores: list["GpuSemaphore"] | None = None,
        fence: GpuFence | None = None,
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

    def wait_idle(self) -> None:
        vkDeviceWaitIdle(self.vk_device)

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


class GpuMemory(BaseResource):
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
        memory_requirements: VkMemoryRequirements,
        device_local: bool,
    ):
        super().__init__(parent=device)
        self.device = device
        self.size = memory_requirements.size
        self.device_local = device_local
        self._mapped_view_use_count = 0
        self._mapped_view = None

        self.vk_device_memory = self._allocate_memory(memory_requirements)

    def __repr__(self) -> str:
        return (
            f"<GpuMemory object at {hex(id(self))} with handle {self.vk_device_memory}>"
        )

    def _allocate_memory(self, requirements: VkMemoryRequirements) -> VkDeviceMemory:
        return vkAllocateMemory(
            device=self.device.vk_device,
            pAllocateInfo=VkMemoryAllocateInfo(
                allocationSize=requirements.size,
                memoryTypeIndex=self.device._find_memory_type(
                    requirements.memoryTypeBits,
                    required_properties=(
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
                        if self.device_local
                        else (
                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                            | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
                        )
                    ),
                ),
            ),
            pAllocator=None,
        )

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

    def write(self, *, data: np.ndarray):
        src_bytes = data.ravel().view(np.uint8)
        with self.map() as host_mem:
            host_mem[: len(src_bytes)] = src_bytes

    def read(self, *, dtype: npt.DTypeLike) -> np.ndarray:
        with self.map() as host_mem:
            return np.frombuffer(host_mem, dtype=dtype).copy()


#
# GpuImage
#


@dataclass
class GpuImageMeta:
    shape: tuple[int, int, int]  # (height, width, channels)
    dtype: npt.DTypeLike
    color_space: GpuColorSpace = "linear"

    @staticmethod
    def from_array(
        array: np.ndarray,
        *,
        color_space: GpuColorSpace = "linear",
    ) -> "GpuImageMeta":
        if array.ndim != 3:
            raise LogicError(
                f"GpuTextureSpec can only be created from 3D arrays, got array with "
                f"{array.ndim} dimensions instead"
            )
        return GpuImageMeta(
            shape=(array.shape[0], array.shape[1], array.shape[2]),
            dtype=array.dtype,
            color_space=color_space,
        )

    def infer_vk_format(
        self,
        usages: list[GpuImageUsage],
    ) -> int:
        dtype = self.dtype
        depth = self.shape[2]
        is_depth_attachment = "depth-attachment" in usages
        color_space = self.color_space

        if is_depth_attachment:
            match (dtype, depth):
                case (np.float32, 1):
                    return VK_FORMAT_D32_SFLOAT
                case _:
                    raise LogicError(
                        f"Invalid depth attachment image meta: {dtype=}, {depth=}"
                    )
        else:
            match (dtype, depth, color_space):
                case (np.uint8, 4, "linear"):
                    return VK_FORMAT_R8G8B8A8_UNORM
                case (np.uint8, 4, "srgb"):
                    return VK_FORMAT_R8G8B8A8_SRGB
                case (np.float32, 1, "linear"):
                    return VK_FORMAT_R32_SFLOAT
                case (np.float32, 4, "linear"):
                    return VK_FORMAT_R32G32B32A32_SFLOAT
                case _:
                    raise LogicError(
                        f"Invalid image meta: "
                        f"{dtype=}, {depth=}, {is_depth_attachment=}, {color_space=}"
                    )

    def into_buffer_meta(self) -> "GpuBufferMeta":
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

GpuImageLayout: TypeAlias = Literal[
    "present-src",
    "color-attachment-optimal",
    "texture-binding",
    "transfer-src-optimal",
    "transfer-dst-optimal",
]

GpuColorSpace: TypeAlias = Literal[
    "srgb",
    "linear",
]


def vk_image_usage(usages: list[GpuImageUsage]) -> int:
    """Map a list of GpuImageUsage to VkImageUsageFlags."""
    # Always include transfer src/dst for copy operations
    result = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
    for usage in usages:
        match usage:
            case "texture-binding":
                result |= VK_IMAGE_USAGE_SAMPLED_BIT
            case "storage-binding":
                result |= VK_IMAGE_USAGE_STORAGE_BIT
            case "color-attachment":
                result |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
            case "depth-attachment":
                result |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
            case _:
                raise LogicError(f"Invalid image usage: {usage!r}")
    return result


def vk_image_layout(layout: GpuImageLayout) -> VkImageLayout:
    """Map a GpuImageLayout to VkImageLayout."""
    match layout:
        case "present-src":
            return VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
        case "color-attachment-optimal":
            return VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
        case "texture-binding":
            return VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        case "transfer-src-optimal":
            return VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
        case "transfer-dst-optimal":
            return VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
        case _:
            raise LogicError(f"Invalid image layout: {layout!r}")


def vk_image_aspect(usages: list[GpuImageUsage]) -> int:
    """Compute VkImageAspectFlags from a list of GpuImageUsage."""
    result = 0
    for usage in usages:
        match usage:
            case "texture-binding" | "storage-binding" | "color-attachment":
                result |= VK_IMAGE_ASPECT_COLOR_BIT
            case "depth-attachment":
                result |= VK_IMAGE_ASPECT_DEPTH_BIT
            case _:
                raise LogicError(f"Invalid image usage: {usage!r}")
    return result


class GpuImage(BaseResource):
    device: GpuDevice
    vk_image: VkImage
    vk_image_view: VkImageView
    vk_format: VkFormat
    memory: GpuMemory | None
    meta: GpuImageMeta
    aspect_mask: int
    usages: list[GpuImageUsage]
    _owns_vk_image: bool
    _owns_vk_image_view: bool
    initial_vk_layout: VkImageLayout
    current_vk_layout: VkImageLayout

    def __init__(
        self,
        *,
        device: GpuDevice,
        usages: list[GpuImageUsage],
        meta: GpuImageMeta,
        custom_vk_image_handle: VkImage | None = None,
        custom_vk_image_view: VkImageView | None = None,
        custom_vk_format: VkFormat | None = None,
    ) -> None:
        super().__init__(parent=device)
        self.device = device
        self.usages = usages
        self.meta = meta

        self._owns_vk_image = custom_vk_image_handle is None
        self._owns_vk_image_view = custom_vk_image_view is None

        self.aspect_mask = vk_image_aspect(usages)
        self.vk_format = (
            custom_vk_format  # user override supplied
            if custom_vk_format is not None
            else meta.infer_vk_format(usages)
        )
        self.vk_image, self.memory = (
            self._help_create_image(
                device=device,
                meta=meta,
                usages=usages,
                vk_format=self.vk_format,
            )
            if custom_vk_image_handle is None
            else (custom_vk_image_handle, None)
        )
        self.vk_image_view = (
            self._help_create_image_view(
                device=device,
                vk_image=self.vk_image,
                vk_format=self.vk_format,
                aspect_mask=self.aspect_mask,
            )
            if custom_vk_image_view is None
            else custom_vk_image_view
        )

        self.initial_vk_layout = VK_IMAGE_LAYOUT_UNDEFINED
        self.current_vk_layout = VK_IMAGE_LAYOUT_UNDEFINED

    @staticmethod
    def _help_create_image(
        *,
        device: GpuDevice,
        meta: GpuImageMeta,
        usages: list[GpuImageUsage],
        vk_format: VkFormat,
    ) -> tuple[VkImage, GpuMemory]:
        """Create a VkImage and allocate/bind memory for it."""
        # Determine which queue families will access the image:
        queue_family_indices = list({idx for _, idx in device.qfis})

        image = vkCreateImage(
            device=device.vk_device,
            pCreateInfo=VkImageCreateInfo(
                flags=0,
                imageType=VK_IMAGE_TYPE_2D,
                format=vk_format,
                extent=VkExtent3D(width=meta.shape[1], height=meta.shape[0], depth=1),
                mipLevels=1,
                arrayLayers=1,
                samples=VK_SAMPLE_COUNT_1_BIT,
                tiling=VK_IMAGE_TILING_OPTIMAL,
                usage=vk_image_usage(usages),
                sharingMode=VK_SHARING_MODE_EXCLUSIVE,
                queueFamilyIndexCount=len(queue_family_indices),
                pQueueFamilyIndices=queue_family_indices,
                initialLayout=VK_IMAGE_LAYOUT_UNDEFINED,
            ),
            pAllocator=None,
        )

        # Allocate and bind memory for the image
        memory_requirements = vkGetImageMemoryRequirements(
            device=device.vk_device,
            image=image,
        )
        memory = GpuMemory(
            device=device,
            memory_requirements=memory_requirements,
            device_local=True,
        )
        vkBindImageMemory(
            device=device.vk_device,
            image=image,
            memory=memory.vk_device_memory,
            memoryOffset=VkDeviceSize(0),
        )

        return image, memory

    @staticmethod
    def _help_create_image_view(
        *,
        device: GpuDevice,
        vk_image: VkImage,
        vk_format: VkFormat,
        aspect_mask: int,
    ) -> VkImageView:
        """Create a VkImageView for the given image."""
        return vkCreateImageView(
            device=device.vk_device,
            pCreateInfo=VkImageViewCreateInfo(
                flags=0,
                image=vk_image,
                viewType=VK_IMAGE_TYPE_2D,
                format=vk_format,
                components=VkComponentMapping(
                    r=VK_COMPONENT_SWIZZLE_IDENTITY,
                    g=VK_COMPONENT_SWIZZLE_IDENTITY,
                    b=VK_COMPONENT_SWIZZLE_IDENTITY,
                    a=VK_COMPONENT_SWIZZLE_IDENTITY,
                ),
                subresourceRange=VkImageSubresourceRange(
                    aspectMask=aspect_mask,
                    baseMipLevel=0,
                    levelCount=1,
                    baseArrayLayer=0,
                    layerCount=1,
                ),
            ),
            pAllocator=None,
        )

    def _on_dispose(self) -> None:
        if self._owns_vk_image_view:
            vkDestroyImageView(
                self.device.vk_device, self.vk_image_view, pAllocator=None
            )
        if self._owns_vk_image:
            vkDestroyImage(self.device.vk_device, self.vk_image, pAllocator=None)

    def __repr__(self) -> str:
        return f"<GpuImage object at {hex(id(self))} with handle {self.vk_image}>"

    @property
    def width(self) -> int:
        return self.meta.shape[1]

    @property
    def height(self) -> int:
        return self.meta.shape[0]

    @property
    def channel_count(self) -> int:
        return self.meta.shape[2]

    def write(self, *, data: np.ndarray):
        staging_buffer = GpuBuffer(
            device=self.device,
            usages=["staging", "copy-src"],
            meta=GpuBufferMeta.from_array(data),
        )
        staging_buffer.memory.write(data=data)

        cmd = GpuCommandEncoder(device=self.device, queue_type="transfer")
        cmd.copy_buffer_to_image(src=staging_buffer, dst=self)
        cmd.submit().wait()


#
# GpuSemaphore
#


class GpuSemaphore(BaseResource):
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

    def __repr__(self) -> str:
        return (
            f"<GpuSemaphore object at {hex(id(self))} with handle {self.vk_semaphore}>"
        )

    def _on_dispose(self) -> None:
        vkDestroySemaphore(self.device.vk_device, self.vk_semaphore, pAllocator=None)


#
# GpuFence
#


class GpuFence(BaseResource):
    device: GpuDevice
    vk_fence: VkFence

    def __init__(self, *, device: GpuDevice, signalled: bool = False) -> None:
        super().__init__(parent=device)
        self.device = device
        self.vk_fence = vkCreateFence(
            device=device.vk_device,
            pCreateInfo=VkFenceCreateInfo(
                flags=VK_FENCE_CREATE_SIGNALED_BIT if signalled else 0
            ),
            pAllocator=None,
        )

    def __repr__(self) -> str:
        return f"<GpuFence object at {hex(id(self))} with handle {self.vk_fence}>"

    def _on_dispose(self) -> None:
        self.wait()
        vkDestroyFence(self.device.vk_device, self.vk_fence, pAllocator=None)

    def wait(self, *, timeout_sec: float = 1.0) -> None:
        """Wait for fence to be signaled.

        Args:
            timeout_sec: Timeout in seconds (default 1s)
        """
        vkWaitForFences(
            device=self.device.vk_device,
            fenceCount=1,
            pFences=[self.vk_fence],
            waitAll=True,
            timeout=int(timeout_sec * 1e9),
        )

    def reset(self) -> None:
        """Reset the fence to unsignaled state."""
        vkResetFences(
            device=self.device.vk_device,
            fenceCount=1,
            pFences=[self.vk_fence],
        )


#
# GpuBuffer
#


@dataclass
class GpuBufferMeta:
    element_count: int
    element_dtype: npt.DTypeLike

    @property
    def size(self) -> int:
        return self.element_count * self.element_size

    @property
    def element_size(self) -> int:
        return np.dtype(self.element_dtype).itemsize

    @staticmethod
    def from_array(array: np.ndarray) -> "GpuBufferMeta":
        return GpuBufferMeta(
            element_count=array.size,
            element_dtype=array.dtype,
        )


GpuBufferUsage: TypeAlias = Literal[
    "staging",
    "copy-src",
    "copy-dst",
    "uniform",
    "storage",
]


def vk_buffer_usage(usages: list[GpuBufferUsage]) -> int:
    """Map a list of GpuBufferUsage to VkBufferUsageFlags."""
    result = 0
    for usage in usages:
        match usage:
            case "copy-src":
                result |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT
            case "copy-dst":
                result |= VK_BUFFER_USAGE_TRANSFER_DST_BIT
            case "uniform":
                result |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
            case "storage":
                result |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
            case "staging":
                pass  # staging doesn't add a usage flag, it affects memory locality
            case _:
                raise LogicError(f"Invalid buffer usage: {usage!r}")
    return result


def vk_buffer_device_local(usages: list[GpuBufferUsage]) -> bool:
    """Determine whether a buffer should be device-local based on its usages."""
    device_local: bool | None = None
    for usage in usages:
        match usage:
            case "staging":
                required = False
            case "uniform" | "storage":
                required = True
            case "copy-src" | "copy-dst":
                continue  # no constraint
            case _:
                raise LogicError(f"Invalid buffer usage: {usage!r}")

        if device_local is not None and required != device_local:
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
        device_local = required

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

    return device_local


class GpuBuffer(BaseResource):
    device: GpuDevice
    vk_buffer: VkBuffer
    memory: GpuMemory
    meta: GpuBufferMeta
    usages: list[GpuBufferUsage]
    device_local: bool

    def __init__(
        self,
        *,
        device: GpuDevice,
        usages: list[GpuBufferUsage],
        meta: GpuBufferMeta,
    ):
        super().__init__(parent=device)
        self.device = device
        self.usages = usages
        self.meta = meta
        self.device_local = vk_buffer_device_local(usages)
        self.vk_buffer, self.memory = self._help_create_buffer(
            device=device,
            usages=usages,
            meta=meta,
            device_local=self.device_local,
        )

    def __repr__(self) -> str:
        return f"<GpuBuffer object at {hex(id(self))} with handle {self.vk_buffer}>"

    @staticmethod
    def _help_create_buffer(
        *,
        device: GpuDevice,
        usages: list[GpuBufferUsage],
        meta: GpuBufferMeta,
        device_local: bool,
    ) -> tuple[VkBuffer, GpuMemory]:
        """Create a VkBuffer and allocate/bind memory for it."""
        # Determine which queue families will access the buffer:
        queue_family_indices = list({idx for _, idx in device.qfis})

        buffer = vkCreateBuffer(
            device=device.vk_device,
            pCreateInfo=VkBufferCreateInfo(
                flags=0,
                size=meta.size,
                usage=vk_buffer_usage(usages),
                sharingMode=VK_SHARING_MODE_EXCLUSIVE,
                queueFamilyIndexCount=len(queue_family_indices),
                pQueueFamilyIndices=queue_family_indices,
            ),
            pAllocator=None,
        )

        # Allocate and bind memory for the buffer:
        memory_requirements = vkGetBufferMemoryRequirements(
            device=device.vk_device,
            buffer=buffer,
        )
        memory = GpuMemory(
            device=device,
            memory_requirements=memory_requirements,
            device_local=device_local,
        )
        vkBindBufferMemory(
            device=device.vk_device,
            buffer=buffer,
            memory=memory.vk_device_memory,
            memoryOffset=0,
        )

        return buffer, memory

    def _on_dispose(self):
        if self.memory is not None:
            self.memory.dispose()

        vkDestroyBuffer(self.device.vk_device, self.vk_buffer, pAllocator=None)

    def write(self, *, data: np.ndarray):
        self.memory.write(data=data)

    def read(self) -> np.ndarray:
        return self.memory.read(dtype=np.dtype(self.meta.element_dtype))


#
# GpuCommandBuffer
#


GpuCommandBufferLevel: TypeAlias = Literal["primary", "secondary"]


class GpuCommandEncoder(BaseResource):
    device: GpuDevice
    vk_command_buffer: VkCommandBuffer
    queue_family_index: int
    submit_queue_type: GpuQueueType
    _submit_fence: GpuFence | None

    def __init__(self, *, device: GpuDevice, queue_type: GpuQueueType) -> None:
        super().__init__(parent=device)
        self.device = device
        self.submit_queue_type = queue_type
        self.queue_family_index = device.qfis[queue_type]
        self.vk_command_buffer = self._allocate_command_buffer(
            device, self.queue_family_index
        )
        self._submit_fence = None
        self._submit_fence_is_owned = False

        self._begin_command_buffer(self.vk_command_buffer)

    @staticmethod
    def _allocate_command_buffer(
        device: GpuDevice, queue_family_index: int
    ) -> VkCommandBuffer:
        vk_command_pool = device.vk_command_pools[queue_family_index]
        return vkAllocateCommandBuffers(
            device=device.vk_device,
            pAllocateInfo=VkCommandBufferAllocateInfo(
                commandPool=vk_command_pool,
                level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                commandBufferCount=1,
            ),
        )[0]

    def _begin_command_buffer(self, vk_command_buffer: VkCommandBuffer) -> None:
        vkBeginCommandBuffer(
            commandBuffer=vk_command_buffer,
            pBeginInfo=VkCommandBufferBeginInfo(
                flags=0,
                pInheritanceInfo=None,
            ),
        )

    def _end_command_buffer(self, vk_command_buffer: VkCommandBuffer) -> None:
        vkEndCommandBuffer(commandBuffer=vk_command_buffer)

    def _on_dispose(self) -> None:
        # If not submitted, end the command buffer:
        if self._submit_fence is None:
            self._end_command_buffer(self.vk_command_buffer)

        # Wait for submission to complete:
        if self._submit_fence is not None:
            if self._submit_fence_is_owned:
                self._submit_fence.dispose()
            else:
                self._submit_fence.wait()
            self._submit_fence = None

        # Free the command buffer
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

    def copy_image_to_buffer(
        self,
        *,
        src: GpuImage,
        dst: GpuBuffer,
    ) -> None:
        vkCmdCopyImageToBuffer(
            commandBuffer=self.vk_command_buffer,
            srcImage=src.vk_image,
            srcImageLayout=VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            dstBuffer=dst.vk_buffer,
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

    def copy_image_to_image(
        self,
        *,
        src: GpuImage,
        dst: GpuImage,
    ) -> None:
        vkCmdCopyImage(
            commandBuffer=self.vk_command_buffer,
            srcImage=src.vk_image,
            srcImageLayout=VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            dstImage=dst.vk_image,
            dstImageLayout=VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            regionCount=1,
            pRegions=[
                VkImageCopy(
                    srcSubresource=VkImageSubresourceLayers(
                        aspectMask=src.aspect_mask,
                        mipLevel=0,
                        baseArrayLayer=0,
                        layerCount=1,
                    ),
                    srcOffset=VkOffset3D(x=0, y=0, z=0),
                    dstSubresource=VkImageSubresourceLayers(
                        aspectMask=dst.aspect_mask,
                        mipLevel=0,
                        baseArrayLayer=0,
                        layerCount=1,
                    ),
                    dstOffset=VkOffset3D(x=0, y=0, z=0),
                    extent=VkExtent3D(
                        width=src.meta.shape[1],
                        height=src.meta.shape[0],
                        depth=1,
                    ),
                )
            ],
        )

    @contextmanager
    def render(
        self,
        *,
        color_attachment: GpuImage | None = None,
        depth_attachment: GpuImage | None = None,
        clear_color: Literal["black", "transparent"] | None = None,
    ):
        color_infos: list[VkRenderingAttachmentInfo] = []
        if color_attachment is not None:
            color_infos.append(
                VkRenderingAttachmentInfo(
                    imageView=color_attachment.vk_image_view,
                    imageLayout=VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    resolveMode=VkResolveModeFlagBits(0),
                    resolveImageView=None,
                    resolveImageLayout=VK_IMAGE_LAYOUT_UNDEFINED,
                    loadOp=(
                        VK_ATTACHMENT_LOAD_OP_CLEAR
                        if clear_color is not None
                        else VK_ATTACHMENT_LOAD_OP_LOAD
                    ),
                    storeOp=VK_ATTACHMENT_STORE_OP_STORE,
                    clearValue=VkClearValue(
                        color=VkClearColorValue(
                            float32=(
                                [0.0, 0.0, 0.0, 1.0]
                                if clear_color == "black"
                                else [0.0, 0.0, 0.0, 0.0]
                            ),
                        )
                    ),
                )
            )

        depth_info = None
        if depth_attachment is not None:
            depth_info = VkRenderingAttachmentInfo(
                imageView=depth_attachment.vk_image_view,
                imageLayout=VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                resolveMode=VkResolveModeFlagBits(0),
                resolveImageView=None,
                resolveImageLayout=VK_IMAGE_LAYOUT_UNDEFINED,
                loadOp=(
                    VK_ATTACHMENT_LOAD_OP_CLEAR
                    if clear_color
                    else VK_ATTACHMENT_LOAD_OP_LOAD
                ),
                storeOp=VK_ATTACHMENT_STORE_OP_STORE,
                clearValue=VkClearValue(
                    depthStencil=VkClearDepthStencilValue(depth=0.0, stencil=0)
                ),
            )

        width = (
            color_attachment.meta.shape[1]
            if color_attachment
            else (depth_attachment.meta.shape[1] if depth_attachment else 1)
        )
        height = (
            color_attachment.meta.shape[0]
            if color_attachment
            else (depth_attachment.meta.shape[0] if depth_attachment else 1)
        )

        info = VkRenderingInfo(
            flags=0,
            renderArea=VkRect2D(
                offset=VkOffset2D(x=0, y=0),
                extent=VkExtent2D(width=width, height=height),
            ),
            layerCount=1,
            viewMask=0,
            colorAttachmentCount=len(color_infos),
            pColorAttachments=(color_infos if color_infos else None),
            pDepthAttachment=depth_info,
            pStencilAttachment=None,
        )

        vkCmdBeginRendering(self.vk_command_buffer, pRenderingInfo=info)

        yield GpuRenderPassCommandEncoder(command_encoder=self, device=self.device)

        vkCmdEndRendering(self.vk_command_buffer)

        # Update tracked layouts after rendering
        if color_attachment is not None:
            color_attachment.current_vk_layout = (
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
            )
        if depth_attachment is not None:
            depth_attachment.current_vk_layout = (
                VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
            )

    def transition_image_layout(
        self,
        *,
        image: GpuImage,
        layout: GpuImageLayout | None,
    ):
        # Determine the new layout based on usage
        new_layout = (
            VK_IMAGE_LAYOUT_UNDEFINED
            if layout is None
            else (
                {
                    "color-attachment-optimal": VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    "present-src": VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                    "copy-dst": VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    "copy-src": VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    "texture-binding": VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    "transfer-src-optimal": VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    "transfer-dst-optimal": VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                }[layout]
            )
        )

        # Skip if already in the correct layout
        if image.current_vk_layout == new_layout:
            return

        # # Determine access masks based on old and new layouts
        # src_access_mask = 0
        # if image.current_vk_layout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
        #     src_access_mask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
        # elif image.current_vk_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
        #     src_access_mask = VK_ACCESS_TRANSFER_WRITE_BIT
        # elif image.current_vk_layout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
        #     src_access_mask = VK_ACCESS_TRANSFER_READ_BIT

        # Determine stage mask based on queue type
        stage_mask = 0
        if self.submit_queue_type == "graphics":
            stage_mask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT
        elif self.submit_queue_type == "compute":
            stage_mask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
        elif self.submit_queue_type == "transfer":
            stage_mask = VK_PIPELINE_STAGE_TRANSFER_BIT
        else:
            raise LogicError(
                f"Unsupported queue type for image layout transition: "
                f"{self.submit_queue_type!r}"
            )

        # Determine dst access mask based on new layout
        # Note: On transfer queues, we can only use transfer-related access flags
        dst_access_mask = 0
        if new_layout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
            dst_access_mask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
        elif new_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
            dst_access_mask = VK_ACCESS_TRANSFER_WRITE_BIT
        elif new_layout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
            dst_access_mask = VK_ACCESS_TRANSFER_READ_BIT
        elif new_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
            # On transfer queue, we can't use VK_ACCESS_SHADER_READ_BIT
            # Use VK_ACCESS_NONE (0) - the sync will happen via semaphore/fence
            if self.submit_queue_type == "transfer":
                dst_access_mask = 0
            else:
                dst_access_mask = VK_ACCESS_SHADER_READ_BIT

        vkCmdPipelineBarrier(
            commandBuffer=self.vk_command_buffer,
            srcStageMask=stage_mask,
            dstStageMask=stage_mask,
            dependencyFlags=0,
            memoryBarrierCount=0,
            pMemoryBarriers=None,
            bufferMemoryBarrierCount=0,
            pBufferMemoryBarriers=None,
            imageMemoryBarrierCount=1,
            pImageMemoryBarriers=[
                VkImageMemoryBarrier(
                    srcAccessMask=VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                    dstAccessMask=dst_access_mask,
                    oldLayout=image.current_vk_layout,
                    newLayout=new_layout,
                    srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,
                    dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,
                    image=image.vk_image,
                    subresourceRange=VkImageSubresourceRange(
                        aspectMask=image.aspect_mask,
                        baseMipLevel=0,
                        levelCount=1,
                        baseArrayLayer=0,
                        layerCount=1,
                    ),
                )
            ],
        )

        # Update tracked layout
        image.current_vk_layout = new_layout

    def submit(
        self,
        fence: GpuFence | None = None,
        wait_semaphores: list["GpuSemaphore"] | None = None,
        signal_semaphores: list["GpuSemaphore"] | None = None,
    ) -> GpuFence:
        self._end_command_buffer(self.vk_command_buffer)

        if self._submit_fence is not None:
            raise LogicError("Each GpuCommandEncoder can only be submitted once.")

        self._submit_fence_is_owned = fence is None
        self._submit_fence = fence or GpuFence(device=self.device)

        self.device.submit(
            queue_type=self.submit_queue_type,
            command_buffers=[self.vk_command_buffer],
            wait_semaphores=wait_semaphores or [],
            signal_semaphores=signal_semaphores or [],
            fence=self._submit_fence,
        )

        return self._submit_fence


#
# GpuShader
#


class GpuShader(BaseResource):
    """Wrapper for VkShaderModule"""

    device: GpuDevice
    vk_shader_module: VkShaderModule
    stage: Literal["vertex", "fragment"]

    def __init__(
        self,
        *,
        device: GpuDevice,
        spirv_path: Path | str,
        stage: Literal["vertex", "fragment"],
    ):
        super().__init__(parent=device)
        self.device = device
        self.stage = stage
        self.vk_shader_module = self._help_create_shader_module(device, spirv_path)

    @staticmethod
    def _help_create_shader_module(
        device: GpuDevice, spirv_path: Path | str
    ) -> VkShaderModule:
        spirv_path = Path(spirv_path)

        # Read SPIR-V bytecode
        with open(spirv_path, "rb") as f:
            spirv_code = f.read()

        # Create shader module using raw Vulkan API
        create_info = VkShaderModuleCreateInfo(
            flags=0,
            codeSize=len(spirv_code),
            pCode=spirv_code,
        )

        return vkCreateShaderModule(
            device=device.vk_device,
            pCreateInfo=create_info,
            pAllocator=None,
        )

    def _on_dispose(self) -> None:
        vkDestroyShaderModule(
            device=self.device.vk_device,
            shaderModule=self.vk_shader_module,
            pAllocator=None,
        )


#
# GpuSampler
#


def vk_sampler_filter(filter: "GpuSamplerFilter") -> int:
    """Convert GpuSamplerFilter to Vulkan filter constant."""
    match filter:
        case "linear":
            return VK_FILTER_LINEAR
        case "nearest":
            return VK_FILTER_NEAREST
        case _:
            raise LogicError(f"Invalid sampler filter: {filter!r}")


def vk_sampler_address_mode(mode: "GpuSamplerAddressMode") -> int:
    """Convert GpuSamplerAddressMode to Vulkan address mode constant."""
    match mode:
        case "repeat":
            return VK_SAMPLER_ADDRESS_MODE_REPEAT
        case "mirrored-repeat":
            return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT
        case "clamp-to-edge":
            return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE
        case "clamp-to-border":
            return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER
        case _:
            raise LogicError(f"Invalid sampler address mode: {mode!r}")


class GpuSampler(BaseResource):
    """Wrapper for VkSampler"""

    device: GpuDevice
    vk_sampler: VkSampler

    def __init__(
        self,
        *,
        device: GpuDevice,
        mag_filter: "GpuSamplerFilter" = "linear",
        min_filter: "GpuSamplerFilter" = "linear",
        address_mode: "GpuSamplerAddressMode" = "clamp-to-edge",
    ):
        super().__init__(parent=device)
        self.device = device
        self.vk_sampler = self._help_create_sampler(
            device, mag_filter, min_filter, address_mode
        )

    @staticmethod
    def _help_create_sampler(
        device: GpuDevice,
        mag_filter: "GpuSamplerFilter",
        min_filter: "GpuSamplerFilter",
        address_mode: "GpuSamplerAddressMode",
    ) -> VkSampler:
        vk_mag_filter = vk_sampler_filter(mag_filter)
        vk_min_filter = vk_sampler_filter(min_filter)
        vk_address_mode = vk_sampler_address_mode(address_mode)

        create_info = VkSamplerCreateInfo(
            flags=0,
            magFilter=vk_mag_filter,
            minFilter=vk_min_filter,
            mipmapMode=VK_SAMPLER_MIPMAP_MODE_LINEAR,
            addressModeU=vk_address_mode,
            addressModeV=vk_address_mode,
            addressModeW=vk_address_mode,
            mipLodBias=0.0,
            anisotropyEnable=False,
            maxAnisotropy=1.0,
            compareEnable=False,
            compareOp=0,
            minLod=0.0,
            maxLod=0.0,
            borderColor=VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
            unnormalizedCoordinates=False,
        )

        return vkCreateSampler(
            device=device.vk_device,
            pCreateInfo=create_info,
            pAllocator=None,
        )

    def _on_dispose(self) -> None:
        vkDestroySampler(
            device=self.device.vk_device,
            sampler=self.vk_sampler,
            pAllocator=None,
        )


GpuSamplerFilter: TypeAlias = Literal[
    "nearest",
    "linear",
]

GpuSamplerAddressMode: TypeAlias = Literal[
    "repeat",
    "mirrored-repeat",
    "clamp-to-edge",
    "clamp-to-border",
]


#
# GpuPipelineLayout, GpuDescriptorSetLayout:
#


class GpuPipelineLayout(BaseResource):
    device: GpuDevice
    vk_pipeline_layout: VkPipelineLayout
    descriptor_set_layouts: list[GpuDescriptorSetLayout]

    def __init__(
        self,
        *,
        device: GpuDevice,
        descriptor_set_layouts: list[GpuDescriptorSetLayout],
    ):
        super().__init__(parent=device)
        self.device = device
        self.descriptor_set_layouts = descriptor_set_layouts
        self.vk_pipeline_layout = self._help_create_pipeline_layout(
            device, descriptor_set_layouts
        )

    @staticmethod
    def _help_create_pipeline_layout(
        device: GpuDevice,
        descriptor_set_layouts: list[GpuDescriptorSetLayout],
    ) -> VkPipelineLayout:
        vk_set_layouts = [
            layout.vk_descriptor_set_layout for layout in descriptor_set_layouts
        ]

        layout_create_info = VkPipelineLayoutCreateInfo(
            flags=0,
            setLayoutCount=len(vk_set_layouts),
            pSetLayouts=vk_set_layouts,
            pushConstantRangeCount=0,
            pPushConstantRanges=None,
        )

        return vkCreatePipelineLayout(
            device=device.vk_device,
            pCreateInfo=layout_create_info,
            pAllocator=None,
        )

    def _on_dispose(self) -> None:
        vkDestroyPipelineLayout(
            device=self.device.vk_device,
            pipelineLayout=self.vk_pipeline_layout,
            pAllocator=None,
        )


class GpuDescriptorSetLayout(BaseResource):
    device: GpuDevice
    vk_descriptor_set_layout: VkDescriptorSetLayout
    bindings: OrderedDict[str, "GpuDescriptorSetLayoutBinding"]

    def __init__(
        self,
        *,
        device: GpuDevice,
        bindings: OrderedDict[str, "GpuDescriptorSetLayoutBinding"],
    ):
        super().__init__(parent=device)
        self.device = device
        self.bindings = bindings
        self.vk_descriptor_set_layout = self._help_create_descriptor_set_layout(
            device, bindings
        )

    @staticmethod
    def _help_create_descriptor_set_layout(
        device: GpuDevice,
        bindings: OrderedDict[str, "GpuDescriptorSetLayoutBinding"],
    ) -> VkDescriptorSetLayout:
        vk_binding_list = [
            VkDescriptorSetLayoutBinding(
                binding=binding_index,
                descriptorType=vk_descriptor_type(binding.type),
                descriptorCount=binding.count,
                stageFlags=vk_shader_stages(binding.stages),
                pImmutableSamplers=None,
            )
            for binding_index, (_, binding) in enumerate(bindings.items())
        ]

        create_info = VkDescriptorSetLayoutCreateInfo(
            flags=0,
            bindingCount=len(vk_binding_list),
            pBindings=vk_binding_list,
        )

        return vkCreateDescriptorSetLayout(
            device=device.vk_device,
            pCreateInfo=create_info,
            pAllocator=None,
        )

    def _on_dispose(self) -> None:
        vkDestroyDescriptorSetLayout(
            device=self.device.vk_device,
            descriptorSetLayout=self.vk_descriptor_set_layout,
            pAllocator=None,
        )


@dataclass
class GpuDescriptorSetLayoutBinding:
    type: GpuDescriptorType
    stages: list[GpuStage] = field(default_factory=lambda: ["vertex", "fragment"])
    count: int = 1


#
# GpuDescriptorSet:
#


class GpuDescriptorSet(BaseResource):
    device: GpuDevice
    vk_descriptor_set: VkDescriptorSet
    layout: GpuDescriptorSetLayout
    bindings: dict[str, GpuDescriptorSetBinding]

    def __init__(
        self,
        *,
        device: GpuDevice,
        layout: GpuDescriptorSetLayout,
        bindings: dict[str, GpuDescriptorSetBinding],
    ):
        super().__init__(parent=device)
        self.device = device
        self.layout = layout
        self.bindings = bindings
        self._help_validate_bindings(bindings, layout)
        self.vk_descriptor_set = self._help_allocate_descriptor_set(device, layout)
        self._help_write_bindings(device, self.vk_descriptor_set, bindings, layout)

    @staticmethod
    def _help_validate_bindings(
        bindings: dict[str, GpuDescriptorSetBinding],
        layout: GpuDescriptorSetLayout,
    ) -> None:
        # Check that the number of bindings matches the layout
        if len(bindings) != len(layout.bindings):
            raise LogicError(
                f"Descriptor set binding count mismatch: "
                f"layout expects {len(layout.bindings)} bindings, "
                f"but got {len(bindings)}"
            )

        # Check that each binding's descriptor type matches the layout
        for binding_name in bindings.keys():
            binding = bindings[binding_name]
            binding_layout = layout.bindings[binding_name]
            ok_desc_types = compatible_descriptor_types_for_binding(binding)
            if binding_layout.type not in ok_desc_types:
                raise LogicError(
                    f"Descriptor set binding type mismatch: "
                    f"layout expects {binding_layout.type}, "
                    f"but got {binding}: "
                    f"expected one of {ok_desc_types}"
                )

    @staticmethod
    def _help_allocate_descriptor_set(
        device: GpuDevice, layout: GpuDescriptorSetLayout
    ) -> VkDescriptorSet:
        alloc_info = VkDescriptorSetAllocateInfo(
            descriptorPool=device.vk_descriptor_pool,
            descriptorSetCount=1,
            pSetLayouts=[layout.vk_descriptor_set_layout],
        )
        vk_sets = vkAllocateDescriptorSets(
            device=device.vk_device,
            pAllocateInfo=alloc_info,
        )
        return vk_sets[0]

    @staticmethod
    def _help_write_bindings(
        device: GpuDevice,
        vk_set: VkDescriptorSet,
        bindings: dict[str, GpuDescriptorSetBinding],
        layout: GpuDescriptorSetLayout,
    ) -> None:
        # Update the descriptor sets by writing the bindings:
        # IMPORTANT: Iterate over the layout's bindings to ensure the correct order, and
        # thus, correct binding indices.
        writes = [
            descriptor_set_write_for_binding(
                vk_set=vk_set,
                binding_index=binding_index,
                binding=bindings[binding_name],
                binding_layout=layout.bindings[binding_name],
            )
            for binding_index, binding_name in enumerate(layout.bindings.keys())
        ]
        vkUpdateDescriptorSets(
            device=device.vk_device,
            descriptorWriteCount=len(writes),
            pDescriptorWrites=writes,
            descriptorCopyCount=0,
            pDescriptorCopies=None,
        )


GpuDescriptorSetBinding: TypeAlias = GpuBuffer | tuple[GpuImage, GpuSampler]


def compatible_descriptor_types_for_binding(
    binding: GpuDescriptorSetBinding,
) -> list[GpuDescriptorType]:
    match binding:
        case GpuBuffer():
            res = []
            if "uniform" in binding.usages:
                res.append("uniform-buffer")
            if "storage" in binding.usages:
                res.append("storage-buffer")
            return res
        case (GpuImage(), GpuSampler()):
            return ["combined-image-sampler"]


def descriptor_set_write_for_binding(
    vk_set: VkDescriptorSet,
    binding_index: int,
    binding: GpuDescriptorSetBinding,
    binding_layout: GpuDescriptorSetLayoutBinding,
):
    match binding_layout.type:
        case "combined-image-sampler":
            assert isinstance(binding, tuple)
            image, sampler = binding
            return VkWriteDescriptorSet(
                dstSet=vk_set,
                dstBinding=binding_index,
                dstArrayElement=0,
                descriptorCount=1,
                descriptorType=vk_descriptor_type(binding_layout.type),
                pImageInfo=[
                    VkDescriptorImageInfo(
                        sampler=sampler.vk_sampler,
                        imageView=image.vk_image_view,
                        imageLayout=VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    )
                ],
                pBufferInfo=None,
                pTexelBufferView=None,
            )
        case "uniform-buffer" | "storage-buffer":
            assert isinstance(binding, GpuBuffer)
            return VkWriteDescriptorSet(
                dstSet=vk_set,
                dstBinding=binding_index,
                dstArrayElement=0,
                descriptorCount=1,
                descriptorType=vk_descriptor_type(binding_layout.type),
                pImageInfo=None,
                pBufferInfo=[
                    VkDescriptorBufferInfo(
                        buffer=binding.vk_buffer,
                        offset=0,
                        range=binding.meta.size,
                    )
                ],
                pTexelBufferView=None,
            )
        case _:
            raise LogicError()


def vk_descriptor_set_binding(binding: GpuDescriptorSetBinding):
    match binding:
        case GpuBuffer():
            return VkDescriptorBufferInfo(
                buffer=binding.vk_buffer,
                offset=0,
                range=binding.meta.size,
            )
        case (GpuImage(), GpuSampler()) as binding:
            image, sampler = binding
            return VkDescriptorImageInfo(
                sampler=sampler.vk_sampler,
                imageView=image.vk_image_view,
                imageLayout=image.current_vk_layout,
            )


GpuDescriptorType: TypeAlias = Literal[
    "combined-image-sampler", "storage-buffer", "uniform-buffer"
]


def vk_descriptor_type(t: GpuDescriptorType):
    return {
        "combined-image-sampler": VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        "uniform-buffer": VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        "storage-buffer": VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
    }[t]


#
# GpuPipeline:
#


class GpuPipeline(BaseResource):
    device: GpuDevice
    vk_pipeline: VkPipeline
    vk_color_format: VkFormat
    viewport_width: int
    viewport_height: int
    layout: GpuPipelineLayout

    def __init__(
        self,
        *,
        device: GpuDevice,
        vertex_shader: GpuShader,
        fragment_shader: GpuShader,
        vk_color_format: VkFormat,
        enable_depth_test: bool,
        enable_alpha_blending: bool,
        viewport_width: int,
        viewport_height: int,
        layout: GpuPipelineLayout,
    ):
        super().__init__(parent=device)
        self.device = device
        self.vk_color_format = vk_color_format
        self.vk_pipeline = self._help_create_pipeline(
            device=device,
            vertex_shader=vertex_shader,
            fragment_shader=fragment_shader,
            vk_color_format=vk_color_format,
            vk_depth_format=VK_FORMAT_D32_SFLOAT,
            enable_depth_test=enable_depth_test,
            enable_alpha_blending=enable_alpha_blending,
            viewport_width=viewport_width,
            viewport_height=viewport_height,
            layout=layout,
        )
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.layout = layout

    @staticmethod
    def _help_create_shader_stages(
        vertex_shader: GpuShader,
        fragment_shader: GpuShader,
    ) -> list:
        return [
            VkPipelineShaderStageCreateInfo(
                flags=0,
                stage=VK_SHADER_STAGE_VERTEX_BIT,
                module=vertex_shader.vk_shader_module,
                pName="main",
                pSpecializationInfo=None,
            ),
            VkPipelineShaderStageCreateInfo(
                flags=0,
                stage=VK_SHADER_STAGE_FRAGMENT_BIT,
                module=fragment_shader.vk_shader_module,
                pName="main",
                pSpecializationInfo=None,
            ),
        ]

    @staticmethod
    def _help_create_vertex_input_state() -> VkPipelineVertexInputStateCreateInfo:
        return VkPipelineVertexInputStateCreateInfo(
            flags=0,
            vertexBindingDescriptionCount=0,
            pVertexBindingDescriptions=None,
            vertexAttributeDescriptionCount=0,
            pVertexAttributeDescriptions=None,
        )

    @staticmethod
    def _help_create_input_assembly_state() -> VkPipelineInputAssemblyStateCreateInfo:
        return VkPipelineInputAssemblyStateCreateInfo(
            flags=0,
            topology=VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            primitiveRestartEnable=False,
        )

    @staticmethod
    def _help_create_viewport_state(
        width: int,
        height: int,
    ) -> VkPipelineViewportStateCreateInfo:
        viewport = VkViewport(
            x=0.0,
            y=0.0,
            width=float(width),
            height=float(height),
            minDepth=0.0,
            maxDepth=1.0,
        )
        scissor = VkRect2D(
            offset=VkOffset2D(x=0, y=0),
            extent=VkExtent2D(width=width, height=height),
        )
        return VkPipelineViewportStateCreateInfo(
            flags=0,
            viewportCount=1,
            pViewports=[viewport],
            scissorCount=1,
            pScissors=[scissor],
        )

    @staticmethod
    def _help_create_rasterization_state() -> VkPipelineRasterizationStateCreateInfo:
        return VkPipelineRasterizationStateCreateInfo(
            flags=0,
            depthClampEnable=False,
            rasterizerDiscardEnable=False,
            polygonMode=VK_POLYGON_MODE_FILL,
            cullMode=VK_CULL_MODE_NONE,
            frontFace=VK_FRONT_FACE_COUNTER_CLOCKWISE,
            depthBiasEnable=False,
            depthBiasConstantFactor=0.0,
            depthBiasClamp=0.0,
            depthBiasSlopeFactor=0.0,
            lineWidth=1.0,
        )

    @staticmethod
    def _help_create_multisample_state() -> VkPipelineMultisampleStateCreateInfo:
        return VkPipelineMultisampleStateCreateInfo(
            flags=0,
            rasterizationSamples=VK_SAMPLE_COUNT_1_BIT,
            sampleShadingEnable=False,
            minSampleShading=1.0,
            pSampleMask=None,
            alphaToCoverageEnable=False,
            alphaToOneEnable=False,
        )

    @staticmethod
    def _help_create_color_blend_state(
        enable_alpha_blending: bool,
    ) -> VkPipelineColorBlendStateCreateInfo:
        color_blend_attachment = VkPipelineColorBlendAttachmentState(
            blendEnable=enable_alpha_blending,
            srcColorBlendFactor=VK_BLEND_FACTOR_SRC_ALPHA,
            dstColorBlendFactor=VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
            colorBlendOp=VK_BLEND_OP_ADD,
            srcAlphaBlendFactor=VK_BLEND_FACTOR_ONE,
            dstAlphaBlendFactor=VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
            alphaBlendOp=VK_BLEND_OP_ADD,
            colorWriteMask=(
                VK_COLOR_COMPONENT_R_BIT
                | VK_COLOR_COMPONENT_G_BIT
                | VK_COLOR_COMPONENT_B_BIT
                | VK_COLOR_COMPONENT_A_BIT
            ),
        )
        return VkPipelineColorBlendStateCreateInfo(
            flags=0,
            logicOpEnable=False,
            logicOp=VK_LOGIC_OP_COPY,
            attachmentCount=1,
            pAttachments=[color_blend_attachment],
            blendConstants=[0.0, 0.0, 0.0, 0.0],
        )

    @staticmethod
    def _help_create_depth_stencil_state(
        enable_depth_test: bool,
    ) -> VkPipelineDepthStencilStateCreateInfo:
        return VkPipelineDepthStencilStateCreateInfo(
            flags=0,
            depthTestEnable=enable_depth_test,
            depthWriteEnable=enable_depth_test,
            depthCompareOp=VK_COMPARE_OP_GREATER,
            depthBoundsTestEnable=True,
            stencilTestEnable=False,
            front=VkStencilOpState(
                failOp=VK_STENCIL_OP_KEEP,
                passOp=VK_STENCIL_OP_KEEP,
                depthFailOp=VK_STENCIL_OP_KEEP,
                compareOp=VK_COMPARE_OP_ALWAYS,
                compareMask=0,
                writeMask=0,
                reference=0,
            ),
            back=VkStencilOpState(
                failOp=VK_STENCIL_OP_KEEP,
                passOp=VK_STENCIL_OP_KEEP,
                depthFailOp=VK_STENCIL_OP_KEEP,
                compareOp=VK_COMPARE_OP_ALWAYS,
                compareMask=0,
                writeMask=0,
                reference=0,
            ),
            minDepthBounds=0.0,
            maxDepthBounds=1.0,
        )

    @staticmethod
    def _help_create_pipeline(
        device: GpuDevice,
        vertex_shader: GpuShader,
        fragment_shader: GpuShader,
        vk_color_format: VkFormat,
        vk_depth_format: VkFormat,
        enable_depth_test: bool,
        enable_alpha_blending: bool,
        viewport_width: int,
        viewport_height: int,
        layout: GpuPipelineLayout,
    ) -> VkPipeline:
        shader_stages = GpuPipeline._help_create_shader_stages(
            vertex_shader, fragment_shader
        )
        vertex_input_state = GpuPipeline._help_create_vertex_input_state()
        input_assembly_state = GpuPipeline._help_create_input_assembly_state()
        viewport_state = GpuPipeline._help_create_viewport_state(
            viewport_width, viewport_height
        )
        rasterization_state = GpuPipeline._help_create_rasterization_state()
        multisample_state = GpuPipeline._help_create_multisample_state()
        color_blend_state = GpuPipeline._help_create_color_blend_state(
            enable_alpha_blending=enable_alpha_blending
        )
        depth_state = GpuPipeline._help_create_depth_stencil_state(
            enable_depth_test=enable_depth_test
        )

        rendering_info = VkPipelineRenderingCreateInfo(
            viewMask=0,
            colorAttachmentCount=1,
            pColorAttachmentFormats=[vk_color_format],
            depthAttachmentFormat=vk_depth_format,
            stencilAttachmentFormat=VK_FORMAT_UNDEFINED,
        )

        pipeline_create_info = VkGraphicsPipelineCreateInfo(
            pNext=rendering_info,
            flags=0,
            stageCount=len(shader_stages),
            pStages=shader_stages,
            pVertexInputState=vertex_input_state,
            pInputAssemblyState=input_assembly_state,
            pTessellationState=None,
            pViewportState=viewport_state,
            pRasterizationState=rasterization_state,
            pMultisampleState=multisample_state,
            pDepthStencilState=depth_state,
            pColorBlendState=color_blend_state,
            pDynamicState=None,
            layout=layout.vk_pipeline_layout,
            renderPass=None,
            subpass=0,
            basePipelineHandle=None,
            basePipelineIndex=-1,
        )

        return vkCreateGraphicsPipelines(
            device=device.vk_device,
            pipelineCache=None,
            createInfoCount=1,
            pCreateInfos=[pipeline_create_info],
            pAllocator=None,
        )[0]

    def _on_dispose(self) -> None:
        vkDestroyPipeline(
            device=self.device.vk_device,
            pipeline=self.vk_pipeline,
            pAllocator=None,
        )


class GpuRenderPassCommandEncoder(BaseResource):
    device: GpuDevice
    command_encoder: GpuCommandEncoder
    bound_pipeline: GpuPipeline | None

    def __init__(self, *, device: GpuDevice, command_encoder: GpuCommandEncoder):
        super().__init__(parent=command_encoder)
        self.device = device
        self.command_encoder = command_encoder
        self.bound_pipeline = None

    def _on_dispose(self) -> None:
        pass

    def bind_pipeline(self, *, pipeline: GpuPipeline) -> None:
        if self.bound_pipeline is pipeline:
            return

        self.bound_pipeline = pipeline

        vkCmdBindPipeline(
            commandBuffer=self.command_encoder.vk_command_buffer,
            pipelineBindPoint=VK_PIPELINE_BIND_POINT_GRAPHICS,
            pipeline=pipeline.vk_pipeline,
        )

    def bind_descriptor_set(
        self,
        *,
        set_index: int,
        descriptor_set: GpuDescriptorSet,
        dynamic_offsets: list[int] | None = None,
    ) -> None:
        self.bind_descriptor_sets(
            first_set=set_index,
            sets=[descriptor_set],
            dynamic_offsets=dynamic_offsets,
        )

    def bind_descriptor_sets(
        self,
        *,
        first_set: int,
        sets: list[GpuDescriptorSet],
        dynamic_offsets: list[int] | None = None,
    ) -> None:
        if self.bound_pipeline is None:
            raise LogicError(
                "Cannot bind descriptor sets: No pipeline is currently bound."
            )
        vkCmdBindDescriptorSets(
            commandBuffer=self.command_encoder.vk_command_buffer,
            pipelineBindPoint=VK_PIPELINE_BIND_POINT_GRAPHICS,
            layout=self.bound_pipeline.layout.vk_pipeline_layout,
            firstSet=first_set,
            descriptorSetCount=len(sets),
            pDescriptorSets=[s.vk_descriptor_set for s in sets],
            dynamicOffsetCount=(len(dynamic_offsets) if dynamic_offsets else 0),
            pDynamicOffsets=(dynamic_offsets if dynamic_offsets else None),
        )

    def draw(
        self,
        *,
        vertex_count: int,
        instance_count: int = 1,
        first_vertex: int = 0,
        first_instance: int = 0,
    ) -> None:
        vkCmdDraw(
            commandBuffer=self.command_encoder.vk_command_buffer,
            vertexCount=vertex_count,
            instanceCount=instance_count,
            firstVertex=first_vertex,
            firstInstance=first_instance,
        )


#
# GpuSurface
#


class GpuSurface(BaseResource):
    context: GpuContext
    vk_surface: VkSurfaceKHR
    width: int
    height: int

    def __init__(
        self,
        *,
        context: GpuContext,
        parent: BaseResource,
        vk_surface: VkSurfaceKHR,
        width: int,
        height: int,
    ):
        super().__init__(parent=parent)
        self.context = context
        self.vk_surface = vk_surface
        self.width = width
        self.height = height

    def _on_dispose(self) -> None:
        self.context.vkDestroySurfaceKHR(
            instance=self.context.vk_instance,
            surface=self.vk_surface,
            pAllocator=None,
        )


#
# GpuSwapChain
#


class GpuSwapChain(BaseResource):
    device: GpuDevice
    vk_swap_chain: VkSwapchainKHR
    images: list[GpuImage]
    slots: list[GpuSwapChainSlot]
    vk_format: VkFormat
    width: int
    height: int
    frame_counter: int

    def __init__(
        self,
        *,
        device: GpuDevice,
        surface: GpuSurface,
        image_count: int,
    ):
        super().__init__(parent=surface)
        self.device = device
        self.frame_counter = 0
        self.width = surface.width
        self.height = surface.height
        vk_format, vk_color_space = self._select_surface_format(device, surface)
        self.vk_format = vk_format
        self.vk_color_space = vk_color_space
        self.vk_swap_chain = self._create_swap_chain(surface, image_count)
        self.images = self._wrap_swap_chain_images(surface)
        self.slots = self._create_slots()

    @property
    def context(self) -> GpuContext:
        return self.device.context

    @staticmethod
    def _select_surface_format(device: GpuDevice, surface: GpuSurface):
        vk_format = VK_FORMAT_B8G8R8A8_SRGB
        vk_colorspace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR
        vk_surface_format_list = device.physical_device.get_surface_formats(surface)
        for surface_format, surface_color_space in vk_surface_format_list:
            if surface_format != vk_format:
                continue
            if surface_color_space != vk_colorspace:
                continue
            break
        else:
            raise PlatformSupportError(
                f"Physical device {device.physical_device.name!r} does not support "
                "the required swapchain format: "
                f"Requires format=VK_FORMAT_R8G8B8A8_SRGB, "
                f"colorSpace=VK_COLOR_SPACE_SRGB_NONLINEAR_KHR."
            )
        return vk_format, vk_colorspace

    def _create_swap_chain(
        self, surface: GpuSurface, image_count: int
    ) -> VkSwapchainKHR:
        return self.device.context.vkCreateSwapchainKHR(
            device=self.device.vk_device,
            pCreateInfo=VkSwapchainCreateInfoKHR(
                flags=0,
                surface=surface.vk_surface,
                minImageCount=image_count,
                imageFormat=self.vk_format,
                imageColorSpace=self.vk_color_space,
                imageExtent=VkExtent2D(width=surface.width, height=surface.height),
                imageArrayLayers=1,
                imageUsage=VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                imageSharingMode=VK_SHARING_MODE_EXCLUSIVE,
                queueFamilyIndexCount=0,
                pQueueFamilyIndices=None,
                preTransform=VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
                compositeAlpha=VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
                presentMode=VK_PRESENT_MODE_FIFO_KHR,
                clipped=True,
                oldSwapchain=None,
            ),
            pAllocator=None,
        )

    def _wrap_swap_chain_images(self, surface: GpuSurface) -> list[GpuImage]:
        return [
            GpuImage(
                device=self.device,
                usages=["color-attachment"],
                meta=GpuImageMeta(
                    shape=(surface.height, surface.width, 4),
                    dtype=np.uint8,
                ),
                custom_vk_image_handle=vk_image,
                custom_vk_format=self.vk_format,
            )
            for vk_image in self.device.context.vkGetSwapchainImagesKHR(
                self.device.vk_device, self.vk_swap_chain
            )
        ]

    def _create_slots(self) -> list[GpuSwapChainSlot]:
        return [GpuSwapChainSlot(swap_chain=self) for _ in self.images]

    def __repr__(self) -> str:
        return (
            f"<GpuSwapChain object at {hex(id(self))} with handle {self.vk_swap_chain}>"
        )

    def _on_dispose(self) -> None:
        self.device.wait_idle()

        for slot in self.slots:
            slot.in_flight_fence.wait()

        self.context.vkDestroySwapchainKHR(
            device=self.device.vk_device,
            swapchain=self.vk_swap_chain,
            pAllocator=None,
        )

    @contextmanager
    def present(self, *, timeout_sec: float = 0.25):
        # See: https://vulkan-tutorial.com/Drawing_a_triangle/Drawing/Frames_in_flight

        global_frame_index = self.frame_counter
        self.frame_counter += 1

        slot_index = global_frame_index % len(self.images)

        slot = self.slots[slot_index]

        slot.in_flight_fence.wait()
        slot.in_flight_fence.reset()

        image_index = self.context.vkAcquireNextImageKHR(
            device=self.device.vk_device,
            swapchain=self.vk_swap_chain,
            timeout=int(timeout_sec * 10**9),
            semaphore=slot.image_available_semaphore.vk_semaphore,
            fence=None,
        )

        yield GpuPresentTarget(
            slot_index=slot_index,
            image_index=image_index,
            image=self.images[image_index],
            render_wait_semaphore=slot.image_available_semaphore,
            render_done_semaphore=slot.render_done_semaphore,
            render_done_fence=slot.in_flight_fence,
        )

        vk_queue = self.device.vk_queues[self.device.qfis["present"]]
        self.context.vkQueuePresentKHR(
            queue=vk_queue,
            pPresentInfo=VkPresentInfoKHR(
                waitSemaphoreCount=1,
                pWaitSemaphores=[slot.render_done_semaphore.vk_semaphore],
                swapchainCount=1,
                pSwapchains=[self.vk_swap_chain],
                pImageIndices=[image_index],
                pResults=None,
            ),
        )


class GpuSwapChainSlot(BaseResource):
    in_flight_fence: GpuFence
    image_available_semaphore: GpuSemaphore
    render_done_semaphore: GpuSemaphore

    def __init__(self, *, swap_chain: GpuSwapChain):
        super().__init__(parent=swap_chain.device)
        # IMPORTANT: Do not store a reference to `swap_chain` as this creates a
        # reference cycle that prevents proper resource disposal.
        self.in_flight_fence = GpuFence(device=swap_chain.device, signalled=True)
        self.image_available_semaphore = GpuSemaphore(device=swap_chain.device)
        self.render_done_semaphore = GpuSemaphore(device=swap_chain.device)

    def _on_dispose(self) -> None:
        self.in_flight_fence.dispose()
        self.image_available_semaphore.dispose()
        self.render_done_semaphore.dispose()


@dataclass
class GpuPresentTarget:
    slot_index: int
    """
    The slot index in the swap chain: frame_index % image_count. 
    May be different from image_index.
    """

    image_index: int
    """
    The index of the swap chain image to present.
    """

    image: GpuImage
    """
    The swap chain image to present to. Use this as a color attachment when rendering.
    """

    render_wait_semaphore: GpuSemaphore
    """
    A semaphore that the rendering should wait on before starting.
    """

    render_done_semaphore: GpuSemaphore
    """
    A semaphore that renderers must signal when rendering is done.
    """

    render_done_fence: GpuFence
    """
    A fence that renderers must signal when rendering is done.
    """


#
# GpuStage
#

GpuStage: TypeAlias = Literal["vertex", "fragment"]


def vk_shader_stages(stages: list[GpuStage]) -> int:
    result = 0
    for stage in stages:
        match stage:
            case "vertex":
                result |= VK_SHADER_STAGE_VERTEX_BIT
            case "fragment":
                result |= VK_SHADER_STAGE_FRAGMENT_BIT
            case _:
                raise LogicError(f"Invalid shader stage: {stage!r}")
    return result
