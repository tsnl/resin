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
    "GpuSwapchain",
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

import glfw
import torch
import numpy as np

from .core import BaseResource
from .excepts import LogicError, PlatformSupportError
from .typed_vulkan import (
    VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
    VK_API_VERSION_1_3,
    VK_API_VERSION_1_4,
    VK_ATTACHMENT_LOAD_OP_CLEAR,
    VK_ATTACHMENT_LOAD_OP_LOAD,
    VK_ATTACHMENT_STORE_OP_STORE,
    VK_BLEND_FACTOR_ONE,
    VK_BLEND_FACTOR_ZERO,
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
    VK_FORMAT_B8G8R8A8_UNORM,
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
# Type aliases for GPU configuration
#


GpuSamplerFilter: TypeAlias = Literal["nearest", "linear"]


#
# GpuContext
#


class GpuResource(BaseResource):
    parent: GpuResource | None
    context: "GpuContext"

    def __init__(
        self,
        *,
        parent: GpuResource | None,
        context: GpuContext | None = None,
    ):
        super().__init__(parent=parent)

        if parent is None:
            assert context is not None
            self.parent = None
            self.context = context
        else:
            assert context is None
            self.parent = parent
            self.context = parent.context


class GpuContext(GpuResource):
    enable_debug_layer_support: bool
    enable_present_support: bool
    enable_portability_subset: bool
    vk_instance: VkInstance
    vk_extra_proc_tab: dict[str, Callable]

    def __init__(
        self,
        app_name: str = "Zero App",
        enable_debug_layer_support: bool = True,
        enable_present_support: bool = True,
        enable_portability_subset_override: bool | None = None,
    ) -> None:
        super().__init__(parent=None, context=self)

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

    def create_surface_from_raw_glfw_window_handle(
        self,
        *,
        raw_glfw_window_handle: glfw._GLFWwindow,
        framebuffer_width: int,
        framebuffer_height: int,
    ) -> GpuSurface:
        """
        Do not call directly: use Window.create_surface() instead.
        """

        if not self.enable_present_support:
            raise LogicError(
                "Cannot create a GpuSurface when GpuContext was created with "
                "enable_present_support=False"
            )

        surface_ptr = raw_ffi.new("VkSurfaceKHR[1]")
        result = glfw.create_window_surface(
            instance=self.vk_instance,
            window=raw_glfw_window_handle,
            allocator=None,
            surface=surface_ptr,
        )
        if result != 0:
            raise RuntimeError(f"Failed to create window surface: VkResult: {result}")
        return GpuSurface(
            context=self,
            vk_surface=surface_ptr[0],
            width=framebuffer_width,
            height=framebuffer_height,
        )

    def create_device(
        self,
        *,
        physical_device: GpuPhysicalDevice,
        surface: GpuSurface | None,
        descriptor_pool_config: dict[GpuDescriptorType, int] | None = None,
    ) -> GpuDevice:
        # Ensure Vulkan 1.3 support
        physical_device.check_vulkan_1_3_support()

        # Compute queue family indices
        qfis = GpuQueueFamilyIndices.find(physical_device, surface=surface)
        queue_create_info_list = qfis.compute_queue_create_info_list()

        # Compute extensions
        extensions = []

        # Add dynamic rendering extension (Vulkan 1.3)
        extensions.append("VK_KHR_dynamic_rendering")

        # Add shader draw parameters extension: needed for Slang shaders
        extensions.append("VK_KHR_shader_draw_parameters")

        # Add present support extension if needed
        if self.enable_present_support:
            extensions.append("VK_KHR_swapchain")

        # Add portability subset extension if needed (macOS)
        if self.enable_portability_subset:
            extensions.append("VK_KHR_portability_subset")

        # Enable dynamic rendering feature (Vulkan 1.3)
        dynamic_rendering_features = VkPhysicalDeviceDynamicRenderingFeatures(
            dynamicRendering=True,
        )

        # Create device:
        vk_device = vkCreateDevice(
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

        # Create a temporary GpuDevice to use for creating the default descriptor pool
        gpu_device = GpuDevice(
            context=self,
            physical_device=physical_device,
            qfis=qfis,
            vk_device=vk_device,
            vk_command_pools=vk_command_pools,
            present_support_enabled=surface is not None,
            descriptor_pool_config=descriptor_pool_config or {},
        )

        return gpu_device

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


class GpuPhysicalDevice(GpuResource):
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

    def get_surface_formats(self, surface: "GpuSurface") -> list[VkSurfaceFormatKHR]:
        assert self.context.enable_present_support
        return list(
            self.context.vkGetPhysicalDeviceSurfaceFormatsKHR(
                self.vk_physical_device,
                surface.vk_surface,
            )
        )


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


class GpuDevice(GpuResource):
    physical_device: GpuPhysicalDevice
    qfis: GpuQueueFamilyIndices
    vk_device: VkDevice
    vk_command_pools: dict[int, VkCommandPool]
    vk_queues: dict[int, VkQueue]
    present_support_enabled: bool
    descriptor_pool_config: dict[GpuDescriptorType, int] | None
    max_descriptor_pool_set_count: int = 1024
    _descriptor_pool: GpuDescriptorPool

    def __init__(
        self,
        *,
        context: GpuContext,
        physical_device: GpuPhysicalDevice,
        qfis: GpuQueueFamilyIndices,
        vk_device: VkDevice,
        vk_command_pools: dict[int, VkCommandPool],
        present_support_enabled: bool,
        descriptor_pool_config: dict[GpuDescriptorType, int],
        max_descriptor_pool_set_count: int = 1024,
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
        self.present_support_enabled = present_support_enabled

        descriptor_pool_config_defaults: dict[GpuDescriptorType, int] = {
            "combined-image-sampler": 1024,
            "storage-buffer": 1024,
            "uniform-buffer": 1024,
        }
        self.descriptor_pool_config = (
            descriptor_pool_config_defaults | descriptor_pool_config
        )

        self.max_descriptor_pool_set_count = max_descriptor_pool_set_count

        self._descriptor_pool = self._create_descriptor_pool(
            max_sets=max_descriptor_pool_set_count,
            pool_sizes=self.descriptor_pool_config,
        )

    def _on_dispose(self) -> None:
        # Destroy default descriptor pool first (before command pools and device):
        if self._descriptor_pool is not None:
            self._descriptor_pool.dispose()

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

        # Infer VkFormat:
        vk_format = meta.infer_vk_format(usages)

        # Determine which queue families will access the image:
        queue_family_indices = list({idx for _, idx in self.qfis})

        # Create the VkImage:
        vk_image = vkCreateImage(
            device=self.vk_device,
            pCreateInfo=VkImageCreateInfo(
                flags=0,
                imageType=VK_IMAGE_TYPE_2D,
                format=vk_format,
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
            vk_format=vk_format,
            memory=memory,
            meta=meta,
            aspect_mask=vk_image_aspect,
            usages=usages,
            skip_image_destroy=False,
            skip_image_view_destroy=False,
            initial_vk_layout=VK_IMAGE_LAYOUT_UNDEFINED,
            current_vk_layout=VK_IMAGE_LAYOUT_UNDEFINED,
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
            device_local=device_local,
        )

        # Done:
        return buffer

    def create_shader(
        self,
        *,
        spirv_path: Path | str,
        stage: Literal["vertex", "fragment"],
    ) -> GpuShader:
        """Create a shader module from a SPIR-V file"""
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

        vk_shader_module = vkCreateShaderModule(
            device=self.vk_device,
            pCreateInfo=create_info,
            pAllocator=None,
        )

        return GpuShader(
            device=self,
            vk_shader_module=vk_shader_module,
            stage=stage,
        )

    def create_sampler(
        self,
        *,
        mag_filter: GpuSamplerFilter = "linear",
        min_filter: GpuSamplerFilter = "linear",
        address_mode: Literal[
            "repeat", "mirrored-repeat", "clamp-to-edge", "clamp-to-border"
        ] = "clamp-to-edge",
    ) -> "GpuSampler":
        """Create a texture sampler"""
        vk_mag_filter = (
            VK_FILTER_LINEAR if mag_filter == "linear" else VK_FILTER_NEAREST
        )
        vk_min_filter = (
            VK_FILTER_LINEAR if min_filter == "linear" else VK_FILTER_NEAREST
        )

        vk_address_mode = {
            "repeat": VK_SAMPLER_ADDRESS_MODE_REPEAT,
            "mirrored-repeat": VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT,
            "clamp-to-edge": VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            "clamp-to-border": VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
        }[address_mode]

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

        vk_sampler = vkCreateSampler(
            device=self.vk_device,
            pCreateInfo=create_info,
            pAllocator=None,
        )

        return GpuSampler(device=self, vk_sampler=vk_sampler)

    def _create_descriptor_pool(
        self,
        *,
        max_sets: int,
        pool_sizes: dict[GpuDescriptorType, int],
    ) -> "GpuDescriptorPool":
        """Create a descriptor pool for allocating descriptor sets."""
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

        vk_pool = vkCreateDescriptorPool(
            device=self.vk_device,
            pCreateInfo=create_info,
            pAllocator=None,
        )

        return GpuDescriptorPool(device=self, vk_descriptor_pool=vk_pool)

    def create_descriptor_set_layout(
        self,
        *,
        bindings: OrderedDict[str, "GpuDescriptorSetLayoutBinding"],
    ) -> "GpuDescriptorSetLayout":
        """Create a descriptor set layout."""

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

        vk_layout = vkCreateDescriptorSetLayout(
            device=self.vk_device,
            pCreateInfo=create_info,
            pAllocator=None,
        )

        return GpuDescriptorSetLayout(
            device=self,
            vk_descriptor_set_layout=vk_layout,
            bindings=bindings,
        )

    def create_descriptor_set(
        self,
        *,
        layout: "GpuDescriptorSetLayout",
        bindings: dict[str, "GpuDescriptorSetBinding"],
    ) -> "GpuDescriptorSet":
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

        # Validation complete.

        # Allocate the descriptor set:
        alloc_info = VkDescriptorSetAllocateInfo(
            descriptorPool=self._descriptor_pool.vk_descriptor_pool,
            descriptorSetCount=1,
            pSetLayouts=[layout.vk_descriptor_set_layout],
        )
        vk_sets = vkAllocateDescriptorSets(
            device=self.vk_device,
            pAllocateInfo=alloc_info,
        )
        vk_set = vk_sets[0]

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
            device=self.vk_device,
            descriptorWriteCount=len(writes),
            pDescriptorWrites=writes,
            descriptorCopyCount=0,
            pDescriptorCopies=None,
        )

        # Done:
        return GpuDescriptorSet(
            device=self,
            vk_descriptor_set=vk_set,
            pool=self._descriptor_pool,
            layout=layout,
            bindings=bindings,
        )

    def create_pipeline_layout(
        self,
        *,
        descriptor_set_layouts: list["GpuDescriptorSetLayout"],
    ) -> "GpuPipelineLayout":
        """Create a pipeline layout."""

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

        vk_pipeline_layout = vkCreatePipelineLayout(
            device=self.vk_device,
            pCreateInfo=layout_create_info,
            pAllocator=None,
        )

        return GpuPipelineLayout(
            device=self,
            vk_pipeline_layout=vk_pipeline_layout,
            descriptor_set_layouts=descriptor_set_layouts,
        )

    def create_pipeline(
        self,
        *,
        vertex_shader: GpuShader,
        fragment_shader: GpuShader,
        vk_color_format: VkFormat,
        viewport_width: int,
        viewport_height: int,
        layout: "GpuPipelineLayout",
    ) -> GpuPipeline:
        """Create a graphics pipeline.

        Args:
            vertex_shader: The vertex shader module.
            fragment_shader: The fragment shader module.
            vk_color_format: The color attachment format.
            viewport_width: Width of the viewport.
            viewport_height: Height of the viewport.
            layout: Optional pipeline layout. If not provided, an empty layout is created.
        """

        # Shader stages
        shader_stages = [
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

        # Vertex input state (empty - hardcoded in shader)
        vertex_input_state = VkPipelineVertexInputStateCreateInfo(
            flags=0,
            vertexBindingDescriptionCount=0,
            pVertexBindingDescriptions=None,
            vertexAttributeDescriptionCount=0,
            pVertexAttributeDescriptions=None,
        )

        # Input assembly state
        input_assembly_state = VkPipelineInputAssemblyStateCreateInfo(
            flags=0,
            topology=VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            primitiveRestartEnable=False,
        )

        # Viewport state
        viewport = VkViewport(
            x=0.0,
            y=0.0,
            width=float(viewport_width),
            height=float(viewport_height),
            minDepth=0.0,
            maxDepth=1.0,
        )

        scissor = VkRect2D(
            offset=VkOffset2D(x=0, y=0),
            extent=VkExtent2D(width=viewport_width, height=viewport_height),
        )

        viewport_state = VkPipelineViewportStateCreateInfo(
            flags=0,
            viewportCount=1,
            pViewports=[viewport],
            scissorCount=1,
            pScissors=[scissor],
        )

        # Rasterization state
        rasterization_state = VkPipelineRasterizationStateCreateInfo(
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

        # Multisample state
        multisample_state = VkPipelineMultisampleStateCreateInfo(
            flags=0,
            rasterizationSamples=VK_SAMPLE_COUNT_1_BIT,
            sampleShadingEnable=False,
            minSampleShading=1.0,
            pSampleMask=None,
            alphaToCoverageEnable=False,
            alphaToOneEnable=False,
        )

        # Color blend state
        color_blend_attachment = VkPipelineColorBlendAttachmentState(
            blendEnable=False,
            srcColorBlendFactor=VK_BLEND_FACTOR_ONE,
            dstColorBlendFactor=VK_BLEND_FACTOR_ZERO,
            colorBlendOp=VK_BLEND_OP_ADD,
            srcAlphaBlendFactor=VK_BLEND_FACTOR_ONE,
            dstAlphaBlendFactor=VK_BLEND_FACTOR_ZERO,
            alphaBlendOp=VK_BLEND_OP_ADD,
            colorWriteMask=(
                VK_COLOR_COMPONENT_R_BIT
                | VK_COLOR_COMPONENT_G_BIT
                | VK_COLOR_COMPONENT_B_BIT
                | VK_COLOR_COMPONENT_A_BIT
            ),
        )

        color_blend_state = VkPipelineColorBlendStateCreateInfo(
            flags=0,
            logicOpEnable=False,
            logicOp=VK_LOGIC_OP_COPY,
            attachmentCount=1,
            pAttachments=[color_blend_attachment],
            blendConstants=[0.0, 0.0, 0.0, 0.0],
        )

        # Dynamic rendering info (Vulkan 1.3)
        rendering_info = VkPipelineRenderingCreateInfo(
            viewMask=0,
            colorAttachmentCount=1,
            pColorAttachmentFormats=[vk_color_format],
            depthAttachmentFormat=VK_FORMAT_UNDEFINED,
            stencilAttachmentFormat=VK_FORMAT_UNDEFINED,
        )

        # Graphics pipeline create info
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
            pDepthStencilState=None,
            pColorBlendState=color_blend_state,
            pDynamicState=None,
            layout=layout.vk_pipeline_layout,
            renderPass=None,  # Using dynamic rendering
            subpass=0,
            basePipelineHandle=None,
            basePipelineIndex=-1,
        )

        # Create pipeline
        vk_pipeline = vkCreateGraphicsPipelines(
            device=self.vk_device,
            pipelineCache=None,
            createInfoCount=1,
            pCreateInfos=[pipeline_create_info],
            pAllocator=None,
        )[0]

        # Wrap:
        return GpuPipeline(
            device=self,
            vk_pipeline=vk_pipeline,
            vk_color_format=vk_color_format,
            viewport_width=viewport_width,
            viewport_height=viewport_height,
            layout=layout,
        )

    def create_command_encoder(
        self,
        *,
        queue_type: GpuQueueType,
        fence: GpuFence | None = None,
        wait_semaphores: list["GpuSemaphore"] | None = None,
        signal_semaphores: list["GpuSemaphore"] | None = None,
    ) -> GpuCommandEncoder:
        """Create a command buffer encoder.

        Args:
            queue_type: Type of queue to submit to.
        """
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

        # Create fence for this command buffer if none was provided
        dispose_fence = fence is None
        fence = fence or self.create_fence()

        vkBeginCommandBuffer(
            commandBuffer=vk_command_buffer,
            pBeginInfo=VkCommandBufferBeginInfo(
                flags=0,
                pInheritanceInfo=None,
            ),
        )
        return GpuCommandEncoder(
            device=self,
            queue_family_index=queue_family_index,
            vk_command_buffer=vk_command_buffer,
            fence=fence,
            submit_queue_type=queue_type,
            dispose_fence=dispose_fence,
            wait_semaphores=wait_semaphores or [],
            signal_semaphores=signal_semaphores or [],
        )

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

    def create_fence(self, *, signalled: bool = False) -> "GpuFence":
        vk_fence = vkCreateFence(
            device=self.vk_device,
            pCreateInfo=VkFenceCreateInfo(
                flags=VK_FENCE_CREATE_SIGNALED_BIT if signalled else 0
            ),
            pAllocator=None,
        )
        return GpuFence(device=self, vk_fence=vk_fence)

    def create_semaphore(self) -> "GpuSemaphore":
        vk_semaphore = vkCreateSemaphore(
            device=self.vk_device,
            pCreateInfo=VkSemaphoreCreateInfo(flags=0),
            pAllocator=None,
        )
        return GpuSemaphore(device=self, vk_semaphore=vk_semaphore)

    def create_swapchain(
        self,
        *,
        surface: GpuSurface,
        image_count: int,
    ) -> "GpuSwapchain":
        vk_format = VK_FORMAT_B8G8R8A8_UNORM
        vk_colorspace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR
        vk_surface_format_list = self.physical_device.get_surface_formats(surface)
        for surface_format in vk_surface_format_list:
            if surface_format.format != vk_format:
                continue
            if surface_format.colorSpace != vk_colorspace:
                continue
            break
        else:
            raise PlatformSupportError(
                f"Physical device {self.physical_device.name!r} does not support "
                "the required swapchain format: "
                f"Requires format=VK_FORMAT_B8G8R8A8_UNORM, "
                f"colorSpace=VK_COLOR_SPACE_SRGB_NONLINEAR_KHR."
            )

        vk_swapchain = self.context.vkCreateSwapchainKHR(
            device=self.vk_device,
            pCreateInfo=VkSwapchainCreateInfoKHR(
                flags=0,
                surface=surface.vk_surface,
                minImageCount=image_count,
                imageFormat=vk_format,
                imageColorSpace=vk_colorspace,
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
        vk_images = self.context.vkGetSwapchainImagesKHR(self.vk_device, vk_swapchain)
        images = []
        in_flight_fences = []
        image_available_semaphores = []
        render_finished_semaphores = []
        for vk_image in vk_images:
            vk_image_view = vkCreateImageView(
                device=self.vk_device,
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
                        aspectMask=VK_IMAGE_ASPECT_COLOR_BIT,
                        baseMipLevel=0,
                        levelCount=1,
                        baseArrayLayer=0,
                        layerCount=1,
                    ),
                ),
                pAllocator=None,
            )
            image = GpuImage(
                device=self,
                vk_image=vk_image,
                vk_image_view=vk_image_view,
                vk_format=vk_format,
                memory=None,
                meta=GpuImageMeta(
                    shape=(surface.height, surface.width, 4),
                    dtype=torch.uint8,
                ),
                aspect_mask=VK_IMAGE_ASPECT_COLOR_BIT,
                usages=["color-attachment"],
                skip_image_destroy=True,
                skip_image_view_destroy=False,
                initial_vk_layout=VK_IMAGE_LAYOUT_UNDEFINED,
                current_vk_layout=VK_IMAGE_LAYOUT_UNDEFINED,
            )
            images.append(image)
            in_flight_fences.append(self.create_fence(signalled=True))
            image_available_semaphores.append(self.create_semaphore())
            render_finished_semaphores.append(self.create_semaphore())
        return GpuSwapchain(
            device=self,
            vk_swapchain=vk_swapchain,
            images=images,
            in_flight_fences=in_flight_fences,
            image_available_semaphores=image_available_semaphores,
            render_finished_semaphores=render_finished_semaphores,
            vk_format=vk_format,
            width=surface.width,
            height=surface.height,
            frame_counter=0,
        )

    def wait_idle(self) -> None:
        vkDeviceWaitIdle(self.vk_device)

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

    def write(self, *, data: torch.Tensor):
        src_bytes = data.ravel().view(torch.uint8).numpy()
        with self.map() as host_mem:
            host_mem[: len(src_bytes)] = src_bytes

    def read(self, *, dtype: torch.dtype) -> torch.Tensor:
        with self.map() as host_mem:
            return torch.frombuffer(host_mem, dtype=dtype).clone()


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

GpuImageLayout: TypeAlias = Literal[
    "present-src",
    "color-attachment-optimal",
    "copy-dst",
    "copy-src",
    "texture-binding",
    "transfer-src",
    "transfer-dst",
]


class GpuImage(GpuResource):
    device: GpuDevice
    vk_image: VkImage
    vk_image_view: VkImageView
    vk_format: VkFormat
    memory: GpuMemory | None
    meta: GpuImageMeta
    aspect_mask: int
    usages: list[GpuImageUsage]
    skip_image_destroy: bool
    skip_image_view_destroy: bool
    initial_vk_layout: VkImageLayout
    current_vk_layout: VkImageLayout

    def __init__(
        self,
        *,
        device: GpuDevice,
        vk_image: VkImage,
        vk_image_view: VkImageView,
        vk_format: VkFormat,
        memory: GpuMemory | None,
        meta: GpuImageMeta,
        aspect_mask: int,
        usages: list[GpuImageUsage],
        skip_image_destroy: bool,
        skip_image_view_destroy: bool,
        initial_vk_layout: VkImageLayout,
        current_vk_layout: VkImageLayout,
    ) -> None:
        super().__init__(parent=device)
        self.device = device
        self.vk_image = vk_image
        self.vk_image_view = vk_image_view
        self.vk_format = vk_format
        self.memory = memory
        self.meta = meta
        self.aspect_mask = aspect_mask
        self.usages = usages
        self.skip_image_destroy = skip_image_destroy
        self.skip_image_view_destroy = skip_image_view_destroy
        self.initial_vk_layout = initial_vk_layout
        self.current_vk_layout = current_vk_layout

    def _on_dispose(self) -> None:
        if not self.skip_image_view_destroy:
            vkDestroyImageView(
                self.device.vk_device, self.vk_image_view, pAllocator=None
            )
        if not self.skip_image_destroy:
            vkDestroyImage(self.device.vk_device, self.vk_image, pAllocator=None)

    @property
    def width(self) -> int:
        return self.meta.shape[1]

    @property
    def height(self) -> int:
        return self.meta.shape[0]

    @property
    def channel_count(self) -> int:
        return self.meta.shape[2]

    def write(self, *, data: torch.Tensor):
        staging_buffer = self.device.create_buffer(
            usages=["copy-src"],
            meta=GpuBufferMeta.from_tensor(data),
        )
        staging_buffer.memory.write(data=data)

        cmd = self.device.create_command_encoder(queue_type="transfer")
        cmd.copy_buffer_to_image(src=staging_buffer, dst=self)
        cmd.submit().wait()


#
# GpuSemaphore
#


class GpuSemaphore(GpuResource):
    device: GpuDevice
    vk_semaphore: VkSemaphore

    def __init__(self, *, device: GpuDevice, vk_semaphore: VkSemaphore) -> None:
        super().__init__(parent=device)
        self.device = device
        self.vk_semaphore = vk_semaphore

    def _on_dispose(self) -> None:
        vkDestroySemaphore(self.device.vk_device, self.vk_semaphore, pAllocator=None)


#
# GpuFence
#


class GpuFence(GpuResource):
    device: GpuDevice
    vk_fence: VkFence

    def __init__(self, *, device: GpuDevice, vk_fence: VkFence) -> None:
        super().__init__(parent=device)
        self.device = device
        self.vk_fence = vk_fence

    def wait(self, *, timeout_ns: int = 10**10) -> None:
        """Wait for fence to be signaled.

        Args:
            timeout_ns: Timeout in nanoseconds (default 10s)
        """
        vkWaitForFences(
            device=self.device.vk_device,
            fenceCount=1,
            pFences=[self.vk_fence],
            waitAll=True,
            timeout=timeout_ns,
        )

    def reset(self) -> None:
        """Reset the fence to unsignaled state."""
        vkResetFences(
            device=self.device.vk_device,
            fenceCount=1,
            pFences=[self.vk_fence],
        )

    def _on_dispose(self) -> None:
        vkDestroyFence(self.device.vk_device, self.vk_fence, pAllocator=None)


#
# GpuBuffer
#


@dataclass
class GpuBufferMeta:
    element_count: int
    element_dtype: torch.dtype | np.dtype

    @property
    def size(self) -> int:
        return self.element_count * self.element_size

    @property
    def element_size(self) -> int:
        return self.element_dtype.itemsize

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
    device_local: bool

    def __init__(
        self,
        device: GpuDevice,
        vk_buffer: VkBuffer,
        memory: GpuMemory,
        meta: GpuBufferMeta,
        usages: list[GpuBufferUsage],
        device_local: bool,
    ):
        super().__init__(parent=device)
        self.device = device
        self.vk_buffer = vk_buffer
        self.memory = memory
        self.meta = meta
        self.usages = usages
        self.device_local = device_local

    def _on_dispose(self):
        vkDestroyBuffer(self.device.vk_device, self.vk_buffer, pAllocator=None)

    def write(self, *, data: torch.Tensor):
        self.memory.write(data=data)

    def read(self) -> torch.Tensor:
        return self.memory.read(dtype=self.meta.element_dtype)


#
# GpuCommandBuffer
#


GpuCommandBufferLevel: TypeAlias = Literal["primary", "secondary"]


class GpuCommandEncoder(GpuResource):
    device: GpuDevice
    vk_command_buffer: VkCommandBuffer
    queue_family_index: int
    fence: GpuFence
    submit_queue_type: GpuQueueType
    dispose_fence: bool
    wait_semaphores: list["GpuSemaphore"] | None
    signal_semaphores: list["GpuSemaphore"] | None
    _submitted: bool

    def __init__(
        self,
        *,
        device: GpuDevice,
        queue_family_index: int,
        vk_command_buffer: VkCommandBuffer,
        fence: GpuFence,
        submit_queue_type: GpuQueueType,
        dispose_fence: bool,
        wait_semaphores: list["GpuSemaphore"],
        signal_semaphores: list["GpuSemaphore"],
    ) -> None:
        super().__init__(parent=device)
        self.device = device
        self.vk_command_buffer = vk_command_buffer
        self.queue_family_index = queue_family_index
        self.fence = fence
        self.submit_queue_type = submit_queue_type
        self.dispose_fence = dispose_fence
        self.wait_semaphores = wait_semaphores
        self.signal_semaphores = signal_semaphores
        self._submitted = False

    def _on_dispose(self) -> None:
        # Wait for command buffer to finish executing before freeing it
        if self._submitted:
            self.fence.wait()

        # Free the command buffer
        vkFreeCommandBuffers(
            device=self.device.vk_device,
            commandPool=self.device.vk_command_pools[self.queue_family_index],
            commandBufferCount=1,
            pCommandBuffers=[self.vk_command_buffer],
        )

        # If specified, dispose the fence as well
        if self.dispose_fence:
            self.fence.dispose()

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
        clear_on_load: bool = False,
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
                        if clear_on_load
                        else VK_ATTACHMENT_LOAD_OP_LOAD
                    ),
                    storeOp=VK_ATTACHMENT_STORE_OP_STORE,
                    clearValue=VkClearValue(
                        color=VkClearColorValue(float32=[0.0, 0.0, 0.0, 0.0])
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
                    if clear_on_load
                    else VK_ATTACHMENT_LOAD_OP_LOAD
                ),
                storeOp=VK_ATTACHMENT_STORE_OP_STORE,
                clearValue=VkClearValue(
                    depthStencil=VkClearDepthStencilValue(depth=1.0, stencil=0)
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
        from .typed_vulkan import (
            VK_ACCESS_SHADER_READ_BIT,
            VK_ACCESS_TRANSFER_READ_BIT,
            VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        )

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
                    "transfer-src": VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    "transfer-dst": VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                }[layout]
            )
        )

        # Skip if already in the correct layout
        if image.current_vk_layout == new_layout:
            return

        # Determine access masks based on old and new layouts
        src_access_mask = 0
        if image.current_vk_layout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
            src_access_mask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
        elif image.current_vk_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
            src_access_mask = VK_ACCESS_TRANSFER_WRITE_BIT
        elif image.current_vk_layout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
            src_access_mask = VK_ACCESS_TRANSFER_READ_BIT

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

    def submit(self) -> GpuFence:
        vkEndCommandBuffer(commandBuffer=self.vk_command_buffer)

        self.device.submit(
            queue_type=self.submit_queue_type,
            command_buffers=[self.vk_command_buffer],
            wait_semaphores=self.wait_semaphores,
            signal_semaphores=self.signal_semaphores,
            fence=self.fence,
        )

        self._submitted = True

        return self.fence


#
# GpuShader
#


class GpuShader(GpuResource):
    """Wrapper for VkShaderModule"""

    device: GpuDevice
    vk_shader_module: VkShaderModule
    stage: Literal["vertex", "fragment"]

    def __init__(
        self,
        *,
        device: GpuDevice,
        vk_shader_module: VkShaderModule,
        stage: Literal["vertex", "fragment"],
    ):
        super().__init__(parent=device)
        self.device = device
        self.vk_shader_module = vk_shader_module
        self.stage = stage

    def _on_dispose(self) -> None:
        vkDestroyShaderModule(
            device=self.device.vk_device,
            shaderModule=self.vk_shader_module,
            pAllocator=None,
        )


#
# GpuSampler
#


class GpuSampler(GpuResource):
    """Wrapper for VkSampler"""

    device: GpuDevice
    vk_sampler: VkSampler

    def __init__(self, *, device: GpuDevice, vk_sampler: VkSampler):
        super().__init__(parent=device)
        self.device = device
        self.vk_sampler = vk_sampler

    def _on_dispose(self) -> None:
        vkDestroySampler(
            device=self.device.vk_device,
            sampler=self.vk_sampler,
            pAllocator=None,
        )


#
# GpuDescriptorPool
#


class GpuDescriptorPool(GpuResource):
    """Wrapper for VkDescriptorPool"""

    device: GpuDevice
    vk_descriptor_pool: VkDescriptorPool

    def __init__(self, *, device: GpuDevice, vk_descriptor_pool: VkDescriptorPool):
        super().__init__(parent=device)
        self.device = device
        self.vk_descriptor_pool = vk_descriptor_pool

    def _on_dispose(self) -> None:
        vkDestroyDescriptorPool(
            device=self.device.vk_device,
            descriptorPool=self.vk_descriptor_pool,
            pAllocator=None,
        )


#
# GpuPipelineLayout, GpuDescriptorSetLayout:
#


class GpuPipelineLayout(GpuResource):
    device: GpuDevice
    vk_pipeline_layout: VkPipelineLayout
    descriptor_set_layouts: list[GpuDescriptorSetLayout]

    def __init__(
        self,
        *,
        device: GpuDevice,
        vk_pipeline_layout: VkPipelineLayout,
        descriptor_set_layouts: list[GpuDescriptorSetLayout],
    ):
        super().__init__(parent=device)
        self.device = device
        self.vk_pipeline_layout = vk_pipeline_layout
        self.descriptor_set_layouts = descriptor_set_layouts

    def _on_dispose(self) -> None:
        vkDestroyPipelineLayout(
            device=self.device.vk_device,
            pipelineLayout=self.vk_pipeline_layout,
            pAllocator=None,
        )


class GpuDescriptorSetLayout(GpuResource):
    device: GpuDevice
    vk_descriptor_set_layout: VkDescriptorSetLayout
    bindings: OrderedDict[str, "GpuDescriptorSetLayoutBinding"]

    def __init__(
        self,
        *,
        device: GpuDevice,
        vk_descriptor_set_layout: VkDescriptorSetLayout,
        bindings: OrderedDict[str, "GpuDescriptorSetLayoutBinding"],
    ):
        super().__init__(parent=device)
        self.device = device
        self.vk_descriptor_set_layout = vk_descriptor_set_layout
        self.bindings = bindings

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


class GpuDescriptorSet(GpuResource):
    device: GpuDevice
    vk_descriptor_set: VkDescriptorSet
    pool: GpuDescriptorPool
    layout: GpuDescriptorSetLayout
    bindings: dict[str, GpuDescriptorSetBinding]

    def __init__(
        self,
        *,
        device: GpuDevice,
        vk_descriptor_set: VkDescriptorSet,
        pool: GpuDescriptorPool,
        layout: GpuDescriptorSetLayout,
        bindings: dict[str, GpuDescriptorSetBinding],
    ):
        super().__init__(parent=pool)
        self.device = device
        self.vk_descriptor_set = vk_descriptor_set
        self.pool = pool
        self.layout = layout
        self.bindings = bindings


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


class GpuPipeline(GpuResource):
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
        vk_pipeline: VkPipeline,
        vk_color_format: VkFormat,
        viewport_width: int,
        viewport_height: int,
        layout: GpuPipelineLayout,
    ):
        super().__init__(parent=device)
        self.device = device
        self.vk_pipeline = vk_pipeline
        self.vk_color_format = vk_color_format
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.layout = layout

    def _on_dispose(self) -> None:
        vkDestroyPipeline(
            device=self.device.vk_device,
            pipeline=self.vk_pipeline,
            pAllocator=None,
        )


class GpuRenderPassCommandEncoder(GpuResource):
    device: GpuDevice
    command_encoder: GpuCommandEncoder

    def __init__(self, *, device: GpuDevice, command_encoder: GpuCommandEncoder):
        super().__init__(parent=command_encoder)
        self.device = device
        self.command_encoder = command_encoder

    def _on_dispose(self) -> None:
        pass

    def bind_pipeline(self, *, pipeline: GpuPipeline) -> None:
        vkCmdBindPipeline(
            commandBuffer=self.command_encoder.vk_command_buffer,
            pipelineBindPoint=VK_PIPELINE_BIND_POINT_GRAPHICS,
            pipeline=pipeline.vk_pipeline,
        )

    def bind_descriptor_sets(
        self,
        *,
        layout: GpuPipelineLayout,
        first_set: int,
        sets: list[GpuDescriptorSet],
        dynamic_offsets: list[int] | None = None,
    ) -> None:
        vkCmdBindDescriptorSets(
            commandBuffer=self.command_encoder.vk_command_buffer,
            pipelineBindPoint=VK_PIPELINE_BIND_POINT_GRAPHICS,
            layout=layout.vk_pipeline_layout,
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


class GpuSurface(GpuResource):
    vk_surface: VkSurfaceKHR
    width: int
    height: int

    def __init__(
        self,
        *,
        context: GpuContext,
        vk_surface: VkSurfaceKHR,
        width: int,
        height: int,
    ):
        super().__init__(parent=context)
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
# GpuSwapchain
#


class GpuSwapchain(GpuResource):
    device: GpuDevice
    vk_swapchain: VkSwapchainKHR
    images: list[GpuImage]
    vk_format: VkFormat
    in_flight_fences: list[GpuFence]
    image_available_semaphores: list[GpuSemaphore]
    render_done_semaphores: list[GpuSemaphore]
    width: int
    height: int
    frame_counter: int

    def __init__(
        self,
        *,
        device: GpuDevice,
        vk_swapchain: VkSwapchainKHR,
        images: list[GpuImage],
        vk_format: VkFormat,
        in_flight_fences: list[GpuFence],
        image_available_semaphores: list[GpuSemaphore],
        render_finished_semaphores: list[GpuSemaphore],
        width: int,
        height: int,
        frame_counter: int,
    ):
        super().__init__(parent=device)
        self.device = device
        self.vk_swapchain = vk_swapchain
        self.images = images
        self.vk_format = vk_format
        self.in_flight_fences = in_flight_fences
        self.image_available_semaphores = image_available_semaphores
        self.render_done_semaphores = render_finished_semaphores
        self.width = width
        self.height = height
        self.frame_counter = frame_counter

    def _on_dispose(self) -> None:
        self.device.wait_idle()

        for fence in self.in_flight_fences:
            fence.wait()

        self.context.vkDestroySwapchainKHR(
            device=self.device.vk_device,
            swapchain=self.vk_swapchain,
            pAllocator=None,
        )

    @contextmanager
    def present(self, *, timeout_sec: float = 0.25):
        # See: https://vulkan-tutorial.com/Drawing_a_triangle/Drawing/Frames_in_flight

        global_frame_index = self.frame_counter
        self.frame_counter += 1

        current_frame = global_frame_index % len(self.in_flight_fences)

        in_flight_fence = self.in_flight_fences[current_frame]
        image_available_semaphore = self.image_available_semaphores[current_frame]
        render_done_semaphore = self.render_done_semaphores[current_frame]

        in_flight_fence.wait()
        in_flight_fence.reset()

        image_index = self.context.vkAcquireNextImageKHR(
            device=self.device.vk_device,
            swapchain=self.vk_swapchain,
            timeout=int(timeout_sec * 10**9),
            semaphore=image_available_semaphore.vk_semaphore,
            fence=None,
        )

        yield GpuPresentTarget(
            swapchain_image_index=image_index,
            global_frame_index=global_frame_index,
            wrapped_frame_index=current_frame,
            in_flight_index=current_frame,
            swapchain_image=self.images[image_index],
            render_wait_semaphores=[image_available_semaphore],
            render_done_semaphores=[render_done_semaphore],
            render_done_fence=in_flight_fence,
        )

        vk_queue = self.device.vk_queues[self.device.qfis["present"]]
        self.context.vkQueuePresentKHR(
            queue=vk_queue,
            pPresentInfo=VkPresentInfoKHR(
                waitSemaphoreCount=1,
                pWaitSemaphores=[render_done_semaphore.vk_semaphore],
                swapchainCount=1,
                pSwapchains=[self.vk_swapchain],
                pImageIndices=[image_index],
                pResults=None,
            ),
        )


@dataclass
class GpuPresentTarget:
    swapchain_image_index: int
    global_frame_index: int
    wrapped_frame_index: int
    in_flight_index: int
    swapchain_image: GpuImage
    render_wait_semaphores: list[GpuSemaphore]
    render_done_semaphores: list[GpuSemaphore]
    render_done_fence: GpuFence


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
