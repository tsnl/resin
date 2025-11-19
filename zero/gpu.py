__all__ = [
    # GpuContext
    "GpuContext",
    # GpuPhysicalDevice
    "GpuPhysicalDevice",
    # GpuDevice
    "GpuDevice",
    # GpuImage
    "GpuImage",
    "GpuTextureUsage",
    "GpuTextureSpec",
]

from dataclasses import dataclass
from collections import defaultdict
from typing import TypeAlias, Literal
from enum import IntFlag, auto
import sys
import os
import functools

import torch

from .core import BaseContext, BaseContextResource, expect
from .excepts import UnsupportedPlatformError, LogicError
from .typed_vulkan import (
    # Basic
    vkGetInstanceProcAddr,
    ffi,
    # Common
    VkDeviceSize,
    VkSampleCountFlags,
    VkExtent3D,
    # Instance
    VkInstance,
    vkCreateInstance,
    vkDestroyInstance,
    VkInstanceCreateInfo,
    VkApplicationInfo,
    vkEnumeratePhysicalDevices,
    VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR,
    # Surfaces
    VkSurfaceKHR,
    # Physical devices
    VkPhysicalDevice,
    vkGetPhysicalDeviceProperties,
    VkPhysicalDeviceProperties,
    VkPhysicalDeviceLimits,
    VK_PHYSICAL_DEVICE_TYPE_OTHER,
    VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU,
    VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU,
    VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU,
    VK_PHYSICAL_DEVICE_TYPE_CPU,
    # Physical device queues
    vkGetPhysicalDeviceQueueFamilyProperties,
    VkDeviceQueueCreateInfo,
    VK_QUEUE_COMPUTE_BIT,
    VK_QUEUE_TRANSFER_BIT,
    VK_QUEUE_GRAPHICS_BIT,
    # Devices
    VkDevice,
    vkCreateDevice,
    vkDestroyDevice,
    VkDeviceCreateInfo,
    # Images
    ## VkImageCreateFlags,
    ## VkImageCreateFlagBits,
    VK_IMAGE_CREATE_SPARSE_BINDING_BIT,
    VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT,
    VK_IMAGE_CREATE_SPARSE_ALIASED_BIT,
    VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT,
    VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT,
    ## VkImageType,
    VK_IMAGE_TYPE_1D,
    VK_IMAGE_TYPE_2D,
    VK_IMAGE_TYPE_3D,
    ## VkFormat,
    VK_FORMAT_R32_SFLOAT,
    VK_FORMAT_R8G8B8A8_UNORM,
    VK_FORMAT_R32G32B32A32_SFLOAT,
    ## VkSampleCountFlags,
    ## VkSampleCountFlagBits,
    VK_SAMPLE_COUNT_1_BIT,
    VK_SAMPLE_COUNT_2_BIT,
    VK_SAMPLE_COUNT_4_BIT,
    VK_SAMPLE_COUNT_8_BIT,
    VK_SAMPLE_COUNT_16_BIT,
    VK_SAMPLE_COUNT_32_BIT,
    VK_SAMPLE_COUNT_64_BIT,
    ## VkImageTiling,
    VK_IMAGE_TILING_OPTIMAL,
    VK_IMAGE_TILING_LINEAR,
    ## VkImageUsageFlags,
    ## VkImageUsageFlagBits,
    VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
    VK_IMAGE_USAGE_TRANSFER_DST_BIT,
    VK_IMAGE_USAGE_SAMPLED_BIT,
    VK_IMAGE_USAGE_STORAGE_BIT,
    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
    ## VkSharingMode,
    VK_SHARING_MODE_EXCLUSIVE,
    VK_SHARING_MODE_CONCURRENT,
    ## VkImageLayout,
    VK_IMAGE_LAYOUT_UNDEFINED,
    VK_IMAGE_LAYOUT_GENERAL,
    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
    VK_IMAGE_LAYOUT_PREINITIALIZED,
    vkCreateImage,
    VkImageCreateInfo,
    vkDestroyImage,
    # Image views
    vkCreateImageView,
    VkImageViewCreateInfo,
    vkDestroyImageView,
    # Samplers
    vkCreateSampler,
    VkSamplerCreateInfo,
    vkDestroySampler,
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
                raise UnsupportedPlatformError(
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
            raise UnsupportedPlatformError(
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
            GpuPhysicalDevice(self, vk_physical_device)
            for vk_physical_device in vkEnumeratePhysicalDevices(self._vk_instance)
        ]

    def create_device(
        self,
        physical_device: GpuPhysicalDevice,
        surface: VkSurfaceKHR | None,
    ) -> GpuDevice:
        return GpuDevice(physical_device, surface)


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
    def __init__(
        self,
        context: GpuContext,
        vk_physical_device: VkPhysicalDevice,
    ) -> None:
        super().__init__(parent=context)
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
        super().__init__(parent=physical_device)

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

    @property
    def vk_device(self) -> VkDevice:
        return self._vk_device

    @property
    def qfis(self) -> GpuQueueFamilyIndices:
        return self._qfis

    def create_texture(
        self,
        usages: tuple[GpuTextureUsage, ...],
        *,
        init: torch.Tensor | None = None,
        spec: GpuTextureSpec | None = None,
    ) -> "GpuImage":
        return GpuImage(device=self, usages=usages, init=init, spec=spec)


#
# GpuImage
#


@dataclass
class GpuTextureSpec:
    shape: tuple[int, int, int]  # (height, width, channels)
    dtype: torch.dtype

    @staticmethod
    def from_tensor(tensor: torch.Tensor) -> "GpuTextureSpec":
        if tensor.ndim != 3:
            raise LogicError(
                f"GpuTextureSpec can only be created from 3D tensors, got tensor with "
                f"{tensor.ndim} dimensions instead"
            )
        return GpuTextureSpec(
            shape=(tensor.shape[0], tensor.shape[1], tensor.shape[2]),
            dtype=tensor.dtype,
        )

    def select_vk_format(self) -> int:
        match (self.dtype, self.shape[2]):
            case (torch.float32, 1):
                return VK_FORMAT_R32_SFLOAT
            case (torch.uint8, 4):
                return VK_FORMAT_R8G8B8A8_UNORM
            case (torch.float32, 4):
                return VK_FORMAT_R32G32B32A32_SFLOAT
            case _:
                raise LogicError(
                    f"GpuTextureSpec.select_vk_format(): unsupported combination of "
                    f"dtype={self.dtype} and channels={self.shape[2]}"
                )


GpuTextureUsage: TypeAlias = Literal[
    "sampled",
    "storage",
    "color-attachment",
    "depth-stencil-attachment",
]


class GpuImage(GpuResource):
    def __init__(
        self,
        device: GpuDevice,
        usages: tuple[GpuTextureUsage, ...],
        *,
        init: torch.Tensor | None = None,
        spec: GpuTextureSpec | None = None,
    ) -> None:
        super().__init__(parent=device)

        self._device: GpuDevice = device

        # _init, _spec
        self._init: torch.Tensor | None = init
        if spec is not None and init is not None:
            raise LogicError("GpuImage(): cannot provide both init and spec")
        elif spec is not None:
            self._spec: GpuTextureSpec = spec
        elif init is not None:
            self._spec = GpuTextureSpec.from_tensor(init)
        else:
            raise LogicError("GpuImage(): either init or spec must be provided")

        # vk_usage
        vk_usage = 0
        for usage in usages:
            vk_usage |= GpuImage._usage_to_vk_format_bit_dict().get(usage, 0)

        # _vk_image
        self._vk_image = vkCreateImage(
            device=self._device.vk_device,
            pCreateInfo=VkImageCreateInfo(
                flags=0,
                imageType=VK_IMAGE_TYPE_2D,
                format=self._spec.select_vk_format(),
                extent=VkExtent3D(
                    width=self._spec.shape[1],
                    height=self._spec.shape[0],
                    depth=1,
                ),
                mipLevels=1,
                arrayLayers=1,
                samples=VK_SAMPLE_COUNT_1_BIT,
                tiling=VK_IMAGE_TILING_OPTIMAL,
                usage=vk_usage,
                sharingMode=VK_SHARING_MODE_EXCLUSIVE,
                queueFamilyIndexCount=1,
                pQueueFamilyIndices=[expect(self._device.qfis.transfer)],
                initialLayout=VK_IMAGE_LAYOUT_UNDEFINED,
            ),
            pAllocator=None,
        )

    @staticmethod
    @functools.cache
    def _usage_to_vk_format_bit_dict() -> dict[GpuTextureUsage, int]:
        return {
            "sampled": VK_IMAGE_USAGE_SAMPLED_BIT,
            "storage": VK_IMAGE_USAGE_STORAGE_BIT,
            "color-attachment": VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
            "depth-stencil-attachment": (VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT),
        }

    def _on_dispose(self) -> None:
        if hasattr(self, "_vk_image"):
            vkDestroyImage(self._device.vk_device, self._vk_image, pAllocator=None)
