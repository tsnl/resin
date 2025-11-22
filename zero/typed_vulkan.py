__all__ = [
    # Basic
    "vkGetInstanceProcAddr",
    "ffi",
    # Common
    "VkDeviceSize",
    "VkFlags",
    "VkSampleCountFlags",
    "VkExtent3D",
    "VK_API_VERSION_1_2",
    "VK_API_VERSION_1_3",
    "VK_API_VERSION_1_4",
    "VK_MAKE_API_VERSION",
    "vk_decompose_api_version",
    "vk_api_version_str",
    # VkInstance
    "VkInstance",
    "vkCreateInstance",
    "vkDestroyInstance",
    "VkApplicationInfo",
    "VkInstanceCreateInfo",
    "VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR",
    # VkSurfaceKHR
    "VkSurfaceKHR",
    # VkPhysicalDevice
    "VkPhysicalDevice",
    "vkEnumeratePhysicalDevices",
    "vkGetPhysicalDeviceProperties",
    "VkPhysicalDeviceProperties",
    "VkPhysicalDeviceLimits",
    "VK_PHYSICAL_DEVICE_TYPE_OTHER",
    "VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU",
    "VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU",
    "VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU",
    "VK_PHYSICAL_DEVICE_TYPE_CPU",
    # VkPhysicalDevice Memory
    "VkPhysicalDeviceMemoryProperties",
    "vkGetPhysicalDeviceMemoryProperties",
    # VkPhysicalDevice queues
    "vkGetPhysicalDeviceQueueFamilyProperties",
    "VkDeviceQueueCreateInfo",
    "VK_QUEUE_GRAPHICS_BIT",
    "VK_QUEUE_COMPUTE_BIT",
    "VK_QUEUE_TRANSFER_BIT",
    # VkDevice
    "VkDevice",
    "vkCreateDevice",
    "vkDestroyDevice",
    "VkDeviceCreateInfo",
    # VkDeviceMemory
    "VkMemoryRequirements",
    "VkMemoryAllocateInfo",
    "vkAllocateMemory",
    "vkFreeMemory",
    "vkMapMemory",
    "vkUnmapMemory",
    "VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT",
    "VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT",
    "VK_MEMORY_PROPERTY_HOST_COHERENT_BIT",
    "VK_MEMORY_PROPERTY_HOST_CACHED_BIT",
    "VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT",
    # VkImage
    "VkImage",
    "VkImageCreateFlags",
    "VkImageCreateFlagBits",
    "VK_IMAGE_CREATE_SPARSE_BINDING_BIT",
    "VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT",
    "VK_IMAGE_CREATE_SPARSE_ALIASED_BIT",
    "VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT",
    "VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT",
    ## "VkImageType"
    "VK_IMAGE_TYPE_1D",
    "VK_IMAGE_TYPE_2D",
    "VK_IMAGE_TYPE_3D",
    ## "VkFormat"
    "VK_FORMAT_R32_SFLOAT",
    "VK_FORMAT_R8G8B8A8_UNORM",
    "VK_FORMAT_R32G32B32A32_SFLOAT",
    "VK_FORMAT_D32_SFLOAT",
    ## "VkSampleCountFlags"
    "VK_SAMPLE_COUNT_1_BIT",
    "VK_SAMPLE_COUNT_2_BIT",
    "VK_SAMPLE_COUNT_4_BIT",
    "VK_SAMPLE_COUNT_8_BIT",
    "VK_SAMPLE_COUNT_16_BIT",
    "VK_SAMPLE_COUNT_32_BIT",
    "VK_SAMPLE_COUNT_64_BIT",
    ## "VkImageTiling"
    "VK_IMAGE_TILING_OPTIMAL",
    "VK_IMAGE_TILING_LINEAR",
    ## "VkImageUsageFlags"
    ## "VkImageUsageFlagBits"
    "VK_IMAGE_USAGE_TRANSFER_SRC_BIT",
    "VK_IMAGE_USAGE_TRANSFER_DST_BIT",
    "VK_IMAGE_USAGE_SAMPLED_BIT",
    "VK_IMAGE_USAGE_STORAGE_BIT",
    "VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT",
    "VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT",
    ## "VkSharingMode"
    "VK_SHARING_MODE_EXCLUSIVE",
    "VK_SHARING_MODE_CONCURRENT",
    ## "VkImageLayout"
    "VK_IMAGE_LAYOUT_UNDEFINED",
    "VK_IMAGE_LAYOUT_GENERAL",
    "VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL",
    "VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL",
    "VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL",
    "VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL",
    "VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL",
    "VK_IMAGE_LAYOUT_PREINITIALIZED",
    "VkImageCreateInfo",
    "vkCreateImage",
    "vkDestroyImage",
    "vkGetImageMemoryRequirements",
    "vkBindImageMemory",
    # VkImageView
    "VK_COMPONENT_SWIZZLE_IDENTITY",
    "VK_IMAGE_ASPECT_DEPTH_BIT",
    "VK_IMAGE_ASPECT_COLOR_BIT",
    "VkImageView",
    "vkCreateImageView",
    "VkImageViewCreateInfo",
    "vkDestroyImageView",
    "VkComponentMapping",
    "VkImageSubresourceRange",
    # VkSampler
    "VkSampler",
    "vkCreateSampler",
    "VkSamplerCreateInfo",
    "vkDestroySampler",
    # VkRenderingAttachmentInfo
    "VkRenderingAttachmentInfo",
    "VkClearValue",
    "VkClearColorValue",
    "VkClearDepthStencilValue",
    # VkBuffer
    "VK_BUFFER_USAGE_TRANSFER_SRC_BIT",
    "VK_BUFFER_USAGE_TRANSFER_DST_BIT",
    "VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT",
    "VK_BUFFER_USAGE_STORAGE_BUFFER_BIT",
    "VkBuffer",
    "VkBufferCreateInfo",
    "vkCreateBuffer",
    "vkDestroyBuffer",
    "vkGetBufferMemoryRequirements",
    "VkBufferUsageFlags",
    "VkBufferUsageFlagBits",
]

from typing import TypeAlias

from vulkan import (
    # Basic
    vkGetInstanceProcAddr,
    ffi,
    # Common
    VkExtent3D,
    # vkInstance
    vkCreateInstance,
    vkDestroyInstance,
    VkInstanceCreateInfo,
    VkApplicationInfo,
    VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR,
    # VkPhysicalDevice
    vkEnumeratePhysicalDevices,
    vkGetPhysicalDeviceProperties,
    VkPhysicalDeviceProperties,
    VkPhysicalDeviceLimits,
    VK_PHYSICAL_DEVICE_TYPE_OTHER,
    VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU,
    VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU,
    VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU,
    VK_PHYSICAL_DEVICE_TYPE_CPU,
    # VkPhysicalDevice Memory
    VkPhysicalDeviceMemoryProperties,
    vkGetPhysicalDeviceMemoryProperties,
    # VkPhysicalDevice Queues
    vkGetPhysicalDeviceQueueFamilyProperties,
    VkDeviceQueueCreateInfo,
    VK_QUEUE_GRAPHICS_BIT,
    VK_QUEUE_COMPUTE_BIT,
    VK_QUEUE_TRANSFER_BIT,
    # VkDevice
    vkCreateDevice,
    vkDestroyDevice,
    VkDeviceCreateInfo,
    # VkDeviceMemory
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
    VK_FORMAT_D32_SFLOAT,
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
    VkImageCreateInfo,
    vkCreateImage,
    vkDestroyImage,
    vkGetImageMemoryRequirements,
    vkBindImageMemory,
    # VkImageView
    VK_COMPONENT_SWIZZLE_IDENTITY,
    VK_IMAGE_ASPECT_DEPTH_BIT,
    VK_IMAGE_ASPECT_COLOR_BIT,
    vkCreateImageView,
    VkImageViewCreateInfo,
    vkDestroyImageView,
    VkComponentMapping,
    VkImageSubresourceRange,
    # VkSampler
    vkCreateSampler,
    VkSamplerCreateInfo,
    vkDestroySampler,
    # VkRenderingAttachmentInfo
    VkRenderingAttachmentInfo,
    VkClearValue,
    VkClearColorValue,
    VkClearDepthStencilValue,
    # VkBuffer
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VkBufferCreateInfo,
    vkCreateBuffer,
    vkDestroyBuffer,
    vkGetBufferMemoryRequirements,
    vkBindBufferMemory,
)


class OpaqueResourceHandle:
    pass


VkDeviceSize: TypeAlias = int
VkFlags: TypeAlias = int
VkSampleCountFlags: TypeAlias = VkFlags
VkSampleCountFlagBits: TypeAlias = int
VkImageCreateFlags: TypeAlias = VkFlags
VkImageCreateFlagBits: TypeAlias = int
VkResolveModeFlags: TypeAlias = VkFlags
VkResolveModeFlagBits: TypeAlias = int
VkAttachmentLoadOp: TypeAlias = int
VkAttachmentStoreOp: TypeAlias = int
VkBufferUsageFlags: TypeAlias = VkFlags
VkBufferUsageFlagBits: TypeAlias = int
VkMemoryMapFlags: TypeAlias = VkFlags
VkMemoryMapFlagBits: TypeAlias = int


def VK_MAKE_API_VERSION(variant: int, major: int, minor: int, patch: int) -> int:
    return (variant << 29) | (major << 22) | (minor << 12) | (patch << 0)


def vk_decompose_api_version(api_version: int) -> tuple[int, int, int, int]:
    variant = (api_version >> 29) & 0x7
    major = (api_version >> 22) & 0x7F
    minor = (api_version >> 12) & 0x3FF
    patch = (api_version >> 0) & 0xFFF
    return variant, major, minor, patch


def vk_api_version_str(api_version: int) -> str:
    variant, major, minor, patch = vk_decompose_api_version(api_version)
    return f"{major}.{minor}.{patch} (variant {variant})"


VK_API_VERSION_1_2 = VK_MAKE_API_VERSION(0, 1, 2, 0)
VK_API_VERSION_1_3 = VK_MAKE_API_VERSION(0, 1, 3, 0)
VK_API_VERSION_1_4 = VK_MAKE_API_VERSION(0, 1, 4, 0)


class VkInstance(OpaqueResourceHandle): ...


class VkSurfaceKHR(OpaqueResourceHandle): ...


class VkPhysicalDevice(OpaqueResourceHandle): ...


class VkDevice(OpaqueResourceHandle): ...


class VkImage(OpaqueResourceHandle): ...


class VkImageView(OpaqueResourceHandle): ...


class VkSampler(OpaqueResourceHandle): ...


class VkBuffer(OpaqueResourceHandle): ...


class VkDeviceMemory(OpaqueResourceHandle): ...
