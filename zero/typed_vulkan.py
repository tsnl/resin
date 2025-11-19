__all__ = [
    # Basic
    "vkGetInstanceProcAddr",
    "ffi",
    # Common
    "VkDeviceSize",
    "VkFlags",
    "VkSampleCountFlags",
    "VkExtent3D",
    # Instance
    "VkInstance",
    "vkCreateInstance",
    "vkDestroyInstance",
    "VkApplicationInfo",
    "VkInstanceCreateInfo",
    "VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR",
    # Surfaces
    "VkSurfaceKHR",
    # Physical devices
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
    # Physical device queues
    "vkGetPhysicalDeviceQueueFamilyProperties",
    "VkDeviceQueueCreateInfo",
    "VK_QUEUE_GRAPHICS_BIT",
    "VK_QUEUE_COMPUTE_BIT",
    "VK_QUEUE_TRANSFER_BIT",
    # Devices
    "VkDevice",
    "vkCreateDevice",
    "vkDestroyDevice",
    "VkDeviceCreateInfo",
    # Images
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
    "VkSampleCountFlags",
    ## "VkSampleCountFlagBits"
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
    # Image views
    "vkCreateImageView",
    "VkImageViewCreateInfo",
    "vkDestroyImageView",
    # Samplers
    "vkCreateSampler",
    "VkSamplerCreateInfo",
    "vkDestroySampler",
]

from typing import TypeAlias

from vulkan import (
    # Basic
    vkGetInstanceProcAddr,
    ffi,
    # Common
    VkExtent3D,
    # Instance:
    vkCreateInstance,
    vkDestroyInstance,
    VkInstanceCreateInfo,
    VkApplicationInfo,
    VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR,
    # Physical devices:
    vkEnumeratePhysicalDevices,
    vkGetPhysicalDeviceProperties,
    VkPhysicalDeviceProperties,
    VkPhysicalDeviceLimits,
    VK_PHYSICAL_DEVICE_TYPE_OTHER,
    VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU,
    VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU,
    VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU,
    VK_PHYSICAL_DEVICE_TYPE_CPU,
    # Surface: (N/A)
    # Queues:
    vkGetPhysicalDeviceQueueFamilyProperties,
    VkDeviceQueueCreateInfo,
    VK_QUEUE_GRAPHICS_BIT,
    VK_QUEUE_COMPUTE_BIT,
    VK_QUEUE_TRANSFER_BIT,
    # Devices:
    vkCreateDevice,
    vkDestroyDevice,
    VkDeviceCreateInfo,
    # Images:
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
    VkImageCreateInfo,
    vkCreateImage,
    vkDestroyImage,
    # Image views:
    vkCreateImageView,
    VkImageViewCreateInfo,
    vkDestroyImageView,
    # Samplers:
    vkCreateSampler,
    VkSamplerCreateInfo,
    vkDestroySampler,
)


class OpaqueResourceHandle:
    pass


VkDeviceSize: TypeAlias = int
VkFlags: TypeAlias = int
VkSampleCountFlags: TypeAlias = VkFlags
VkImageCreateFlags: TypeAlias = VkFlags
VkImageCreateFlagBits: TypeAlias = int


class VkInstance(OpaqueResourceHandle): ...


class VkSurfaceKHR(OpaqueResourceHandle): ...


class VkPhysicalDevice(OpaqueResourceHandle): ...


class VkDevice(OpaqueResourceHandle): ...


class VkImage(OpaqueResourceHandle): ...
