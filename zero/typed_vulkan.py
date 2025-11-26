__all__ = [
    "vkGetInstanceProcAddr",
    "ffi",
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
    "VkInstance",
    "vkCreateInstance",
    "vkDestroyInstance",
    "VkApplicationInfo",
    "VkInstanceCreateInfo",
    "VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR",
    "VkSurfaceKHR",
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
    "VkPhysicalDeviceMemoryProperties",
    "vkGetPhysicalDeviceMemoryProperties",
    "vkGetPhysicalDeviceQueueFamilyProperties",
    "VkDeviceQueueCreateInfo",
    "VK_QUEUE_GRAPHICS_BIT",
    "VK_QUEUE_COMPUTE_BIT",
    "VK_QUEUE_TRANSFER_BIT",
    "VkDevice",
    "vkCreateDevice",
    "vkDestroyDevice",
    "VkDeviceCreateInfo",
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
    "VkImage",
    "VkImageCreateFlags",
    "VkImageCreateFlagBits",
    "VK_IMAGE_CREATE_SPARSE_BINDING_BIT",
    "VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT",
    "VK_IMAGE_CREATE_SPARSE_ALIASED_BIT",
    "VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT",
    "VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT",
    "VK_IMAGE_TYPE_1D",
    "VK_IMAGE_TYPE_2D",
    "VK_IMAGE_TYPE_3D",
    "VK_FORMAT_R32_SFLOAT",
    "VK_FORMAT_R8G8B8A8_UNORM",
    "VK_FORMAT_R32G32B32A32_SFLOAT",
    "VK_FORMAT_D32_SFLOAT",
    "VK_SAMPLE_COUNT_1_BIT",
    "VK_SAMPLE_COUNT_2_BIT",
    "VK_SAMPLE_COUNT_4_BIT",
    "VK_SAMPLE_COUNT_8_BIT",
    "VK_SAMPLE_COUNT_16_BIT",
    "VK_SAMPLE_COUNT_32_BIT",
    "VK_SAMPLE_COUNT_64_BIT",
    "VK_IMAGE_TILING_OPTIMAL",
    "VK_IMAGE_TILING_LINEAR",
    "VK_IMAGE_USAGE_TRANSFER_SRC_BIT",
    "VK_IMAGE_USAGE_TRANSFER_DST_BIT",
    "VK_IMAGE_USAGE_SAMPLED_BIT",
    "VK_IMAGE_USAGE_STORAGE_BIT",
    "VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT",
    "VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT",
    "VK_SHARING_MODE_EXCLUSIVE",
    "VK_SHARING_MODE_CONCURRENT",
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
    "VK_COMPONENT_SWIZZLE_IDENTITY",
    "VK_IMAGE_ASPECT_DEPTH_BIT",
    "VK_IMAGE_ASPECT_COLOR_BIT",
    "VkImageView",
    "vkCreateImageView",
    "VkImageViewCreateInfo",
    "vkDestroyImageView",
    "VkComponentMapping",
    "VkImageSubresourceRange",
    "VkSampler",
    "vkCreateSampler",
    "VkSamplerCreateInfo",
    "vkDestroySampler",
    "VkRenderingAttachmentInfo",
    "VkClearValue",
    "VkClearColorValue",
    "VkClearDepthStencilValue",
    "VK_BUFFER_USAGE_TRANSFER_SRC_BIT",
    "VK_BUFFER_USAGE_TRANSFER_DST_BIT",
    "VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT",
    "VK_BUFFER_USAGE_STORAGE_BUFFER_BIT",
    "VkBuffer",
    "VkBufferCreateInfo",
    "vkCreateBuffer",
    "vkDestroyBuffer",
    "vkGetBufferMemoryRequirements",
    "vkBindBufferMemory",
    "VkBufferUsageFlags",
    "VkBufferUsageFlagBits",
    "vkCreateBufferView",
    "VkBufferViewCreateInfo",
    "vkDestroyBufferView",
    "VkCommandPool",
    "vkCreateCommandPool",
    "VkCommandPoolCreateInfo",
    "vkDestroyCommandPool",
    "VK_COMMAND_POOL_CREATE_TRANSIENT_BIT",
    "VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT",
    "VkCommandBuffer",
    "vkAllocateCommandBuffers",
    "VkCommandBufferAllocateInfo",
    "vkFreeCommandBuffers",
    "vkResetCommandBuffer",
    "vkBeginCommandBuffer",
    "VkCommandBufferBeginInfo",
    "vkEndCommandBuffer",
    "vkCmdCopyBuffer",
    "VkBufferCopy",
    "vkCmdCopyBufferToImage",
    "VkBufferImageCopy",
    "vkCmdCopyImageToBuffer",
    "VkImageCopy",
    "VkImageSubresourceLayers",
    "VkOffset3D",
    "VK_COMMAND_BUFFER_LEVEL_PRIMARY",
    "VK_COMMAND_BUFFER_LEVEL_SECONDARY",
    "VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT",
    "VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT",
    "VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT",
    "VkSemaphore",
    "VkSemaphoreCreateInfo",
    "vkCreateSemaphore",
    "vkDestroySemaphore",
    "VkFence",
    "VkFenceCreateInfo",
    "vkCreateFence",
    "vkDestroyFence",
    "VkQueue",
    "vkGetDeviceQueue",
    "VkSubmitInfo",
    "vkQueueSubmit",
    "vkCmdCopyImage",
    "VkImageCopy",
    "VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT",
]

from typing import TypeAlias

from vulkan import (
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
    VK_COMMAND_BUFFER_LEVEL_PRIMARY,
    VK_COMMAND_BUFFER_LEVEL_SECONDARY,
    VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT,
    VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT,
    VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
    VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
    VK_COMPONENT_SWIZZLE_IDENTITY,
    VK_FORMAT_D32_SFLOAT,
    VK_FORMAT_R8G8B8A8_UNORM,
    VK_FORMAT_R32_SFLOAT,
    VK_FORMAT_R32G32B32A32_SFLOAT,
    VK_IMAGE_ASPECT_COLOR_BIT,
    VK_IMAGE_ASPECT_DEPTH_BIT,
    VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT,
    VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT,
    VK_IMAGE_CREATE_SPARSE_ALIASED_BIT,
    VK_IMAGE_CREATE_SPARSE_BINDING_BIT,
    VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT,
    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    VK_IMAGE_LAYOUT_GENERAL,
    VK_IMAGE_LAYOUT_PREINITIALIZED,
    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
    VK_IMAGE_LAYOUT_UNDEFINED,
    VK_IMAGE_TILING_LINEAR,
    VK_IMAGE_TILING_OPTIMAL,
    VK_IMAGE_TYPE_1D,
    VK_IMAGE_TYPE_2D,
    VK_IMAGE_TYPE_3D,
    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
    VK_IMAGE_USAGE_SAMPLED_BIT,
    VK_IMAGE_USAGE_STORAGE_BIT,
    VK_IMAGE_USAGE_TRANSFER_DST_BIT,
    VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
    VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
    VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT,
    VK_PHYSICAL_DEVICE_TYPE_CPU,
    VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU,
    VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU,
    VK_PHYSICAL_DEVICE_TYPE_OTHER,
    VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU,
    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
    VK_QUEUE_COMPUTE_BIT,
    VK_QUEUE_GRAPHICS_BIT,
    VK_QUEUE_TRANSFER_BIT,
    VK_SAMPLE_COUNT_1_BIT,
    VK_SAMPLE_COUNT_2_BIT,
    VK_SAMPLE_COUNT_4_BIT,
    VK_SAMPLE_COUNT_8_BIT,
    VK_SAMPLE_COUNT_16_BIT,
    VK_SAMPLE_COUNT_32_BIT,
    VK_SAMPLE_COUNT_64_BIT,
    VK_SHARING_MODE_CONCURRENT,
    VK_SHARING_MODE_EXCLUSIVE,
    VkApplicationInfo,
    VkBufferCopy,
    VkBufferCreateInfo,
    VkBufferImageCopy,
    VkBufferViewCreateInfo,
    VkClearColorValue,
    VkClearDepthStencilValue,
    VkClearValue,
    VkCommandBufferAllocateInfo,
    VkCommandBufferBeginInfo,
    VkCommandPoolCreateInfo,
    VkComponentMapping,
    VkDeviceCreateInfo,
    VkDeviceQueueCreateInfo,
    VkExtent3D,
    VkFenceCreateInfo,
    VkImageCopy,
    VkImageCreateInfo,
    VkImageSubresourceLayers,
    VkImageSubresourceRange,
    VkImageViewCreateInfo,
    VkInstanceCreateInfo,
    VkMemoryAllocateInfo,
    VkMemoryRequirements,
    VkOffset3D,
    VkPhysicalDeviceLimits,
    VkPhysicalDeviceMemoryProperties,
    VkPhysicalDeviceProperties,
    VkRenderingAttachmentInfo,
    VkSamplerCreateInfo,
    VkSemaphoreCreateInfo,
    VkSubmitInfo,
    ffi,
    vkAllocateCommandBuffers,
    vkAllocateMemory,
    vkBeginCommandBuffer,
    vkBindBufferMemory,
    vkBindImageMemory,
    vkCmdCopyBuffer,
    vkCmdCopyBufferToImage,
    vkCmdCopyImage,
    vkCmdCopyImageToBuffer,
    vkCreateBuffer,
    vkCreateBufferView,
    vkCreateCommandPool,
    vkCreateDevice,
    vkCreateFence,
    vkCreateImage,
    vkCreateImageView,
    vkCreateInstance,
    vkCreateSampler,
    vkCreateSemaphore,
    vkDestroyBuffer,
    vkDestroyBufferView,
    vkDestroyCommandPool,
    vkDestroyDevice,
    vkDestroyFence,
    vkDestroyImage,
    vkDestroyImageView,
    vkDestroyInstance,
    vkDestroySampler,
    vkDestroySemaphore,
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
    vkResetCommandBuffer,
    vkUnmapMemory,
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


class VkCommandPool(OpaqueResourceHandle): ...


class VkCommandBuffer(OpaqueResourceHandle): ...


class VkSemaphore(OpaqueResourceHandle): ...


class VkFence(OpaqueResourceHandle): ...


class VkQueue(OpaqueResourceHandle): ...
