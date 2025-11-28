__all__ = [
    "VK_ACCESS_INDIRECT_COMMAND_READ_BIT",
    "VK_ACCESS_INDEX_READ_BIT",
    "VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT",
    "VK_ACCESS_UNIFORM_READ_BIT",
    "VK_ACCESS_INPUT_ATTACHMENT_READ_BIT",
    "VK_ACCESS_SHADER_READ_BIT",
    "VK_ACCESS_SHADER_WRITE_BIT",
    "VK_ACCESS_COLOR_ATTACHMENT_READ_BIT",
    "VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT",
    "VK_ACCESS_NONE",
    "VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT",
    "VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT",
    "VK_ACCESS_TRANSFER_READ_BIT",
    "VK_ACCESS_TRANSFER_WRITE_BIT",
    "VK_ACCESS_HOST_READ_BIT",
    "VK_ACCESS_HOST_WRITE_BIT",
    "VK_ACCESS_MEMORY_READ_BIT",
    "VK_ACCESS_MEMORY_WRITE_BIT",
    "VK_API_VERSION_1_2",
    "VK_API_VERSION_1_3",
    "VK_API_VERSION_1_4",
    "VK_ATTACHMENT_LOAD_OP_CLEAR",
    "VK_ATTACHMENT_LOAD_OP_DONT_CARE",
    "VK_ATTACHMENT_LOAD_OP_LOAD",
    "VK_ATTACHMENT_STORE_OP_DONT_CARE",
    "VK_ATTACHMENT_STORE_OP_STORE",
    "VK_BLEND_FACTOR_ONE",
    "VK_BLEND_FACTOR_ZERO",
    "VK_BLEND_OP_ADD",
    "VK_BUFFER_USAGE_STORAGE_BUFFER_BIT",
    "VK_BUFFER_USAGE_TRANSFER_DST_BIT",
    "VK_BUFFER_USAGE_TRANSFER_SRC_BIT",
    "VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT",
    "VK_COLOR_COMPONENT_A_BIT",
    "VK_COLOR_COMPONENT_B_BIT",
    "VK_COLOR_COMPONENT_G_BIT",
    "VK_COLOR_COMPONENT_R_BIT",
    "VK_COLOR_SPACE_SRGB_NONLINEAR_KHR",
    "VK_COMMAND_BUFFER_LEVEL_PRIMARY",
    "VK_COMMAND_BUFFER_LEVEL_SECONDARY",
    "VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT",
    "VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT",
    "VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT",
    "VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT",
    "VK_COMMAND_POOL_CREATE_TRANSIENT_BIT",
    "VK_COMPONENT_SWIZZLE_IDENTITY",
    "VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR",
    "VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR",
    "VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR",
    "VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR",
    "VK_DEPENDENCY_BY_REGION_BIT",
    "VK_CULL_MODE_BACK_BIT",
    "VK_CULL_MODE_FRONT_BIT",
    "VK_CULL_MODE_NONE",
    "VK_FENCE_CREATE_SIGNALED_BIT",
    "VK_FORMAT_D32_SFLOAT",
    "VK_FORMAT_R32G32B32A32_SFLOAT",
    "VK_FORMAT_R32_SFLOAT",
    "VK_FORMAT_R8G8B8A8_UNORM",
    "VK_FORMAT_UNDEFINED",
    "VK_FRONT_FACE_CLOCKWISE",
    "VK_FRONT_FACE_COUNTER_CLOCKWISE",
    "VK_IMAGE_ASPECT_COLOR_BIT",
    "VK_IMAGE_ASPECT_DEPTH_BIT",
    "VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT",
    "VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT",
    "VK_IMAGE_CREATE_SPARSE_ALIASED_BIT",
    "VK_IMAGE_CREATE_SPARSE_BINDING_BIT",
    "VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT",
    "VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL",
    "VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL",
    "VK_IMAGE_LAYOUT_GENERAL",
    "VK_IMAGE_LAYOUT_PREINITIALIZED",
    "VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL",
    "VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL",
    "VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL",
    "VK_IMAGE_LAYOUT_UNDEFINED",
    "VK_IMAGE_LAYOUT_PRESENT_SRC_KHR",
    "VK_IMAGE_TILING_LINEAR",
    "VK_IMAGE_TILING_OPTIMAL",
    "VK_IMAGE_TYPE_1D",
    "VK_IMAGE_TYPE_2D",
    "VK_IMAGE_TYPE_3D",
    "VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT",
    "VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT",
    "VK_IMAGE_USAGE_SAMPLED_BIT",
    "VK_IMAGE_USAGE_STORAGE_BIT",
    "VK_IMAGE_USAGE_TRANSFER_DST_BIT",
    "VK_IMAGE_USAGE_TRANSFER_SRC_BIT",
    "VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR",
    "VK_LOGIC_OP_COPY",
    "VK_MAKE_API_VERSION",
    "VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT",
    "VK_MEMORY_PROPERTY_HOST_CACHED_BIT",
    "VK_MEMORY_PROPERTY_HOST_COHERENT_BIT",
    "VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT",
    "VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT",
    "VK_PHYSICAL_DEVICE_TYPE_CPU",
    "VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU",
    "VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU",
    "VK_PHYSICAL_DEVICE_TYPE_OTHER",
    "VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU",
    "VK_PIPELINE_BIND_POINT_GRAPHICS",
    "VK_PIPELINE_BIND_POINT_COMPUTE",
    "VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT",
    "VK_PIPELINE_STAGE_ALL_COMMANDS_BIT",
    "VK_POLYGON_MODE_FILL",
    "VK_POLYGON_MODE_LINE",
    "VK_POLYGON_MODE_POINT",
    "VK_PRIMITIVE_TOPOLOGY_LINE_LIST",
    "VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST",
    "VK_PRESENT_MODE_IMMEDIATE_KHR",
    "VK_PRESENT_MODE_MAILBOX_KHR",
    "VK_PRESENT_MODE_FIFO_KHR",
    "VK_PRESENT_MODE_FIFO_RELAXED_KHR",
    "VK_QUEUE_COMPUTE_BIT",
    "VK_QUEUE_GRAPHICS_BIT",
    "VK_QUEUE_TRANSFER_BIT",
    "VK_SAMPLE_COUNT_16_BIT",
    "VK_SAMPLE_COUNT_1_BIT",
    "VK_SAMPLE_COUNT_2_BIT",
    "VK_SAMPLE_COUNT_32_BIT",
    "VK_SAMPLE_COUNT_4_BIT",
    "VK_SAMPLE_COUNT_64_BIT",
    "VK_SAMPLE_COUNT_8_BIT",
    "VK_SHADER_STAGE_FRAGMENT_BIT",
    "VK_SHADER_STAGE_VERTEX_BIT",
    "VK_SHARING_MODE_CONCURRENT",
    "VK_SHARING_MODE_EXCLUSIVE",
    "VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR",
    "VK_SURFACE_TRANSFORM_ROTATE_90_BIT_KHR",
    "VK_SURFACE_TRANSFORM_ROTATE_180_BIT_KHR",
    "VK_SURFACE_TRANSFORM_ROTATE_270_BIT_KHR",
    "VK_SURFACE_TRANSFORM_HORIZONTAL_MIRROR_BIT_KHR",
    "VK_SURFACE_TRANSFORM_HORIZONTAL_MIRROR_ROTATE_90_BIT_KHR",
    "VK_SURFACE_TRANSFORM_HORIZONTAL_MIRROR_ROTATE_180_BIT_KHR",
    "VK_SURFACE_TRANSFORM_HORIZONTAL_MIRROR_ROTATE_270_BIT_KHR",
    "VK_SURFACE_TRANSFORM_INHERIT_BIT_KHR",
    "VkApplicationInfo",
    "VkBuffer",
    "VkBufferCopy",
    "VkBufferCreateInfo",
    "VkBufferImageCopy",
    "VkBufferUsageFlagBits",
    "VkBufferUsageFlags",
    "VkBufferViewCreateInfo",
    "VkClearColorValue",
    "VkClearDepthStencilValue",
    "VkClearValue",
    "VkColorComponentFlags",
    "VkCommandBuffer",
    "VkCommandBufferAllocateInfo",
    "VkCommandBufferBeginInfo",
    "VkCommandPool",
    "VkCommandPoolCreateInfo",
    "VkComponentMapping",
    "VkDescriptorSet",
    "VkDescriptorSetLayout",
    "VkDevice",
    "VkDeviceCreateInfo",
    "VkDeviceQueueCreateInfo",
    "VkDeviceSize",
    "VkExtent2D",
    "VkExtent3D",
    "VkFence",
    "VkFenceCreateInfo",
    "VkFlags",
    "VkGraphicsPipelineCreateInfo",
    "VkImage",
    "VkImageCopy",
    "VkImageCopy",
    "VkImageCreateFlagBits",
    "VkImageCreateFlags",
    "VkImageCreateInfo",
    "VkImageSubresourceLayers",
    "VkImageSubresourceRange",
    "VkImageView",
    "VkImageViewCreateInfo",
    "VkInstance",
    "VkInstanceCreateInfo",
    "VkMemoryAllocateInfo",
    "VkMemoryRequirements",
    "VkOffset2D",
    "VkOffset3D",
    "VkPhysicalDevice",
    "VkPhysicalDeviceDynamicRenderingFeatures",
    "VkPhysicalDeviceLimits",
    "VkPhysicalDeviceMemoryProperties",
    "VkPhysicalDeviceProperties",
    "VkPipeline",
    "VkPipelineBindPoint",
    "VkPipelineColorBlendAttachmentState",
    "VkPipelineColorBlendStateCreateInfo",
    "VkPipelineInputAssemblyStateCreateInfo",
    "VkPipelineLayout",
    "VkPipelineLayoutCreateInfo",
    "VkPipelineMultisampleStateCreateInfo",
    "VkPipelineRasterizationStateCreateInfo",
    "VkPipelineRenderingCreateInfo",
    "VkPipelineShaderStageCreateInfo",
    "VkPipelineVertexInputStateCreateInfo",
    "VkPipelineViewportStateCreateInfo",
    "VkQueue",
    "VkRect2D",
    "VkRenderingAttachmentInfo",
    "VkRenderingInfo",
    "VkSampleCountFlags",
    "VkSampler",
    "VkSamplerCreateInfo",
    "VkSemaphore",
    "VkSemaphoreCreateInfo",
    "VkShaderModule",
    "VkShaderModuleCreateInfo",
    "VkSubmitInfo",
    "VkSurfaceKHR",
    "VkViewport",
    "raw_ffi",
    "vkAllocateCommandBuffers",
    "vkAllocateMemory",
    "vkBeginCommandBuffer",
    "vkBindBufferMemory",
    "vkBindImageMemory",
    "vkCmdBeginRendering",
    "vkCmdBindDescriptorSets",
    "vkCmdBindPipeline",
    "vkCmdCopyBuffer",
    "vkCmdCopyBufferToImage",
    "vkCmdCopyImage",
    "vkCmdCopyImageToBuffer",
    "VkDependencyFlags",
    "vkCmdDraw",
    "vkCmdEndRendering",
    "vkCreateBuffer",
    "vkCreateBufferView",
    "vkCreateCommandPool",
    "vkCreateDevice",
    "vkCreateFence",
    "vkCreateGraphicsPipelines",
    "vkCreateImage",
    "vkCreateImageView",
    "vkCreateInstance",
    "vkCreatePipelineLayout",
    "vkCreateSampler",
    "vkCreateSemaphore",
    "vkCreateShaderModule",
    "vkDestroyBuffer",
    "vkDestroyBufferView",
    "vkDestroyCommandPool",
    "vkDestroyDevice",
    "vkDestroyFence",
    "vkDestroyImage",
    "vkDestroyImageView",
    "vkDestroyInstance",
    "vkDestroyPipeline",
    "vkDestroyPipelineLayout",
    "vkDestroySampler",
    "vkDestroySemaphore",
    "vkDestroyShaderModule",
    "vkEndCommandBuffer",
    "vkEnumeratePhysicalDevices",
    "vkFreeCommandBuffers",
    "vkFreeMemory",
    "vkGetBufferMemoryRequirements",
    "vkGetDeviceQueue",
    "vkGetImageMemoryRequirements",
    "vkGetInstanceProcAddr",
    "vkGetPhysicalDeviceMemoryProperties",
    "vkGetPhysicalDeviceProperties",
    "vkGetPhysicalDeviceQueueFamilyProperties",
    "vkMapMemory",
    "vkQueueSubmit",
    "vkResetCommandBuffer",
    "VkShaderModule",
    "VkSwapchainKHR",
    "VkSwapchainCreateInfoKHR",
    "VkSurfaceFormatKHR",
    "vkUnmapMemory",
    "vkWaitForFences",
    "VkPresentInfoKHR",
    "vkResetFences",
    "vk_api_version_str",
    "vk_decompose_api_version",
    "vkDeviceWaitIdle",
    "vkCmdPipelineBarrier",
    "VkImageMemoryBarrier",
]


from typing import TypeAlias
from vulkan import (
    VK_ACCESS_INDIRECT_COMMAND_READ_BIT,
    VK_ACCESS_INDEX_READ_BIT,
    VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
    VK_ACCESS_UNIFORM_READ_BIT,
    VK_ACCESS_INPUT_ATTACHMENT_READ_BIT,
    VK_ACCESS_SHADER_READ_BIT,
    VK_ACCESS_SHADER_WRITE_BIT,
    VK_ACCESS_COLOR_ATTACHMENT_READ_BIT,
    VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
    VK_ACCESS_NONE,
    VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
    VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
    VK_ACCESS_TRANSFER_READ_BIT,
    VK_ACCESS_TRANSFER_WRITE_BIT,
    VK_ACCESS_HOST_READ_BIT,
    VK_ACCESS_HOST_WRITE_BIT,
    VK_ACCESS_MEMORY_READ_BIT,
    VK_ACCESS_MEMORY_WRITE_BIT,
    VK_BLEND_FACTOR_ONE,
    VK_BLEND_FACTOR_ZERO,
    VK_BLEND_OP_ADD,
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
    VK_COMMAND_BUFFER_LEVEL_SECONDARY,
    VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT,
    VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT,
    VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
    VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
    VK_COMPONENT_SWIZZLE_IDENTITY,
    VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
    VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
    VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR,
    VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR,
    VK_CULL_MODE_BACK_BIT,
    VK_CULL_MODE_FRONT_BIT,
    VK_CULL_MODE_NONE,
    VK_FENCE_CREATE_SIGNALED_BIT,
    VK_FRONT_FACE_CLOCKWISE,
    VK_FRONT_FACE_COUNTER_CLOCKWISE,
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
    VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
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
    VK_LOGIC_OP_COPY,
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
    VK_PIPELINE_BIND_POINT_GRAPHICS,
    VK_PIPELINE_BIND_POINT_COMPUTE,
    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
    VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
    VK_POLYGON_MODE_FILL,
    VK_POLYGON_MODE_LINE,
    VK_POLYGON_MODE_POINT,
    VK_PRIMITIVE_TOPOLOGY_LINE_LIST,
    VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
    VK_PRESENT_MODE_IMMEDIATE_KHR,
    VK_PRESENT_MODE_MAILBOX_KHR,
    VK_PRESENT_MODE_FIFO_KHR,
    VK_PRESENT_MODE_FIFO_RELAXED_KHR,
    VK_QUEUE_COMPUTE_BIT,
    VK_QUEUE_GRAPHICS_BIT,
    VK_QUEUE_TRANSFER_BIT,
    VK_QUEUE_FAMILY_IGNORED,
    VK_SAMPLE_COUNT_16_BIT,
    VK_SAMPLE_COUNT_1_BIT,
    VK_SAMPLE_COUNT_2_BIT,
    VK_SAMPLE_COUNT_32_BIT,
    VK_SAMPLE_COUNT_4_BIT,
    VK_SAMPLE_COUNT_64_BIT,
    VK_SAMPLE_COUNT_8_BIT,
    VK_SHADER_STAGE_FRAGMENT_BIT,
    VK_SHADER_STAGE_VERTEX_BIT,
    VK_SHARING_MODE_CONCURRENT,
    VK_SHARING_MODE_EXCLUSIVE,
    VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
    VK_SURFACE_TRANSFORM_ROTATE_90_BIT_KHR,
    VK_SURFACE_TRANSFORM_ROTATE_180_BIT_KHR,
    VK_SURFACE_TRANSFORM_ROTATE_270_BIT_KHR,
    VK_SURFACE_TRANSFORM_HORIZONTAL_MIRROR_BIT_KHR,
    VK_SURFACE_TRANSFORM_HORIZONTAL_MIRROR_ROTATE_90_BIT_KHR,
    VK_SURFACE_TRANSFORM_HORIZONTAL_MIRROR_ROTATE_180_BIT_KHR,
    VK_SURFACE_TRANSFORM_HORIZONTAL_MIRROR_ROTATE_270_BIT_KHR,
    VK_SURFACE_TRANSFORM_INHERIT_BIT_KHR,
    VK_DEPENDENCY_BY_REGION_BIT,
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
    VkExtent2D,
    VkExtent3D,
    VkFenceCreateInfo,
    VkGraphicsPipelineCreateInfo,
    VkImageCopy,
    VkImageCreateInfo,
    VkImageSubresourceLayers,
    VkImageSubresourceRange,
    VkImageViewCreateInfo,
    VkInstanceCreateInfo,
    VkMemoryAllocateInfo,
    VkMemoryRequirements,
    VkOffset2D,
    VkOffset3D,
    VkPhysicalDeviceDynamicRenderingFeatures,
    VkPhysicalDeviceLimits,
    VkPhysicalDeviceMemoryProperties,
    VkPhysicalDeviceProperties,
    VkPipelineColorBlendAttachmentState,
    VkPipelineColorBlendStateCreateInfo,
    VkPipelineInputAssemblyStateCreateInfo,
    VkPipelineLayoutCreateInfo,
    VkPipelineMultisampleStateCreateInfo,
    VkPipelineRasterizationStateCreateInfo,
    VkPipelineRenderingCreateInfo,
    VkPipelineShaderStageCreateInfo,
    VkPipelineVertexInputStateCreateInfo,
    VkPipelineViewportStateCreateInfo,
    VkRect2D,
    VkRenderingAttachmentInfo,
    VkRenderingInfo,
    VkSamplerCreateInfo,
    VkSemaphoreCreateInfo,
    VkShaderModuleCreateInfo,
    VkSubmitInfo,
    VkSurfaceFormatKHR,
    VkViewport,
    ffi as raw_ffi,
    vkAllocateCommandBuffers,
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
    vkCreateBuffer,
    vkCreateBufferView,
    vkCreateCommandPool,
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
    vkDestroyBufferView,
    vkDestroyCommandPool,
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
    vkWaitForFences,
    VkPresentInfoKHR,
    vkResetFences,
    VkSwapchainCreateInfoKHR,
    vkDeviceWaitIdle,
    vkCmdPipelineBarrier,
    VkImageMemoryBarrier,
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
VkFormat: TypeAlias = int
VkDependencyFlags: TypeAlias = VkFlags
VkDependencyFlagBits: TypeAlias = int


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


class VkPipeline(OpaqueResourceHandle): ...


class VkPipelineLayout(OpaqueResourceHandle): ...


class VkDescriptorSetLayout(OpaqueResourceHandle): ...


class VkDescriptorSet(OpaqueResourceHandle): ...


class VkShaderModule(OpaqueResourceHandle): ...


class VkSwapchainKHR(OpaqueResourceHandle): ...


# Bind point enum
VkPipelineBindPoint: TypeAlias = int  # 0 = GRAPHICS


# Attachment load/store ops (match .pyi values)
VK_ATTACHMENT_LOAD_OP_LOAD = 0
VK_ATTACHMENT_LOAD_OP_CLEAR = 1
VK_ATTACHMENT_LOAD_OP_DONT_CARE = 2
VK_ATTACHMENT_STORE_OP_STORE = 0
VK_ATTACHMENT_STORE_OP_DONT_CARE = 1

# Format constants
VK_FORMAT_UNDEFINED: VkFormat = 0
VK_FORMAT_R4G4_UNORM_PACK8: VkFormat = 1
VK_FORMAT_R4G4B4A4_UNORM_PACK16: VkFormat = 2
VK_FORMAT_B4G4R4A4_UNORM_PACK16: VkFormat = 3
VK_FORMAT_R5G6B5_UNORM_PACK16: VkFormat = 4
VK_FORMAT_B5G6R5_UNORM_PACK16: VkFormat = 5
VK_FORMAT_R5G5B5A1_UNORM_PACK16: VkFormat = 6
VK_FORMAT_B5G5R5A1_UNORM_PACK16: VkFormat = 7
VK_FORMAT_A1R5G5B5_UNORM_PACK16: VkFormat = 8
VK_FORMAT_R8_UNORM: VkFormat = 9
VK_FORMAT_R8_SNORM: VkFormat = 10
VK_FORMAT_R8_USCALED: VkFormat = 11
VK_FORMAT_R8_SSCALED: VkFormat = 12
VK_FORMAT_R8_UINT: VkFormat = 13
VK_FORMAT_R8_SINT: VkFormat = 14
VK_FORMAT_R8_SRGB: VkFormat = 15
VK_FORMAT_R8G8_UNORM: VkFormat = 16
VK_FORMAT_R8G8_SNORM: VkFormat = 17
VK_FORMAT_R8G8_USCALED: VkFormat = 18
VK_FORMAT_R8G8_SSCALED: VkFormat = 19
VK_FORMAT_R8G8_UINT: VkFormat = 20
VK_FORMAT_R8G8_SINT: VkFormat = 21
VK_FORMAT_R8G8_SRGB: VkFormat = 22
VK_FORMAT_R8G8B8_UNORM: VkFormat = 23
VK_FORMAT_R8G8B8_SNORM: VkFormat = 24
VK_FORMAT_R8G8B8_USCALED: VkFormat = 25
VK_FORMAT_R8G8B8_SSCALED: VkFormat = 26
VK_FORMAT_R8G8B8_UINT: VkFormat = 27
VK_FORMAT_R8G8B8_SINT: VkFormat = 28
VK_FORMAT_R8G8B8_SRGB: VkFormat = 29
VK_FORMAT_B8G8R8_UNORM: VkFormat = 30
VK_FORMAT_B8G8R8_SNORM: VkFormat = 31
VK_FORMAT_B8G8R8_USCALED: VkFormat = 32
VK_FORMAT_B8G8R8_SSCALED: VkFormat = 33
VK_FORMAT_B8G8R8_UINT: VkFormat = 34
VK_FORMAT_B8G8R8_SINT: VkFormat = 35
VK_FORMAT_B8G8R8_SRGB: VkFormat = 36
VK_FORMAT_R8G8B8A8_UNORM: VkFormat = 37
VK_FORMAT_R8G8B8A8_SNORM: VkFormat = 38
VK_FORMAT_R8G8B8A8_USCALED: VkFormat = 39
VK_FORMAT_R8G8B8A8_SSCALED: VkFormat = 40
VK_FORMAT_R8G8B8A8_UINT: VkFormat = 41
VK_FORMAT_R8G8B8A8_SINT: VkFormat = 42
VK_FORMAT_R8G8B8A8_SRGB: VkFormat = 43
VK_FORMAT_B8G8R8A8_UNORM: VkFormat = 44
VK_FORMAT_B8G8R8A8_SNORM: VkFormat = 45
VK_FORMAT_B8G8R8A8_USCALED: VkFormat = 46
VK_FORMAT_B8G8R8A8_SSCALED: VkFormat = 47
VK_FORMAT_B8G8R8A8_UINT: VkFormat = 48
VK_FORMAT_B8G8R8A8_SINT: VkFormat = 49
VK_FORMAT_B8G8R8A8_SRGB: VkFormat = 50
VK_FORMAT_A8B8G8R8_UNORM_PACK32: VkFormat = 51
VK_FORMAT_A8B8G8R8_SNORM_PACK32: VkFormat = 52
VK_FORMAT_A8B8G8R8_USCALED_PACK32: VkFormat = 53
VK_FORMAT_A8B8G8R8_SSCALED_PACK32: VkFormat = 54
VK_FORMAT_A8B8G8R8_UINT_PACK32: VkFormat = 55
VK_FORMAT_A8B8G8R8_SINT_PACK32: VkFormat = 56
VK_FORMAT_A8B8G8R8_SRGB_PACK32: VkFormat = 57
VK_FORMAT_A2R10G10B10_UNORM_PACK32: VkFormat = 58
VK_FORMAT_A2R10G10B10_SNORM_PACK32: VkFormat = 59
VK_FORMAT_A2R10G10B10_USCALED_PACK32: VkFormat = 60
VK_FORMAT_A2R10G10B10_SSCALED_PACK32: VkFormat = 61
VK_FORMAT_A2R10G10B10_UINT_PACK32: VkFormat = 62
VK_FORMAT_A2R10G10B10_SINT_PACK32: VkFormat = 63
VK_FORMAT_A2B10G10R10_UNORM_PACK32: VkFormat = 64
VK_FORMAT_A2B10G10R10_SNORM_PACK32: VkFormat = 65
VK_FORMAT_A2B10G10R10_USCALED_PACK32: VkFormat = 66
VK_FORMAT_A2B10G10R10_SSCALED_PACK32: VkFormat = 67
VK_FORMAT_A2B10G10R10_UINT_PACK32: VkFormat = 68
VK_FORMAT_A2B10G10R10_SINT_PACK32: VkFormat = 69
VK_FORMAT_R16_UNORM: VkFormat = 70
VK_FORMAT_R16_SNORM: VkFormat = 71
VK_FORMAT_R16_USCALED: VkFormat = 72
VK_FORMAT_R16_SSCALED: VkFormat = 73
VK_FORMAT_R16_UINT: VkFormat = 74
VK_FORMAT_R16_SINT: VkFormat = 75
VK_FORMAT_R16_SFLOAT: VkFormat = 76
VK_FORMAT_R16G16_UNORM: VkFormat = 77
VK_FORMAT_R16G16_SNORM: VkFormat = 78
VK_FORMAT_R16G16_USCALED: VkFormat = 79
VK_FORMAT_R16G16_SSCALED: VkFormat = 80
VK_FORMAT_R16G16_UINT: VkFormat = 81
VK_FORMAT_R16G16_SINT: VkFormat = 82
VK_FORMAT_R16G16_SFLOAT: VkFormat = 83
VK_FORMAT_R16G16B16_UNORM: VkFormat = 84
VK_FORMAT_R16G16B16_SNORM: VkFormat = 85
VK_FORMAT_R16G16B16_USCALED: VkFormat = 86
VK_FORMAT_R16G16B16_SSCALED: VkFormat = 87
VK_FORMAT_R16G16B16_UINT: VkFormat = 88
VK_FORMAT_R16G16B16_SINT: VkFormat = 89
VK_FORMAT_R16G16B16_SFLOAT: VkFormat = 90
VK_FORMAT_R16G16B16A16_UNORM: VkFormat = 91
VK_FORMAT_R16G16B16A16_SNORM: VkFormat = 92
VK_FORMAT_R16G16B16A16_USCALED: VkFormat = 93
VK_FORMAT_R16G16B16A16_SSCALED: VkFormat = 94
VK_FORMAT_R16G16B16A16_UINT: VkFormat = 95
VK_FORMAT_R16G16B16A16_SINT: VkFormat = 96
VK_FORMAT_R16G16B16A16_SFLOAT: VkFormat = 97
VK_FORMAT_R32_UINT: VkFormat = 98
VK_FORMAT_R32_SINT: VkFormat = 99
VK_FORMAT_R32_SFLOAT: VkFormat = 100
VK_FORMAT_R32G32_UINT: VkFormat = 101
VK_FORMAT_R32G32_SINT: VkFormat = 102
VK_FORMAT_R32G32_SFLOAT: VkFormat = 103
VK_FORMAT_R32G32B32_UINT: VkFormat = 104
VK_FORMAT_R32G32B32_SINT: VkFormat = 105
VK_FORMAT_R32G32B32_SFLOAT: VkFormat = 106
VK_FORMAT_R32G32B32A32_UINT: VkFormat = 107
VK_FORMAT_R32G32B32A32_SINT: VkFormat = 108
VK_FORMAT_R32G32B32A32_SFLOAT: VkFormat = 109
VK_FORMAT_R64_UINT: VkFormat = 110
VK_FORMAT_R64_SINT: VkFormat = 111
VK_FORMAT_R64_SFLOAT: VkFormat = 112
VK_FORMAT_R64G64_UINT: VkFormat = 113
VK_FORMAT_R64G64_SINT: VkFormat = 114
VK_FORMAT_R64G64_SFLOAT: VkFormat = 115
VK_FORMAT_R64G64B64_UINT: VkFormat = 116
VK_FORMAT_R64G64B64_SINT: VkFormat = 117
VK_FORMAT_R64G64B64_SFLOAT: VkFormat = 118
VK_FORMAT_R64G64B64A64_UINT: VkFormat = 119
VK_FORMAT_R64G64B64A64_SINT: VkFormat = 120
VK_FORMAT_R64G64B64A64_SFLOAT: VkFormat = 121
VK_FORMAT_B10G11R11_UFLOAT_PACK32: VkFormat = 122
VK_FORMAT_E5B9G9R9_UFLOAT_PACK32: VkFormat = 123
VK_FORMAT_D16_UNORM: VkFormat = 124
VK_FORMAT_X8_D24_UNORM_PACK32: VkFormat = 125
VK_FORMAT_D32_SFLOAT: VkFormat = 126
VK_FORMAT_S8_UINT: VkFormat = 127
VK_FORMAT_D16_UNORM_S8_UINT: VkFormat = 128
VK_FORMAT_D24_UNORM_S8_UINT: VkFormat = 129
VK_FORMAT_D32_SFLOAT_S8_UINT: VkFormat = 130
VK_FORMAT_BC1_RGB_UNORM_BLOCK: VkFormat = 131
VK_FORMAT_BC1_RGB_SRGB_BLOCK: VkFormat = 132
VK_FORMAT_BC1_RGBA_UNORM_BLOCK: VkFormat = 133
VK_FORMAT_BC1_RGBA_SRGB_BLOCK: VkFormat = 134
VK_FORMAT_BC2_UNORM_BLOCK: VkFormat = 135
VK_FORMAT_BC2_SRGB_BLOCK: VkFormat = 136
VK_FORMAT_BC3_UNORM_BLOCK: VkFormat = 137
VK_FORMAT_BC3_SRGB_BLOCK: VkFormat = 138
VK_FORMAT_BC4_UNORM_BLOCK: VkFormat = 139
VK_FORMAT_BC4_SNORM_BLOCK: VkFormat = 140
VK_FORMAT_BC5_UNORM_BLOCK: VkFormat = 141
VK_FORMAT_BC5_SNORM_BLOCK: VkFormat = 142
VK_FORMAT_BC6H_UFLOAT_BLOCK: VkFormat = 143
VK_FORMAT_BC6H_SFLOAT_BLOCK: VkFormat = 144
VK_FORMAT_BC7_UNORM_BLOCK: VkFormat = 145
VK_FORMAT_BC7_SRGB_BLOCK: VkFormat = 146
VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK: VkFormat = 147
VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK: VkFormat = 148
VK_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK: VkFormat = 149
VK_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK: VkFormat = 150
VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK: VkFormat = 151
VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK: VkFormat = 152
VK_FORMAT_EAC_R11_UNORM_BLOCK: VkFormat = 153
VK_FORMAT_EAC_R11_SNORM_BLOCK: VkFormat = 154
VK_FORMAT_EAC_R11G11_UNORM_BLOCK: VkFormat = 155
VK_FORMAT_EAC_R11G11_SNORM_BLOCK: VkFormat = 156
VK_FORMAT_ASTC_4x4_UNORM_BLOCK: VkFormat = 157
VK_FORMAT_ASTC_4x4_SRGB_BLOCK: VkFormat = 158
VK_FORMAT_ASTC_5x4_UNORM_BLOCK: VkFormat = 159
VK_FORMAT_ASTC_5x4_SRGB_BLOCK: VkFormat = 160
VK_FORMAT_ASTC_5x5_UNORM_BLOCK: VkFormat = 161
VK_FORMAT_ASTC_5x5_SRGB_BLOCK: VkFormat = 162
VK_FORMAT_ASTC_6x5_UNORM_BLOCK: VkFormat = 163
VK_FORMAT_ASTC_6x5_SRGB_BLOCK: VkFormat = 164
VK_FORMAT_ASTC_6x6_UNORM_BLOCK: VkFormat = 165
VK_FORMAT_ASTC_6x6_SRGB_BLOCK: VkFormat = 166
VK_FORMAT_ASTC_8x5_UNORM_BLOCK: VkFormat = 167
VK_FORMAT_ASTC_8x5_SRGB_BLOCK: VkFormat = 168
VK_FORMAT_ASTC_8x6_UNORM_BLOCK: VkFormat = 169
VK_FORMAT_ASTC_8x6_SRGB_BLOCK: VkFormat = 170
VK_FORMAT_ASTC_8x8_UNORM_BLOCK: VkFormat = 171
VK_FORMAT_ASTC_8x8_SRGB_BLOCK: VkFormat = 172
VK_FORMAT_ASTC_10x5_UNORM_BLOCK: VkFormat = 173
VK_FORMAT_ASTC_10x5_SRGB_BLOCK: VkFormat = 174
VK_FORMAT_ASTC_10x6_UNORM_BLOCK: VkFormat = 175
VK_FORMAT_ASTC_10x6_SRGB_BLOCK: VkFormat = 176
VK_FORMAT_ASTC_10x8_UNORM_BLOCK: VkFormat = 177
VK_FORMAT_ASTC_10x8_SRGB_BLOCK: VkFormat = 178
VK_FORMAT_ASTC_10x10_UNORM_BLOCK: VkFormat = 179
VK_FORMAT_ASTC_10x10_SRGB_BLOCK: VkFormat = 180
VK_FORMAT_ASTC_12x10_UNORM_BLOCK: VkFormat = 181
VK_FORMAT_ASTC_12x10_SRGB_BLOCK: VkFormat = 182
VK_FORMAT_ASTC_12x12_UNORM_BLOCK: VkFormat = 183
VK_FORMAT_ASTC_12x12_SRGB_BLOCK: VkFormat = 184

VkColorComponentFlags: TypeAlias = int

VkColorSpaceKHR: TypeAlias = int
