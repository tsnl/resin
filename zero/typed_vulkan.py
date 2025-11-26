__all__ = [
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
    "VK_COMMAND_BUFFER_LEVEL_PRIMARY",
    "VK_COMMAND_BUFFER_LEVEL_SECONDARY",
    "VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT",
    "VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT",
    "VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT",
    "VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT",
    "VK_COMMAND_POOL_CREATE_TRANSIENT_BIT",
    "VK_COMPONENT_SWIZZLE_IDENTITY",
    "VK_CULL_MODE_BACK_BIT",
    "VK_CULL_MODE_FRONT_BIT",
    "VK_CULL_MODE_NONE",
    "VK_FALSE",
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
    "VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT",
    "VK_POLYGON_MODE_FILL",
    "VK_POLYGON_MODE_LINE",
    "VK_POLYGON_MODE_POINT",
    "VK_PRIMITIVE_TOPOLOGY_LINE_LIST",
    "VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST",
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
    "VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO",
    "VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES",
    "VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO",
    "VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO",
    "VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO",
    "VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO",
    "VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO",
    "VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO",
    "VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO",
    "VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO",
    "VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO",
    "VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO",
    "VK_TRUE",
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
    "ffi",
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
    "vkUnmapMemory",
    "vkWaitForFences",
    "vk_api_version_str",
    "vk_decompose_api_version",
]


from typing import TypeAlias
from vulkan import (
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
    VK_COMMAND_BUFFER_LEVEL_PRIMARY,
    VK_COMMAND_BUFFER_LEVEL_SECONDARY,
    VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT,
    VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT,
    VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
    VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
    VK_COMPONENT_SWIZZLE_IDENTITY,
    VK_CULL_MODE_BACK_BIT,
    VK_CULL_MODE_FRONT_BIT,
    VK_CULL_MODE_NONE,
    VK_FORMAT_D32_SFLOAT,
    VK_FORMAT_R32G32B32A32_SFLOAT,
    VK_FORMAT_R32_SFLOAT,
    VK_FORMAT_R8G8B8A8_UNORM,
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
    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
    VK_POLYGON_MODE_FILL,
    VK_POLYGON_MODE_LINE,
    VK_POLYGON_MODE_POINT,
    VK_PRIMITIVE_TOPOLOGY_LINE_LIST,
    VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
    VK_QUEUE_COMPUTE_BIT,
    VK_QUEUE_GRAPHICS_BIT,
    VK_QUEUE_TRANSFER_BIT,
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
    VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES,
    VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
    VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
    VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
    VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
    VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
    VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
    VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
    VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
    VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
    VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
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
    VkViewport,
    ffi,
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


# Graphics pipeline related opaque handles
class VkPipeline(OpaqueResourceHandle): ...


class VkPipelineLayout(OpaqueResourceHandle): ...


class VkDescriptorSetLayout(OpaqueResourceHandle): ...


class VkDescriptorSet(OpaqueResourceHandle): ...


# Bind point enum
VkPipelineBindPoint: TypeAlias = int  # 0 = GRAPHICS

# Attachment load/store ops (match .pyi values)
VK_ATTACHMENT_LOAD_OP_LOAD = 0
VK_ATTACHMENT_LOAD_OP_CLEAR = 1
VK_ATTACHMENT_LOAD_OP_DONT_CARE = 2
VK_ATTACHMENT_STORE_OP_STORE = 0
VK_ATTACHMENT_STORE_OP_DONT_CARE = 1

# Boolean constants
VK_TRUE = 1
VK_FALSE = 0

# Format constants
VK_FORMAT_UNDEFINED = 0

VkColorComponentFlags: TypeAlias = int


class VkShaderModule(OpaqueResourceHandle): ...
