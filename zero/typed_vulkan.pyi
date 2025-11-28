from dataclasses import dataclass
from typing import Any, Callable, List, Optional, list, TypeAlias

import cffi

#
# Basic:
#

class OpaqueResourceHandle: ...

def vkGetInstanceProcAddr(instance: "VkInstance", pName: str) -> Callable:
    """
    vkGetInstanceProcAddr retrieves a function pointer for a Vulkan command.
    """

raw_ffi: cffi.FFI

#
# Common:
#

def VK_MAKE_API_VERSION(variant: int, major: int, minor: int, patch: int) -> int: ...
def vk_decompose_api_version(api_version: int) -> tuple[int, int, int, int]: ...
def vk_api_version_str(api_version: int) -> str: ...

VK_API_VERSION_1_2: int
VK_API_VERSION_1_3: int
VK_API_VERSION_1_4: int

# VkBool32
VkBool32: TypeAlias = bool
VK_TRUE: VkBool32 = True
VK_FALSE: VkBool32 = False

VkDeviceSize: TypeAlias = int
VkFlags: TypeAlias = int

# VkExtent3D
# https://docs.vulkan.org/refpages/latest/refpages/source/VkExtent3D.html
@dataclass
class VkExtent3D:
    width: int
    height: int
    depth: int

# VkOffset3D
# https://docs.vulkan.org/refpages/latest/refpages/source/VkOffset3D.html
@dataclass
class VkOffset3D:
    x: int
    y: int
    z: int

# VkOffset2D
# https://docs.vulkan.org/refpages/latest/refpages/source/VkOffset2D.html
@dataclass
class VkOffset2D:
    x: int
    y: int

# VkExtent2D
# https://docs.vulkan.org/refpages/latest/refpages/source/VkExtent2D.html
@dataclass
class VkExtent2D:
    width: int
    height: int

# VkRect2D
# https://docs.vulkan.org/refpages/latest/refpages/source/VkRect2D.html
@dataclass
class VkRect2D:
    offset: VkOffset2D
    extent: VkExtent2D

# VkViewport
# https://docs.vulkan.org/refpages/latest/refpages/source/VkViewport.html
@dataclass
class VkViewport:
    x: float
    y: float
    width: float
    height: float
    minDepth: float
    maxDepth: float

#
# VkInstance
#

# VkInstance
# https://docs.vulkan.org/refpages/latest/refpages/source/VkInstance.html
class VkInstance(OpaqueResourceHandle): ...

# VkApplicationInfo
# https://docs.vulkan.org/refpages/latest/refpages/source/VkApplicationInfo.html
@dataclass
class VkApplicationInfo:
    pApplicationName: str | None = None
    pEngineName: str | None = None
    apiVersion: int = 0

# VkInstanceCreateInfo
# https://docs.vulkan.org/refpages/latest/refpages/source/VkInstanceCreateInfo.html
@dataclass
class VkInstanceCreateInfo:
    pApplicationInfo: VkApplicationInfo | None
    enabledLayerCount: int = 0
    ppEnabledLayerNames: list[str] = ()
    enabledExtensionCount: int = 0
    ppEnabledExtensionNames: list[str] = ()
    flags: int = 0

# vkCreateInstance
# https://docs.vulkan.org/refpages/latest/refpages/source/vkCreateInstance.html
def vkCreateInstance(
    pCreateInfo: VkInstanceCreateInfo,
    pAllocator: Optional[Any] = None,
) -> VkInstance:
    """
    vkCreateInstance creates a Vulkan instance.
    """

# vkDestroyInstance
# https://docs.vulkan.org/refpages/latest/refpages/source/vkDestroyInstance.html
def vkDestroyInstance(
    instance: VkInstance,
    pAllocator: Optional[Any] = None,
) -> None:
    """
    vkDestroyInstance destroys a Vulkan instance.
    """

# VkInstanceCreateFlags
# https://docs.vulkan.org/refpages/latest/refpages/source/VkInstanceCreateFlagBits.html
VkInstanceCreateFlagBits: TypeAlias = VkFlags
VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR: VkFlags = 0x00000001

#
# VkSurfaceKHR
#

# VkSurfaceKHR
# https://docs.vulkan.org/refpages/latest/refpages/source/VkSurfaceKHR.html
class VkSurfaceKHR(OpaqueResourceHandle): ...

# vkGetPhysicalDeviceSurfaceSupportKHR
# https://docs.vulkan.org/refpages/latest/refpages/source/vkGetPhysicalDeviceSurfaceSupportKHR.html
def vkGetPhysicalDeviceSurfaceSupportKHR(
    physicalDevice: VkPhysicalDevice,
    queueFamilyIndex: int,
    surface: VkSurfaceKHR,
) -> bool:
    """
    vkGetPhysicalDeviceSurfaceSupportKHR queries if a queue family of a physical device
    supports presentation to a given surface.
    """

# VkSurfaceFormatKHR
# https://docs.vulkan.org/refpages/latest/refpages/source/VkSurfaceFormatKHR.html
@dataclass
class VkSurfaceFormatKHR:
    format: VkFormat
    colorSpace: VkColorSpaceKHR

# vkGetPhysicalDeviceSurfaceFormatsKHR
# https://docs.vulkan.org/refpages/latest/refpages/source/vkGetPhysicalDeviceSurfaceFormatsKHR.html
def vkGetPhysicalDeviceSurfaceFormatsKHR(
    physicalDevice: VkPhysicalDevice,
    surface: VkSurfaceKHR,
) -> list[VkSurfaceFormatKHR]:
    """
    vkGetPhysicalDeviceSurfaceFormatsKHR queries the supported surface formats for a given physical device and surface.
    """

#
# VkPhysicalDevice
#

# VkGetPhysicalDevice
# https://docs.vulkan.org/refpages/latest/refpages/source/VkPhysicalDevice.html
class VkPhysicalDevice(OpaqueResourceHandle): ...

# VkPhysicalDeviceProperties
# https://docs.vulkan.org/refpages/latest/refpages/source/VkPhysicalDeviceProperties.html
@dataclass
class VkPhysicalDeviceProperties:
    apiVersion: int
    driverVersion: int
    vendorID: int
    deviceID: int
    deviceType: int
    deviceName: str
    limits: VkPhysicalDeviceLimits

# VkPhysicalDeviceLimits
# https://docs.vulkan.org/refpages/latest/refpages/source/VkPhysicalDeviceLimits.html
@dataclass
class VkPhysicalDeviceLimits:
    maxImageDimension1D: int
    maxImageDimension2D: int
    maxImageDimension3D: int
    maxImageDimensionCube: int
    maxImageArrayLayers: int
    maxTexelBufferElements: int
    maxUniformBufferRange: int
    maxStorageBufferRange: int
    maxPushConstantsSize: int
    maxMemoryAllocationCount: int
    maxSamplerAllocationCount: int
    bufferImageGranularity: VkDeviceSize
    sparseAddressSpaceSize: VkDeviceSize
    maxBoundDescriptorSets: int
    maxPerStageDescriptorSamplers: int
    maxPerStageDescriptorUniformBuffers: int
    maxPerStageDescriptorStorageBuffers: int
    maxPerStageDescriptorSampledImages: int
    maxPerStageDescriptorStorageImages: int
    maxPerStageDescriptorInputAttachments: int
    maxPerStageResources: int
    maxDescriptorSetSamplers: int
    maxDescriptorSetUniformBuffers: int
    maxDescriptorSetUniformBuffersDynamic: int
    maxDescriptorSetStorageBuffers: int
    maxDescriptorSetStorageBuffersDynamic: int
    maxDescriptorSetSampledImages: int
    maxDescriptorSetStorageImages: int
    maxDescriptorSetInputAttachments: int
    maxVertexInputAttributes: int
    maxVertexInputBindings: int
    maxVertexInputAttributeOffset: int
    maxVertexInputBindingStride: int
    maxVertexOutputComponents: int
    maxTessellationGenerationLevel: int
    maxTessellationPatchSize: int
    maxTessellationControlPerVertexInputComponents: int
    maxTessellationControlPerVertexOutputComponents: int
    maxTessellationControlPerPatchOutputComponents: int
    maxTessellationControlTotalOutputComponents: int
    maxTessellationEvaluationInputComponents: int
    maxTessellationEvaluationOutputComponents: int
    maxGeometryShaderInvocations: int
    maxGeometryInputComponents: int
    maxGeometryOutputComponents: int
    maxGeometryOutputVertices: int
    maxGeometryTotalOutputComponents: int
    maxFragmentInputComponents: int
    maxFragmentOutputAttachments: int
    maxFragmentDualSrcAttachments: int
    maxFragmentCombinedOutputResources: int
    maxComputeSharedMemorySize: int
    maxComputeWorkGroupCount: tuple[int, int, int]
    maxComputeWorkGroupInvocations: int
    maxComputeWorkGroupSize: tuple[int, int, int]
    subPixelPrecisionBits: int
    subTexelPrecisionBits: int
    mipmapPrecisionBits: int
    maxDrawIndexedIndexValue: int
    maxDrawIndirectCount: int
    maxSamplerLodBias: float
    maxSamplerAnisotropy: float
    maxViewports: int
    maxViewportDimensions: tuple[int, int]
    viewportBoundsRange: tuple[float, float]
    viewportSubPixelBits: int
    minMemoryMapAlignment: int
    minTexelBufferOffsetAlignment: VkDeviceSize
    minUniformBufferOffsetAlignment: VkDeviceSize
    minStorageBufferOffsetAlignment: VkDeviceSize
    minTexelOffset: int
    maxTexelOffset: int
    minTexelGatherOffset: int
    maxTexelGatherOffset: int
    minInterpolationOffset: float
    maxInterpolationOffset: float
    subPixelInterpolationOffsetBits: int
    maxFramebufferWidth: int
    maxFramebufferHeight: int
    maxFramebufferLayers: int
    framebufferColorSampleCounts: VkSampleCountFlags
    framebufferDepthSampleCounts: VkSampleCountFlags
    framebufferStencilSampleCounts: VkSampleCountFlags
    framebufferNoAttachmentsSampleCounts: VkSampleCountFlags
    maxColorAttachments: int
    sampledImageColorSampleCounts: VkSampleCountFlags
    sampledImageIntegerSampleCounts: VkSampleCountFlags
    sampledImageDepthSampleCounts: VkSampleCountFlags
    sampledImageStencilSampleCounts: VkSampleCountFlags
    storageImageSampleCounts: VkSampleCountFlags
    maxSampleMaskWords: int
    timestampComputeAndGraphics: bool
    timestampPeriod: float
    maxClipDistances: int
    maxCullDistances: int
    maxCombinedClipAndCullDistances: int
    discreteQueuePriorities: int
    pointSizeRange: tuple[float, float]
    lineWidthRange: tuple[float, float]
    pointSizeGranularity: float
    lineWidthGranularity: float
    strictLines: bool
    standardSampleLocations: bool
    optimalBufferCopyOffsetAlignment: VkDeviceSize
    optimalBufferCopyRowPitchAlignment: VkDeviceSize
    nonCoherentAtomSize: VkDeviceSize

# VkPhysicalDeviceDynamicRenderingFeatures
# https://docs.vulkan.org/refpages/latest/refpages/source/VkPhysicalDeviceDynamicRenderingFeatures.html
@dataclass
class VkPhysicalDeviceDynamicRenderingFeatures:
    dynamicRendering: bool

# VkPhysicalDeviceFeatures
# https://docs.vulkan.org/refpages/latest/refpages/source/VkPhysicalDeviceFeatures.html
@dataclass
class VkPhysicalDeviceFeatures:
    pNext: Any = None
    robustBufferAccess: bool = False
    fullDrawIndexUint32: bool = False
    imageCubeArray: bool = False
    independentBlend: bool = False
    geometryShader: bool = False
    tessellationShader: bool = False
    sampleRateShading: bool = False
    dualSrcBlend: bool = False
    logicOp: bool = False
    multiDrawIndirect: bool = False
    drawIndirectFirstInstance: bool = False
    depthClamp: bool = False
    depthBiasClamp: bool = False
    fillModeNonSolid: bool = False
    depthBounds: bool = False
    wideLines: bool = False
    largePoints: bool = False
    alphaToOne: bool = False
    multiViewport: bool = False
    samplerAnisotropy: bool = False
    textureCompressionETC2: bool = False
    textureCompressionASTC_LDR: bool = False
    textureCompressionBC: bool = False
    occlusionQueryPrecise: bool = False
    pipelineStatisticsQuery: bool = False
    vertexPipelineStoresAndAtomics: bool = False
    fragmentStoresAndAtomics: bool = False
    shaderTessellationAndGeometryPointSize: bool = False
    shaderImageGatherExtended: bool = False
    shaderStorageImageExtendedFormats: bool = False
    shaderStorageImageMultisample: bool = False
    shaderStorageImageReadWithoutFormat: bool = False
    shaderStorageImageWriteWithoutFormat: bool = False
    shaderUniformBufferArrayDynamicIndexing: bool = False
    shaderSampledImageArrayDynamicIndexing: bool = False
    shaderStorageBufferArrayDynamicIndexing: bool = False
    shaderStorageImageArrayDynamicIndexing: bool = False
    shaderClipDistance: bool = False
    shaderCullDistance: bool = False
    shaderFloat64: bool = False
    shaderInt64: bool = False
    shaderInt16: bool = False
    shaderResourceResidency: bool = False
    shaderResourceMinLod: bool = False
    sparseBinding: bool = False
    sparseResidencyBuffer: bool = False
    sparseResidencyImage2D: bool = False
    sparseResidencyImage3D: bool = False
    sparseResidency2Samples: bool = False
    sparseResidency4Samples: bool = False
    sparseResidency8Samples: bool = False
    sparseResidency16Samples: bool = False
    sparseResidencyAliased: bool = False
    variableMultisampleRate: bool = False
    inheritedQueries: bool = False

# VkPhysicalDeviceType
# https://docs.vulkan.org/refpages/latest/refpages/source/VkPhysicalDeviceType.html
VkPhysicalDeviceType: TypeAlias = int
VK_PHYSICAL_DEVICE_TYPE_OTHER: VkPhysicalDeviceType = 0
VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: VkPhysicalDeviceType = 1
VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: VkPhysicalDeviceType = 2
VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: VkPhysicalDeviceType = 3
VK_PHYSICAL_DEVICE_TYPE_CPU: VkPhysicalDeviceType = 4

# vkEnumeratePhysicalDevices
# https://docs.vulkan.org/refpages/latest/refpages/source/vkEnumeratePhysicalDevices.html
def vkEnumeratePhysicalDevices(instance: VkInstance) -> List[VkPhysicalDevice]:
    """
    vkEnumeratePhysicalDevices enumerates the physical devices available on the system.
    """

# vkGetPhysicalDeviceProperties
# https://docs.vulkan.org/refpages/latest/refpages/source/vkGetPhysicalDeviceProperties.html
def vkGetPhysicalDeviceProperties(
    physicalDevice: VkPhysicalDevice,
) -> VkPhysicalDeviceProperties:
    """
    vkGetPhysicalDeviceProperties retrieves the properties of a physical device.
    """

# vkGetPhysicalDeviceQueueFamilyProperties
# https://docs.vulkan.org/refpages/latest/refpages/source/vkGetPhysicalDeviceQueueFamilyProperties.html
def vkGetPhysicalDeviceQueueFamilyProperties(
    physicalDevice: VkPhysicalDevice,
) -> list["VkQueueFamilyProperties"]:
    """
    vkGetPhysicalDeviceQueueFamilyProperties retrieves the queue family properties of a physical device.
    """

#
# VkPhysicalDevice Memory
#

# VkPhysicalDeviceMemoryProperties
# https://docs.vulkan.org/refpages/latest/refpages/source/VkPhysicalDeviceMemoryProperties.html
@dataclass
class VkPhysicalDeviceMemoryProperties:
    memoryTypeCount: int
    memoryTypes: List[Any]
    memoryHeapCount: int
    memoryHeaps: List[Any]

# vkGetPhysicalDeviceMemoryProperties
# https://docs.vulkan.org/refpages/latest/refpages/source/vkGetPhysicalDeviceMemoryProperties.html
def vkGetPhysicalDeviceMemoryProperties(
    physicalDevice: VkPhysicalDevice,
) -> VkPhysicalDeviceMemoryProperties:
    """
    vkGetPhysicalDeviceMemoryProperties retrieves the memory properties of a physical device.
    """

#
# VkPhysicalDevice Queues
#

# VkQueueFlags
# https://docs.vulkan.org/refpages/latest/refpages/source/VkQueueFlagBits.html
VkQueueFlags: TypeAlias = VkFlags
VkQueueFlagBits: TypeAlias = int
VK_QUEUE_GRAPHICS_BIT: VkQueueFlagBits = 0x00000001
VK_QUEUE_COMPUTE_BIT: VkQueueFlagBits = 0x00000002
VK_QUEUE_TRANSFER_BIT: VkQueueFlagBits = 0x00000004

VK_QUEUE_FAMILY_IGNORED: int = -1

# VkQueueFamilyProperties
# https://docs.vulkan.org/refpages/latest/refpages/source/VkQueueFamilyProperties.html
@dataclass
class VkQueueFamilyProperties:
    queueFlags: VkQueueFlags
    queueCount: int
    timestampValidBits: int
    minImageTransferGranularity: VkExtent3D

# VkDeviceQueueCreateInfo
# https://docs.vulkan.org/refpages/latest/refpages/source/VkDeviceQueueCreateInfo.html
@dataclass
class VkDeviceQueueCreateInfo:
    queueFamilyIndex: int
    queueCount: int
    pQueuePriorities: list[float] | None = None

#
# VkDevice
#

# VkDevice
# https://docs.vulkan.org/refpages/latest/refpages/source/VkDevice.html
class VkDevice(OpaqueResourceHandle): ...

# VkDeviceCreateFlags
# https://docs.vulkan.org/refpages/latest/refpages/source/VkDeviceCreateFlags.html
VkDeviceCreateFlags: TypeAlias = VkFlags

# VkDeviceCreateInfo
# https://docs.vulkan.org/refpages/latest/refpages/source/VkDeviceCreateInfo.html
@dataclass
class VkDeviceCreateInfo:
    pNext: VkPhysicalDeviceDynamicRenderingFeatures | None = None
    flags: VkDeviceCreateFlags = 0
    queueCreateInfoCount: int = 0
    pQueueCreateInfos: list[Any] = ()
    enabledExtensionCount: int = 0
    ppEnabledExtensionNames: list[str] = ()
    pEnabledFeatures: list[Any] = ()

# vkCreateDevice
# https://docs.vulkan.org/refpages/latest/refpages/source/vkCreateDevice.html
def vkCreateDevice(
    physicalDevice: VkPhysicalDevice,
    pCreateInfo: VkDeviceCreateInfo,
    pAllocator: Any,
) -> VkDevice:
    """
    vkCreateDevice creates a logical device from a physical device.
    """

# vkDestroyDevice
# https://docs.vulkan.org/refpages/latest/refpages/source/vkDestroyDevice.html
def vkDestroyDevice(device: VkDevice, pAllocator: Any) -> None:
    """
    vkDestroyDevice destroys a logical device.
    """

# vkDeviceWaitIdle
# https://docs.vulkan.org/refpages/latest/refpages/source/vkDeviceWaitIdle.html
def vkDeviceWaitIdle(device: VkDevice) -> None:
    """
    vkDeviceWaitIdle waits until the device is idle.
    """

#
# VkDeviceMemory
#

class VkDeviceMemory(OpaqueResourceHandle): ...

@dataclass
class VkMemoryRequirements:
    size: VkDeviceSize
    alignment: VkDeviceSize
    memoryTypeBits: int

@dataclass
class VkMemoryAllocateInfo:
    allocationSize: VkDeviceSize
    memoryTypeIndex: int

# vkAllocateMemory
# https://docs.vulkan.org/refpages/latest/refpages/source/vkAllocateMemory.html
def vkAllocateMemory(
    device: VkDevice,
    pAllocateInfo: VkMemoryAllocateInfo,
    pAllocator: Any,
) -> VkDeviceMemory:
    """
    vkAllocateMemory allocates device memory.
    """

# vkFreeMemory
# https://docs.vulkan.org/refpages/latest/refpages/source/vkFreeMemory.html
def vkFreeMemory(
    device: VkDevice,
    memory: VkDeviceMemory,
    pAllocator: Any,
) -> None:
    """
    vkFreeMemory frees device memory.
    """

def vkMapMemory(
    device: VkDevice,
    memory: VkDeviceMemory,
    offset: VkDeviceSize,
    size: VkDeviceSize,
    flags: int,
) -> memoryview:
    pass

def vkUnmapMemory(
    device: VkDevice,
    memory: VkDeviceMemory,
) -> None:
    pass

# VkMemoryPropertyFlags
# https://docs.vulkan.org/refpages/latest/refpages/source/VkMemoryPropertyFlagBits.html
VkMemoryPropertyFlags: TypeAlias = VkFlags
VkMemoryPropertyFlagBits: TypeAlias = int
VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT: VkMemoryPropertyFlagBits = 0x00000001
VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT: VkMemoryPropertyFlagBits = 0x00000002
VK_MEMORY_PROPERTY_HOST_COHERENT_BIT: VkMemoryPropertyFlagBits = 0x00000004
VK_MEMORY_PROPERTY_HOST_CACHED_BIT: VkMemoryPropertyFlagBits = 0x00000008
VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT: VkMemoryPropertyFlagBits = 0x00000010
VK_MEMORY_PROPERTY_PROTECTED_BIT: VkMemoryPropertyFlagBits = 0x00000020

# VkMemoryMapFlags
# https://docs.vulkan.org/refpages/latest/refpages/source/VkMemoryMapFlagBits.html
VkMemoryMapFlags: TypeAlias = VkFlags
VkMemoryMapFlagBits: TypeAlias = int

#
# VkImage
#

# VkImage
# https://docs.vulkan.org/refpages/latest/refpages/source/VkImage.html
class VkImage(OpaqueResourceHandle): ...

# VkImageCreateFlagBits:
# https://docs.vulkan.org/refpages/latest/refpages/source/VkImageCreateFlagBits.html
VkImageCreateFlags: TypeAlias = VkFlags
VkImageCreateFlagBits: TypeAlias = int
VK_IMAGE_CREATE_SPARSE_BINDING_BIT = 0x00000001
VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT = 0x00000002
VK_IMAGE_CREATE_SPARSE_ALIASED_BIT = 0x00000004
VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT = 0x00000008
VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT = 0x00000010

# VkImageType
# https://docs.vulkan.org/refpages/latest/refpages/source/VkImageType.html
VkImageType: TypeAlias = int
VK_IMAGE_TYPE_1D: VkImageType = 0
VK_IMAGE_TYPE_2D: VkImageType = 1
VK_IMAGE_TYPE_3D: VkImageType = 2

# VkFormat
# https://docs.vulkan.org/refpages/latest/refpages/source/VkFormat.html
VkFormat: TypeAlias = int
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

# VkSampleCountFlagBits
# https://docs.vulkan.org/refpages/latest/refpages/source/VkSampleCountFlagBits.html
VkSampleCountFlags: TypeAlias = VkFlags
VkSampleCountFlagBits: TypeAlias = int
VK_SAMPLE_COUNT_1_BIT: VkSampleCountFlagBits = 0x00000001
VK_SAMPLE_COUNT_2_BIT: VkSampleCountFlagBits = 0x00000002
VK_SAMPLE_COUNT_4_BIT: VkSampleCountFlagBits = 0x00000004
VK_SAMPLE_COUNT_8_BIT: VkSampleCountFlagBits = 0x00000008
VK_SAMPLE_COUNT_16_BIT: VkSampleCountFlagBits = 0x00000010
VK_SAMPLE_COUNT_32_BIT: VkSampleCountFlagBits = 0x00000020
VK_SAMPLE_COUNT_64_BIT: VkSampleCountFlagBits = 0x00000040

# VkImageTiling
# https://docs.vulkan.org/refpages/latest/refpages/source/VkImageTiling.html
VkImageTiling: TypeAlias = int
VK_IMAGE_TILING_OPTIMAL: VkImageTiling = 0
VK_IMAGE_TILING_LINEAR: VkImageTiling = 1

# VkImageUsageFlags
# https://docs.vulkan.org/refpages/latest/refpages/source/VkImageUsageFlagBits
VkImageUsageFlags: TypeAlias = VkFlags
VkImageUsageFlagBits: TypeAlias = int
VK_IMAGE_USAGE_TRANSFER_SRC_BIT: VkImageUsageFlagBits = 0x00000001
VK_IMAGE_USAGE_TRANSFER_DST_BIT: VkImageUsageFlagBits = 0x00000002
VK_IMAGE_USAGE_SAMPLED_BIT: VkImageUsageFlagBits = 0x00000004
VK_IMAGE_USAGE_STORAGE_BIT: VkImageUsageFlagBits = 0x00000008
VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT: VkImageUsageFlagBits = 0x00000010
VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT: VkImageUsageFlagBits = 0x00000020

# VkSharingMode
# https://docs.vulkan.org/refpages/latest/refpages/source/VkSharingMode.html
VkSharingMode: TypeAlias = int
VK_SHARING_MODE_EXCLUSIVE: VkSharingMode = 0
VK_SHARING_MODE_CONCURRENT: VkSharingMode = 1

# VkImageLayout
# https://docs.vulkan.org/refpages/latest/refpages/source/VkImageLayout.html
VkImageLayout: TypeAlias = int
VK_IMAGE_LAYOUT_UNDEFINED: VkImageLayout = 0
VK_IMAGE_LAYOUT_GENERAL: VkImageLayout = 1
VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL: VkImageLayout = 2
VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL: VkImageLayout = 3
VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL: VkImageLayout = 4
VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL: VkImageLayout = 5
VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL: VkImageLayout = 6
VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL: VkImageLayout = 7
VK_IMAGE_LAYOUT_PREINITIALIZED: VkImageLayout = 8

# VkImageCreateInfo
# https://docs.vulkan.org/refpages/latest/refpages/source/VkImageCreateInfo.html
@dataclass
class VkImageCreateInfo:
    flags: VkImageCreateFlags
    imageType: VkImageType
    format: VkFormat
    extent: VkExtent3D
    mipLevels: int
    arrayLayers: int
    samples: VkSampleCountFlagBits
    tiling: VkImageTiling
    usage: VkImageUsageFlags
    sharingMode: VkSharingMode
    queueFamilyIndexCount: int
    pQueueFamilyIndices: list[int] | None
    initialLayout: VkImageLayout

# vkCreateImage
# https://docs.vulkan.org/refpages/latest/refpages/source/vkCreateImage.html
def vkCreateImage(
    device: VkDevice,
    pCreateInfo: VkImageCreateInfo,
    pAllocator: Any,
) -> VkImage:
    """
    vkCreateImage creates a new image object.
    """

# vkDestroyImage
# https://docs.vulkan.org/refpages/latest/refpages/source/vkDestroyImage.html
def vkDestroyImage(
    device: VkDevice,
    image: VkImage,
    pAllocator: Any,
) -> None:
    """
    vkDestroyImage destroys an image object.
    """

def vkGetImageMemoryRequirements(
    device: VkDevice, image: VkImage
) -> VkMemoryRequirements:
    """
    vkGetImageMemoryRequirements retrieves the memory requirements for an image object.
    """

def vkBindImageMemory(
    device: VkDevice,
    image: VkImage,
    memory: VkDeviceMemory,
    memoryOffset: int,
) -> None:
    """
    vkBindImageMemory binds device memory to an image object.
    """

#
# VkImageView
#

# VkImageView
# https://docs.vulkan.org/refpages/latest/refpages/source/VkImageView.html
class VkImageView(OpaqueResourceHandle): ...

# VkImageViewCreateFlags
# https://docs.vulkan.org/refpages/latest/refpages/source/VkImageViewCreateFlags.html
VkImageViewCreateFlags: TypeAlias = VkFlags
VkImageViewCreateFlagBits: TypeAlias = int

# VkImageViewType
# https://docs.vulkan.org/refpages/latest/refpages/source/VkImageViewType.html
VkImageViewType: TypeAlias = int
VK_IMAGE_VIEW_TYPE_1D: VkImageViewType = 0
VK_IMAGE_VIEW_TYPE_2D: VkImageViewType = 1
VK_IMAGE_VIEW_TYPE_3D: VkImageViewType = 2
VK_IMAGE_VIEW_TYPE_CUBE: VkImageViewType = 3
VK_IMAGE_VIEW_TYPE_1D_ARRAY: VkImageViewType = 4
VK_IMAGE_VIEW_TYPE_2D_ARRAY: VkImageViewType = 5
VK_IMAGE_VIEW_TYPE_CUBE_ARRAY: VkImageViewType = 6

# VkComponentMapping
# https://docs.vulkan.org/refpages/latest/refpages/source/VkComponentMapping.html
@dataclass
class VkComponentMapping:
    r: VkComponentSwizzle
    g: VkComponentSwizzle
    b: VkComponentSwizzle
    a: VkComponentSwizzle

# VkComponentSwizzle
# https://docs.vulkan.org/refpages/latest/refpages/source/VkComponentSwizzle.html
VkComponentSwizzle: TypeAlias = int
VK_COMPONENT_SWIZZLE_IDENTITY: VkComponentSwizzle = 0
VK_COMPONENT_SWIZZLE_ZERO: VkComponentSwizzle = 1
VK_COMPONENT_SWIZZLE_ONE: VkComponentSwizzle = 2
VK_COMPONENT_SWIZZLE_R: VkComponentSwizzle = 3
VK_COMPONENT_SWIZZLE_G: VkComponentSwizzle = 4
VK_COMPONENT_SWIZZLE_B: VkComponentSwizzle = 5
VK_COMPONENT_SWIZZLE_A: VkComponentSwizzle = 6

# VkImageSubresourceRange
# https://docs.vulkan.org/refpages/latest/refpages/source/VkImageSubresourceRange.html
@dataclass
class VkImageSubresourceRange:
    aspectMask: VkImageAspectFlags
    baseMipLevel: int
    levelCount: int
    baseArrayLayer: int
    layerCount: int

# VkImageAspectFlags
# https://docs.vulkan.org/refpages/latest/refpages/source/VkImageAspectFlagBits.html
VkImageAspectFlags: TypeAlias = VkFlags
VkImageAspectFlagBits: TypeAlias = int
VK_IMAGE_ASPECT_COLOR_BIT: VkImageAspectFlagBits = 0x00000001
VK_IMAGE_ASPECT_DEPTH_BIT: VkImageAspectFlagBits = 0x00000002
VK_IMAGE_ASPECT_STENCIL_BIT: VkImageAspectFlagBits = 0x00000004
VK_IMAGE_ASPECT_METADATA_BIT: VkImageAspectFlagBits = 0x00000008

# VkImageViewCreateInfo
# https://docs.vulkan.org/refpages/latest/refpages/source/VkImageViewCreateInfo.html
@dataclass
class VkImageViewCreateInfo:
    flags: VkImageViewCreateFlags
    image: VkImage
    viewType: VkImageViewType
    format: VkFormat
    components: VkComponentMapping
    subresourceRange: VkImageSubresourceRange

# vkCreateImageView
# https://docs.vulkan.org/refpages/latest/refpages/source/vkCreateImageView.html
def vkCreateImageView(
    device: VkDevice,
    pCreateInfo: VkImageViewCreateInfo,
    pAllocator: Any,
) -> VkImageView:
    """
    vkCreateImageView creates an image view object.
    """

# vkDestroyImageView
# https://docs.vulkan.org/refpages/latest/refpages/source/vkDestroyImageView.html
def vkDestroyImageView(
    device: VkDevice,
    imageView: VkImageView,
    pAllocator: Any,
) -> None:
    """
    vkDestroyImageView destroys an image view object.
    """

#
# VkSampler
#

# VkSampler
# https://docs.vulkan.org/refpages/latest/refpages/source/VkSampler.html
class VkSampler(OpaqueResourceHandle): ...

# VkSamplerCreateFlags
# https://docs.vulkan.org/refpages/latest/refpages/source/VkSamplerCreateFlagBits.html
VkSamplerCreateFlags: TypeAlias = VkFlags
VkSamplerCreateFlagBits: TypeAlias = int

# VkFilter
# https://docs.vulkan.org/refpages/latest/refpages/source/VkFilter.html
VkFilter: TypeAlias = int
VK_FILTER_NEAREST: VkFilter = 0
VK_FILTER_LINEAR: VkFilter = 1

# VkSamplerMipmapMode
# https://docs.vulkan.org/refpages/latest/refpages/source/VkSamplerMipmapMode.html
VkSamplerMipmapMode: TypeAlias = int
VK_SAMPLER_MIPMAP_MODE_NEAREST: VkSamplerMipmapMode = 0
VK_SAMPLER_MIPMAP_MODE_LINEAR: VkSamplerMipmapMode = 1

# VkSamplerAddressMode
# https://docs.vulkan.org/refpages/latest/refpages/source/VkSamplerAddressMode.html
VkSamplerAddressMode: TypeAlias = int
VK_SAMPLER_ADDRESS_MODE_REPEAT: VkSamplerAddressMode = 0
VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT: VkSamplerAddressMode = 1
VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE: VkSamplerAddressMode = 2
VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER: VkSamplerAddressMode = 3
VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE: VkSamplerAddressMode = 4

# VkCompareOp
# https://docs.vulkan.org/refpages/latest/refpages/source/VkCompareOp.html
VkCompareOp: TypeAlias = int
VK_COMPARE_OP_NEVER: VkCompareOp = 0
VK_COMPARE_OP_LESS: VkCompareOp = 1
VK_COMPARE_OP_EQUAL: VkCompareOp = 2
VK_COMPARE_OP_LESS_OR_EQUAL: VkCompareOp = 3
VK_COMPARE_OP_GREATER: VkCompareOp = 4
VK_COMPARE_OP_NOT_EQUAL: VkCompareOp = 5
VK_COMPARE_OP_GREATER_OR_EQUAL: VkCompareOp = 6
VK_COMPARE_OP_ALWAYS: VkCompareOp = 7

# VkBorderColor
# https://docs.vulkan.org/refpages/latest/refpages/source/VkBorderColor.html
VkBorderColor: TypeAlias = int
VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK: VkBorderColor = 0
VK_BORDER_COLOR_INT_TRANSPARENT_BLACK: VkBorderColor = 1
VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK: VkBorderColor = 2
VK_BORDER_COLOR_INT_OPAQUE_BLACK: VkBorderColor = 3
VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE: VkBorderColor = 4
VK_BORDER_COLOR_INT_OPAQUE_WHITE: VkBorderColor = 5

# VkSamplerCreateInfo
# https://docs.vulkan.org/refpages/latest/refpages/source/VkSamplerCreateInfo.html
@dataclass
class VkSamplerCreateInfo:
    flags: VkSamplerCreateFlags
    magFilter: VkFilter
    minFilter: VkFilter
    mipmapMode: VkSamplerMipmapMode
    addressModeU: VkSamplerAddressMode
    addressModeV: VkSamplerAddressMode
    addressModeW: VkSamplerAddressMode
    mipLodBias: float
    anisotropyEnable: bool
    maxAnisotropy: float
    compareEnable: bool
    compareOp: VkCompareOp
    minLod: float
    maxLod: float
    borderColor: VkBorderColor
    unnormalizedCoordinates: bool

# vkCreateSampler
# https://docs.vulkan.org/refpages/latest/refpages/source/vkCreateSampler.html
def vkCreateSampler(
    device: VkDevice,
    pCreateInfo: VkSamplerCreateInfo,
    pAllocator: Any,
) -> VkSampler:
    """
    vkCreateSampler creates a new sampler object.
    """

# vkDestroySampler
# https://docs.vulkan.org/refpages/latest/refpages/source/vkDestroySampler.html
def vkDestroySampler(
    device: VkDevice,
    sampler: VkSampler,
    pAllocator: Any,
) -> None:
    """
    vkDestroySampler destroys a sampler object.
    """

#
# VkRenderingAttachmentInfo
#

# VkResolveModeFlagBits
# https://docs.vulkan.org/refpages/latest/refpages/source/VkResolveModeFlagBits.html
VkResolveModeFlag: TypeAlias = VkFlags
VkResolveModeFlagBits: TypeAlias = int
VK_RESOLVE_MODE_NONE: VkResolveModeFlagBits = 0x00000000
VK_RESOLVE_MODE_SAMPLE_ZERO_BIT: VkResolveModeFlagBits = 0x00000001
VK_RESOLVE_MODE_AVERAGE_BIT: VkResolveModeFlagBits = 0x00000002
VK_RESOLVE_MODE_MIN_BIT: VkResolveModeFlagBits = 0x00000004
VK_RESOLVE_MODE_MAX_BIT: VkResolveModeFlagBits = 0x00000008

# VkAttachmentLoadOp
# https://docs.vulkan.org/refpages/latest/refpages/source/VkAttachmentLoadOp.html
VkAttachmentLoadOp: TypeAlias = int
VK_ATTACHMENT_LOAD_OP_LOAD: VkAttachmentLoadOp = 0
VK_ATTACHMENT_LOAD_OP_CLEAR: VkAttachmentLoadOp = 1
VK_ATTACHMENT_LOAD_OP_DONT_CARE: VkAttachmentLoadOp = 2

# VkAttachmentStoreOp
# https://docs.vulkan.org/refpages/latest/refpages/source/VkAttachmentStoreOp.html
VkAttachmentStoreOp: TypeAlias = int
VK_ATTACHMENT_STORE_OP_STORE: VkAttachmentStoreOp = 0
VK_ATTACHMENT_STORE_OP_DONT_CARE: VkAttachmentStoreOp = 1

# VkClearColorValue
# https://docs.vulkan.org/refpages/latest/refpages/source/VkClearColorValue.html
@dataclass
class VkClearColorValue:
    float32: list[float] = (0.0, 0.0, 0.0, 0.0)
    int32: list[int] = (0, 0, 0, 0)
    uint32: list[int] = (0, 0, 0, 0)

# VkClearDepthStencilValue
# https://docs.vulkan.org/refpages/latest/refpages/source/VkClearDepthStencilValue.html
@dataclass
class VkClearDepthStencilValue:
    depth: float = 1.0
    stencil: int = 0

# VkClearValue
# https://docs.vulkan.org/refpages/latest/refpages/source/VkClearValue.html
@dataclass
class VkClearValue:
    color: VkClearColorValue | None = None
    depthStencil: VkClearDepthStencilValue | None = None

# VkRenderingAttachmentInfo
# https://docs.vulkan.org/refpages/latest/refpages/source/VkRenderingAttachmentInfo.html
@dataclass
class VkRenderingAttachmentInfo:
    imageView: VkImageView
    imageLayout: VkImageLayout
    resolveMode: VkResolveModeFlagBits = 0
    resolveImageView: VkImageView | None = None
    resolveImageLayout: VkImageLayout = VK_IMAGE_LAYOUT_UNDEFINED
    loadOp: VkAttachmentLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE
    storeOp: VkAttachmentStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE
    clearValue: VkClearValue | None = None

#
# VkBuffer
#

# VkBuffer
# https://docs.vulkan.org/refpages/latest/refpages/source/VkBuffer.html
class VkBuffer(OpaqueResourceHandle): ...

# VkBufferCreateFlags
# https://docs.vulkan.org/refpages/latest/refpages/source/VkBufferCreateFlagBits.html
VkBufferCreateFlags: TypeAlias = VkFlags
VkBufferCreateFlagBits: TypeAlias = int
VK_BUFFER_CREATE_SPARSE_BINDING_BIT: VkBufferCreateFlagBits = 0x00000001
VK_BUFFER_CREATE_SPARSE_RESIDENCY_BIT: VkBufferCreateFlagBits = 0x00000002
VK_BUFFER_CREATE_SPARSE_ALIASED_BIT: VkBufferCreateFlagBits = 0x00000004

# VkBufferUsageFlags
# https://docs.vulkan.org/refpages/latest/refpages/source/VkBufferUsageFlagBits.html
VkBufferUsageFlags: TypeAlias = VkFlags
VkBufferUsageFlagBits: TypeAlias = int
VK_BUFFER_USAGE_TRANSFER_SRC_BIT: VkBufferUsageFlagBits = 0x00000001
VK_BUFFER_USAGE_TRANSFER_DST_BIT: VkBufferUsageFlagBits = 0x00000002
VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT: VkBufferUsageFlagBits = 0x00000004
VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT: VkBufferUsageFlagBits = 0x00000008
VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT: VkBufferUsageFlagBits = 0x00000010
VK_BUFFER_USAGE_STORAGE_BUFFER_BIT: VkBufferUsageFlagBits = 0x00000020
VK_BUFFER_USAGE_INDEX_BUFFER_BIT: VkBufferUsageFlagBits = 0x00000040
VK_BUFFER_USAGE_VERTEX_BUFFER_BIT: VkBufferUsageFlagBits = 0x00000080
VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT: VkBufferUsageFlagBits = 0x00000100

# VkBufferCreateInfo
# https://docs.vulkan.org/refpages/latest/refpages/source/VkBufferCreateInfo.html
@dataclass
class VkBufferCreateInfo:
    flags: VkBufferCreateFlags
    size: VkDeviceSize
    usage: VkBufferUsageFlags
    sharingMode: VkSharingMode
    queueFamilyIndexCount: int
    pQueueFamilyIndices: list[int] | None

# vkCreateBuffer
# https://docs.vulkan.org/refpages/latest/refpages/source/vkCreateBuffer.html
def vkCreateBuffer(
    device: VkDevice,
    pCreateInfo: VkBufferCreateInfo,
    pAllocator: Any,
) -> VkBuffer:
    """
    vkCreateBuffer creates a new buffer object.
    """

# vkDestroyBuffer
# https://docs.vulkan.org/refpages/latest/refpages/source/vkDestroyBuffer.html
def vkDestroyBuffer(
    device: VkDevice,
    buffer: VkBuffer,
    pAllocator: Any,
) -> None:
    """
    vkDestroyBuffer destroys a buffer object.
    """

# vkGetBufferMemoryRequirements
# https://docs.vulkan.org/refpages/latest/refpages/source/vkGetBufferMemoryRequirements.html
def vkGetBufferMemoryRequirements(
    device: VkDevice,
    buffer: VkBuffer,
) -> VkMemoryRequirements:
    """
    vkGetBufferMemoryRequirements retrieves the memory requirements for a buffer object.
    """

# vkBindBufferMemory
# https://docs.vulkan.org/refpages/latest/refpages/source/vkBindBufferMemory.html
def vkBindBufferMemory(
    device: VkDevice,
    buffer: VkBuffer,
    memory: VkDeviceMemory,
    memoryOffset: VkDeviceSize,
):
    """
    vkBindBufferMemory associates a piece of memory with the given buffer.
    """

#
# VkBufferView
#

# VkBufferView
# https://docs.vulkan.org/refpages/latest/refpages/source/VkBufferView.html
class VkBufferView(OpaqueResourceHandle): ...

# VkBufferViewCreateFlags
VkBufferViewCreateFlags: TypeAlias = VkFlags
VkBufferViewCreateFlagBits: TypeAlias = int

# vkCreateBufferView
# https://docs.vulkan.org/refpages/latest/refpages/source/vkCreateBufferView.html
def vkCreateBufferView(
    device: VkDevice,
    pCreateInfo: VkBufferViewCreateInfo,
    pAllocator: Any,
) -> VkBufferView:
    """
    vkCreateBufferView creates a new buffer view object.
    """

# vkDestroyBufferView
# https://docs.vulkan.org/refpages/latest/refpages/source/vkDestroyBufferView.html
def vkDestroyBufferView(
    device: VkDevice,
    bufferView: VkBufferView,
    pAllocator: Any,
) -> None:
    """
    vkDestroyBufferView destroys a buffer view object.
    """

# VkBufferViewCreateInfo
# https://docs.vulkan.org/refpages/latest/refpages/source/VkBufferViewCreateInfo
@dataclass
class VkBufferViewCreateInfo:
    flags: VkBufferViewCreateFlags
    buffer: VkBuffer
    format: VkFormat
    offset: VkDeviceSize
    range: VkDeviceSize

#
# VkCommandPool
#

class VkCommandPool(OpaqueResourceHandle): ...

# VkCommandPoolCreateFlags
# https://docs.vulkan.org/refpages/latest/refpages/source/VkCommandPoolCreateFlags
VkCommandPoolCreateFlags: TypeAlias = VkFlags
VkCommandPoolCreateFlagBits: TypeAlias = int
VK_COMMAND_POOL_CREATE_TRANSIENT_BIT: VkCommandPoolCreateFlagBits = (  #
    0x00000001
)
VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT: VkCommandPoolCreateFlagBits = (
    0x00000002
)

# vkCreateCommandPool
# https://docs.vulkan.org/refpages/latest/refpages/source/vkCreateCommandPool.html
def vkCreateCommandPool(
    device: VkDevice,
    pCreateInfo: VkCommandPoolCreateInfo,
    pAllocator: Any,
) -> VkCommandPool:
    """
    vkCreateCommandPool creates a new command pool object.
    """

# vkDestroyCommandPool
# https://docs.vulkan.org/refpages/latest/refpages/source/vkDestroyCommandPool.html
def vkDestroyCommandPool(
    device: VkDevice,
    commandPool: VkCommandPool,
    pAllocator: Any,
) -> None:
    """
    vkDestroyCommandPool destroys a command pool object.
    """

# VkCommandPoolCreateFlags
# https://docs.vulkan.org/refpages/latest/refpages/source/VkCommandPoolCreateInfo.html
@dataclass
class VkCommandPoolCreateInfo:
    flags: VkCommandPoolCreateFlags
    queueFamilyIndex: int

#
# VkCommandBuffer
#

# VkCommandBuffer
# https://docs.vulkan.org/refpages/latest/refpages/source/VkCommandBuffer.html
class VkCommandBuffer(OpaqueResourceHandle): ...

# VkCommandBufferLevel
# https://docs.vulkan.org/refpages/latest/refpages/source/VkCommandBufferLevel.html
VkCommandBufferLevel: TypeAlias = int
VK_COMMAND_BUFFER_LEVEL_PRIMARY: VkCommandBufferLevel = 0
VK_COMMAND_BUFFER_LEVEL_SECONDARY: VkCommandBufferLevel = 1

# VkCommandBufferResetFlags
# https://docs.vulkan.org/refpages/latest/refpages/source/VkCommandBufferResetFlags.html
VkCommandBufferResetFlags: TypeAlias = VkFlags
VkCommandBufferResetFlagBits: TypeAlias = int

# VkCommandBufferAllocateInfo
# https://docs.vulkan.org/refpages/latest/refpages/source/VkCommandBufferAllocateInfo.html
@dataclass
class VkCommandBufferAllocateInfo:
    commandPool: VkCommandPool
    level: VkCommandBufferLevel
    commandBufferCount: int

# vkAllocateCommandBuffers
# https://docs.vulkan.org/refpages/latest/refpages/source/vkAllocateCommandBuffers.html
def vkAllocateCommandBuffers(
    device: VkDevice,
    pAllocateInfo: VkCommandBufferAllocateInfo,
) -> list[VkCommandBuffer]:
    """
    vkAllocateCommandBuffers allocates command buffers from a command pool.
    """

# vkFreeCommandBuffers
# https://docs.vulkan.org/refpages/latest/refpages/source/vkFreeCommandBuffers.html
def vkFreeCommandBuffers(
    device: VkDevice,
    commandPool: VkCommandPool,
    commandBufferCount: int,
    pCommandBuffers: list[VkCommandBuffer],
) -> None:
    """
    vkFreeCommandBuffers frees command buffers back to the command pool.
    """

def vkResetCommandBuffer(
    commandBuffer: VkCommandBuffer,
    flags: VkCommandBufferResetFlags,
) -> None:
    """
    vkResetCommandBuffer resets a command buffer to the initial state.
    """

VkCommandBufferUsageFlags: TypeAlias = VkFlags
VkCommandBufferUsageFlagBits: TypeAlias = int
VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT: VkCommandBufferUsageFlagBits = (  #
    0x00000001
)
VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT: VkCommandBufferUsageFlagBits = (  #
    0x00000002
)
VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT: VkCommandBufferUsageFlagBits = (  #
    0x00000004
)

# VkCommandBufferBeginInfo
# https://docs.vulkan.org/refpages/latest/refpages/source/VkCommandBufferBeginInfo
@dataclass
class VkCommandBufferBeginInfo:
    flags: VkCommandBufferUsageFlags
    pInheritanceInfo: Any | None

def vkBeginCommandBuffer(
    commandBuffer: VkCommandBuffer,
    pBeginInfo: VkCommandBufferBeginInfo,
) -> None:
    """
    vkBeginCommandBuffer starts the recording of a command buffer.
    """

def vkEndCommandBuffer(commandBuffer: VkCommandBuffer) -> None:
    """
    vkEndCommandBuffer ends the recording of a command buffer.
    """

# VkBufferCopy
# https://docs.vulkan.org/refpages/latest/refpages/source/VkBufferCopy.html
@dataclass
class VkBufferCopy:
    srcOffset: VkDeviceSize
    dstOffset: VkDeviceSize
    size: VkDeviceSize

def vkCmdCopyBuffer(
    commandBuffer: VkCommandBuffer,
    srcBuffer: VkBuffer,
    dstBuffer: VkBuffer,
    regionCount: int,
    pRegions: list[VkBufferCopy],
) -> None:
    """
    vkCmdCopyBuffer copies data between buffer regions.
    """

# VkImageCopy
# https://docs.vulkan.org/refpages/latest/refpages/source/VkImageCopy.html
@dataclass
class VkImageCopy:
    srcSubresource: VkImageSubresourceLayers
    srcOffset: VkOffset3D
    dstSubresource: VkImageSubresourceLayers
    dstOffset: VkOffset3D
    extent: VkExtent3D

def vkCmdCopyImage(
    commandBuffer: VkCommandBuffer,
    srcImage: VkImage,
    srcImageLayout: VkImageLayout,
    dstImage: VkImage,
    dstImageLayout: VkImageLayout,
    regionCount: int,
    pRegions: list[VkImageCopy],
) -> None:
    """
    vkCmdCopyImage copies data between image regions.
    """

# VkImageSubresourceLayers
# https://docs.vulkan.org/refpages/latest/refpages/source/VkImageSubresourceLayers.html
@dataclass
class VkImageSubresourceLayers:
    aspectMask: VkImageAspectFlags
    mipLevel: int
    baseArrayLayer: int
    layerCount: int

# VkBufferImageCopy
# https://docs.vulkan.org/refpages/latest/refpages/source/VkBufferImageCopy.html
@dataclass
class VkBufferImageCopy:
    bufferOffset: VkDeviceSize
    bufferRowLength: int
    bufferImageHeight: int
    imageSubresource: VkImageSubresourceLayers
    imageOffset: VkOffset3D
    imageExtent: VkExtent3D

def vkCmdCopyBufferToImage(
    commandBuffer: VkCommandBuffer,
    srcBuffer: VkBuffer,
    dstImage: VkImage,
    dstImageLayout: VkImageLayout,
    regionCount: int,
    pRegions: list[VkBufferImageCopy],
) -> None:
    """
    vkCmdCopyBufferToImage copies data from a buffer into an image.
    """

# vkCmdCopyImageToBuffer
# https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdCopyImageToBuffer.html
def vkCmdCopyImageToBuffer(
    commandBuffer: VkCommandBuffer,
    srcImage: VkImage,
    srcImageLayout: VkImageLayout,
    dstBuffer: VkBuffer,
    regionCount: int,
    pRegions: list[VkBufferImageCopy],
) -> None:
    """
    vkCmdCopyImageToBuffer copies data from an image into a buffer.
    """

# VkDependencyFlags
# https://docs.vulkan.org/refpages/latest/refpages/source/VkDependencyFlagBits.html
VkDependencyFlags: TypeAlias = VkFlags
VkDependencyFlagBits: TypeAlias = int

# VkAccessFlags
# https://docs.vulkan.org/refpages/latest/refpages/source/VkAccessFlagBits.html
VkAccessFlags: TypeAlias = VkFlags
VkAccessFlagBits: TypeAlias = int
VK_ACCESS_INDIRECT_COMMAND_READ_BIT: VkAccessFlagBits = 0x00000001
VK_ACCESS_INDEX_READ_BIT: VkAccessFlagBits = 0x00000002
VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT: VkAccessFlagBits = 0x00000004
VK_ACCESS_UNIFORM_READ_BIT: VkAccessFlagBits = 0x00000008
VK_ACCESS_INPUT_ATTACHMENT_READ_BIT: VkAccessFlagBits = 0x00000010
VK_ACCESS_SHADER_READ_BIT: VkAccessFlagBits = 0x00000020
VK_ACCESS_SHADER_WRITE_BIT: VkAccessFlagBits = 0x00000040
VK_ACCESS_COLOR_ATTACHMENT_READ_BIT: VkAccessFlagBits = 0x00000080
VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT: VkAccessFlagBits = 0x00000100
VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT: VkAccessFlagBits = 0x00000200
VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT: VkAccessFlagBits = 0x00000400
VK_ACCESS_TRANSFER_READ_BIT: VkAccessFlagBits = 0x00000800
VK_ACCESS_TRANSFER_WRITE_BIT: VkAccessFlagBits = 0x00001000
VK_ACCESS_HOST_READ_BIT: VkAccessFlagBits = 0x00002000
VK_ACCESS_HOST_WRITE_BIT: VkAccessFlagBits = 0x00004000
VK_ACCESS_MEMORY_READ_BIT: VkAccessFlagBits = 0x00008000
VK_ACCESS_MEMORY_WRITE_BIT: VkAccessFlagBits = 0x00010000

# VkMemoryBarrier
# https://docs.vulkan.org/refpages/latest/refpages/source/VkMemoryBarrier.html
@dataclass
class VkMemoryBarrier:
    srcAccessMask: VkAccessFlags
    dstAccessMask: VkAccessFlags

# VkBufferMemoryBarrier
# https://docs.vulkan.org/refpages/latest/refpages/source/VkBufferMemoryBarrier.html
@dataclass
class VkBufferMemoryBarrier:
    srcAccessMask: VkAccessFlags
    dstAccessMask: VkAccessFlags
    srcQueueFamilyIndex: int
    dstQueueFamilyIndex: int
    buffer: VkBuffer
    offset: VkDeviceSize
    size: VkDeviceSize

# VkImageMemoryBarrier
# https://docs.vulkan.org/refpages/latest/refpages/source/VkImageMemoryBarrier.html
@dataclass
class VkImageMemoryBarrier:
    srcAccessMask: VkAccessFlags
    dstAccessMask: VkAccessFlags
    oldLayout: VkImageLayout
    newLayout: VkImageLayout
    srcQueueFamilyIndex: int
    dstQueueFamilyIndex: int
    image: VkImage
    subresourceRange: VkImageSubresourceRange

# vkCmdPipelineBarrier
# https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdPipelineBarrier.html
def vkCmdPipelineBarrier(
    commandBuffer: VkCommandBuffer,
    srcStageMask: VkPipelineStageFlags,
    dstStageMask: VkPipelineStageFlags,
    dependencyFlags: VkDependencyFlags,
    memoryBarrierCount: int,
    pMemoryBarriers: list[VkMemoryBarrier] | None,
    bufferMemoryBarrierCount: int,
    pBufferMemoryBarriers: list[VkBufferMemoryBarrier] | None,
    imageMemoryBarrierCount: int,
    pImageMemoryBarriers: list[VkImageMemoryBarrier] | None,
) -> None:
    """
    vkCmdPipelineBarrier inserts a pipeline barrier into the command buffer.
    """

# Pipeline and drawing commands
#

# VkRenderingInfo
# https://docs.vulkan.org/refpages/latest/refpages/source/VkRenderingInfo.html
@dataclass
class VkRenderingInfo:
    flags: int
    renderArea: VkRect2D
    layerCount: int
    viewMask: int
    colorAttachmentCount: int
    pColorAttachments: list[VkRenderingAttachmentInfo] | None
    pDepthAttachment: VkRenderingAttachmentInfo | None
    pStencilAttachment: VkRenderingAttachmentInfo | None

# vkCmdBeginRendering
# https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdBeginRendering.html
def vkCmdBeginRendering(
    commandBuffer: "VkCommandBuffer",
    pRenderingInfo: VkRenderingInfo,
) -> None: ...

# vkCmdEndRendering
# https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdEndRendering.html
def vkCmdEndRendering(commandBuffer: "VkCommandBuffer") -> None: ...

# vkCmdBindPipeline
# http://docs.vulkan.org/refpages/latest/refpages/source/vkCmdBindPipeline.html
def vkCmdBindPipeline(
    commandBuffer: "VkCommandBuffer",
    pipelineBindPoint: VkPipelineBindPoint,
    pipeline: VkPipeline,
) -> None: ...

# vkCmdBindDescriptorSets
# http://docs.vulkan.org/refpages/latest/refpages/source/vkCmdBindDescriptorSets
def vkCmdBindDescriptorSets(
    commandBuffer: "VkCommandBuffer",
    pipelineBindPoint: VkPipelineBindPoint,
    layout: VkPipelineLayout,
    firstSet: int,
    descriptorSetCount: int,
    pDescriptorSets: list[VkDescriptorSet],
    dynamicOffsetCount: int,
    pDynamicOffsets: list[int] | None,
) -> None: ...

# vkCmdBindVertexBuffers
# http://docs.vulkan.org/refpages/latest/refpages/source/vkCmdBindVertexBuffers
def vkCmdDraw(
    commandBuffer: "VkCommandBuffer",
    vertexCount: int,
    instanceCount: int,
    firstVertex: int,
    firstInstance: int,
) -> None: ...

#
# VkFence
#

# VkFence
# https://docs.vulkan.org/refpages/latest/refpages/source/VkFence.html
class VkFence(OpaqueResourceHandle): ...

# VkFenceCreateFlags
# https://docs.vulkan.org/refpages/latest/refpages/source/VkFenceCreateFlags.html
VkFenceCreateFlags: TypeAlias = VkFlags
VkFenceCreateFlagBits: TypeAlias = int
VK_FENCE_CREATE_SIGNALED_BIT: VkFenceCreateFlagBits = 0x00000001

# VkFenceCreateInfo
# https://docs.vulkan.org/refpages/latest/refpages/source/VkFenceCreateInfo.html
@dataclass
class VkFenceCreateInfo:
    flags: int = 0

# vkCreateFence
# https://docs.vulkan.org/refpages/latest/refpages/source/vkCreateFence.html
def vkCreateFence(
    device: VkDevice,
    pCreateInfo: VkFenceCreateInfo,
    pAllocator: Any | None,
) -> VkFence: ...

# vkDestroyFence
# https://docs.vulkan.org/refpages/latest/refpages/source/vkDestroyFence.html
def vkDestroyFence(
    device: VkDevice,
    fence: VkFence,
    pAllocator: Any | None,
) -> None: ...

#
# VkSemaphore
#

# VkSemaphore
# https://docs.vulkan.org/refpages/latest/refpages/source/VkSemaphore.html
class VkSemaphore(OpaqueResourceHandle): ...

# VkSemaphoreCreateFlags
# https://docs.vulkan.org/refpages/latest/refpages/source/VkSemaphoreCreateFlags.html
@dataclass
class VkSemaphoreCreateInfo:
    flags: int = 0

# vkCreateSemaphore
# https://docs.vulkan.org/refpages/latest/refpages/source/vkCreateSemaphore.html
def vkCreateSemaphore(
    device: VkDevice,
    pCreateInfo: VkSemaphoreCreateInfo,
    pAllocator: Any | None,
) -> VkSemaphore: ...

# vkDestroySemaphore
# https://docs.vulkan.org/refpages/latest/refpages/source/vkDestroySemaphore.html
def vkDestroySemaphore(
    device: VkDevice,
    semaphore: VkSemaphore,
    pAllocator: Any | None,
) -> None: ...

#
# VkQueue
#

# VkQueue
# https://docs.vulkan.org/refpages/latest/refpages/source/VkQueue.html
class VkQueue(OpaqueResourceHandle): ...

# vkGetDeviceQueue
# https://docs.vulkan.org/refpages/latest/refpages/source/vkGetDeviceQueue.html
def vkGetDeviceQueue(
    device: VkDevice, queueFamilyIndex: int, queueIndex: int
) -> VkQueue: ...

# VkPipelineStageFlags
# https://docs.vulkan.org/refpages/latest/refpages/source/VkPipelineStageFlagBits.html
VkPipelineStageFlags: TypeAlias = VkFlags
VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT: int = 0x00000001
VK_PIPELINE_STAGE_ALL_COMMANDS_BIT: int = 0xFFFFFFFF

# VkSubmitInfo
# https://docs.vulkan.org/refpages/latest/refpages/source/VkSubmitInfo.html
@dataclass
class VkSubmitInfo:
    waitSemaphoreCount: int
    pWaitSemaphores: list[VkSemaphore] | None
    pWaitDstStageMask: list[VkPipelineStageFlags] | None
    commandBufferCount: int
    pCommandBuffers: list[VkCommandBuffer] | None
    signalSemaphoreCount: int
    pSignalSemaphores: list[VkSemaphore] | None

# vkQueueSubmit
# https://docs.vulkan.org/refpages/latest/refpages/source/vkQueueSubmit.html
def vkQueueSubmit(
    queue: VkQueue,
    submitCount: int,
    pSubmits: list[VkSubmitInfo],
    fence: VkFence | None,
) -> None: ...

# vkWaitForFences
# https://docs.vulkan.org/refpages/latest/refpages/source/vkWaitForFences.html
def vkWaitForFences(
    device: VkDevice,
    fenceCount: int,
    pFences: list[VkFence],
    waitAll: bool,
    timeout: int,
) -> None: ...

# vkResetFences
# https://docs.vulkan.org/refpages/latest/refpages/source/vkResetFences.html
def vkResetFences(
    device: VkDevice,
    fenceCount: int,
    pFences: list[VkFence],
) -> None: ...

#
# VkShaderModule
#

# VkShaderModule
# https://docs.vulkan.org/refpages/latest/refpages/source/VkShaderModule.html
class VkShaderModule(OpaqueResourceHandle): ...

# VkShaderModuleCreateFlags
# https://docs.vulkan.org/refpages/latest/refpages/source/VkShaderModuleCreateFlags.html
VkShaderModuleCreateFlags: TypeAlias = VkFlags
VkShaderModuleCreateFlagBits: TypeAlias = int

# VkShaderModuleCreateInfo
# https://docs.vulkan.org/refpages/latest/refpages/source/VkShaderModuleCreateInfo
@dataclass
class VkShaderModuleCreateInfo:
    flags: VkShaderModuleCreateFlags
    codeSize: int
    pCode: bytes

# vkCreateShaderModule
# https://docs.vulkan.org/refpages/latest/refpages/source/vkCreateShaderModule.html
def vkCreateShaderModule(
    device: VkDevice,
    pCreateInfo: VkShaderModuleCreateInfo,
    pAllocator: Any,
) -> VkShaderModule:
    """
    vkCreateShaderModule creates a new shader module object.
    """

# vkDestroyShaderModule
# https://docs.vulkan.org/refpages/latest/refpages/source/vkDestroyShaderModule.html
def vkDestroyShaderModule(
    device: VkDevice,
    shaderModule: VkShaderModule,
    pAllocator: Any,
) -> None:
    """
    vkDestroyShaderModule destroys a shader module object.
    """

#
# VkPipeline
#

# VkPipeline
# https://docs.vulkan.org/refpages/latest/refpages/source/VkPipeline.html
class VkPipeline(OpaqueResourceHandle): ...

# VkPipelineCache
# https://docs.vulkan.org/refpages/latest/refpages/source/VkPipelineCache.html
class VkPipelineCache(OpaqueResourceHandle): ...

# VkPipelineLayout
# https://docs.vulkan.org/refpages/latest/refpages/source/VkPipelineLayout.html
class VkPipelineLayout(OpaqueResourceHandle): ...

# VkDescriptorSetLayout
# https://docs.vulkan.org/refpages/latest/refpages/source/VkDescriptorSetLayout.html
class VkDescriptorSetLayout(OpaqueResourceHandle): ...

# VkDescriptorSet
# https://docs.vulkan.org/refpages/latest/refpages/source/VkDescriptorSet.html
class VkDescriptorSet(OpaqueResourceHandle): ...

# VkRenderPass
# https://docs.vulkan.org/refpages/latest/refpages/source/VkRenderPass.html
class VkRenderPass(OpaqueResourceHandle): ...

# VkBlendFactor
# https://docs.vulkan.org/refpages/latest/refpages/source/VkBlendFactor.html
VkBlendFactor: TypeAlias = int
VK_BLEND_FACTOR_ZERO: VkBlendFactor = 0
VK_BLEND_FACTOR_ONE: VkBlendFactor = 1
VK_BLEND_FACTOR_SRC_COLOR: VkBlendFactor = 2
VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR: VkBlendFactor = 3
VK_BLEND_FACTOR_DST_COLOR: VkBlendFactor = 4
VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR: VkBlendFactor = 5
VK_BLEND_FACTOR_SRC_ALPHA: VkBlendFactor = 6
VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA: VkBlendFactor = 7
VK_BLEND_FACTOR_DST_ALPHA: VkBlendFactor = 8
VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA: VkBlendFactor = 9
VK_BLEND_FACTOR_CONSTANT_COLOR: VkBlendFactor = 10
VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR: VkBlendFactor = 11
VK_BLEND_FACTOR_CONSTANT_ALPHA: VkBlendFactor = 12
VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA: VkBlendFactor = 13
VK_BLEND_FACTOR_SRC_ALPHA_SATURATE: VkBlendFactor = 14
VK_BLEND_FACTOR_SRC1_COLOR: VkBlendFactor = 15
VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR: VkBlendFactor = 16
VK_BLEND_FACTOR_SRC1_ALPHA: VkBlendFactor = 17
VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA: VkBlendFactor = 18

# VkBlendOp
# https://docs.vulkan.org/refpages/latest/refpages/source/VkBlendOp.html
VkBlendOp: TypeAlias = int
VK_BLEND_OP_ADD: VkBlendOp = 0
VK_BLEND_OP_SUBTRACT: VkBlendOp = 1
VK_BLEND_OP_REVERSE_SUBTRACT: VkBlendOp = 2
VK_BLEND_OP_MIN: VkBlendOp = 3
VK_BLEND_OP_MAX: VkBlendOp = 4

# VkColorComponentFlags
# https://docs.vulkan.org/refpages/latest/refpages/source/VkColorComponentFlagBits.html
VkColorComponentFlags: TypeAlias = VkFlags
VkColorComponentFlagBits: TypeAlias = int
VK_COLOR_COMPONENT_R_BIT: VkColorComponentFlagBits = 0x00000001
VK_COLOR_COMPONENT_G_BIT: VkColorComponentFlagBits = 0x00000002
VK_COLOR_COMPONENT_B_BIT: VkColorComponentFlagBits = 0x00000004
VK_COLOR_COMPONENT_A_BIT: VkColorComponentFlagBits = 0x00000008

# VkCullModeFlags
# https://docs.vulkan.org/refpages/latest/refpages/source/VkCullModeFlagBits.html
VkCullModeFlags: TypeAlias = VkFlags
VkCullModeFlagBits: TypeAlias = int
VK_CULL_MODE_NONE: int = 0
VK_CULL_MODE_FRONT_BIT: int = 0x00000001
VK_CULL_MODE_BACK_BIT: int = 0x00000002
VK_CULL_MODE_FRONT_AND_BACK: int = 0x00000003

# VkPipelineColorBlendAttachmentState
# https://docs.vulkan.org/refpages/latest/refpages/source/VkPipelineColorBlendAttachment
@dataclass
class VkPipelineColorBlendAttachmentState:
    blendEnable: bool
    srcColorBlendFactor: VkBlendFactor
    dstColorBlendFactor: VkBlendFactor
    colorBlendOp: VkBlendOp
    srcAlphaBlendFactor: VkBlendFactor
    dstAlphaBlendFactor: VkBlendFactor
    alphaBlendOp: VkBlendOp
    colorWriteMask: VkColorComponentFlags

# VkPipelineCreateFlags
# https://docs.vulkan.org/refpages/latest/refpages/source/VkPipelineCreateFlagBits.html
VkPipelineCreateFlags: TypeAlias = VkFlags
VkPipelineCreateFlagBits: TypeAlias = int
VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT: VkPipelineCreateFlagBits = 0x00000001
VK_PIPELINE_CREATE_ALLOW_DERIVATIVES_BIT: VkPipelineCreateFlagBits = 0x00000002
VK_PIPELINE_CREATE_DERIVATIVE_BIT: VkPipelineCreateFlagBits = 0x00000004

# VkPipelineShaderStageCreateFlags
# https://docs.vulkan.org/refpages/latest/refpages/source/VkPipelineShaderStageCreateFlags.html
VkPipelineShaderStageCreateFlags: TypeAlias = VkFlags
VkPipelineShaderStageCreateFlagBits: TypeAlias = int
VK_PIPELINE_SHADER_STAGE_CREATE_ALLOW_VARYING_SUBGROUP_SIZE_BIT: VkPipelineShaderStageCreateFlagBits = 0x00000001
VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT: VkPipelineShaderStageCreateFlagBits = 0x00000002

# VkShaderStageFlagBits
# https://docs.vulkan.org/refpages/latest/refpages/source/VkShaderStageFlagBits.html
VkShaderStageFlags: TypeAlias = VkFlags
VkShaderStageFlagBits: TypeAlias = int
VK_SHADER_STAGE_VERTEX_BIT: VkShaderStageFlagBits = 0x00000001
VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT: VkShaderStageFlagBits = 0x00000002
VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT: VkShaderStageFlagBits = 0x00000004
VK_SHADER_STAGE_GEOMETRY_BIT: VkShaderStageFlagBits = 0x00000008
VK_SHADER_STAGE_FRAGMENT_BIT: VkShaderStageFlagBits = 0x00000010
VK_SHADER_STAGE_COMPUTE_BIT: VkShaderStageFlagBits = 0x00000020
VK_SHADER_STAGE_ALL_GRAPHICS: VkShaderStageFlagBits = 0x0000001F
VK_SHADER_STAGE_ALL: VkShaderStageFlagBits = 0x7FFFFFFF

# VkPrimitiveTopology
# https://docs.vulkan.org/refpages/latest/refpages/source/VkPrimitiveTopology.html
VkPrimitiveTopology: TypeAlias = int
VK_PRIMITIVE_TOPOLOGY_POINT_LIST: VkPrimitiveTopology = 0
VK_PRIMITIVE_TOPOLOGY_LINE_LIST: VkPrimitiveTopology = 1
VK_PRIMITIVE_TOPOLOGY_LINE_STRIP: VkPrimitiveTopology = 2
VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST: VkPrimitiveTopology = 3
VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP: VkPrimitiveTopology = 4
VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN: VkPrimitiveTopology = 5
VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY: VkPrimitiveTopology = 6
VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY: VkPrimitiveTopology = 7
VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY: VkPrimitiveTopology = 8
VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY: VkPrimitiveTopology = 9
VK_PRIMITIVE_TOPOLOGY_PATCH_LIST: VkPrimitiveTopology = 10

# VkPipelineBindPoint
# https://docs.vulkan.org/refpages/latest/refpages/source/VkPipelineBindPoint.html
VkPipelineBindPoint: TypeAlias = int
VK_PIPELINE_BIND_POINT_GRAPHICS: VkPipelineBindPoint = 0
VK_PIPELINE_BIND_POINT_COMPUTE: VkPipelineBindPoint = 1

# VkFrontFace
# https://docs.vulkan.org/refpages/latest/refpages/source/VkFrontFace.html
VkFrontFace: TypeAlias = int
VK_FRONT_FACE_COUNTER_CLOCKWISE: VkFrontFace = 0
VK_FRONT_FACE_CLOCKWISE: VkFrontFace = 1

# VkPolygonMode
# https://docs.vulkan.org/refpages/latest/refpages/source/VkPolygonMode.html
VkPolygonMode: TypeAlias = int
VK_POLYGON_MODE_FILL: VkPolygonMode = 0
VK_POLYGON_MODE_LINE: VkPolygonMode = 1
VK_POLYGON_MODE_POINT: VkPolygonMode = 2

# VkLogicOp
# https://docs.vulkan.org/refpages/latest/refpages/source/VkLogicOp.html
VkLogicOp: TypeAlias = int
VK_LOGIC_OP_CLEAR: VkLogicOp = 0
VK_LOGIC_OP_AND: VkLogicOp = 1
VK_LOGIC_OP_AND_REVERSE: VkLogicOp = 2
VK_LOGIC_OP_COPY: VkLogicOp = 3
VK_LOGIC_OP_AND_INVERTED: VkLogicOp = 4
VK_LOGIC_OP_NO_OP: VkLogicOp = 5
VK_LOGIC_OP_XOR: VkLogicOp = 6
VK_LOGIC_OP_OR: VkLogicOp = 7
VK_LOGIC_OP_NOR: VkLogicOp = 8
VK_LOGIC_OP_EQUIVALENT: VkLogicOp = 9
VK_LOGIC_OP_INVERT: VkLogicOp = 10
VK_LOGIC_OP_OR_REVERSE: VkLogicOp = 11
VK_LOGIC_OP_COPY_INVERTED: VkLogicOp = 12
VK_LOGIC_OP_OR_INVERTED: VkLogicOp = 13
VK_LOGIC_OP_NAND: VkLogicOp = 14
VK_LOGIC_OP_SET: VkLogicOp = 15

# VkDynamicState
# https://docs.vulkan.org/refpages/latest/refpages/source/VkDynamicState.html
VkDynamicState: TypeAlias = int
VK_DYNAMIC_STATE_VIEWPORT: VkDynamicState = 0
VK_DYNAMIC_STATE_SCISSOR: VkDynamicState = 1
VK_DYNAMIC_STATE_LINE_WIDTH: VkDynamicState = 2
VK_DYNAMIC_STATE_DEPTH_BIAS: VkDynamicState = 3
VK_DYNAMIC_STATE_BLEND_CONSTANTS: VkDynamicState = 4
VK_DYNAMIC_STATE_DEPTH_BOUNDS: VkDynamicState = 5
VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK: VkDynamicState = 6
VK_DYNAMIC_STATE_STENCIL_WRITE_MASK: VkDynamicState = 7
VK_DYNAMIC_STATE_STENCIL_REFERENCE: VkDynamicState = 8

# VkPipelineShaderStageCreateInfo
# https://docs.vulkan.org/refpages/latest/refpages/source/VkPipelineShaderStageCreate
@dataclass
class VkPipelineShaderStageCreateInfo:
    flags: VkPipelineShaderStageCreateFlags
    stage: VkShaderStageFlagBits
    module: VkShaderModule
    pName: str
    pSpecializationInfo: Any | None

# VkPipelineVertexInputStateCreateInfo
# https://docs.vulkan.org/refpages/latest/refpages/source/VkPipelineVertexInputStateCreateInfo.html
@dataclass
class VkPipelineVertexInputStateCreateInfo:
    flags: int
    vertexBindingDescriptionCount: int
    pVertexBindingDescriptions: Any | None
    vertexAttributeDescriptionCount: int
    pVertexAttributeDescriptions: Any | None

# VkPipelineInputAssemblyStateCreateInfo
# https://docs.vulkan.org/refpages/latest/refpages/source/VkPipelineInputAssemblyStateCreateInfo.html
@dataclass
class VkPipelineInputAssemblyStateCreateInfo:
    flags: int
    topology: VkPrimitiveTopology
    primitiveRestartEnable: bool

# VkPipelineTessellationStateCreateInfo
# https://docs.vulkan.org/refpages/latest/refpages/source/VkPipelineTessellationStateCreateInfo.html
@dataclass
class VkPipelineTessellationStateCreateInfo:
    flags: int
    patchControlPoints: int

# VkPipelineViewportStateCreateInfo
# https://docs.vulkan.org/refpages/latest/refpages/source/VkPipelineViewportStateCreateInfo.html
@dataclass
class VkPipelineViewportStateCreateInfo:
    flags: int
    viewportCount: int
    pViewports: Any | None
    scissorCount: int
    pScissors: Any | None

# VkPipelineRasterizationStateCreateInfo
# https://docs.vulkan.org/refpages/latest/refpages/source/VkPipelineRasterizationStateCreateInfo.html
@dataclass
class VkPipelineRasterizationStateCreateInfo:
    flags: int
    depthClampEnable: bool
    rasterizerDiscardEnable: bool
    polygonMode: VkPolygonMode
    cullMode: VkCullModeFlags
    frontFace: VkFrontFace
    depthBiasEnable: bool
    depthBiasConstantFactor: float
    depthBiasClamp: float
    depthBiasSlopeFactor: float
    lineWidth: float

# VkPipelineMultisampleStateCreateInfo
# https://docs.vulkan.org/refpages/latest/refpages/source/VkPipelineMultisampleStateCreateInfo.html
@dataclass
class VkPipelineMultisampleStateCreateInfo:
    flags: int
    rasterizationSamples: VkSampleCountFlagBits
    sampleShadingEnable: bool
    minSampleShading: float
    pSampleMask: Any | None
    alphaToCoverageEnable: bool
    alphaToOneEnable: bool

# VkPipelineDepthStencilStateCreateInfo
# https://docs.vulkan.org/refpages/latest/refpages/source/VkPipelineDepthStencilStateCreateInfo.html
@dataclass
class VkPipelineDepthStencilStateCreateInfo:
    flags: int
    depthTestEnable: bool
    depthWriteEnable: bool
    depthCompareOp: VkCompareOp
    depthBoundsTestEnable: bool
    stencilTestEnable: bool
    front: Any
    back: Any
    minDepthBounds: float
    maxDepthBounds: float

# VkPipelineColorBlendStateCreateInfo
# https://docs.vulkan.org/refpages/latest/refpages/source/VkPipelineColorBlendStateCreateInfo.html
@dataclass
class VkPipelineColorBlendStateCreateInfo:
    flags: int
    logicOpEnable: bool
    logicOp: VkLogicOp
    attachmentCount: int
    pAttachments: list[VkPipelineColorBlendAttachmentState]
    blendConstants: list[float]

# VkPipelineDynamicStateCreateInfo
# https://docs.vulkan.org/refpages/latest/refpages/source/VkPipelineDynamicStateCreateInfo.html
@dataclass
class VkPipelineDynamicStateCreateInfo:
    flags: int
    dynamicStateCount: int
    pDynamicStates: list[VkDynamicState] | None

# VkPipelineRenderingCreateInfo
# https://docs.vulkan.org/refpages/latest/refpages/source/VkPipelineRenderingCreateInfo.html
@dataclass
class VkPipelineRenderingCreateInfo:
    viewMask: int = 0
    colorAttachmentCount: int = 0
    pColorAttachmentFormats: list[VkFormat] | None = None
    depthAttachmentFormat: VkFormat | None = None
    stencilAttachmentFormat: VkFormat | None = None

# VkGraphicsPipelineCreateInfo
# https://docs.vulkan.org/refpages/latest/refpages/source/VkGraphicsPipelineCreateInfo.html
@dataclass
class VkGraphicsPipelineCreateInfo:
    pNext: VkPipelineRenderingCreateInfo | None
    flags: VkPipelineCreateFlags
    stageCount: int
    pStages: list[VkPipelineShaderStageCreateInfo]
    pVertexInputState: VkPipelineVertexInputStateCreateInfo | None
    pInputAssemblyState: VkPipelineInputAssemblyStateCreateInfo | None
    pTessellationState: VkPipelineTessellationStateCreateInfo | None
    pViewportState: VkPipelineViewportStateCreateInfo | None
    pRasterizationState: VkPipelineRasterizationStateCreateInfo | None
    pMultisampleState: VkPipelineMultisampleStateCreateInfo | None
    pDepthStencilState: VkPipelineDepthStencilStateCreateInfo | None
    pColorBlendState: VkPipelineColorBlendStateCreateInfo | None
    pDynamicState: VkPipelineDynamicStateCreateInfo | None
    layout: VkPipelineLayout
    renderPass: VkRenderPass | None  # pass None for dynamic rendering
    subpass: int
    basePipelineHandle: VkPipeline | None
    basePipelineIndex: int

# vkCreateGraphicsPipelines
# https://docs.vulkan.org/refpages/latest/refpages/source/vkCreateGraphicsPipelines.html
def vkCreateGraphicsPipelines(
    device: VkDevice,
    pipelineCache: VkPipelineCache | None,
    createInfoCount: int,
    pCreateInfos: list[VkGraphicsPipelineCreateInfo],
    pAllocator: Any | None,
) -> list[VkPipeline]:
    """
    vkCreateGraphicsPipelines creates graphics pipeline objects.
    """

# vkDestroyPipeline
# https://docs.vulkan.org/refpages/latest/refpages/source/vkDestroyPipeline.html
def vkDestroyPipeline(
    device: VkDevice,
    pipeline: VkPipeline,
    pAllocator: Any | None,
) -> None:
    """
    vkDestroyPipeline destroys a pipeline object.
    """

#
# VkPipelineLayout
#

# VkPipelineLayoutCreateFlags
# https://docs.vulkan.org/refpages/latest/refpages/source/VkPipelineLayoutCreateFlags.html
VkPipelineLayoutCreateFlags: TypeAlias = VkFlags
VkPipelineLayoutCreateFlagBits: TypeAlias = int

# VkPipelineLayoutCreateInfo
# https://docs.vulkan.org/refpages/latest/refpages/source/VkPipelineLayoutCreateInfo.html
@dataclass
class VkPipelineLayoutCreateInfo:
    flags: VkPipelineLayoutCreateFlags
    setLayoutCount: int
    pSetLayouts: list[VkDescriptorSetLayout] | None
    pushConstantRangeCount: int
    pPushConstantRanges: Any | None

# vkCreatePipelineLayout
# https://docs.vulkan.org/refpages/latest/refpages/source/vkCreatePipelineLayout.html
def vkCreatePipelineLayout(
    device: VkDevice,
    pCreateInfo: VkPipelineLayoutCreateInfo,
    pAllocator: Any | None,
) -> VkPipelineLayout:
    """
    vkCreatePipelineLayout creates a pipeline layout object.
    """

# vkDestroyPipelineLayout
# https://docs.vulkan.org/refpages/latest/refpages/source/vkDestroyPipelineLayout.html
def vkDestroyPipelineLayout(
    device: VkDevice,
    pipelineLayout: VkPipelineLayout,
    pAllocator: Any | None,
) -> None:
    """
    vkDestroyPipelineLayout destroys a pipeline layout object.
    """

#
# VkSwapchainKHR
#

# VkSwapchainKHR
# https://docs.vulkan.org/refpages/latest/refpages/source/VkSwapchainKHR.html
class VkSwapchainKHR(OpaqueResourceHandle): ...

# VkSwapchainCreateFlags
# https://docs.vulkan.org/refpages/latest/refpages/source/VkSwapchainCreateFlags.html
VkSwapchainCreateFlags: TypeAlias = VkFlags
VkSwapchainCreateFlagBits: TypeAlias = int

# VkColorSpaceKHR
# https://docs.vulkan.org/refpages/latest/refpages/source/VkColorSpaceKHR.html
VkColorSpaceKHR: TypeAlias = int
VK_COLOR_SPACE_SRGB_NONLINEAR_KHR: VkColorSpaceKHR = 0

# VkSurfaceTransformFlagBitsKHR
# https://docs.vulkan.org/refpages/latest/refpages/source/VkSurfaceTransformFlagBitsKHR.html
VkSurfaceTransformFlagBitsKHR: TypeAlias = int
VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR: VkSurfaceTransformFlagBitsKHR = 0x00000001
VK_SURFACE_TRANSFORM_ROTATE_90_BIT_KHR: VkSurfaceTransformFlagBitsKHR = 0x00000002
VK_SURFACE_TRANSFORM_ROTATE_180_BIT_KHR: VkSurfaceTransformFlagBitsKHR = 0x00000004
VK_SURFACE_TRANSFORM_ROTATE_270_BIT_KHR: VkSurfaceTransformFlagBitsKHR = 0x00000008
VK_SURFACE_TRANSFORM_HORIZONTAL_MIRROR_BIT_KHR: VkSurfaceTransformFlagBitsKHR = (
    0x00000010
)
VK_SURFACE_TRANSFORM_HORIZONTAL_MIRROR_ROTATE_90_BIT_KHR: VkSurfaceTransformFlagBitsKHR = 0x00000020
VK_SURFACE_TRANSFORM_HORIZONTAL_MIRROR_ROTATE_180_BIT_KHR: VkSurfaceTransformFlagBitsKHR = 0x00000040
VK_SURFACE_TRANSFORM_HORIZONTAL_MIRROR_ROTATE_270_BIT_KHR: VkSurfaceTransformFlagBitsKHR = 0x00000080
VK_SURFACE_TRANSFORM_INHERIT_BIT_KHR: VkSurfaceTransformFlagBitsKHR = 0x00000100

# VkCompositeAlphaFlagBitsKHR
# https://docs.vulkan.org/refpages/latest/refpages/source/VkCompositeAlphaFlagBitsKHR.html
VkCompositeAlphaFlagBitsKHR: TypeAlias = int
VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR: VkCompositeAlphaFlagBitsKHR = 0x00000001
VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR: VkCompositeAlphaFlagBitsKHR = 0x00000002
VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR: VkCompositeAlphaFlagBitsKHR = 0x00000004
VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR: VkCompositeAlphaFlagBitsKHR = 0x00000008

# VkPresentModeKHR
# https://docs.vulkan.org/refpages/latest/refpages/source/VkPresentModeKHR.html
VkPresentModeKHR: TypeAlias = int
VK_PRESENT_MODE_IMMEDIATE_KHR: VkPresentModeKHR = 0
VK_PRESENT_MODE_MAILBOX_KHR: VkPresentModeKHR = 1
VK_PRESENT_MODE_FIFO_KHR: VkPresentModeKHR = 2
VK_PRESENT_MODE_FIFO_RELAXED_KHR: VkPresentModeKHR = 3

# VkSwapchainCreateInfoKHR
# https://docs.vulkan.org/refpages/latest/refpages/source/VkSwapchainCreateInfo
@dataclass
class VkSwapchainCreateInfoKHR:
    flags: VkSwapchainCreateFlags
    surface: VkSurfaceKHR
    minImageCount: int
    imageFormat: VkFormat
    imageColorSpace: VkColorSpaceKHR
    imageExtent: VkExtent2D
    imageArrayLayers: int
    imageUsage: VkImageUsageFlags
    imageSharingMode: VkSharingMode
    queueFamilyIndexCount: int
    pQueueFamilyIndices: list[int] | None
    preTransform: VkSurfaceTransformFlagBitsKHR
    compositeAlpha: VkCompositeAlphaFlagBitsKHR
    presentMode: VkPresentModeKHR
    clipped: bool
    oldSwapchain: VkSwapchainKHR | None

# vkCreateSwapchainKHR
# https://docs.vulkan.org/refpages/latest/refpages/source/vkCreateSwapchainKHR.html
def vkCreateSwapchainKHR(
    device: VkDevice,
    pCreateInfo: VkSwapchainCreateInfoKHR,
    pAllocator: Any | None,
) -> VkSwapchainKHR:
    """
    vkCreateSwapchainKHR creates a new swapchain object.
    """

# vkDestroySwapchainKHR
# https://docs.vulkan.org/refpages/latest/refpages/source/vkDestroySwapchainKHR.html
def vkDestroySwapchainKHR(
    device: VkDevice,
    swapchain: VkSwapchainKHR,
    pAllocator: Any | None,
) -> None:
    """
    vkDestroySwapchainKHR destroys a swapchain object.
    """

# vkGetSwapchainImagesKHR
# https://docs.vulkan.org/refpages/latest/refpages/source/vkGetSwapchainImagesKHR.html
def vkGetSwapchainImagesKHR(
    device: VkDevice,
    swapchain: VkSwapchainKHR,
) -> list[VkImage]:
    """
    vkGetSwapchainImagesKHR retrieves the array of presentable images associated with a swapchain.
    """

# vkAcquireNextImageKHR
# https://docs.vulkan.org/refpages/latest/refpages/source/vkAcquireNextImageKHR.html
def vkAcquireNextImageKHR(
    device: VkDevice,
    swapchain: VkSwapchainKHR,
    timeout: int,
    semaphore: VkSemaphore | None,
    fence: VkFence | None,
) -> int:
    """
    vkAcquireNextImageKHR acquires the next available presentable image.
    """

# VkPresentInfoKHR
# https://docs.vulkan.org/refpages/latest/refpages/source/VkPresentInfoKHR.html
@dataclass
class VkPresentInfoKHR:
    waitSemaphoreCount: int
    pWaitSemaphores: list[VkSemaphore] | None
    swapchainCount: int
    pSwapchains: list[VkSwapchainKHR]
    pImageIndices: list[int]
    pResults: list[int] | None

# vkQueuePresentKHR
# https://docs.vulkan.org/refpages/latest/refpages/source/vkQueuePresentKHR.html
def vkQueuePresentKHR(
    queue: VkQueue,
    pPresentInfo: VkPresentInfoKHR,
):
    """
    vkQueuePresentKHR queues an image for presentation.
    """
