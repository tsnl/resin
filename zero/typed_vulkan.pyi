from enum import IntEnum
from typing import Any, List, Optional, Sequence, TypeAlias
from dataclasses import dataclass

#
# Common:
#

class OpaqueResourceHandle:
    pass

VkDeviceSize: TypeAlias = int
VkFlags: TypeAlias = int

@dataclass
class VkExtent3D:
    width: int
    height: int
    depth: int

#
# VkInstance
#

VkInstance: TypeAlias = OpaqueResourceHandle

@dataclass
class VkApplicationInfo:
    pApplicationName: str | None = None
    pEngineName: str | None = None
    apiVersion: int = 0

@dataclass
class VkInstanceCreateInfo:
    pApplicationInfo: VkApplicationInfo | None
    enabledLayerCount: int = 0
    ppEnabledLayerNames: Sequence[str] = ()
    enabledExtensionCount: int = 0
    ppEnabledExtensionNames: Sequence[str] = ()

@dataclass
class VkPhysicalDeviceProperties:
    apiVersion: int
    driverVersion: int
    vendorID: int
    deviceID: int
    deviceType: int
    deviceName: str
    limits: VkPhysicalDeviceLimits

def vkCreateInstance(
    pCreateInfo: VkInstanceCreateInfo,
    pAllocator: Optional[Any] = None,
) -> VkInstance:
    """
    vkCreateInstance creates a Vulkan instance.
    """

def vkEnumeratePhysicalDevices(instance: VkInstance) -> List[VkPhysicalDevice]:
    """
    vkEnumeratePhysicalDevices enumerates the physical devices available on the system.
    """

#
# Physical Devices
#

VkPhysicalDevice: TypeAlias = OpaqueResourceHandle

def vkGetPhysicalDeviceProperties(
    physicalDevice: VkPhysicalDevice,
) -> VkPhysicalDeviceProperties:
    """
    vkGetPhysicalDeviceProperties retrieves the properties of a physical device.
    """

def vkGetPhysicalDeviceQueueFamilyProperties(
    physicalDevice: VkPhysicalDevice,
) -> list["VkQueueFamilyProperties"]:
    """
    vkGetPhysicalDeviceQueueFamilyProperties retrieves the queue family properties of a physical device.
    """

VkSampleCountFlags: TypeAlias = VkFlags

class VkPhysicalDeviceLimits:
    """
    Structure specifying physical device limits.
    """

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

VkPhysicalDeviceType: TypeAlias = int
VK_PHYSICAL_DEVICE_TYPE_OTHER: VkPhysicalDeviceType = 0
VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: VkPhysicalDeviceType = 1
VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: VkPhysicalDeviceType = 2
VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: VkPhysicalDeviceType = 3
VK_PHYSICAL_DEVICE_TYPE_CPU: VkPhysicalDeviceType = 4

#
# VkDevice
#

VkDevice: TypeAlias = OpaqueResourceHandle
VkDeviceCreateFlags: TypeAlias = VkFlags

@dataclass
class VkDeviceCreateInfo:
    """
    VkStructureType                    sType;
    const void*                        pNext;
    VkDeviceCreateFlags                flags;
    uint32_t                           queueCreateInfoCount;
    const VkDeviceQueueCreateInfo*     pQueueCreateInfos;
    // enabledLayerCount is legacy and should not be used
    uint32_t                           enabledLayerCount;
    // ppEnabledLayerNames is legacy and should not be used
    const char* const*                 ppEnabledLayerNames;
    uint32_t                           enabledExtensionCount;
    const char* const*                 ppEnabledExtensionNames;
    const VkPhysicalDeviceFeatures*    pEnabledFeatures;
    """

    flags: VkDeviceCreateFlags = 0
    queueCreateInfoCount: int = 0
    pQueueCreateInfos: Sequence[Any] = ()
    enabledExtensionCount: int = 0
    ppEnabledExtensionNames: Sequence[str] = ()
    pEnabledFeatures: Sequence[Any] = ()

@dataclass
class VkPhysicalDeviceFeatures:
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

#
# VkQueue
#

VkQueueFlags: TypeAlias = VkFlags
VkQueueFlagBits: TypeAlias = int
VK_QUEUE_GRAPHICS_BIT: VkQueueFlagBits = 0x00000001
VK_QUEUE_COMPUTE_BIT: VkQueueFlagBits = 0x00000002
VK_QUEUE_TRANSFER_BIT: VkQueueFlagBits = 0x00000004

class VkQueueFamilyProperties:
    """
    Structure specifying properties of a queue family.
    """

    queueFlags: VkQueueFlags
    queueCount: int
    timestampValidBits: int
    minImageTransferGranularity: VkExtent3D

#
# VkDevice
#

def vkCreateDevice(
    physicalDevice: VkPhysicalDevice,
    pCreateInfo: VkDeviceCreateInfo,
    pAllocator: Optional[Any] = None,
) -> VkDevice:
    """
    vkCreateDevice creates a logical device from a physical device.
    """
