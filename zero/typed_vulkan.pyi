from typing import Any, List, Optional, Sequence, TypeAlias
from dataclasses import dataclass

#
# Common:
#

VkDeviceSize: TypeAlias = int
VkFlags: TypeAlias = int
VkSampleCountFlags: TypeAlias = VkFlags

#
# VkInstance
#

VkInstance: TypeAlias = Any

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

VkPhysicalDevice: TypeAlias = Any

def vkGetPhysicalDeviceProperties(
    physicalDevice: VkPhysicalDevice,
) -> VkPhysicalDeviceProperties:
    """
    vkGetPhysicalDeviceProperties retrieves the properties of a physical device.
    """

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

#
# VkDevice
#

VkDevice: TypeAlias = Any
