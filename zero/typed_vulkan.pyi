import cffi
from typing import Any, List, Optional, Sequence, TypeAlias, Callable
from dataclasses import dataclass

#
# Basic:
#

class OpaqueResourceHandle:
    pass

def vkGetInstanceProcAddr(instance: "VkInstance", pName: str) -> Callable:
    """
    vkGetInstanceProcAddr retrieves a function pointer for a Vulkan command.
    """

ffi: cffi.FFI

#
# Common:
#

def VK_MAKE_API_VERSION(variant: int, major: int, minor: int, patch: int) -> int: ...
def vk_decompose_api_version(api_version: int) -> tuple[int, int, int, int]: ...
def vk_api_version_str(api_version: int) -> str: ...

VK_API_VERSION_1_2: int
VK_API_VERSION_1_3: int
VK_API_VERSION_1_4: int

VkDeviceSize: TypeAlias = int
VkFlags: TypeAlias = int

# VkExtent3D
# https://docs.vulkan.org/refpages/latest/refpages/source/VkExtent3D.html
@dataclass
class VkExtent3D:
    width: int
    height: int
    depth: int

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
    ppEnabledLayerNames: Sequence[str] = ()
    enabledExtensionCount: int = 0
    ppEnabledExtensionNames: Sequence[str] = ()
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
# Physical Devices
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

# VkPhysicalDeviceFeatures
# https://docs.vulkan.org/refpages/latest/refpages/source/VkPhysicalDeviceFeatures.html
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

#
# VkQueue
#

# VkQueueFlags
# https://docs.vulkan.org/refpages/latest/refpages/source/VkQueueFlagBits.html
VkQueueFlags: TypeAlias = VkFlags
VkQueueFlagBits: TypeAlias = int
VK_QUEUE_GRAPHICS_BIT: VkQueueFlagBits = 0x00000001
VK_QUEUE_COMPUTE_BIT: VkQueueFlagBits = 0x00000002
VK_QUEUE_TRANSFER_BIT: VkQueueFlagBits = 0x00000004

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
    pQueuePriorities: Sequence[float] | None = None

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
    flags: VkDeviceCreateFlags = 0
    queueCreateInfoCount: int = 0
    pQueueCreateInfos: Sequence[Any] = ()
    enabledExtensionCount: int = 0
    ppEnabledExtensionNames: Sequence[str] = ()
    pEnabledFeatures: Sequence[Any] = ()

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
    pQueueFamilyIndices: Sequence[int] | None
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
    float32: Sequence[float] = (0.0, 0.0, 0.0, 0.0)
    int32: Sequence[int] = (0, 0, 0, 0)
    uint32: Sequence[int] = (0, 0, 0, 0)

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
    """
    // Provided by VK_VERSION_1_3
    typedef struct VkRenderingAttachmentInfo {
        VkStructureType          sType;
        const void*              pNext;
        VkImageView              imageView;
        VkImageLayout            imageLayout;
        VkResolveModeFlagBits    resolveMode;
        VkImageView              resolveImageView;
        VkImageLayout            resolveImageLayout;
        VkAttachmentLoadOp       loadOp;
        VkAttachmentStoreOp      storeOp;
        VkClearValue             clearValue;
    } VkRenderingAttachmentInfo;
    """

    imageView: VkImageView
    imageLayout: VkImageLayout
    resolveMode: VkResolveModeFlagBits = 0
    resolveImageView: VkImageView | None = None
    resolveImageLayout: VkImageLayout = VK_IMAGE_LAYOUT_UNDEFINED
    loadOp: VkAttachmentLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE
    storeOp: VkAttachmentStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE
    clearValue: VkClearValue | None = None
