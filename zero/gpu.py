__all__ = [
    # GpuContext
    "GpuContext",
    # GpuPhysicalDevice
    "GpuPhysicalDevice",
    "GpuPhysicalDeviceType",
    "GpuPhysicalDeviceLimits",
    # GpuDevice
    "GpuDevice",
]

from enum import IntEnum

import glfw

from .core import ensure_glfw_init
from .typed_vulkan import (
    # Common:
    VkDeviceSize,
    VkSampleCountFlags,
    # Instance:
    VkInstance,
    vkCreateInstance,
    VkInstanceCreateInfo,
    VkApplicationInfo,
    vkEnumeratePhysicalDevices,
    # Physical devices:
    VkPhysicalDevice,
    vkGetPhysicalDeviceProperties,
    VkPhysicalDeviceProperties,
    VkPhysicalDeviceLimits,
    # Devices:
    VkDevice,
)


#
# GpuContext
#


class GpuContext:
    def __init__(
        self,
        app_name: str = "Unnamed Zero App",
        enable_debug_layer_support: bool = True,
        enable_present_support: bool = True,
    ) -> None:
        super().__init__()

        ensure_glfw_init()

        self._vk_instance = GpuContext._create_instance(
            app_name,
            enable_debug_layer_support,
            enable_present_support,
        )

    @staticmethod
    def _create_instance(
        app_name: str,
        enable_debug_layers: bool,
        enable_present_support: bool,
    ) -> VkInstance:
        layers = []
        extensions = []

        if enable_debug_layers:
            layers.append("VK_LAYER_KHRONOS_validation")
            extensions.append("VK_EXT_debug_utils")
            extensions.append("VK_EXT_debug_report")

        if enable_present_support:
            extensions += glfw.get_required_instance_extensions()

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
            ),
            pAllocator=None,
        )

    def enumerate_physical_devices(self) -> list[GpuPhysicalDevice]:
        return [
            GpuPhysicalDevice(vk_physical_device)
            for vk_physical_device in vkEnumeratePhysicalDevices(self._vk_instance)
        ]


#
# GpuPhysicalDevice
#


class GpuPhysicalDevice:
    def __init__(self, vk_physical_device: VkPhysicalDevice) -> None:
        super().__init__()
        self.vk_physical_device = vk_physical_device
        raw_properties = vkGetPhysicalDeviceProperties(vk_physical_device)
        self.properties = GpuPhysicalDeviceProperties(raw_properties)


class GpuPhysicalDeviceProperties:
    api_version: int
    driver_version: int
    vendor_id: int
    device_id: int
    device_type: GpuPhysicalDeviceType
    device_name: str
    limits: GpuPhysicalDeviceLimits

    def __init__(self, raw: VkPhysicalDeviceProperties) -> None:
        self.api_version = int(raw.apiVersion)
        self.driver_version = int(raw.driverVersion)
        self.vendor_id = int(raw.vendorID)
        self.device_id = int(raw.deviceID)
        self.device_type = GpuPhysicalDeviceType(raw.deviceType)
        self.device_name = str(raw.deviceName)
        self.limits = GpuPhysicalDeviceLimits(raw.limits)


class GpuPhysicalDeviceType(IntEnum):
    OTHER = 0
    INTEGRATED_GPU = 1
    DISCRETE_GPU = 2
    VIRTUAL_GPU = 3
    CPU = 4


class GpuPhysicalDeviceLimits:
    max_image_dimension1d: int
    max_image_dimension2d: int
    max_image_dimension3d: int
    max_image_dimension_cube: int
    max_image_array_layers: int
    max_texel_buffer_elements: int
    max_uniform_buffer_range: int
    max_storage_buffer_range: int
    max_push_constants_size: int
    max_memory_allocation_count: int
    max_sampler_allocation_count: int
    buffer_image_granularity: VkDeviceSize
    sparse_address_space_size: VkDeviceSize
    max_bound_descriptor_sets: int
    max_per_stage_descriptor_samplers: int
    max_per_stage_descriptor_uniform_buffers: int
    max_per_stage_descriptor_storage_buffers: int
    max_per_stage_descriptor_sampled_images: int
    max_per_stage_descriptor_storage_images: int
    max_per_stage_descriptor_input_attachments: int
    max_per_stage_resources: int
    max_descriptor_set_samplers: int
    max_descriptor_set_uniform_buffers: int
    max_descriptor_set_uniform_buffers_dynamic: int
    max_descriptor_set_storage_buffers: int
    max_descriptor_set_storage_buffers_dynamic: int
    max_descriptor_set_sampled_images: int
    max_descriptor_set_storage_images: int
    max_descriptor_set_input_attachments: int
    max_vertex_input_attributes: int
    max_vertex_input_bindings: int
    max_vertex_input_attribute_offset: int
    max_vertex_input_binding_stride: int
    max_vertex_output_components: int
    max_tessellation_generation_level: int
    max_tessellation_patch_size: int
    max_tessellation_control_per_vertex_input_components: int
    max_tessellation_control_per_vertex_output_components: int
    max_tessellation_control_per_patch_output_components: int
    max_tessellation_control_total_output_components: int
    max_tessellation_evaluation_input_components: int
    max_tessellation_evaluation_output_components: int
    max_geometry_shader_invocations: int
    max_geometry_input_components: int
    max_geometry_output_components: int
    max_geometry_output_vertices: int
    max_geometry_total_output_components: int
    max_fragment_input_components: int
    max_fragment_output_attachments: int
    max_fragment_dual_src_attachments: int
    max_fragment_combined_output_resources: int
    max_compute_shared_memory_size: int
    max_compute_work_group_count: tuple[int, int, int]
    max_compute_work_group_invocations: int
    max_compute_work_group_size: tuple[int, int, int]
    sub_pixel_precision_bits: int
    sub_texel_precision_bits: int
    mipmap_precision_bits: int
    max_draw_indexed_index_value: int
    max_draw_indirect_count: int
    max_sampler_lod_bias: float
    max_sampler_anisotropy: float
    max_viewports: int
    max_viewport_dimensions: tuple[int, int]
    viewport_bounds_range: tuple[float, float]
    viewport_sub_pixel_bits: int
    min_memory_map_alignment: int
    min_texel_buffer_offset_alignment: VkDeviceSize
    min_uniform_buffer_offset_alignment: VkDeviceSize
    min_storage_buffer_offset_alignment: VkDeviceSize
    min_texel_offset: int
    max_texel_offset: int
    min_texel_gather_offset: int
    max_texel_gather_offset: int
    min_interpolation_offset: float
    max_interpolation_offset: float
    sub_pixel_interpolation_offset_bits: int
    max_framebuffer_width: int
    max_framebuffer_height: int
    max_framebuffer_layers: int
    framebuffer_color_sample_counts: VkSampleCountFlags
    framebuffer_depth_sample_counts: VkSampleCountFlags
    framebuffer_stencil_sample_counts: VkSampleCountFlags
    framebuffer_no_attachments_sample_counts: VkSampleCountFlags
    max_color_attachments: int
    sampled_image_color_sample_counts: VkSampleCountFlags
    sampled_image_integer_sample_counts: VkSampleCountFlags
    sampled_image_depth_sample_counts: VkSampleCountFlags
    sampled_image_stencil_sample_counts: VkSampleCountFlags
    storage_image_sample_counts: VkSampleCountFlags
    max_sample_mask_words: int
    timestamp_compute_and_graphics: bool
    timestamp_period: float
    max_clip_distances: int
    max_cull_distances: int
    max_combined_clip_and_cull_distances: int
    discrete_queue_priorities: int
    point_size_range: tuple[float, float]
    line_width_range: tuple[float, float]
    point_size_granularity: float
    line_width_granularity: float
    strict_lines: bool
    standard_sample_locations: bool
    optimal_buffer_copy_offset_alignment: VkDeviceSize
    optimal_buffer_copy_row_pitch_alignment: VkDeviceSize
    non_coherent_atom_size: VkDeviceSize

    def __init__(self, raw: VkPhysicalDeviceLimits) -> None:
        self.max_image_dimension1d = int(raw.maxImageDimension1D)
        self.max_image_dimension2d = int(raw.maxImageDimension2D)
        self.max_image_dimension3d = int(raw.maxImageDimension3D)
        self.max_image_dimension_cube = int(raw.maxImageDimensionCube)
        self.max_image_array_layers = int(raw.maxImageArrayLayers)
        self.max_texel_buffer_elements = int(raw.maxTexelBufferElements)
        self.max_uniform_buffer_range = int(raw.maxUniformBufferRange)
        self.max_storage_buffer_range = int(raw.maxStorageBufferRange)
        self.max_push_constants_size = int(raw.maxPushConstantsSize)
        self.max_memory_allocation_count = int(raw.maxMemoryAllocationCount)
        self.max_sampler_allocation_count = int(raw.maxSamplerAllocationCount)
        self.buffer_image_granularity = VkDeviceSize(raw.bufferImageGranularity)
        self.sparse_address_space_size = VkDeviceSize(raw.sparseAddressSpaceSize)
        self.max_bound_descriptor_sets = int(raw.maxBoundDescriptorSets)
        self.max_per_stage_descriptor_samplers = int(raw.maxPerStageDescriptorSamplers)
        self.max_per_stage_descriptor_uniform_buffers = int(
            raw.maxPerStageDescriptorUniformBuffers
        )
        self.max_per_stage_descriptor_storage_buffers = int(
            raw.maxPerStageDescriptorStorageBuffers
        )
        self.max_per_stage_descriptor_sampled_images = int(
            raw.maxPerStageDescriptorSampledImages
        )
        self.max_per_stage_descriptor_storage_images = int(
            raw.maxPerStageDescriptorStorageImages
        )
        self.max_per_stage_descriptor_input_attachments = int(
            raw.maxPerStageDescriptorInputAttachments
        )
        self.max_per_stage_resources = int(raw.maxPerStageResources)
        self.max_descriptor_set_samplers = int(raw.maxDescriptorSetSamplers)
        self.max_descriptor_set_uniform_buffers = int(
            raw.maxDescriptorSetUniformBuffers
        )
        self.max_descriptor_set_uniform_buffers_dynamic = int(
            raw.maxDescriptorSetUniformBuffersDynamic
        )
        self.max_descriptor_set_storage_buffers = int(
            raw.maxDescriptorSetStorageBuffers
        )
        self.max_descriptor_set_storage_buffers_dynamic = int(
            raw.maxDescriptorSetStorageBuffersDynamic
        )
        self.max_descriptor_set_sampled_images = int(raw.maxDescriptorSetSampledImages)
        self.max_descriptor_set_storage_images = int(raw.maxDescriptorSetStorageImages)
        self.max_descriptor_set_input_attachments = int(
            raw.maxDescriptorSetInputAttachments
        )
        self.max_vertex_input_attributes = int(raw.maxVertexInputAttributes)
        self.max_vertex_input_bindings = int(raw.maxVertexInputBindings)
        self.max_vertex_input_attribute_offset = int(raw.maxVertexInputAttributeOffset)
        self.max_vertex_input_binding_stride = int(raw.maxVertexInputBindingStride)
        self.max_vertex_output_components = int(raw.maxVertexOutputComponents)
        self.max_tessellation_generation_level = int(raw.maxTessellationGenerationLevel)
        self.max_tessellation_patch_size = int(raw.maxTessellationPatchSize)
        self.max_tessellation_control_per_vertex_input_components = int(
            raw.maxTessellationControlPerVertexInputComponents
        )
        self.max_tessellation_control_per_vertex_output_components = int(
            raw.maxTessellationControlPerVertexOutputComponents
        )
        self.max_tessellation_control_per_patch_output_components = int(
            raw.maxTessellationControlPerPatchOutputComponents
        )
        self.max_tessellation_control_total_output_components = int(
            raw.maxTessellationControlTotalOutputComponents
        )
        self.max_tessellation_evaluation_input_components = int(
            raw.maxTessellationEvaluationInputComponents
        )
        self.max_tessellation_evaluation_output_components = int(
            raw.maxTessellationEvaluationOutputComponents
        )
        self.max_geometry_shader_invocations = int(raw.maxGeometryShaderInvocations)
        self.max_geometry_input_components = int(raw.maxGeometryInputComponents)
        self.max_geometry_output_components = int(raw.maxGeometryOutputComponents)
        self.max_geometry_output_vertices = int(raw.maxGeometryOutputVertices)
        self.max_geometry_total_output_components = int(
            raw.maxGeometryTotalOutputComponents
        )
        self.max_fragment_input_components = int(raw.maxFragmentInputComponents)
        self.max_fragment_output_attachments = int(raw.maxFragmentOutputAttachments)
        self.max_fragment_dual_src_attachments = int(raw.maxFragmentDualSrcAttachments)
        self.max_fragment_combined_output_resources = int(
            raw.maxFragmentCombinedOutputResources
        )
        self.max_compute_shared_memory_size = int(raw.maxComputeSharedMemorySize)
        self.max_compute_work_group_count = (
            int(raw.maxComputeWorkGroupCount[0]),
            int(raw.maxComputeWorkGroupCount[1]),
            int(raw.maxComputeWorkGroupCount[2]),
        )
        self.max_compute_work_group_invocations = int(
            raw.maxComputeWorkGroupInvocations
        )
        self.max_compute_work_group_size = (
            int(raw.maxComputeWorkGroupSize[0]),
            int(raw.maxComputeWorkGroupSize[1]),
            int(raw.maxComputeWorkGroupSize[2]),
        )
        self.sub_pixel_precision_bits = int(raw.subPixelPrecisionBits)
        self.sub_texel_precision_bits = int(raw.subTexelPrecisionBits)
        self.mipmap_precision_bits = int(raw.mipmapPrecisionBits)
        self.max_draw_indexed_index_value = int(raw.maxDrawIndexedIndexValue)
        self.max_draw_indirect_count = int(raw.maxDrawIndirectCount)
        self.max_sampler_lod_bias = float(raw.maxSamplerLodBias)
        self.max_sampler_anisotropy = float(raw.maxSamplerAnisotropy)
        self.max_viewports = int(raw.maxViewports)
        self.max_viewport_dimensions = (
            int(raw.maxViewportDimensions[0]),
            int(raw.maxViewportDimensions[1]),
        )
        self.viewport_bounds_range = (
            float(raw.viewportBoundsRange[0]),
            float(raw.viewportBoundsRange[1]),
        )
        self.viewport_sub_pixel_bits = int(raw.viewportSubPixelBits)
        self.min_memory_map_alignment = int(raw.minMemoryMapAlignment)
        self.min_texel_buffer_offset_alignment = VkDeviceSize(
            raw.minTexelBufferOffsetAlignment
        )
        self.min_uniform_buffer_offset_alignment = VkDeviceSize(
            raw.minUniformBufferOffsetAlignment
        )
        self.min_storage_buffer_offset_alignment = VkDeviceSize(
            raw.minStorageBufferOffsetAlignment
        )
        self.min_texel_offset = int(raw.minTexelOffset)
        self.max_texel_offset = int(raw.maxTexelOffset)
        self.min_texel_gather_offset = int(raw.minTexelGatherOffset)
        self.max_texel_gather_offset = int(raw.maxTexelGatherOffset)
        self.min_interpolation_offset = float(raw.minInterpolationOffset)
        self.max_interpolation_offset = float(raw.maxInterpolationOffset)
        self.sub_pixel_interpolation_offset_bits = int(
            raw.subPixelInterpolationOffsetBits
        )
        self.max_framebuffer_width = int(raw.maxFramebufferWidth)
        self.max_framebuffer_height = int(raw.maxFramebufferHeight)
        self.max_framebuffer_layers = int(raw.maxFramebufferLayers)
        self.framebuffer_color_sample_counts = VkSampleCountFlags(
            raw.framebufferColorSampleCounts
        )
        self.framebuffer_depth_sample_counts = VkSampleCountFlags(
            raw.framebufferDepthSampleCounts
        )
        self.framebuffer_stencil_sample_counts = VkSampleCountFlags(
            raw.framebufferStencilSampleCounts
        )
        self.framebuffer_no_attachments_sample_counts = VkSampleCountFlags(
            raw.framebufferNoAttachmentsSampleCounts
        )
        self.max_color_attachments = int(raw.maxColorAttachments)
        self.sampled_image_color_sample_counts = VkSampleCountFlags(
            raw.sampledImageColorSampleCounts
        )
        self.sampled_image_integer_sample_counts = VkSampleCountFlags(
            raw.sampledImageIntegerSampleCounts
        )
        self.sampled_image_depth_sample_counts = VkSampleCountFlags(
            raw.sampledImageDepthSampleCounts
        )
        self.sampled_image_stencil_sample_counts = VkSampleCountFlags(
            raw.sampledImageStencilSampleCounts
        )
        self.storage_image_sample_counts = VkSampleCountFlags(
            raw.storageImageSampleCounts
        )
        self.max_sample_mask_words = int(raw.maxSampleMaskWords)
        self.timestamp_compute_and_graphics = bool(raw.timestampComputeAndGraphics)
        self.timestamp_period = float(raw.timestampPeriod)
        self.max_clip_distances = int(raw.maxClipDistances)
        self.max_cull_distances = int(raw.maxCullDistances)
        self.max_combined_clip_and_cull_distances = int(
            raw.maxCombinedClipAndCullDistances
        )
        self.discrete_queue_priorities = int(raw.discreteQueuePriorities)
        self.point_size_range = (
            float(raw.pointSizeRange[0]),
            float(raw.pointSizeRange[1]),
        )
        self.line_width_range = (
            float(raw.lineWidthRange[0]),
            float(raw.lineWidthRange[1]),
        )
        self.point_size_granularity = float(raw.pointSizeGranularity)
        self.line_width_granularity = float(raw.lineWidthGranularity)
        self.strict_lines = bool(raw.strictLines)
        self.standard_sample_locations = bool(raw.standardSampleLocations)
        self.optimal_buffer_copy_offset_alignment = VkDeviceSize(
            raw.optimalBufferCopyOffsetAlignment
        )
        self.optimal_buffer_copy_row_pitch_alignment = VkDeviceSize(
            raw.optimalBufferCopyRowPitchAlignment
        )
        self.non_coherent_atom_size = VkDeviceSize(raw.nonCoherentAtomSize)


#
# GpuDevice
#


class GpuDevice:
    pass
