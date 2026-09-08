#pragma once

#include "resin_runtime/status.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ResinGpu ResinGpu;
typedef struct ResinAllocation ResinAllocation;
typedef struct ResinImage ResinImage;
typedef struct ResinPipeline ResinPipeline;
typedef struct ResinCommandBuffer ResinCommandBuffer;

typedef uint64_t ResinDeviceAddress;

#define RESIN_GPU_DEVICE_NAME_MAX 256

typedef enum ResinGpuDeviceType {
    RESIN_GPU_DEVICE_OTHER = 0,
    RESIN_GPU_DEVICE_INTEGRATED = 1,
    RESIN_GPU_DEVICE_DISCRETE = 2,
    RESIN_GPU_DEVICE_VIRTUAL = 3,
    RESIN_GPU_DEVICE_CPU = 4
} ResinGpuDeviceType;

typedef struct ResinGpuDeviceInfo {
    uint32_t index;
    ResinGpuDeviceType kind;
    uint32_t vendor_id;
    uint32_t device_id;
    uint32_t api_version;
    uint32_t driver_version;
    uint32_t suitable;
    uint32_t reserved;
    uint64_t device_local_bytes;
    uint64_t host_visible_device_local_bytes;
    uint64_t max_buffer_size;
    char name[RESIN_GPU_DEVICE_NAME_MAX];
} ResinGpuDeviceInfo;

typedef enum ResinMemory {
    /* Persistently mapped, coherent memory; device-local when the hardware permits. */
    RESIN_MEMORY_DEFAULT = 0,
    /* Device-local memory, which need not be visible to the host. */
    RESIN_MEMORY_GPU = 1,
    /* Persistently mapped, coherent memory intended for device-to-host results. */
    RESIN_MEMORY_READBACK = 2
} ResinMemory;

/* Vulkan device order; writes at most count entries, returning INCOMPLETE if truncated. */
ResinStatus resin_gpu_device_count(uint32_t *count);
ResinStatus resin_gpu_enumerate_devices(ResinGpuDeviceInfo *infos, uint32_t count);

/* Every object created from a GPU must be destroyed before that GPU.
   All operations on a GPU and its children must be externally synchronized.
   Resources referenced by a recording must remain live until it is cancelled
   or submission completes; host accesses must not overlap device writes.
   `resin_gpu_create` picks the highest-scoring suitable device. */
ResinStatus resin_gpu_create(ResinGpu **out_gpu);
ResinStatus resin_gpu_create_at(uint32_t index, ResinGpu **out_gpu);
void resin_gpu_destroy(ResinGpu *gpu);

/* `alignment` applies to the device address; 0 means 16. */
ResinStatus resin_gpu_malloc(
    ResinGpu *gpu,
    size_t bytes,
    size_t alignment,
    ResinMemory memory,
    ResinAllocation **out_allocation);
void resin_gpu_free(ResinGpu *gpu, ResinAllocation *allocation);
void *resin_allocation_host_pointer(const ResinAllocation *allocation);
ResinDeviceAddress resin_allocation_device_pointer(const ResinAllocation *allocation);
size_t resin_allocation_size(const ResinAllocation *allocation);

/* Translate any pointer inside a mapped heap block, preserving its byte offset. */
ResinStatus resin_gpu_host_to_device_pointer(
    const ResinGpu *gpu,
    const void *host_pointer,
    ResinDeviceAddress *out_device_pointer);

/* SPIR-V byte lengths must be multiples of four. */
ResinStatus resin_gpu_create_compute_pipeline(
    ResinGpu *gpu,
    const void *spv_bytes,
    size_t spv_length,
    ResinPipeline **out_pipeline);
ResinStatus resin_gpu_create_graphics_pipeline(
    ResinGpu *gpu,
    const void *vertex_spv_bytes,
    size_t vertex_spv_length,
    const void *fragment_spv_bytes,
    size_t fragment_spv_length,
    ResinPipeline **out_pipeline);
void resin_gpu_free_pipeline(ResinGpu *gpu, ResinPipeline *pipeline);

/* RGBA8 color attachment, transfer-src. */
ResinStatus resin_gpu_create_image(
    ResinGpu *gpu,
    uint32_t width,
    uint32_t height,
    ResinImage **out_image);
void resin_gpu_free_image(ResinGpu *gpu, ResinImage *image);
uint32_t resin_image_width(const ResinImage *image);
uint32_t resin_image_height(const ResinImage *image);

ResinStatus resin_gpu_start_command_recording(
    ResinGpu *gpu,
    ResinCommandBuffer **out_command_buffer);
ResinStatus resin_gpu_set_pipeline(
    ResinCommandBuffer *command_buffer,
    const ResinPipeline *pipeline);

/* `root_data` is supplied as one 64-bit push constant. */
ResinStatus resin_gpu_dispatch(
    ResinCommandBuffer *command_buffer,
    ResinDeviceAddress root_data,
    uint32_t group_count_x,
    uint32_t group_count_y,
    uint32_t group_count_z);
ResinStatus resin_gpu_begin_rendering(
    ResinCommandBuffer *command_buffer,
    ResinImage *color,
    float clear_r,
    float clear_g,
    float clear_b,
    float clear_a);
ResinStatus resin_gpu_end_rendering(ResinCommandBuffer *command_buffer);
ResinStatus resin_gpu_draw(
    ResinCommandBuffer *command_buffer,
    ResinDeviceAddress root_data,
    uint32_t vertex_count);
ResinStatus resin_gpu_copy_image_to_buffer(
    ResinCommandBuffer *command_buffer,
    ResinImage *image,
    const ResinAllocation *dst);

/* Submits, waits until this command buffer's work completes, then frees it.
   Returns INVALID_ARGUMENT if another submitted recording has changed an
   image layout assumed by this recording; record its commands again.
   A non-null command buffer is consumed even when the status is not success. */
ResinStatus resin_gpu_submit(ResinGpu *gpu, ResinCommandBuffer *command_buffer);
/* A non-null command buffer is consumed. Pending image layout changes are discarded. */
void resin_gpu_cancel_command_buffer(ResinGpu *gpu, ResinCommandBuffer *command_buffer);

#ifdef __cplusplus
}
#endif
