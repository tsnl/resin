#ifndef RESIN_RUNTIME_H
#define RESIN_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ResinGpu ResinGpu;
typedef struct ResinAllocation ResinAllocation;
typedef struct ResinPipeline ResinPipeline;
typedef struct ResinCommandBuffer ResinCommandBuffer;

typedef uint64_t ResinDeviceAddress;

typedef enum ResinStatus {
    RESIN_STATUS_SUCCESS = 0,
    RESIN_STATUS_INVALID_ARGUMENT = 1,
    RESIN_STATUS_VULKAN_UNAVAILABLE = 2,
    RESIN_STATUS_UNSUPPORTED = 3,
    RESIN_STATUS_OUT_OF_MEMORY = 4,
    RESIN_STATUS_VULKAN_ERROR = 5
} ResinStatus;

typedef enum ResinMemory {
    /* Persistently mapped, coherent memory; device-local when the hardware permits. */
    RESIN_MEMORY_DEFAULT = 0,
    /* Device-local memory, which need not be visible to the host. */
    RESIN_MEMORY_GPU = 1,
    /* Persistently mapped, coherent memory intended for device-to-host results. */
    RESIN_MEMORY_READBACK = 2
} ResinMemory;

/* Every object created from a GPU must be destroyed before that GPU. */
ResinStatus resin_gpu_create(ResinGpu **out_gpu);
void resin_gpu_destroy(ResinGpu *gpu);

/* The returned allocation owns both the Vulkan buffer and its memory. */
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

/* Translate any pointer within a live mapped allocation, preserving its byte offset. */
ResinStatus resin_gpu_host_to_device_pointer(
    const ResinGpu *gpu,
    const void *host_pointer,
    ResinDeviceAddress *out_device_pointer);

/* `spv_bytes` is a SPIR-V module, not a filename; its length must be a multiple of four. */
ResinStatus resin_gpu_create_compute_pipeline(
    ResinGpu *gpu,
    const void *spv_bytes,
    size_t spv_length,
    ResinPipeline **out_pipeline);
void resin_gpu_free_pipeline(ResinGpu *gpu, ResinPipeline *pipeline);

ResinStatus resin_gpu_start_command_recording(
    ResinGpu *gpu,
    ResinCommandBuffer **out_command_buffer);
ResinStatus resin_gpu_set_pipeline(
    ResinCommandBuffer *command_buffer,
    const ResinPipeline *pipeline);

/* `root_data` is supplied to the compute shader as one 64-bit push constant. */
ResinStatus resin_gpu_dispatch(
    ResinCommandBuffer *command_buffer,
    ResinDeviceAddress root_data,
    uint32_t group_count_x,
    uint32_t group_count_y,
    uint32_t group_count_z);

/* The initial implementation submits, waits for the queue to become idle, then frees the command buffer. */
ResinStatus resin_gpu_submit(ResinGpu *gpu, ResinCommandBuffer *command_buffer);
void resin_gpu_cancel_command_buffer(ResinGpu *gpu, ResinCommandBuffer *command_buffer);

const char *resin_status_string(ResinStatus status);

#ifdef __cplusplus
}
#endif

#endif
