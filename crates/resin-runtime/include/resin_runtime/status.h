#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Match the runtime's Rust #[repr(i32)] status ABI on every C target. */
typedef int32_t ResinStatus;
enum {
    RESIN_STATUS_SUCCESS = 0,
    RESIN_STATUS_INVALID_ARGUMENT = 1,
    RESIN_STATUS_VULKAN_UNAVAILABLE = 2,
    RESIN_STATUS_UNSUPPORTED = 3,
    RESIN_STATUS_OUT_OF_MEMORY = 4,
    RESIN_STATUS_VULKAN_ERROR = 5,
    RESIN_STATUS_IO_ERROR = 6,
    RESIN_STATUS_INCOMPLETE = 7,
    RESIN_STATUS_WINDOW_UNAVAILABLE = 8
};

const char *resin_status_string(ResinStatus status);

#ifdef __cplusplus
}
#endif
