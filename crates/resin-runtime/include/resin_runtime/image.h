#pragma once

#include "resin_runtime/status.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Packed 8-bit channels, top-left origin. 1=Y, 2=YA, 3=RGB, 4=RGBA.
   `row_stride` is bytes per row; 0 means `width * channels`. */
ResinStatus resin_image_write_png(
    const char *path,
    uint32_t width,
    uint32_t height,
    uint32_t channels,
    const void *pixels,
    size_t row_stride);

/* Decode a PNG to packed 8-bit pixels. `requested_channels` 0 keeps the
   file's channel count; 1-4 converts. Free `*out_pixels` with resin_image_free. */
ResinStatus resin_image_read_png(
    const char *path,
    uint32_t requested_channels,
    uint32_t *out_width,
    uint32_t *out_height,
    uint32_t *out_channels,
    void **out_pixels);

void resin_image_free(void *pixels);

#ifdef __cplusplus
}
#endif
