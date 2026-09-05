#pragma once

#include "resin_runtime/gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ResinWindow ResinWindow;

#define RESIN_KEY_ESCAPE 256

/* All window operations, and operations on window-associated GPUs, run on the
   process main thread. The runtime owns GLFW initialization and termination;
   do not mix these calls with an independently managed GLFW context.
   GLFW is bundled and initialized lazily. Initialization or window creation
   failures print GLFW's error to stderr and return WINDOW_UNAVAILABLE. */
ResinStatus resin_window_create(uint32_t width, uint32_t height, const char *title, ResinWindow **out_window);
void resin_window_destroy(ResinWindow *window);
ResinStatus resin_window_poll_events(const ResinWindow *window);
int resin_window_should_close(const ResinWindow *window);
ResinStatus resin_window_set_should_close(const ResinWindow *window, int close);
ResinStatus resin_window_framebuffer_size(const ResinWindow *window, uint32_t *width, uint32_t *height);
ResinStatus resin_window_set_size(const ResinWindow *window, uint32_t width, uint32_t height);
/* GLFW key codes; returns 1 if pressed, otherwise 0. */
int resin_window_key_pressed(const ResinWindow *window, int key);

/* Selects a graphics/compute/present-capable device with swapchain maintenance1.
   One GPU per window. The GPU retains the native window until it is destroyed.
   Destroy GPU resources, then the GPU, then the window. */
ResinStatus resin_gpu_create_for_window(const ResinWindow *window, ResinGpu **out_gpu);

/* Synchronously blits an initialized image to the GPU's window, scaling to its
   framebuffer. FIFO presentation; resize recreates the swapchain automatically.
   INCOMPLETE means no frame was presented (minimized, timed out, or out of date):
   poll events and retry. Submit rendering before presenting. */
ResinStatus resin_gpu_present(ResinGpu *gpu, ResinImage *image);

#ifdef __cplusplus
}
#endif
