#pragma once

#include <webgpu.h>

struct resin_interp {
    WGPUDevice device;
    WGPUQueue queue;
    WGPUBuffer* buffers;
};