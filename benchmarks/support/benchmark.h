#ifndef RESIN_BENCHMARK_H
#define RESIN_BENCHMARK_H

#include <errno.h>
#include <inttypes.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <time.h>
#endif

static void benchmark_fail(const char *message) {
    fprintf(stderr, "CPU benchmark: %s\n", message);
    exit(EXIT_FAILURE);
}

static uint64_t benchmark_argument(int32_t argc, uint8_t **argv, uint32_t index) {
    if (argc != 6 || index >= (uint32_t)argc) {
        benchmark_fail("expected count, iterations, samples, warmup, and seed");
    }
    const char *text = (const char *)argv[index];
    char *end = NULL;
    errno = 0;
    uint64_t value = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || *text == '-') {
        benchmark_fail("invalid unsigned argument");
    }
    return value;
}

static uint32_t *benchmark_allocate(uint32_t count) {
    uint32_t *data = malloc((size_t)count * sizeof(uint32_t));
    if (data == NULL) {
        benchmark_fail("buffer allocation failed");
    }
    return data;
}

static void benchmark_prepare(uint32_t *input, uint32_t *output, uint32_t count, uint32_t seed) {
    for (uint32_t index = 0; index < count; index++) {
        input[index] = index * UINT32_C(747796405) + seed;
        output[index] = UINT32_C(0xdeadbeef);
    }
}

// Fences keep the compiler from moving input reads or output writes across either
// timer call. Checksums make every output observable, outside the measured region.
static uint64_t benchmark_clock(void) {
    atomic_signal_fence(memory_order_seq_cst);
#if defined(_WIN32)
    LARGE_INTEGER time;
    if (!QueryPerformanceCounter(&time)) {
        benchmark_fail("QueryPerformanceCounter failed");
    }
    uint64_t ticks = (uint64_t)time.QuadPart;
#else
    struct timespec time;
    if (clock_gettime(CLOCK_MONOTONIC, &time) != 0) {
        benchmark_fail("clock_gettime failed");
    }
    uint64_t ticks = (uint64_t)time.tv_sec * UINT64_C(1000000000) + (uint64_t)time.tv_nsec;
#endif
    atomic_signal_fence(memory_order_seq_cst);
    return ticks;
}

static double benchmark_seconds(uint64_t start, uint64_t end) {
    if (end <= start) {
        benchmark_fail("clock did not advance; increase the workload size");
    }
#if defined(_WIN32)
    LARGE_INTEGER frequency;
    if (!QueryPerformanceFrequency(&frequency) || frequency.QuadPart <= 0) {
        benchmark_fail("QueryPerformanceFrequency failed");
    }
    return (double)(end - start) / (double)frequency.QuadPart;
#else
    return (double)(end - start) / 1e9;
#endif
}

static void benchmark_report(uint64_t start, uint64_t end, uint32_t *output, uint32_t count, int32_t measured) {
    uint64_t checksum = UINT64_C(14695981039346656037);
    for (uint32_t index = 0; index < count; index++) {
        checksum = (checksum ^ output[index]) * UINT64_C(1099511628211);
    }
    printf("%s,%.17g,%" PRIu64 "\n", measured ? "sample" : "warmup",
        benchmark_seconds(start, end), checksum);
}

static void benchmark_free(uint32_t *data) {
    free(data);
}

#endif
