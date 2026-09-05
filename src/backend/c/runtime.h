#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void **r_allocations;
static size_t r_count, r_capacity;

_Noreturn void r_fail(const char *message) {
    fprintf(stderr, "resin: %s\n", message);
    exit(1);
}

void *r_alloc(size_t size) {
    if (r_count == r_capacity) {
        size_t capacity = r_capacity ? r_capacity * 2 : 16;
        if (capacity < r_capacity || capacity > SIZE_MAX / sizeof(void *))
            r_fail("allocation size overflow");
        void **items = realloc(r_allocations, capacity * sizeof(void *));
        if (!items) r_fail("out of memory");
        r_allocations = items;
        r_capacity = capacity;
    }
    void *value = calloc(1, size);
    if (!value) r_fail("out of memory");
    r_allocations[r_count++] = value;
    return value;
}

void r_cleanup(void) {
    for (size_t i = 0; i < r_count; ++i) free(r_allocations[i]);
    free(r_allocations);
}

#define R_SIGNED(bits) \
    int##bits##_t r_i##bits(uint64_t value) { \
        uint##bits##_t bits_value = (uint##bits##_t)value; \
        int##bits##_t result; \
        memcpy(&result, &bits_value, sizeof(result)); \
        return result; \
    }
R_SIGNED(8)
R_SIGNED(16)
R_SIGNED(32)
R_SIGNED(64)
#undef R_SIGNED

int64_t r_idiv(int64_t a, int64_t b) {
    if (!b) r_fail("division by zero");
    if (a == INT64_MIN && b == -1) return a;
    return a / b;
}

int64_t r_imod(int64_t a, int64_t b) {
    if (!b) r_fail("division by zero");
    if (a == INT64_MIN && b == -1) return 0;
    return a % b;
}

uint64_t r_udiv(uint64_t a, uint64_t b) {
    if (!b) r_fail("division by zero");
    return a / b;
}

uint64_t r_umod(uint64_t a, uint64_t b) {
    if (!b) r_fail("division by zero");
    return a % b;
}

unsigned r_shift(uint64_t count, unsigned width) {
    if (count >= width) r_fail("shift count out of range");
    return (unsigned)count;
}

size_t r_index(uint64_t index, size_t length) {
    if (index >= length) r_fail("array index out of bounds");
    return (size_t)index;
}
