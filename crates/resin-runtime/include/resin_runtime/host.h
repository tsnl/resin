#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
[[noreturn]] void resin_fail(const char *message);
#else
_Noreturn void resin_fail(const char *message);
#endif

/* Zeroed, suitably aligned storage retained until resin_cleanup. */
void *resin_alloc(size_t size);
void resin_cleanup(void);

int8_t resin_i8(uint64_t value);
int16_t resin_i16(uint64_t value);
int32_t resin_i32(uint64_t value);
int64_t resin_i64(uint64_t value);
int64_t resin_idiv(int64_t a, int64_t b);
int64_t resin_imod(int64_t a, int64_t b);
uint64_t resin_udiv(uint64_t a, uint64_t b);
uint64_t resin_umod(uint64_t a, uint64_t b);
unsigned resin_shift(uint64_t count, unsigned width);
size_t resin_index(uint64_t index, size_t length);

#ifdef __cplusplus
}
#endif
