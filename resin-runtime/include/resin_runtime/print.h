#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    RESIN_PRINT_UNIT = 0,
    RESIN_PRINT_BOOL = 1,
    RESIN_PRINT_SIGNED = 2,
    RESIN_PRINT_UNSIGNED = 3,
    RESIN_PRINT_FLOAT32 = 4,
    RESIN_PRINT_FLOAT64 = 5,
    RESIN_PRINT_BYTES = 6,
    RESIN_PRINT_POINTER = 7,
};

typedef struct ResinPrintBytes {
    const uint8_t *data;
    size_t length;
} ResinPrintBytes;

typedef union ResinPrintData {
    int64_t signed_value;
    uint64_t unsigned_value;
    double float_value;
    ResinPrintBytes bytes;
} ResinPrintData;

typedef struct ResinPrintArg {
    uint32_t kind;
    ResinPrintData value;
} ResinPrintArg;

// Buffers must remain readable during the call; NULL is allowed only for zero lengths.
// Invalid formats terminate the program. No newline is added.
void resin_print(const uint8_t *format, size_t length, const ResinPrintArg *args, size_t count);

#ifdef __cplusplus
}
#endif
