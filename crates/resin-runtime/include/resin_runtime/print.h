#pragma once

#include <stddef.h>
#include <stdint.h>
#include "shared.h"

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
    RESIN_PRINT_REPR = 8,
};

typedef struct ResinPrintBytes {
    const uint8_t *data;
    size_t length;
} ResinPrintBytes;

// Descriptors refer to immutable compiler-emitted tables and initialized values.
// Offsets and strides describe the actual native layout; pointers are never followed.
enum {
    RESIN_REPR_UNIT, RESIN_REPR_NONE, RESIN_REPR_BOOL,
    RESIN_REPR_I8, RESIN_REPR_I16, RESIN_REPR_I32, RESIN_REPR_I64,
    RESIN_REPR_U8, RESIN_REPR_U16, RESIN_REPR_U32, RESIN_REPR_U64,
    RESIN_REPR_F32, RESIN_REPR_F64, RESIN_REPR_STR, RESIN_REPR_POINTER,
    RESIN_REPR_OPAQUE, RESIN_REPR_NAMED, RESIN_REPR_ARRAY,
    RESIN_REPR_TUPLE, RESIN_REPR_RECORD, RESIN_REPR_UNION, RESIN_REPR_TEXT,
};
typedef struct ResinReprType ResinReprType;
typedef struct ResinReprField {
    const char *name;
    const ResinReprType *type;
    size_t offset;
    uint32_t tag;
} ResinReprField;
struct ResinReprType {
    uint32_t kind;
    const char *name;
    size_t size;
    size_t length;
    const ResinReprField *fields;
    size_t field_count;
    ResinPrintBytes (*text)(const void *);
};
typedef struct ResinReprValue {
    const ResinReprType *type;
    const void *data;
} ResinReprValue;
ResinArc *resin_repr(const ResinReprType *type, const void *data);
void resin_report_error(const ResinReprType *type, const void *data);

typedef union ResinPrintData {
    int64_t signed_value;
    uint64_t unsigned_value;
    double float_value;
    ResinPrintBytes bytes;
    ResinReprValue repr;
} ResinPrintData;

typedef struct ResinPrintArg {
    uint32_t kind;
    ResinPrintData value;
} ResinPrintArg;

// Buffers must remain readable during the call; NULL is allowed only for zero lengths.
// Writes bytes verbatim to stdout, flushes, and adds no newline. Failure terminates.
void resin_print(const uint8_t *data, size_t length);
// Writes and flushes stdout (0) or stderr (1). Returns zero on success, -1 on failure.
int32_t resin_stream_write(uint32_t stream, const uint8_t *data, size_t length);

// Return owned ArcSpan<ubyte> storage. resin_arc_data borrows the bytes;
// resin_arc_span_length returns their logical length, excluding a trailing NUL.
// Copies bytes verbatim; the source need not be NUL-terminated.
ResinArc *resin_string_from_str(const uint8_t *data, size_t length);
// Formats arguments; invalid formats terminate the program.
ResinArc *resin_format(const uint8_t *format, size_t length, const ResinPrintArg *args, size_t count);

#ifdef __cplusplus
}
#endif
