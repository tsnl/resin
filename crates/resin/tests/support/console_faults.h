#pragma once

#include <resin_runtime.h>
#include <stdlib.h>

static int allocations;
static int frees;
static int reads;
static void *live_allocation;

static int console_test_mode(void) { return TEST_MODE; }
static int console_test_frees(void) { return frees; }
static int console_test_getchar(void) {
    if (TEST_MODE == 2) return EOF;
    return reads++ < 300 ? 'A' : EOF;
}
static int console_test_error(void) { return TEST_MODE == 2 || TEST_MODE == 3; }
static void *console_test_realloc(void *memory, size_t size) {
    ++allocations;
    if (TEST_MODE == 0 || (TEST_MODE == 1 && allocations == 2)) return NULL;
    void *resized = realloc(memory, size);
    if (resized) live_allocation = resized;
    return resized;
}
static void console_test_free(void *memory) {
    if (!memory || memory != live_allocation) abort();
    ++frees;
    live_allocation = NULL;
    free(memory);
}
static int console_test_putchar(uint8_t byte) { return TEST_MODE == 4 ? EOF : byte; }
static int console_test_flush(void) { return TEST_MODE == 5 ? EOF : 0; }

#define getchar console_test_getchar
#define realloc console_test_realloc
#define free console_test_free
#define resin_stdin_error console_test_error
#define resin_console_putchar console_test_putchar
#define resin_stdout_flush console_test_flush
