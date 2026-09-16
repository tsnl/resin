#ifndef RESIN_CONSOLE_FAULTS_H
#define RESIN_CONSOLE_FAULTS_H

#include <resin_runtime.h>
#include <stdio.h>
#include <stdlib.h>

static int allocations;
static int frees;
static int reads;
static void *live_allocation;

int console_test_mode(void);
int console_test_mode(void) { return TEST_MODE; }
int console_test_frees(void);
int console_test_frees(void) { return frees; }
int console_test_getchar(void);
int console_test_getchar(void) {
    if (TEST_MODE == 2) return EOF;
    return reads++ < 300 ? 'A' : EOF;
}
int console_test_error(void);
int console_test_error(void) { return TEST_MODE == 2 || TEST_MODE == 3; }
void *console_test_realloc(void *memory, size_t size);
void *console_test_realloc(void *memory, size_t size) {
    ++allocations;
    if (TEST_MODE == 0 || (TEST_MODE == 1 && allocations == 2)) return NULL;
    void *resized = realloc(memory, size);
    if (resized) live_allocation = resized;
    return resized;
}
void console_test_free(void *memory);
void console_test_free(void *memory) {
    if (!memory || memory != live_allocation) abort();
    ++frees;
    live_allocation = NULL;
    free(memory);
}
int console_test_putchar(int byte);
int console_test_putchar(int byte) { return TEST_MODE == 4 ? EOF : byte; }
int console_test_flush(void);
int console_test_flush(void) { return TEST_MODE == 5 ? EOF : 0; }
#endif
