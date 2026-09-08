#pragma once

#include <stdio.h>
#include <stdint.h>

// Standard streams can be C macros, so expose only the operations Resin needs.
static inline int resin_stdin_error(void) { return ferror(stdin); }
static inline int resin_stdout_flush(void) { return fflush(stdout); }

// Resin type ascriptions do not yet convert between integer widths.
// Call resin_console_byte only after checking getchar's result for EOF.
static inline uint8_t resin_console_byte(int byte) { return (uint8_t)byte; }
static inline int resin_console_putchar(uint8_t byte) { return putchar(byte); }
