#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/* Access the same C streams used by getchar/putchar. Returns ferror(stdin)
   or fflush(stdout), respectively; these are exported runtime functions. */
int resin_stdin_error(void);
int resin_stdout_flush(void);

#ifdef __cplusplus
}
#endif
