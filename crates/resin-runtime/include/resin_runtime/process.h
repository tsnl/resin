#pragma once
#include "resin_runtime/host.h"
#ifdef __cplusplus
extern "C" {
#endif

/* Called by generated main before entering Resin, before foreign threads or
   environment mutations. Deep copies both argument/environment arrays and their
   strings; both arrays end with NULL. Storage is borrowed until resin_cleanup.
   Unix preserves native bytes. Windows uses UTF-8, replacing unpaired UTF-16
   surrogates. Returns the argument count. argv[0] is the invocation name.
   Treat the returned data as read-only; do not free or replace its pointers. */
int32_t resin_process_init(int32_t argc, const char *const *argv, char ***out_argv, char ***out_envp);
size_t resin_string_array_length(const char *const *values);
/* Exact, case-sensitive byte lookup in the supplied snapshot on every platform.
   NULL means absent/invalid name; an empty string is a present empty value.
   Does not consult or modify the live process environment. */
const char *resin_environment_get(const char *const *envp, const char *name);

#ifdef __cplusplus
}
#endif
