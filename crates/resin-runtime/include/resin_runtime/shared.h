#pragma once
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct ResinArc ResinArc;
ResinArc *resin_arc_new(size_t size, size_t alignment, void (*destroy)(void *));
/* Returns NULL on overflow or allocation failure. Initialize every element before
   publishing or releasing the owner. A non-NULL destroy_element runs once per
   element in reverse order; length excludes any external sentinel convention. */
ResinArc *resin_arc_span_try_new(size_t length, size_t stride, size_t alignment, void (*destroy_element)(void *));
/* Borrows the payload, or the first element of a sequence, from a live owner. */
void *resin_arc_data(ResinArc *owner);
/* Requires a live sequence owner. Zero-sized elements still contribute to length. */
size_t resin_arc_span_length(ResinArc *owner);
void resin_arc_retain(ResinArc *owner);
void resin_arc_release(ResinArc *owner);
void resin_weak_retain(ResinArc *owner);
void resin_weak_release(ResinArc *owner);
ResinArc *resin_weak_upgrade(ResinArc *owner);
#ifdef __cplusplus
}
#endif
