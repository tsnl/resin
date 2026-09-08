#pragma once
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct ResinArc ResinArc;
ResinArc *resin_arc_new(size_t size, size_t alignment, void (*destroy)(void *));
void *resin_arc_data(ResinArc *owner);
void resin_arc_retain(ResinArc *owner);
void resin_arc_release(ResinArc *owner);
void resin_weak_retain(ResinArc *owner);
void resin_weak_release(ResinArc *owner);
ResinArc *resin_weak_upgrade(ResinArc *owner);
#ifdef __cplusplus
}
#endif
