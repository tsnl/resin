#pragma once
#include "host.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct ResinGltfScene ResinGltfScene;
ResinStatus resin_gltf_load(const char *path, ResinGltfScene **output);
size_t resin_gltf_count(const ResinGltfScene *scene, uint32_t table);
void resin_gltf_free(ResinGltfScene *scene);
/* Borrow immutable tables until scene is freed: 0 vertices, 1 materials,
 * 2 textures, 3 pixel bytes. See the matching records in resin/gltf.resin. */
const void *resin_gltf_table(const ResinGltfScene *scene, uint32_t table, size_t *length);

#ifdef __cplusplus
}
#endif
