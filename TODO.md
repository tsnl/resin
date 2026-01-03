# `TODO`

## `draw_3d`

### Basic Ray Tracer

- [x] Basic setup
  - [x] API, planning CPU-side code to set things up.
  - [x] Dispatch compute workgroups, write test gradient to output image.
- [ ] Primary rays
  - [x] Naive ray-triangle intersection with primary rays, incl. instanced meshes.
  - [x] Construct BVHs for BLAS
  - [ ] BVH-accelerated ray-triangle intersection with primary rays, ensure no regressions.
  - [ ] TLAS support, including per-frame TLAS rebuilds, ensure no regressions.
- [ ] Basic shading
  - [ ] Compute barycentric coordinates
  - [ ] Interpolate normals, get smooth lambertian shading.
  - [ ] Texture mapping support (?)
  - [ ] Sample environment map on ray miss.
- [ ] Secondary rays, PBR shading

### Baked Global Illumination

Add support for probe-based GI and lightmaps for indirect GI.

We can run the above ray-tracer offline to produce really high-quality baked lighting data.

We can then run a fast rasterizer or ray-tracer with only primary rays at run-time to then sample 
the baked lighting data.

### Wavefront-based Rewrite

Move away from megakernel design to wavefront design, with multiple specialized kernels.

## `window`

TODO: add windowing support.

## `python`

Python bindings.
