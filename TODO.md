# `TODO`

## `draw_3d`

### Basic Ray Tracer

- [x] Basic setup
  - [x] API, planning CPU-side code to set things up.
  - [x] Dispatch compute workgroups, write test gradient to output image.
- [ ] Primary rays
  - [x] Naive ray-triangle intersection with primary rays, incl. instanced meshes.
  - [x] Construct BVHs for BLAS
  - [x] BVH-accelerated ray-triangle intersection with primary rays, ensure no regressions.
- [x] Basic shading
  - [x] Compute barycentric coordinates
  - [x] Interpolate normals, get smooth lambertian shading.
  - [x] Texture mapping support
  - [x] Sample environment map on ray miss.
- [ ] Texture heap overhaul
  - [ ] Implement BC4, BC6H texture encoding.
  - [ ] Use a large BC6H texture array (and a separate BC4 for mono) instead of a 
        storage buffer for all textures, using a GPU linear sampler to read data.
  - [ ] Move the big environment map into a singleton cube map.
  - [ ] (Future) add a separate cube map heap for light probes.
- [ ] Secondary rays, PBR shading:
    - [ ] IBL (environment map): prefiltered environment map.
    - [ ] Monte Carlo path tracing, offline rendering with ∞ samples.
- [ ] Optimization
  - [ ] Texture heaps: BC4 for mono, BC6H for color, paged atlas allocator, 
        deallocation.
  - [ ] Linear buffer deallocation, free lists for resource allocator heaps.
  - [ ] TLAS support, including per-frame TLAS rebuilds, ensure no regressions.
- [ ] Bigger Changes
  - [ ] Rewrite ray tracer to use wavefront design.
  - [ ] Switch back to Vulkan: maybe even rewrite in Rust at this point.
    - CUDA Interop for PyTorch tensors.
    - Bindless, buffer device address.
    - Multiple frames in flight.

### Vulkan for CUDA Interop

Rewrite from WebGPU to Vulkan for CUDA interop, especially with PyTorch tensors.

Would also give us better performance, more frames in flight.

Old Vulkan wrapper:

```bash
git checkout origin/archive/main-v2 -- src/zfw/typed_vulkan.py  src/zfw/typed_vulkan.pyi  
```

## Sharp Edges

Stuff that needs to be cleaned up with research.
- [ ] Instead of KiwiSolver, use Google OR-Tools (or better still don't use constraint 
      solving for GUI)
