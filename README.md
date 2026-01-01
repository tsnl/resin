# `zfw`

A game engine for fun and profit, written in pure Python.

## Quick Start

If you're on NixOS, just use the provided `shell.nix` to get started.

## Resources

-   Rendering
    -   [Vulkan `VK_KHR_dynamic_rendering_local_read` in 1.4](https://docs.vulkan.org/spec/latest/appendices/legacy.html#_render_pass_objects_superseded_via_dynamic_rendering)
    -   [Dynamic rendering blog post](https://www.khronos.org/blog/streamlining-render-passes)
    -   [Irradiance Caching](https://www.ludicon.com/castano/blog/articles/irradiance-caching-part-1/)
    -   [Vulkan `VK_KHR_ray_tracing_extension` tutorial](https://nvpro-samples.github.io/vk_raytracing_tutorial_KHR/)
    -   [Vulkan mini path-tracer](https://github.com/nvpro-samples/vk_mini_path_tracer/)
    -   [Megakernels Considered Harmful](https://research.nvidia.com/sites/default/files/pubs/2013-07_Megakernels-Considered-Harmful/laine2013hpg_paper.pdf)
    -   [Jacco's Blog - How to Build a BVH](https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/)
    -   [PBR Book - Bounding Volume Hierarchies - The Surface Area Heuristic](https://www.pbr-book.org/3ed-2018/Primitives_and_Intersection_Acceleration/Bounding_Volume_Hierarchies#TheSurfaceAreaHeuristic)


## Conventions

All units conform to the International System of Units (SI). We use meters for distance,
seconds for time, radians for angles, and so on.

All coordinate systems are **right-handed**.

| Name        | Dim | Units          | Interpretation                                |
| ----------- | --- | -------------- | --------------------------------------------- |
| World space | 3D  | Meters         | +Z up, +Y forward, +X right                   |
| Clip space  | 3D  | NDC, [-1,+1]^3 | +X right, +Y up, -Z forward                   |
| Image space | 2D  | Pixels         | +Y down (rows), +X right (cols), origin at TL |

The above table means that **cameras look down -Z**, and that the projection matrix maps
the near plane to Z=-1 and the far plane to Z=+1 in clip space.

**Colors are in linear space using sRGB primaries** unless otherwise noted. APIs 
explicitly specify when colors are in sRGB space, and we try to convert to linear space
as early as possible.
