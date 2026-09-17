# Arris renderer and SVGF

`$/renderer.resin` ports the static, headless rendering path of
[Arris main-v1](https://github.com/tsnl/enlighten/tree/archive/main-v1) to Resin.
The port follows the architecture at upstream commit
`4b8b32e7c82520fc397806977df8789bdbb8e39d`; the shaders are expressed as Resin
functions and use RT pipelines. The [complete example](arris.md) loads a glTF scene and an HDR environment and
writes both a tone-mapped PNG and a linear OpenEXR image.

## Persistent resources and frames

Create a GPU, load a `GltfScene`, and call `resource_pack(gpu, scene)` once.
The resulting `ResourcePack` owns uploaded geometry, material/texture tables, and
an acceleration structure. `environment(gpu, image)` uploads a floating-point
lat-long environment and builds its sampling distribution. Set its `yaw` in
radians and linear RGB `tint` before constructing the renderer.

`renderer_create(gpu, resources, environment, width, height, quality)` returns a
`Renderer`. The camera follows Arris's focal-length/sensor-size model. Its
row-major camera-to-world transform looks along local −Z; `look_at` constructs a
rigid transform. `camera(transform, aspect)` supplies sensible defaults. Camera
exposure is a linear multiplier and gamma affects only the PNG/display buffer.

`renderer:render(camera)` waits for GPU completion and returns a `RenderFrame`:

| Field | GPU contents |
| --- | --- |
| `radiance` | Linear HDR `GpuSpan<Vec4>`; RGB is radiance, W is filter variance when denoising |
| `positions` | World XYZ and stable triangle ID + 1; W = 0 means background |
| `normals` | Unit shading normal XYZ; W = 0 means background |
| `pixels` | Packed RGBA8 after exposure, Reinhard tone mapping, and gamma |

Handles retain their allocations. The next render reuses those allocations, so
copy the data to preserve a frame. `frame:write_png(path)` and
`frame:write_exr(path)` read back the appropriate buffer; EXR writes alpha = 1.
Pipeline inputs and outputs use ordinary GPU spans, so downstream compute work
can consume the buffers without CPU readback.

`Renderer` and `Svgf` are move-only: accidental copies cannot alias mutable history.
Keep one renderer per camera history. Recreate it to change resolution.
`renderer:reset()` discards temporal history and restarts the random sequence.
Use it after a camera cut or scene/lighting change; ordinary camera movement is
reprojected. Rendering is synchronous, matching Resin's current command API.

## Rendering passes

1. Rasterize triangles into a floating-point visibility attachment and a depth
   attachment. Visibility stores a triangle ID and perspective-correct
   barycentrics. `MASK` pixels below their texture alpha cutoff return `None`;
   discarded fragments do not write depth. Single-sided backfaces are discarded.
2. Reconstruct each visible surface in a ray-generation shader. Diffuse cosine
   sampling and GGX visible-normal sampling share a mixture PDF. Environment
   next-event samples use a luminance × solid-angle distribution, shadow rays,
   and multiple importance sampling. Secondary intersections interpolate glTF
   textures, normals, tangents, colors, and metallic/roughness parameters.
   Russian roulette starts after three scatters.
3. Optionally run SVGF on the linear HDR result, then tone map into the display
   buffer. Environment/background pixels bypass spatial filtering.

All traversal uses a ray tracing pipeline with recursion depth one. The
ray-generation shader traces sequentially; closest-hit returns distance,
triangle ID, and barycentrics. Secondary alpha rejection advances traversal
without consuming a scattering bounce. There are no ray queries. The source
library depends on Resin's pipeline abstraction; a future Metal backend would
need to implement that abstraction, as discussed in [ray tracing pipelines](ray-tracing-pipelines.md).

`Quality` controls `samples_per_frame`, `max_bounces`, and `denoise`. Disable
denoising and increase the sample count for an unfiltered reference image.
Frames use independent samples; with denoising off, each result is that frame's
sample average, not a progressive accumulation of previous frames.

## Standalone SVGF middleware

`$/svgf.resin` imports no renderer module. `svgf_create(gpu, width, height)`
allocates history and compiles its compute pipelines. Call
`denoiser:filter(radiance, positions, normals, view_projection)` with same-sized
GPU spans and the current Vulkan zero-to-one view-projection matrix. It returns
a retained linear RGB buffer whose W component holds estimated variance.

The filter reprojects history with bilinear taps, rejects inconsistent triangle
IDs, world positions and normals, and clamps history to a current 3 × 3 color
neighborhood. It accumulates temporal luminance moments, estimates short-history
variance over a 7 × 7 neighborhood, and runs five edge-aware à-trous levels.
Only the first spatial level feeds temporal color history. `denoiser:reset()`
invalidates history without reallocating the buffers.

The initial filter uses fixed world-space thresholds (5 cm for reprojection,
5 mm scaled by filter step for plane distance). Use meter-scale geometry.
It filters combined radiance, so fine texture/specular detail can soften. There
is no motion-vector input for moving geometry in this static-scene port.

## glTF loading and current scope

`$/gltf.resin` is independent of the renderer. `gltf_load(path)` reads `.gltf` or
`.glb`, external files and embedded data, selecting the default scene or the
first scene. Its immutable `vertices`, `materials`, `textures`, and
`texture_pixels` spans borrow storage owned by the `GltfScene`; retain the owner
through every use. `:clone()` shares ownership.

The loader bakes node transforms, triangulates triangle strips/fans, computes
missing flat normals, and generates missing UV tangents with MikkTSpace.
Nonuniform and reflected transforms are handled. Texture data is packed RGBA8;
albedo and emissive sampling decode sRGB before interpolation, while normal and
metallic/roughness maps remain linear. Sampler wrapping and nearest/bilinear
magnification are respected; mipmaps and anisotropic filtering are not included.

This first port supports static core glTF metallic/roughness scenes and HDRI
lighting. Skins, morph targets, non-triangle topology, UV sets other than zero
when used by supported material textures, and required extensions are rejected.
Animation is not evaluated. Optional material extensions use the core material
fallback. glTF cameras/lights are not imported: supply the camera explicitly;
lighting comes from the environment and emissive surfaces. Emissive triangles
are encountered by secondary rays but are not directly importance sampled.

`BLEND` surfaces are opaque to primary rasterization. Secondary rays use
stochastic alpha coverage; this is coverage transparency, not dielectric
refraction or a transmission BSDF. `MASK` is binary at every bounce.

Geometry and the acceleration structure are static after upload. Rebuild a
resource pack to change them. The visibility encoding supports fewer than
2²⁴ triangles, and the image must fit one 65,535-group compute launch at the
selected GPU's workgroup width. Host loaders work without Vulkan; rendering
requires Resin's Vulkan baseline plus ray tracing pipeline support.
