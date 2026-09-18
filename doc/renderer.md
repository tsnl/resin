# Rendering primitives and SVGF

`$/renderer.resin` exposes reusable rendering operations ported from
[Arris main-v1](https://github.com/tsnl/enlighten/tree/archive/main-v1).
Your program owns the buffers, pipelines, integrator, and pass schedule. The
[Arris tutorial](arris.md) composes these pieces into a headless glTF + HDRI
renderer; its entire frame loop and path integrator are in the example file.

## Choose the pieces

| Primitive | Inputs and result |
| --- | --- |
| `upload_scene` | A loaded glTF scene → GPU vertex, material, texture, and pixel tables |
| `scene_acceleration` | A CPU world-space triangle list → a static RT acceleration structure |
| `primary_vertex`, `primary_fragment` | Scene tables and view projection → depth-tested triangle IDs and barycentrics |
| `surface_at`, `accepts_surface` | Triangle ID and barycentrics → interpolated surface and MASK acceptance |
| `trace_surface` | Scene tables, ray origin/direction, distance bounds, and RNG → closest accepted hit or miss |
| `bsdf_value`, `bsdf_pdf`, `bsdf_sample` | Surface and directions/random samples → scattering value, PDF, or weighted direction |
| `environment_value`, `environment_pdf`, `environment_sample` | HDR environment and direction/random samples → radiance, PDF, or sampled direction |
| `tone_map` | Caller-owned linear HDR input and RGBA8 output → exposure, Reinhard mapping, and gamma |
| `svgf_create`, `filter`, `reset` | Independent temporal denoising over GPU radiance/position/normal buffers |

The umbrella import re-exports scene, environment, and BSDF operations. These can
also be imported individually from `$/renderer/scene.resin`,
`$/renderer/environment.resin`, and `$/renderer/brdf.resin`. glTF loading, image
codecs, and SVGF remain separate libraries. Importing the renderer module does
not require creating an RT pipeline: raster and tone mapping work on their own.

## Data and ownership

`DeviceScene` is a plain record of retained read-only `GpuSpan` views: `vertices`, `materials`,
`textures`, and `pixels`. `upload_scene(gpu, asset)` is a convenience for uploading
loader output. You can construct the record directly from your own GPU buffers;
no `GltfScene`, file loader, renderer object, or denoiser is required. Allocations
return `GpuSpanMut`; obtain an input view with `:read_only()` and retain a writable
owner separately when you intend to update its contents. The shader
view is `ShaderScene`, using borrowed spans with the same glTF table layout.
Material indices, texture indices, and packed byte offsets must stay in bounds.

Acceleration construction is separate from uploading those tables.
`scene_acceleration(gpu, vertices)` builds one static triangle list with one
identity instance. `trace_surface` and `surface_at` require its triangle order and
world positions to match the shader vertex table. The current acceleration
builder takes a CPU vertex span; GPU-only construction and refits remain future
work. Callers can already supply GPU material, texture, and shader-output buffers.

`environment(gpu, image)` uploads a linear HDR map and builds a luminance ×
solid-angle CDF. The returned `DeviceEnvironment` exposes `pixels`, `cdf`,
`width`, `height`, `yaw` in radians, and linear RGB `tint`. It can likewise be
constructed from caller-owned GPU buffers. The CDF must have `width*height+1`
nondecreasing entries, beginning at zero and ending at one. Shader evaluation and
sampling use `ShaderEnvironment`; they do not trace rays or choose a light path.

`Camera` only describes projection: a rigid camera-to-world `transform`,
`focal_length`, `sensor_size`, `near`, and `far`. It looks down local −Z;
`look_at` constructs the transform and `camera(transform, aspect)` supplies
pinhole defaults. Use positive aspect, focal and sensor sizes, and
`0 < near < far`. Exposure and gamma belong to the separate tone-mapping pass.

## Compose the GPU work

Create the pipelines you need, allocate their inputs and outputs, then record
ordinary GPU commands. The tutorial's frame schedule is explicit:

```text
commands:begin_rendering(visibility_image, depth_image, 0, 0, 0, 0)?;
commands:draw(raster, raster_root, vertex_count)?;
commands:end_rendering()?;
commands:copy_image_to_buffer(visibility_image, visibility)?;
commands:trace_rays(tracing, trace_root, width, height, 1)?;
commands:submit()?;

let linear_input = radiance:read_only();
let position_input = positions:read_only();
let normal_input = normals:read_only();
let filtered = denoiser:filter(linear_input, position_input, normal_input, view_projection)?;
// Dispatch tone_map into a caller-owned RGBA8 buffer, or consume filtered HDR directly.
```

`RasterRoot<DeviceScene>` carries scene tables, a view-projection matrix, and the
eye position. The supplied visibility shaders target RGBA32F with D32 LESS depth
testing. Clear to zero; visible pixels contain `(triangle + 1, u, v, 1)`, and zero
means background. Alpha-tested `MASK` fragments and single-sided backfaces are
discarded before depth commits. You can replace these shaders or consume their
visibility buffer from your own compute or ray-generation entry.

The example defines its own `TraceRoot` and ray-generation shader. Small local
miss/closest-hit entries return the library's `miss_hit()` and `closest_hit()`
payloads. `trace_surface(scene, origin, direction, minimum, maximum, rng)` traces
sequentially within that interval and skips rejected alpha/backface hits. Use a
normalized direction and finite nonnegative distance bounds; a miss has
`triangle == ~u32(0)`. Finite bounds support point-light shadow segments as well
as indirect paths. Rejected intersections do not spend a scattering bounce.

The BSDF combines Lambert diffuse and GGX visible-normal sampling. Directions
are normalized world-space directions pointing away from the surface; orient a
surface toward the outgoing direction with `orient_surface`. The sample's
`weight` already includes BSDF × cosine / PDF. Environment and BSDF PDFs are per
steradian, so callers can combine them with `power_heuristic` for MIS. The
example chooses next-event lighting, bounce limits, and Russian roulette; these
policies live in its integrator and can be changed independently of the library.

All traversal uses RT pipelines with recursion depth one, with sequential traces
from ray generation. There are no ray queries. This remains compatible with a
future Metal implementation of Resin's pipeline abstraction, discussed in
[ray tracing pipelines](ray-tracing-pipelines.md).

`ToneRoot` accepts a read-only HDR input and writable RGBA8 output
(`GpuSpan<Vec4>` / `GpuSpanMut<u8>` on the host, `Span<Vec4>` / `SpanMut<u8>`
in the shader), a pixel count, exposure, and
gamma. Dispatch enough invocations for that count, keep input and output storage
distinct, and provide at least `count` linear RGBA values and `4*count` output
bytes. Exposure is nonnegative and gamma is positive. Tone mapping leaves the HDR
input intact. PNG/EXR readback is ordinary image-library code in the example.

Command submission currently waits for GPU completion. GPU spans retain their
allocations, but writing a buffer changes what all its handles observe; cloning
is not a frame snapshot. The example chooses to reuse buffers on each frame.
With SVGF skipped, each frame is that frame's independent sample average.

## Standalone SVGF middleware

`$/svgf.resin` imports no renderer module. `svgf_create(gpu, width, height)`
allocates history and compiles its compute pipelines. Call
`denoiser:filter(radiance, positions, normals, view_projection)` with same-sized
read-only `GpuSpan<Vec4>` views and the current Vulkan zero-to-one view-projection
matrix. For writable allocations, bind their `:read_only()` views to locals first.
The result is a retained read-only linear RGB buffer whose W component holds
estimated variance.

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
first scene. Pass a bounded path such as `bytes("scene.glb")`; embedded NULs are
rejected. Its immutable `vertices`, `materials`, `textures`, and
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

The acceleration builder currently accepts CPU triangles and creates static
geometry. Rebuild it when vertex positions change, keeping the shader vertex table
and acceleration geometry in the same order. Material and lighting buffers can be
replaced independently; invalidate any temporal history after such edits. The visibility encoding supports fewer than
2²⁴ triangles, and the image must fit one 65,535-group compute launch at the
selected GPU's workgroup width. Host loaders work without Vulkan; rendering
requires Resin's Vulkan baseline plus ray tracing pipeline support.
