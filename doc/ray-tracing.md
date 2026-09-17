# Ray tracing pipelines

Import `$/gpu.resin` to build triangle scenes and trace them with Vulkan ray
tracing pipelines. See the complete
[triangle example](../examples/eg013_ray_tracing.resin).

Ray tracing is an optional device capability. Check
`gpu:supports_ray_tracing()` after creating a GPU. Unsupported devices still
support their ordinary compute and graphics operations; scene creation returns
`Err<Unsupported>` when ray tracing is unavailable. This implementation requires
`VK_KHR_acceleration_structure`, `VK_KHR_ray_tracing_pipeline`, and
`VK_KHR_deferred_host_operations`, in addition to Resin's existing GPU requirements.
There is no ray query API or software fallback.

## Scene ownership

`gpu:create_ray_scene(vertices, transforms)` accepts two `Span<f32>` values:

- Vertices contain tightly packed XYZ triples. Each consecutive three vertices
  form one opaque triangle. The vertex span must be nonempty and its length a
  multiple of nine floats.
- Transforms contain row-major affine 3×4 matrices, twelve floats per instance.
  Supply at least one finite, invertible transform. Instance indices follow input
  order. An identity transform is `[1,0,0,0, 0,1,0,0, 0,0,1,0]`.

Creation copies the inputs, builds one bottom-level acceleration structure for
the mesh and one top-level structure for its instances, and waits for completion.
The input spans can then be released. Scenes are immutable; changing geometry or
transforms requires creating another scene. Build scratch storage is temporary.

`scene:create_ray_tracing_pipeline(generation, miss, closest)` takes direct
shader declarations and returns a `GpuRayTracingPipeline<Root, Owner>`. The pipeline
retains its scene and GPU; it remains valid after the original scene handle is
released. The runtime owns shader groups, the shader binding table, alignment,
and the fixed recursion-depth setting.

## Stage interfaces

All three shaders use the same `Ptr<Root>` second parameter:

| Decorator | Signature | Meaning |
| --- | --- | --- |
| `@ray_generation_shader` | `(u64, Ptr<Root>) -> ()` | Generate rays and write output. |
| `@miss_shader` | `(Payload, Ptr<Root>) -> Payload` | Return the payload for a miss. |
| `@closest_hit_shader` | `(Payload, Ptr<Root>) -> Payload` | Return the payload for the nearest triangle hit. |

Ray generation receives the flattened launch index `x + width * (y + height * z)`.
Dispatch dimensions count rays, not compute workgroups. Pass dimensions through
`Root` when the shader needs to recover coordinates.

A pipeline has one payload type. It must match the miss/closest-hit input and
result types and every reachable `trace_ray` call. Payloads contain `f32`, `i32`,
`u32`, or arrays/records composed of them. Pointers, managed values, unions, and
other scalar widths are not supported in payloads. The compiler gives payloads
their own shader interface; they are not ordinary device pointers.

```resin
let result = trace_ray(ox, oy, oz, dx, dy, dz, minimum, maximum, initial);
```

The first eight arguments are `f32`; `initial` determines the payload type.
The ray is `origin + t * direction`. Use finite inputs, a nonzero direction, and
`0 <= minimum <= maximum`. Direction need not be normalized; hit distance is the
parameter `t`. Every instance is visible to every ray in this initial API.

`trace_ray` is available only from ray generation and its helpers. Sequential
traces in a loop are supported; tracing from miss or closest-hit is rejected.
This keeps Vulkan's maximum pipeline ray recursion depth at one.

`ray_hit_info()` is available only from closest-hit and its helpers. It returns
`(f32, u32, u32, f32, f32)`: hit distance, primitive index, instance index, and
triangle barycentric coordinates U/V. The remaining barycentric weight is
`1 - U - V`.

## Recording and limits

`commands:trace_rays(pipeline, arguments, width, height, depth)` projects host
arguments to the pipeline's `Root`, just like compute dispatch. Width, height,
and depth are `u32`. Recording checks the device's dispatch limits and retains
the pipeline and projected allocations until submission or cancellation.
`commands:submit()` waits for completion before host readback.

The initial implementation supports one mesh with multiple instances, one miss
shader, one triangle hit group, and opaque triangles. It does not expose any-hit,
procedural intersections, callable shaders, nested tracing, per-material shader
binding records, acceleration-structure updates, or compaction.

Compiler and SPIR-V validation tests run without ray tracing hardware. To require
hardware execution instead of permitting capability skips:

```sh
nix-shell --run 'RESIN_REQUIRE_SPIRV_TOOLS=1 RESIN_REQUIRE_RAY_TRACING=1 VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation cargo test --features gpu --test ray_tracing'
```
