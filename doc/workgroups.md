# Workgroup storage and synchronization

Compute functions run once per invocation. An explicit workgroup parameter gives
invocations in the same group shared state and a barrier. Ordinary host calls
execute the same function as a single lane.

## One parameter, ordinary references

Import `$/workgroup.resin` for `Workgroup<T>`, `sync`, `lane_index`, and `lane_count`.
`Workgroup<T>` is a transparent alias for `RefMut<T>`, not a new owning resource.
A compute entry may take it as an optional third parameter:

```resin
{{#include ../examples/workgroup.resin:workgroup}}
```

The GPU supplies one zero-initialized `State` per workgroup and passes its mutable
reference to every lane. A barrier publishes that initialization before the entry
runs. Locals declared inside the function remain private to their invocation.
Helpers receive ordinary references, including references to shared fields;
no pointer is materialized from a local reference.

The entry's first argument remains the global X invocation index. Dispatch still
counts physical workgroups, and the runtime specializes the physical lane count
for the device. Existing two-parameter shaders and launch code keep their behavior.
The host launch record supplies only the existing root argument; the GPU wrapper
allocates the third argument internally.

`group:lane_index()` returns a `u64` in `[0, group:lane_count())`. The example gives
two lanes producer jobs and has every lane join the barrier. Logical jobs need
not equal physical lanes, but the program must assign those jobs explicitly.

## Host calls

An ordinary host call borrows a caller-supplied initialized local, as in `host_main`
above. It runs with lane index zero and lane count one; `sync()` is a no-op.
It neither resets the supplied state nor creates threads. The example explicitly
assigns both producer jobs to the sole lane in that case.

This is a serial reference execution, not a simulation of an arbitrary number of
GPU lanes. Algorithms that assume two or more lanes must provide a one-lane path.
A multi-threaded host implementation would need an execution context or a richer
workgroup handle; that is outside the current implementation.

## Synchronization contract

`group:sync()` waits for every invocation in the current workgroup and publishes
earlier writes to shared state and device-pointer storage to that group. It is
not a barrier between workgroups. Vulkan defines this scope in its
[compute execution model](https://docs.vulkan.org/spec/latest/chapters/shaders.html#shaders-scope-workgroup).

Every lane must reach the same barriers in the same order. Keep barriers outside
lane-dependent branches; inactive lanes still participate. Do not return early
from only some lanes before a later barrier. The compiler does **not** prove
uniform participation or race freedom. Violating this contract can hang execution.
Helpers do not create new groups: their `sync()` joins the caller's group.

Checked arithmetic, assertions, and trapping unwraps can terminate an invocation.
This first version conservatively rejects any emitted checked-failure path in a
shader with explicit workgroup state, including paths inside helpers. It does not
silently permit a trap to strand peers. Supporting collective failure or proving
that a failure occurs after the last barrier is future work. Floating-point
arithmetic, wrapping integer arithmetic, and widening casts remain available.

On the GPU, these operations require a reference originating in the actual shared
state. A private local or a device-buffer pointer cannot impersonate a group.
Because the alias is transparent, this origin restriction is checked during shader
emission; on the host, any suitable mutable local represents a one-lane group.

## Supported state and limits

State supports plain numeric values, booleans, unit, pointers, tuples, and named
records. Compiler-level array types are supported, but the current source language
cannot spell a fixed array's element count in a type annotation; array syntax is
an independent follow-up. Managed resources, unions, and empty arrays are rejected.
Zero initialization sets pointers to null; initialize them before dereferencing.

The compiler applies a conservative 16 KiB shared-state budget, reserving at least
eight bytes per scalar slot including padding. This fits the baseline
[Vulkan shared-memory limit](https://docs.vulkan.org/refpages/latest/refpages/source/Required_Limits.html).
Device-specific larger allocations, multidimensional groups, subgroup primitives,
cooperative matrices, and concurrent host execution remain open.

The implementation adds three primitive operations and a third-argument compute
convention. Existing reference specialization carries the Workgroup storage class
through helpers. There are no new parser productions, HIR block forms, LIR control
regions, or automatic scalar broadcasts.
