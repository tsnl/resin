# Cooperative execution proposal

**Status: draft design with an initial implementation.**
[Parallel blocks](parallel.md) documents the implemented syntax, capture checks,
serial host execution, and top-level cooperative compute execution. Everything described as planned below remains future
work. The motivating limitations are tracked in [issue #235](https://github.com/tsnl/resin/issues/235).

## Ordinary functions, explicit parallel regions

The goal is to call the same algorithm directly on the host and use it as a
compute entry. A function should not need GPU-only `Workgroup<T>` or
`WorkgroupSync` arguments to declare scratch memory or become callable.

Parallelism belongs in the language’s execution model on both targets. The
OpenMP influence is scoped work distribution and completion; the Futhark
influence is explicit map and reduction operations. This proposal gives ordinary
scalar control flow one logical execution. In OpenMP, a `parallel` construct
instead runs its associated block on every thread in the team. See the
[OpenMP parallel construct](https://www.openmp.org/spec-html/5.2/openmpse57.html).

Blocks have explicit argument bindings and read-only captures. They are not
first-class functions, and capture checking does not require whole-program effect
analysis. Pointer targets and helper effects remain the programmer’s
responsibility. Reduction laws are supplied by the programmer without compiler
proofs; a stricter numerical contract can follow experience with implementations.
[Futhark’s reduction example](https://futhark-lang.org/examples/scan-reduce.html)
illustrates the role of the identity and combiner.

## Logical execution scopes

| Context | Planned meaning |
| --- | --- |
| Ordinary host call | One logical control flow that can distribute parallel regions across host workers |
| Compute entry | One logical invocation for an entire workgroup |
| Map body | One computation per logical element; not a physical lane identity |
| Reduce body | One combining operation selected by the schedule |
| Vertex, fragment, and future ray-tracing entries | One shader invocation |

Helpers inherit the context in which they are called. Observable scalar effects
outside a parallel block happen once per logical call. A GPU backend must preserve
this even if it replicates private arithmetic across lanes. Nested parallel
regions describe nested work, without implicitly creating additional Vulkan
workgroups or requiring a fresh host thread team.

The host may share a bounded worker pool across calls. Each operation has scoped
completion, and captures remain alive until its workers finish. This does not
impose a process-wide barrier. NUMA placement can influence scheduling without
changing the program’s meaning. A serial schedule remains valid.

**Implemented migration:** compute entries now receive the workgroup X index,
and dispatch counts logical groups. Examples distribute fixed batches explicitly.
Only parallel regions written directly in the entry cooperate; helpers and nested
blocks remain serial. There is no task or mesh shader support.

## Workgroup size and storage

A logical batch can be larger or smaller than its physical worker team. The
launcher should choose workgroup size through pipeline specialization, independent
of the function’s algorithm and array extent. Vulkan’s `LocalSizeId` allows
specialization constants to set the local size; `vkCmdDispatch` supplies the
number of workgroups. The size is fixed for a pipeline variant, rather than
changed by a dispatch command. See the [Vulkan compute guide](https://docs.vulkan.org/guide/latest/compute_shaders.html).

Ordinary locals remain the source representation. Iteration-local temporaries
are private. Captured scalars can be replicated, and intermediate array values
can use distributed registers or workgroup storage according to their consumers.
A shared span descriptor does not implicitly stage its external allocation into
workgroup memory. The compiler must plan storage and synchronization together,
respect device limits, and keep collective participation well-defined. There is
no implicit barrier across workgroups; that requires a dispatch/API boundary.

Fusion may remove a map result consumed only by a reduction. It is an optimization,
not a prerequisite or a promise that all intermediate arrays disappear.

## Compiler path and implementation stages

The initial implementation retains explicit map/reduce nodes through AST, typed
checking, completed HIR, and concrete specialization. HIR establishes block
bindings, capture permissions, copy requirements, and control-flow boundaries;
completed nodes list their captured binding identities. The host schedule is
selected before storage lowering expands the bodies into ordinary verified LIR.
Source lookup and ownership inference do not move into the backend.

The following stages build on that representation:

1. **Completed here:** syntax, inference, diagnostics, formatting and editor
   support, read-only captures, a serial host schedule over inline arrays, and an
   executable manual example.
2. **Host scheduling:** choose a runtime loop representation and bounded parallel
   tasks with scoped joins. Audit runtime reference counting and native resources
   before sharing them across workers: copyability is not thread safety. Resolve
   dynamic input/output allocation and failure behavior before exposing spans as
   map inputs.
3. **Completed prototype:** migrate the compute entry and dispatch contracts;
   retain explicit LIR parallel regions and iteration-local storage ranges.
   SPIR-V assigns strided map jobs and tree-reduction pairs, shares group locals,
   and emits barriers. Leader execution and broadcasts make scalar decisions
   uniform; a group failure flag converges traps before collective exits.
   Nested blocks and helper calls remain serial. The first storage plan uses a
   conservative portable 32 KiB budget; storage reuse remains future work.
4. **Optimization and specialized operations:** subgroup reductions, fusion,
   distributed values, and matrix primitives after the baseline schedules agree.

Future scheduling should establish explicit guarantees in a concrete translation
before storage lowering. It should not recover parallelism from scalarized loops
or hide scheduling state in source scopes. CPU runtime workers are program
execution machinery, separate from `resin-executor`, which schedules compiler and
native-tool work.

## Subgroups and cooperative matrices

Workgroups are the program’s cooperative scope; subgroup width is a backend
property. A reduction may combine values inside subgroups and then combine their
partials across the workgroup. Programs cannot assume that logical element 17 is
physical lane 17.

A future typed matrix multiply-add operation can similarly tile workgroup work
into subgroup fragments. `VK_KHR_cooperative_matrix` properties report supported
shapes, component types, and scopes; without the optional workgroup-scope feature,
the scope is subgroup. See [cooperative matrix properties](https://docs.vulkan.org/refpages/latest/refpages/source/VkCooperativeMatrixPropertiesKHR.html).
The compiler must honor that reported scope and participation contract. Workgroup
matrix support can be an additional backend path; it should not require a
vendor-specific extension for ordinary map/reduce programs.

## Decisions still needed

- How dynamic spans and output ownership fit the map result, and how callers
  express large logical batches without array literals.
- How the launcher selects and caches workgroup-size pipeline variants, and additional
  launch tuning controls. Full residualized const generics are not a
  prerequisite for the first backend.
- How nested or divergent regions select legal collective schedules, including
  error/trap propagation and resource-limit diagnostics.
- Which host values can cross worker boundaries safely, beyond ordinary Copy.

Validation must cover direct host calls, capture rejection through generic
helpers, ownership and empty inputs, partial and non-power-of-two batches,
multiple workgroup/subgroup widths, and CPU/GPU result agreement. Vertex and
fragment invocation semantics must remain unchanged. The Mandelbrot example
should demonstrate the compute migration when that backend is added.
