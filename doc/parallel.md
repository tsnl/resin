# Parallel blocks

`parallel_map` and `parallel_reduce` express a batch of work inside an ordinary
function. Their bodies have explicit parameters and may read enclosing bindings.
They are language expressions, not lambda values: a block cannot be stored,
returned, or passed to another function.

**Prototype:** a serial host schedule and cooperative GPU execution for blocks
written directly in a compute entry. The host does not create worker threads.
[Cooperative execution](cooperative-execution.md) explains the design and remaining work.

## Map and reduce

This is a complete, executable program:

```resin
{{#include ../examples/parallel_blocks.resin:parallel_blocks}}
```

`parallel_map(input) |element| { body }` evaluates `input` once and produces an
inline array with the same length and order. Each element initializes a fresh
block parameter. The body’s tail value determines the output element type, which
may differ from the input element type. An empty input produces an empty output;
the body is still type-checked.

`parallel_reduce(input, identity) |left, right| { body }` evaluates the input and
then the identity, once each. Both parameters and the body result have the input
element type. Empty input returns the identity. Supply a neutral value, such as
zero for addition or one for multiplication. The compiler does not prove
associativity or neutrality. Grouping may change with the execution schedule;
floating-point results need not match across schedules. The current host backend
uses a left fold starting with the supplied identity.

The initial forms require an inline array whose length is known during HIR
construction. Spans and generic parameters hiding the whole array shape are not
supported yet. Element types can contain ordinary generic parameters. Input
elements must be [copyable](lifetimes.md); block results are owned values. A map
may construct move-only results, with ordinary scope cleanup. References cannot
be stored as map results. An empty map may need an annotation inside its body to
determine the input element type.

## Captures and local mutation

A *capture* is a use of a local binding defined outside the block. Captures are
read-only places, even if the original binding has `mut` or exposes a `RefMut`.
The block cannot assign to the captured place or one of its inline fields,
borrow it as `RefMut`, or move a noncopyable value out of it. These restrictions
also apply through generic calls. Read-only borrowing and ordinary value copies
remain available.

Block parameters are immutable unless marked `mut`, and locals inside the block
follow the usual [binding rules](syntax.md). Each iteration has its own parameter
and local storage. A nested parallel block sees an outer iteration’s bindings as
read-only captures. Shadowing introduces a separate binding.

Read-only captures are **not an effect system or a race-freedom guarantee**. A
captured raw pointer still allows writes through that pointer. Copying a span or
shared handle does not freeze its backing allocation. Ordinary helpers can have
external effects. Programs intended for concurrent scheduling must avoid
conflicting accesses through these aliases; the capture check cannot detect them.

Each operation completes before the containing function continues. Its body uses
a tail value; `return` and postfix `?` cannot exit the enclosing function from
inside a parallel block. `break` and `continue` may target loops introduced inside
the block, but cannot escape it. Handle error values inside the block or produce
them as ordinary results. On the host, traps terminate the process without unwinding scopes. GPU failure
behavior is described below.

## Compute execution

A compute entry receives the workgroup's X index. Its scalar code runs once per
group; each top-level map distributes indices `lane, lane + width, ...` across
the physical invocations. Reduce uses a pairwise tree in shared scratch storage,
including the identity once. Batch lengths need not match the workgroup width.
The runtime still specializes that width for the GPU.

The compiler places group locals and intermediate arrays in workgroup memory;
iteration locals are private. Barriers publish results before the next region,
including writes through device pointers. Scalar branches and loops containing
parallel regions take uniform decisions. A checked failure stops the affected
group at the next collective boundary; other groups continue. Effects already
performed are not rolled back. This is not a cross-workgroup barrier.

Helpers and nested blocks use the serial schedule on each calling lane. This
prototype distributes only regions written directly in the compute entry;
calling a helper that contains a map does not introduce another collective.
Vertex and fragment helpers also use serial execution.

## Current implementation limits

The serial schedule expands each array element, so host and nested code size
grows with the input extent. GPU arrays must be nonempty, following the existing
shader array restriction; the host supports empty arrays. Captured references
retain the shader backend's existing restrictions on merging distinct local
addresses.

The initial GPU storage plan reserves all group locals rather than minimizing
lifetimes or fusing operations. It enforces a conservative 32 KiB shared-storage
budget, counting at least eight bytes per scalar slot and padding. Large batches
receive a compilation diagnostic. This is a correctness prototype, not an
optimized parallel-array implementation.

There is no `parallel_filter`, general closure facility, or automatic purity
analysis. Host workers, dynamic spans, fusion, subgroup operations, and nested
cooperative scheduling remain follow-up work.
