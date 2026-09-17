# Parallel blocks

`parallel_map` and `parallel_reduce` express a batch of work inside an ordinary
function. Their bodies have explicit parameters and may read enclosing bindings.
They are language expressions, not lambda values: a block cannot be stored,
returned, or passed to another function.

**Initial implementation:** inline arrays and a serial host backend. These forms
currently provide reference behavior for the cooperative execution design; they
do not yet create CPU workers or run in shaders. See the
[cooperative execution proposal](cooperative-execution.md) for the planned GPU
and host scheduling model.

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
them as ordinary results. Traps retain the existing process-termination behavior;
they do not unwind scopes.

## Current implementation limits

The serial backend expands each source-known array element into an execution of
the block. Code size therefore grows with the input extent and nested operations.
This is a small-array reference implementation, not a performance feature yet.
Shader specialization rejects parallel regions, including those reached through
helpers, with an explicit unsupported-backend diagnostic.

There is no `parallel_filter`, general closure facility, or automatic purity
analysis. Scheduling, fusion, and cooperative shader support are follow-up work.
