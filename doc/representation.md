# Representation and conversions

## Representation details

### Byte-array storage

An ordinary `ubyte` array contains exactly its N declared bytes, with alignment 1 and
stride N when nested in another array. Host and device layouts agree. Whole-array copies
copy those elements without an extra sentinel; embedded zeros remain ordinary data.
`size_of([1_ub, 2_ub])` is 2, and `size_of([[1_ub, 2_ub], [3_ub, 4_ub]])` is 4.
Empty arrays reserve a C storage placeholder for portability; it is not an accessible
array element, and empty arrays have no shared host/device layout.

For C calls, obtain an element pointer with `:lea(0_ul)` on a pointer to an
allocated, nonempty byte array, and pass its logical length. An inline local array
cannot expose its address; use explicitly allocated storage when a C API needs a
pointer. Raw arrays and spans do not promise NUL termination. Use a `str` literal's `.data`
or an owned `String`'s `:get().data` when a C function requires a terminator. `string_from_bytes(span)`
copies raw bytes and appends that terminator outside the logical length.

### Shared size and alignment

`size_of(T)` and `align_of(T)` return `ulong` constants for the shared host/device
layout of a concrete type. They use the same layout rules as C assertions and SPIR-V
storage emission. Scalars in the shared profile, padded/nested records, pointers,
spans, and nonempty arrays are supported; unsupported layouts produce a source error.
For an inferred array type, `size_of(array_expression)` queries its type. Expression
operands are checked but not executed, as with C `sizeof`; side effects do not run.
Holes in an explicit type argument are rejected. Empty arrays/records,
booleans, function values, and other types outside the shared profile are rejected.

Use `gpu:create(value)?` to allocate and initialize one GPU element, with its type
inferred from the value or result context. `gpu:alloc::<T>(count)?` allocates
uninitialized elements and checks the multiplication of count by element size.
The byte allocator `gpu:alloc_in::<ubyte>(bytes, memory)?` returns `GpuSpan<ubyte>`.

### Explicit numeric conversions

`T(value)` converts an already typed numeric value to numeric type T. Integer-to-integer
conversions preserve the mathematical value and trap if it is outside the destination
range, including narrowing and signedness changes. Float-to-integer conversions truncate
toward zero, then range-check; NaNs, infinities, and out-of-range results trap. Checks use
exclusive power-of-two upper bounds, including for 64-bit integer destinations.

Integer-to-float and float narrowing use round-to-nearest, ties-to-even in the normal
floating-point environment. Float32-to-float64 is exact. Float64-to-float32 overflow
produces signed infinity; results below the smallest normal float32 magnitude become
signed zero. NaNs remain NaNs without a payload guarantee. Signed zero is preserved.
C traps abort the process; shader traps stop that invocation and propagate failure
through helper calls. Traps do not unwind automatic cleanup.
A shader trap is not a host-visible Err value, and earlier writes remain visible.

The shared shader profile supports `ubyte`, `int`, `uint`, `ulong`, and `float32` conversions.
Other numeric types remain available on the host and are diagnosed when reached by a
shader. There are no implicit numeric conversions. `float32(1)` still contextually types
a direct unsuffixed literal; suffixes fix source literal types. A conversion of a local
value does not narrow that local's storage merely because of the destination type.
Pointer reinterpretation and nominal record ascription remain separate IR operations.

### Eliminating Never

`absurd(value)` consumes a value of `Never` and has no returning execution path.
Its result type comes from its context; annotate the enclosing result or binding
if that type cannot otherwise be inferred. An inhabited struct or union is rejected.
For example, an infallible error union can be unwrapped without inventing an error value:

```resin
fn unwrap(r: (int | Err<Never>)) -> int {
	match (r) {
		int(n) => {
			n
		},
		Err(impossible) => {
			absurd(impossible)
		}
	}
}
```

The IR explicitly marks elimination as divergent. Its continuation type is only for
checking unreachable code; neither backend constructs a value of that type. C aborts
and shaders stop the invocation if invalid external memory somehow supplies a `Never`.
This defensive trap does not unwind cleanup. Reachable success and `?` paths retain normal
scope destruction. Matches over inhabited variants still require exhaustive, unique arms.

[`T | None`](options.md) supports direct widening, exhaustive matching,
and postfix `!` to exclude `None` or trap.

Operations are [visible free functions](methods.md); struct bodies contain only fields.
