# Modules and libraries

## Files and the standard library

Each file has its own scope. Optional `export`, grouped `extern`, and `import`
clauses appear in that order before declarations. Each clause may appear only once:

```resin
export { answer };
import { "helpers.resin", "$/status.resin" };

fn answer() -> int  { helper() }
```
Imports bring only the dependency's exported names into the file's flat namespace. Without
an export clause (or with `export {}`), everything is private. An exported function can use
its private helpers and types. Imported bindings may be explicitly re-exported; dependencies
are not implicitly re-exported. Functions, types, and constants can be exported.

Different functions with the same name form an overload set, including imported
functions. Calls must select exactly one signature. Conflicting types, constants,
and nonfunction bindings remain errors. Nested scopes can still shadow names. Re-importing the same binding through
multiple paths is harmless. Import paths without a leading `$` resolve relative to the
importing file; each canonical file is loaded once. Imports never execute code. Import cycles are errors; mutually recursive
functions within one file remain supported.
`include` has been replaced by `import`.

Syntax keywords (`export`, `import`, `extern`, `type`, `struct`, `fn`, `let`, `mut`, `const`, `sizeof`, `if`,
`else`, `while`, and `match`), primitive type names, `Never`, and
`Ptr`, `Err`, `None`, and the opaque compiler handle types are reserved,
including in parameters and field names. Wrapper names such as `Span`, `ArcPtr`,
and `GpuSpan` are ordinary source type names.
Names such as `if_value` are ordinary identifiers. `size_of`, `align_of`, `iota`, and `absurd` are unshadowable compiler builtins, not syntax
keywords: definitions and parameters cannot use those names, but record fields can.

Imports beginning with `$/` resolve from the service's frozen [`resin/`](../resin/) library root,
independent of the source file or working directory. For example, `$/gpu.resin`
loads `resin/gpu.resin`; a future `$/math/matrix.resin` would load `resin/math/matrix.resin`.
Set `RESIN_LIBRARY_ROOT` or `--library-root` on the service to choose that snapshot.
Pinned packages are also service-owned; see [managed dependencies](compiler-service.md#pinned-dependencies).
Other imports resolve relative to the importing file. Files have explicit imports and exports;
directories need no manifest or special entry file.
The native Rust crate lives separately at `crates/resin-runtime/`; it has no dependency on the standard
library. Programs use standard-library wrappers; the integer-status C ABI stays private
to those modules. Public operations are exported free functions, including constructors and colon-call operations:

- `$/gpu.resin`: devices, allocations, images, pipelines, and command recording.
- `$/window.resin`: windows and input; `gpu_new_for_window(window)` and `gpu:present(image)` live in `$/gpu.resin`.
- `$/image.resin`: PNG reading and writing.
- `$/status.resin`: Native status conversion functions and the `RuntimeError` union and its variants.
- `$/graphics.resin`: shared `Position`, `Color`, and `Vertex` types.
- `$/io.resin`: `io_stdout():write(text)` and `io_stderr():write(text)`.
- `$/span.resin`: borrowed `Span<T>` and `bytes(text)` for literal byte views.
- `$/shared.resin`: `arc_ptr_alloc::<T>(initial)?`, `arc_span_alloc::<T>(count, initial)?`, and weak owners.
- `$/string.resin`: owned `String`, `string_from_str`, `string_from_bytes`, `fmt`, and `print`.
- `$/console.resin`: `console_read_byte()`, `console_read_line()`, and shared `InputLine` owners with `console_print(line)`.

Pass decorated shader declarations directly to GPU pipeline creation. Compiled shader
representations belong to code generation and the runtime; functions expose no bytecode property.
GPU memory modes are exported `int` constants: `memory_default`, `memory_gpu`, and
`memory_readback`. Pass them directly, for example `gpu:alloc_in::<uint>(count, memory_readback)`.
Run `cargo run -- examples/eg009_imports.resin` for an explicitly owned counter, or append
`:independent` to run a second entry that uses two independent counters.

```resin
export { main };
import { "$/gpu.resin", "$/string.resin" };

fn main() -> (() | Err<_>) {
	let gpu = gpu_new()?;
	print("GPU ready\n");
	(())
}
```
