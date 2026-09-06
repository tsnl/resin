# Standard library

Resin modules wrapping the native `resin-runtime` C ABI. Import only the functionality a file
uses; the `std/` prefix resolves here. `RESIN_STDLIB` overrides this directory for installations.

```resin
import { "std/gpu.resin", "std/status.resin" };
```

- `gpu.resin`: devices, allocations, pipelines, GPU images, and commands.
- `window.resin`: windows and presentation.
- `image.resin`: PNG I/O.
- `status.resin`: `status(code)` returns `Result<(), RuntimeError>`; `check(code)` still exits on failure.
- `graphics.resin`: shared shader input/output types.

Foreign bindings can be exported directly when no wrapper is useful. Native declarations
remain unchecked: callers must uphold pointer validity, lifetimes, and buffer sizes. Importing
a module does not re-export its dependencies. `print`, `shader`, `ok`, and `err` are unshadowable compiler builtins.

Use `status(native_call(...))?` in a Result-returning function. After each successful
resource acquisition, register its release with `defer release(resource);` before another fallible
operation. Cleanup runs in reverse order on scope exit, including through `?`.
Deferred expressions discard their values and must handle errors locally; they cannot
propagate with `?` themselves. Blocks can group multiple cleanup actions.
Keep GPU resources live until pending work completes, and avoid releasing handles whose
ownership was transferred. The existing `check` helper terminates the process on failure,
bypassing defers; existing graphics examples still use it with explicit cleanup.
