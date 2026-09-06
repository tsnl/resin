# Standard library

Resin modules wrapping the native `resin-runtime` C ABI. Import only the functionality a file
uses; the `std/` prefix resolves here. `RESIN_STDLIB` overrides this directory for installations.

```resin
import { "std/gpu.resin", "std/status.resin" };
```

- `gpu.resin`: devices, allocations, pipelines, GPU images, and commands.
- `window.resin`: windows and presentation.
- `image.resin`: PNG I/O.
- `status.resin`: status handling and `check`, which exits on failure.
- `graphics.resin`: shared shader input/output types.

Foreign bindings can be exported directly when no wrapper is useful. Native declarations
remain unchecked: callers must uphold pointer validity, lifetimes, and buffer sizes. Importing
a module does not re-export its dependencies. `print` and `shader` are unshadowable compiler builtins.
