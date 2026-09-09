# Resin libraries

Libraries written in Resin live in this directory, one library per subdirectory.

- [`std/`](std/): the core platform-facing library, built on C and the native
  `resin-runtime` ABI. Programs import its modules through `$/std/` paths.

Add libraries such as math or rendering as siblings of `std/`. Keep platform
wrappers in `std/` and put higher-level functionality in the library that owns it.

Imports beginning with `$/` resolve from this directory. For example,
`$/std/gpu.resin` loads `std/gpu.resin`, and a future `$/math/matrix.resin` would
load `math/matrix.resin`. `RESIN_LIBRARY_ROOT` overrides the whole library root.
Other imports resolve relative to the importing file. No manifests or special
entry files are required; existing explicit imports and exports apply.
