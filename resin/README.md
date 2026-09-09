# Resin libraries

Libraries written in Resin live in this directory, one library per subdirectory.

- [`std/`](std/): the core platform-facing library, built on C and the native
  `resin-runtime` ABI. Programs import its modules through `$/std/` paths.

Add libraries such as math or rendering as siblings of `std/`. Keep platform
wrappers in `std/` and put higher-level functionality in the library that owns it.

The compiler resolves `$/std/` imports to this checkout's `resin/std/` by default;
`RESIN_STDLIB` can override that directory. Other libraries currently use relative
file imports.
