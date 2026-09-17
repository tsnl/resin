# Developing Resin

## Development

Builds target 64-bit Linux, macOS, and Windows. GPU execution requires a compatible Vulkan
driver; host-only programs do not require Vulkan or a GPU.

New to the implementation? Start with the [guided repository tour](repository-tour.md) and the
[compiler architecture](architecture.md), including the phase crates and their public APIs.

The root is both the `resin` CLI package and a Cargo workspace. Reusable libraries
live under `crates/`; the Zed extension has its own workspace under `editors/zed/`.
Root Cargo commands select `resin`, and `--workspace` builds or tests all native
packages. The `resin` client runs downloaded programs, requests builds, formats source,
and serves stdio LSP. The separate `resin-server` application owns semantic compilation
and native tools. Build/run and LSP require an explicit `RESIN_SERVER` HTTP(S) URL;
formatting works locally without a service.

On Linux or macOS, enter `nix-shell` for Rustup (using `rust-toolchain.toml`), a C compiler,
CMake, Ninja, GLFW's native build dependencies, SPIR-V Tools, and `glslc` for handwritten
test fixtures. On Linux it also supplies Vulkan tools, validation layers, and RenderDoc.
The parser is included in `crates/tree-sitter-resin/`; run `cargo test --workspace` directly.
Non-interactive commands work too: `nix-shell --run 'cargo test --workspace'`.

Cargo builds and statically links the GLFW source bundled in `glfw-sys`; no GLFW installation
or library search path is needed. Cargo uses `rust-toolchain.toml` to install the project's
Rust toolchain. Outside Nix:

- Linux: install Rustup, a C compiler, CMake, Ninja, SPIR-V Tools (`spirv-opt`, `spirv-val`),
  `glslc` for tests, pkg-config, and the X11, Wayland, and xkbcommon development packages,
  including `wayland-scanner`. For GPU execution, add the Vulkan
  loader and a Vulkan driver.
- macOS: install Xcode Command Line Tools (`xcode-select --install`), Rustup, CMake, Ninja, and SPIR-V Tools
  (`brew install cmake ninja spirv-tools shaderc`; Shaderc supplies `glslc` for tests).
  Start the compiler service below before requesting a host program.
  For GPU programs, install the [Vulkan SDK](https://vulkan.lunarg.com/sdk/home#mac), which supplies
  `spirv-opt`, the Vulkan loader, and MoltenVK; use its `setup-env.sh` before running Resin. If Cargo
  strips the loader's search path, run `target/debug/resin` directly from that configured shell.
- Windows: install Rustup's **x86_64-pc-windows-msvc** toolchain, Visual Studio's **Desktop
  development with C++** workload (including a Windows SDK), LLVM Clang, CMake, Ninja, and SPIR-V Tools. Open a
  **Developer PowerShell for VS** targeting x64 and put `clang.exe`, `cmake.exe`, `ninja.exe`, and `spirv-opt.exe` on PATH.
  Start the compiler service below. For GPU programs, install the
  [Vulkan SDK](https://vulkan.lunarg.com/sdk/home#windows) for SPIR-V Tools and a Vulkan-capable GPU driver.

Start a service in one development shell:

```sh
cargo build -p resin -p resin-server -p resin-runtime
cargo run -p resin-server -- --listen 127.0.0.1:7412
```

In another shell, set `RESIN_SERVER=http://127.0.0.1:7412` before running `resin`
or your editor. On Windows PowerShell, use `$env:RESIN_SERVER = "http://127.0.0.1:7412"`.
The [compiler service guide](compiler-service.md) covers remote deployment, native
installation paths, cache settings, header bundles, and pinned dependencies.
The service never starts automatically and the client has no local compiler fallback.

Windows emitted C uses the GNU-style `clang` driver with the MSVC ABI, not `cl` or `clang-cl`.
MinGW and cross-compiling Resin programs are not tested. macOS enables Vulkan portability
enumeration and the portability-subset extension when available, but does not relax the runtime's
Vulkan 1.3 feature requirements. Some MoltenVK devices and presentation paths may still report
`unsupported`; macOS build support is not a promise that every GPU demo works.
Native CI builds and tests the compiler, runtime, and language server without opening windows.
Pull requests and pushes to `main` run on Linux; manually dispatching the `Build` workflow
also checks Windows and macOS. CI caches Rust dependencies and checks Clippy before building
executables. Example formatting, `cargo test --doc`, and a host smoke check then run before
`cargo nextest` executes integration and GPU tests across suites concurrently. A failed check
skips the remaining steps in its job; grammar checks run in parallel with the native job.
Nextest uses the available CPU count for its total worker limit; `--test-threads N`
overrides it. Every suite that takes the GPU test lock belongs to a group with at most
eight tests in flight, also bounded by that total worker limit. CI selects `--profile ci`,
which keeps the GPU limit at two for the hosted runners and their software Vulkan driver.
GPU pointer fixtures build only their requested host entry and prepare their private
native projects before taking that lock, so compilation can overlap device execution.
The lock still serializes GPU access. `cargo test` uses the same fixture
separation; Nextest also overlaps work across test binaries. Each GPU pointer test thread
retains a lazily initialized GPU singleton to establish device availability.
CI omits Rust debug symbols to reduce compile and native-link work while retaining debug
assertions and overflow checks. Local Cargo profile defaults are unchanged.
CI sets `RESIN_TEST_PARTICLE_COUNT=10000` for the particle compute/render test.
Its default remains one million particles for local stress testing; the particle example
also keeps its one-million default. The smaller test spans the same cloud volume and
checks synchronization, bounds, visible output, and sphere lighting.

For parser development, install `cargo install --locked tree-sitter-cli --version 0.27.0`.
After changing `crates/tree-sitter-resin/grammar.js`, regenerate from that directory with
`tree-sitter generate --js-runtime native`.
Commit grammar changes and generated files together in this repository.

The service's native builds require Ninja and a C compiler, selected with `CC` or server `--cc` (default `cc`
on Unix, `clang` on Windows MSVC). Shader builds use the `spirv-opt` binary from
[SPIR-V Tools](https://github.com/KhronosGroup/SPIRV-Tools),
selected with `SPIRV_OPT` or server `--spirv-opt`; `NINJA` selects the build runner. Resin performs
no tool preflight; required commands report errors when executed. Host-only builds never
invoke `spirv-opt`. SPIR-V embedding invokes the running service executable through the platform
`current_exe` API, so it neither searches PATH for Resin nor mixes compiler versions.
Backend tests build generated projects through the same Ninja toolchain.
Generated shaders are validated with `spirv-val` and optimized with `spirv-opt`. Set
`RESIN_REQUIRE_SPIRV_TOOLS=1` to require these tools in tests. Handwritten GLSL fixtures
in runtime tests still use `glslc` from `PATH` and skip if absent, unless GPU tests are required.
The `gpu` feature enables compiler-to-image integration tests, not a different execution mode:

```sh
RESIN_REQUIRE_GPU=1 RESIN_REQUIRE_SPIRV_TOOLS=1 RESIN_REQUIRE_GLSLC=1 cargo test --workspace --all-features
cargo clippy --workspace --all-targets --all-features -- -D warnings
```

Runtime GPU tests expect `glslc` on `PATH` and a working system Vulkan driver.
The development shell supplies the Vulkan loader and window-system libraries through
`LD_LIBRARY_PATH`.
`VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation` enables installed validation layers.

Dedicated [CPU and GPU benchmarks](benchmarks.md) live under `benchmarks/`.
Run them separately with `cargo bench --bench cpu` and `cargo bench --bench gpu`
inside the development environment. Both measure compiled Resin workloads, validate
their outputs, and can save raw timings and hardware metadata with `-- --json PATH`.

Window integration tests use the `gpu` feature and run on a desktop display or Xvfb. Set `RESIN_REQUIRE_WINDOW=1`
to fail instead of skipping when windowing or presentation is unavailable.
When using Xvfb, set `DISPLAY` to its display and `XDG_SESSION_TYPE=x11` to select GLFW's X11
backend. Unsetting `WAYLAND_DISPLAY` alone is insufficient: GLFW can still connect to a default
Wayland socket. With Xvfb already running, run the full suite without excluding window tests:

```sh
XDG_SESSION_TYPE=x11 RESIN_REQUIRE_GPU=1 RESIN_REQUIRE_SPIRV_TOOLS=1 RESIN_REQUIRE_GLSLC=1 RESIN_REQUIRE_WINDOW=1 \
  cargo test --workspace --all-features
```
