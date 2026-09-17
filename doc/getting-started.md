# Getting started

Resin consists of a **compiler service** that compiles programs and performs editor
analysis with shared caches, and a **CLI tool** that sends it source code, runs the
resulting programs locally, and connects editors through LSP.

You start one service and point the CLI and your editor at it with `RESIN_SERVER`;
keeping compilation and caches in that process lets separate builds and editor
requests reuse work. The service may run locally or on another machine. The steps
below use a local service bound to the loopback interface.

A future installer is intended to register and start the compiler as a system-wide
service automatically. Today, a checkout requires the manual startup below; the
CLI does not start or discover a service for you.

## Setup

These instructions assume you have cloned the repository. Building Resin requires
Rustup (the checkout pins a Rust toolchain), a C compiler, CMake, Ninja, and the
platform's native development libraries. The compiler service also needs SPIR-V
Tools for shader builds. Install the [dependencies for your platform](development.md#development)
before continuing. Linux, macOS, and Windows are supported build hosts; on Windows,
use the documented Visual Studio developer PowerShell with LLVM Clang.

CPU programs need no GPU. The later graphics chapters require a compatible Vulkan
device and driver, and interactive windows need a display. A remote compiler does
not provide a GPU for programs executed on your own machine.

### For Nix users

On Linux or macOS, the repository includes a `shell.nix` with the development tools
and libraries. Run `nix-shell` in each terminal used below to enter that environment;
it is a convenience alternative to installing the dependencies yourself.

```sh
nix-shell
```

## Build and start

From the repository root, build the client, service, and runtime, then leave the
service running in this terminal:

```sh
cargo build -p resin -p resin-server -p resin-runtime
./target/debug/resin-server --listen 127.0.0.1:7412
```

The service handles compilation and owns the native tools and caches. It does not
run your application. Its [configuration guide](compiler-service.md) covers library
paths, cache settings, remote deployment, and manually installing a system service.

## Run a program

In another terminal at the repository root, tell the client where the service is:

```sh
export RESIN_SERVER=http://127.0.0.1:7412
cargo run -- examples/tutorial/basics/hello.resin
```

In PowerShell, use `$env:RESIN_SERVER = "http://127.0.0.1:7412"` for that first line.
`cargo run` runs the Resin CLI from this checkout. The CLI uploads the source,
downloads the compiled executable, and runs it locally with your working directory,
arguments, and environment. You should see `Hello, Resin!`.

## Build an executable

To save an optimized executable instead of immediately running a debug build:

```sh
cargo run -- examples/tutorial/basics/hello.resin -o hello
./hello
```

On Windows use `-o hello.exe`, then `./hello.exe`. Once built, the executable does
not contact the compiler service. [Tools](tools.md) explains entry selection,
program arguments after `--`, formatting, and editor setup. Formatting and source
documentation generation are local operations and do not need `RESIN_SERVER`.

Continue with [Your first Resin program](tutorial/basics.md). If a build cannot reach
the service, check `RESIN_SERVER` in the client terminal and the server's output.
Client and server must agree on the negotiated protocol version.
