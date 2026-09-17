# Getting started

These commands assume a checkout of Resin on Linux or macOS with Nix installed.
The [development guide](development.md) lists native dependencies and the Windows
setup. CPU programs need no GPU; later tutorial chapters need a compatible Vulkan
device and driver.

## 1. Build and start the compiler service

From the repository root, in one terminal:

```sh
nix-shell
cargo build -p resin -p resin-server -p resin-runtime
./target/debug/resin-server --listen 127.0.0.1:7412
```

Leave that terminal running. The compiler service owns the compiler and native
tools. The client downloads and runs the resulting executable locally.

## 2. Run a program

Open a second terminal at the repository root:

```sh
nix-shell
export RESIN_SERVER=http://127.0.0.1:7412
cargo run -- examples/tutorial/01_orbit.resin
```

The program checks several Mandelbrot orbits with assertions and exits successfully.
A failed assertion stops execution. Open the source and change an expected result
to see the difference, then restore it.

## 3. Build an executable

```sh
cargo run -- examples/tutorial/01_orbit.resin -o /tmp/resin-orbits
/tmp/resin-orbits
```

`-o` selects a release build and writes it without running it. Without `-o`, Resin
requests a debug build and runs it. [Tools](tools.md) explains entry selection,
program arguments after `--`, formatting, and editor support.

Continue with [Follow one orbit](tutorial/orbit.md). If the service cannot be
reached, check `RESIN_SERVER` in this terminal and the server terminal's output.
Client and server must use the same negotiated protocol version.
