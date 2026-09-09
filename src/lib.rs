//! Resin's command-line interface. Compilation and editor services live in library crates.

use resin_compiler as compiler;
use resin_cst as cst;
use resin_platform_toolchain as toolchain;
mod cli;

pub fn main() -> ! {
    cli::main()
}
