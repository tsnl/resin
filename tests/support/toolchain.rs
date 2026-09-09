#![allow(dead_code)]
use resin_toolchain::Environment;
use resin_toolchain::Toolchain;
use std::{ffi::OsStr, fs, path::Path};
use tempfile::TempDir;

pub fn c(compiler: &OsStr) -> Toolchain {
    environment().toolchain(Some(compiler), None)
}

pub fn spirv(compiler: &OsStr) -> Toolchain {
    environment().toolchain(None, Some(compiler))
}

fn environment() -> Environment {
    let mut environment = Environment::capture().unwrap();
    environment.executable = env!("CARGO_BIN_EXE_resin").into();
    environment
}

/// Handwritten C runtime tests use the same Ninja execution boundary as Resin.
pub fn compile_c(
    source: &str,
    output: &Path,
    compiler: &OsStr,
) -> Result<(), Box<dyn std::error::Error>> {
    let directory = TempDir::new()?;
    fs::write(directory.path().join("main.c"), source)?;
    fs::write(
        directory.path().join("build.ninja"),
        "include toolchain.ninja\nrule c\n  command = $cc $cflags $in $ldflags -o $out\nbuild program: c main.c | toolchain.state\ndefault program\n",
    )?;
    let built = c(compiler).build(
        directory.path(),
        "native-test",
        "main",
        resin_toolchain::CProfile::Release,
    )?;
    built.executable("program")?.copy_to(output)?;
    Ok(())
}

/// Malformed native input belongs to the native-tool boundary, without any LIR.
pub fn optimize_spirv(
    source: &[u8],
    compiler: &OsStr,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let directory = TempDir::new()?;
    fs::write(directory.path().join("shader.unoptimized.spv"), source)?;
    fs::write(
        directory.path().join("build.ninja"),
        "include toolchain.ninja\nrule optimize\n  command = $spirv_opt --target-env=vulkan1.3 -O $in -o $out\nbuild shader.spv: optimize shader.unoptimized.spv | toolchain.state\ndefault shader.spv\n",
    )?;
    let built = spirv(compiler).build(
        directory.path(),
        "shader-test",
        "shader",
        resin_toolchain::CProfile::Release,
    )?;
    Ok(fs::read(built.path("shader.spv"))?)
}
