use resin_common::prelude::*;
use std::{
    collections::hash_map::DefaultHasher,
    fs,
    hash::{Hash, Hasher},
    path::Path,
    process::Command,
};

use crate::Error;

use super::{Settings, c, write_output};

/// Compile GLSL with an environment-sensitive, locked SPIR-V cache.
pub(super) fn build_glsl(
    source: &str,
    stage: Stage,
    settings: &Settings,
) -> Result<Vec<u8>, Error> {
    let key = fingerprint(source, stage, settings)?;
    let directory = settings.cache.join("shaders").join(&key);
    let _lock = super::files::lock_directory(&directory)?;
    let output = directory.join("shader.spv");
    if let Ok(bytes) = fs::read(&output)
        && valid_spirv(&bytes)
    {
        return Ok(bytes);
    }
    let bytes = compile_glsl(source, stage, settings)?;
    if fingerprint(source, stage, settings)? == key {
        write_output(source.as_bytes(), &directory.join("shader.glsl"))?;
        write_output(&bytes, &output)?;
    }
    Ok(bytes)
}

fn fingerprint(source: &str, stage: Stage, settings: &Settings) -> Result<String, Error> {
    let compiler = settings.glslc()?;
    let mut hash = DefaultHasher::new();
    source.hash(&mut hash);
    stage.name().hash(&mut hash);
    settings.environment.hash(&mut hash);
    settings.directory.hash(&mut hash);
    c::metadata(&settings.executable, &mut hash)?;
    compiler.hash(&mut hash);
    c::metadata(compiler, &mut hash)?;
    Ok(format!("{:016x}", hash.finish()))
}

pub(super) fn compile_glsl(
    source: &str,
    stage: Stage,
    settings: &Settings,
) -> Result<Vec<u8>, Error> {
    let compiler = settings.glslc()?;
    let temp = TempDir::new(&settings.temporary)?;
    let input = temp.path().join("shader.glsl");
    let output = temp.path().join("shader.spv");
    fs::write(&input, source)?;
    super::process::run(
        command(settings, compiler, stage, &input, &output),
        "shader compiler",
    )?;
    read_spirv(&output)
}

fn command(
    settings: &Settings,
    compiler: &Path,
    stage: Stage,
    input: &Path,
    output: &Path,
) -> Command {
    let mut command = settings.command(compiler);
    command
        .arg(format!("-fshader-stage={}", stage.name()))
        .args(["--target-env=vulkan1.3", "-O", "-Werror"])
        .arg(input)
        .arg("-o")
        .arg(output);
    command
}

fn read_spirv(output: &Path) -> Result<Vec<u8>, Error> {
    let bytes = fs::read(output)?;
    if !valid_spirv(&bytes) {
        return Err(Error("shader compiler returned invalid SPIR-V".into()));
    }
    Ok(bytes)
}

fn valid_spirv(bytes: &[u8]) -> bool {
    bytes.len() >= 20 && bytes.len().is_multiple_of(4) && bytes[..4] == [3, 2, 35, 7]
}
