use std::{
    collections::hash_map::DefaultHasher,
    fs,
    hash::{Hash, Hasher},
};

use crate::{codegen::Error, glsl};

use super::{Settings, c, compile_glsl, io_error, write_output};

/// Compile GLSL with an environment-sensitive, locked SPIR-V cache.
pub fn build_glsl(source: &str, stage: glsl::Stage, settings: &Settings) -> Result<Vec<u8>, Error> {
    let compiler = settings.glslc()?;
    let fingerprint = || {
        let mut hash = DefaultHasher::new();
        source.hash(&mut hash);
        stage.name().hash(&mut hash);
        settings.environment.hash(&mut hash);
        settings.directory.hash(&mut hash);
        c::metadata(&settings.executable, &mut hash)?;
        compiler.hash(&mut hash);
        c::metadata(compiler, &mut hash)?;
        Ok::<_, Error>(format!("{:016x}", hash.finish()))
    };
    let key = fingerprint()?;
    let directory = settings.cache.join("shaders").join(&key);
    fs::create_dir_all(&directory).map_err(io_error)?;
    let lock = fs::File::options()
        .create(true)
        .truncate(false)
        .write(true)
        .read(true)
        .open(directory.join("lock"))
        .map_err(io_error)?;
    lock.lock().map_err(io_error)?;
    let output = directory.join("shader.spv");
    if let Ok(bytes) = fs::read(&output)
        && super::valid_spirv(&bytes)
    {
        return Ok(bytes);
    }
    let bytes = compile_glsl(source, stage, settings)?;
    if fingerprint()? == key {
        write_output(source.as_bytes(), &directory.join("shader.glsl"))?;
        write_output(&bytes, &output)?;
    }
    Ok(bytes)
}
