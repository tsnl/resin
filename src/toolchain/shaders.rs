use std::{
    collections::hash_map::DefaultHasher,
    ffi::OsStr,
    fs,
    hash::{Hash, Hasher},
};

use crate::{
    backend::{Error, c::Shader, glsl},
    ir::{Instr, Module},
};

use super::{c, compile_glsl, io_error, write_output};

pub fn build_shaders(module: &Module, compiler: &OsStr) -> Result<Vec<Shader>, Error> {
    let mut shaders: Vec<Shader> = Vec::new();
    for instr in module
        .functions
        .iter()
        .flat_map(|f| &f.blocks)
        .flat_map(|b| &b.instrs)
    {
        let Instr::Shader { function, stage } = instr else {
            continue;
        };
        if shaders
            .iter()
            .any(|shader| shader.function == *function && shader.stage.name() == stage.as_ref())
        {
            continue;
        }
        let stage = stage.parse()?;
        let source = glsl::emit_function(module, *function, stage)?;
        let bytes = cached(&source, stage, compiler)?;
        shaders.push(Shader {
            function: *function,
            stage,
            words: bytes
                .chunks_exact(4)
                .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
                .collect(),
        });
    }
    Ok(shaders)
}

fn cached(source: &str, stage: glsl::Stage, compiler: &OsStr) -> Result<Vec<u8>, Error> {
    let compiler = c::resolve(compiler)?;
    let fingerprint = || {
        let mut hash = DefaultHasher::new();
        source.hash(&mut hash);
        stage.name().hash(&mut hash);
        let mut environment: Vec<_> = std::env::vars_os().collect();
        environment.sort();
        environment.hash(&mut hash);
        c::metadata(&std::env::current_exe().map_err(io_error)?, &mut hash)?;
        compiler.hash(&mut hash);
        c::metadata(&compiler, &mut hash)?;
        Ok::<_, Error>(format!("{:016x}", hash.finish()))
    };
    let key = fingerprint()?;
    let directory = std::env::current_dir()
        .map_err(io_error)?
        .join("build/shaders")
        .join(&key);
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
    let bytes = compile_glsl(source, stage, compiler.as_os_str())?;
    if fingerprint()? == key {
        write_output(source.as_bytes(), &directory.join("shader.glsl"))?;
        write_output(&bytes, &output)?;
    }
    Ok(bytes)
}
