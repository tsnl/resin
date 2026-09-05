use std::{
    ffi::OsStr,
    fs, io,
    path::{Path, PathBuf},
    process::Command,
    sync::atomic::{AtomicU64, Ordering},
};

use crate::backend::Error;
use crate::backend::glsl::Stage;

pub fn compile_glsl(source: &str, stage: Stage, compiler: &OsStr) -> Result<Vec<u8>, Error> {
    let temp = TempDir::new(&std::env::temp_dir()).map_err(io_error)?;
    let input = temp.path().join("shader.glsl");
    let output = temp.path().join("shader.spv");
    fs::write(&input, source).map_err(io_error)?;
    let result = Command::new(compiler)
        .arg(format!("-fshader-stage={}", stage.name()))
        .args(["--target-env=vulkan1.3", "-O", "-Werror"])
        .arg(&input)
        .arg("-o")
        .arg(&output)
        .output()
        .map_err(|error| {
            Error(format!(
                "cannot run {}: {error}",
                compiler.to_string_lossy()
            ))
        })?;
    if !result.status.success() {
        return Err(Error(format!(
            "shader compiler failed ({}):\n{}",
            result.status,
            String::from_utf8_lossy(&result.stderr)
        )));
    }
    let bytes = fs::read(output).map_err(io_error)?;
    if bytes.len() < 20 || bytes.len() % 4 != 0 || bytes[..4] != [3, 2, 35, 7] {
        return Err(Error("shader compiler returned invalid SPIR-V".into()));
    }
    Ok(bytes)
}

pub fn compile_c(source: &str, output: &Path, compiler: &OsStr) -> Result<(), Error> {
    let temp = TempDir::new(parent(output)).map_err(io_error)?;
    let input = temp.path().join("program.c");
    let binary = temp.path().join("program");
    fs::write(&input, source).map_err(io_error)?;
    let result = Command::new(compiler)
        .args([
            "-std=c11",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pedantic",
        ])
        .arg(&input)
        .arg("-o")
        .arg(&binary)
        .output()
        .map_err(|error| {
            Error(format!(
                "cannot run {}: {error}",
                compiler.to_string_lossy()
            ))
        })?;
    if !result.status.success() {
        return Err(Error(format!(
            "C compiler failed ({}):\n{}",
            result.status,
            String::from_utf8_lossy(&result.stderr)
        )));
    }
    fs::rename(binary, output).map_err(io_error)
}

pub fn write_output(bytes: &[u8], output: &Path) -> Result<(), Error> {
    let temp = TempDir::new(parent(output)).map_err(io_error)?;
    let path = temp.path().join("output");
    fs::write(&path, bytes).map_err(io_error)?;
    fs::rename(path, output).map_err(io_error)
}

fn parent(path: &Path) -> &Path {
    path.parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or(Path::new("."))
}

fn io_error(error: io::Error) -> Error {
    Error(error.to_string())
}

pub struct TempDir(PathBuf);

impl TempDir {
    pub fn new(parent: &Path) -> io::Result<Self> {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        for _ in 0..1000 {
            let id = NEXT.fetch_add(1, Ordering::Relaxed);
            let path = parent.join(format!(".resin-{}-{id}", std::process::id()));
            match fs::create_dir(&path) {
                Ok(()) => return Ok(Self(path)),
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error),
            }
        }
        Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            "cannot create a temporary directory",
        ))
    }

    pub fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
