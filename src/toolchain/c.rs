use std::{
    collections::{HashSet, hash_map::DefaultHasher},
    ffi::OsStr,
    fs,
    hash::{Hash, Hasher},
    path::{Path, PathBuf},
    process::Command,
};

use crate::backend::Error;

use super::{TempDir, dependencies, io_error, parent, write_output};

const FLAGS: &[&str] = &[
    "-std=c11",
    "-fno-strict-aliasing",
    "-Wall",
    "-Wextra",
    "-Werror",
    "-pedantic",
];
const LIBRARIES: &[&str] = &["-ldl", "-lpthread", "-lm", "-lrt", "-lutil"];

#[derive(Clone, Copy)]
pub enum CProfile {
    Debug,
    Release,
}

impl CProfile {
    fn directory(self) -> &'static str {
        match self {
            Self::Debug => "debug",
            Self::Release => "release",
        }
    }

    fn flags(self) -> &'static [&'static str] {
        match self {
            Self::Debug => &["-O0"],
            Self::Release => &["-O3"],
        }
    }
}

/// Keeps the artifact locked through execution and copying.
pub struct CBuild {
    executable: PathBuf,
    _lock: fs::File,
}

impl CBuild {
    pub fn executable(&self) -> &Path {
        &self.executable
    }
}

pub fn build_c(
    file: &Path,
    entry: &str,
    source: &str,
    compiler: &OsStr,
    profile: CProfile,
) -> Result<CBuild, Error> {
    let file = fs::canonicalize(file).map_err(io_error)?;
    let mut hash = DefaultHasher::new();
    file.hash(&mut hash);
    entry.hash(&mut hash);
    let mut name = file
        .file_stem()
        .unwrap_or(OsStr::new("program"))
        .to_os_string();
    name.push(format!("-{:016x}", hash.finish()));
    let directory = std::env::current_dir()
        .map_err(io_error)?
        .join("build")
        .join(name);
    fs::create_dir_all(&directory).map_err(io_error)?;
    let lock = fs::File::options()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(directory.join("lock"))
        .map_err(io_error)?;
    lock.lock().map_err(io_error)?;
    let directory = directory.join(profile.directory());
    fs::create_dir_all(&directory).map_err(io_error)?;
    let build = CBuild {
        executable: directory.join(format!("program{}", std::env::consts::EXE_SUFFIX)),
        _lock: lock,
    };
    let compiler = Compiler::new(compiler, profile)?;
    let dependency_file = directory.join("dependencies");
    let dependencies: Vec<_> = fs::read_to_string(&dependency_file)
        .unwrap_or_default()
        .split('\0')
        .filter(|path| !path.is_empty())
        .map(PathBuf::from)
        .collect();
    let fingerprint = compiler.fingerprint(source, &dependencies)?;
    let stamp = directory.join("fingerprint");
    if build.executable.is_file() && fs::read_to_string(&stamp).ok().as_ref() == Some(&fingerprint)
    {
        return Ok(build);
    }
    match fs::remove_file(&stamp) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(io_error(error)),
    }
    write_output(source.as_bytes(), &directory.join("program.c"))?;
    let started = std::time::SystemTime::now();
    let found = compiler.compile(source, &build.executable)?;
    let paths = found
        .iter()
        .map(|path| path.to_string_lossy())
        .collect::<Vec<_>>()
        .join("\0");
    write_output(paths.as_bytes(), &dependency_file)?;
    if compiler.fingerprint(source, &dependencies)? == fingerprint
        && found.iter().all(|path| {
            fs::metadata(path)
                .and_then(|m| m.modified())
                .is_ok_and(|time| time <= started)
        })
    {
        write_output(compiler.fingerprint(source, &found)?.as_bytes(), &stamp)?;
    }
    Ok(build)
}

pub fn compile_c(source: &str, output: &Path, compiler: &OsStr) -> Result<(), Error> {
    Compiler::new(compiler, CProfile::Release)?
        .compile(source, output)
        .map(|_| ())
}

struct Compiler {
    profile: CProfile,
    executable: PathBuf,
    include: PathBuf,
    library: PathBuf,
}

impl Compiler {
    fn new(compiler: &OsStr, profile: CProfile) -> Result<Self, Error> {
        Ok(Self {
            profile,
            executable: resolve(compiler)?,
            include: std::env::var_os("RESIN_RUNTIME_INCLUDE")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from(resin_runtime::INCLUDE_DIR)),
            library: runtime_library()?,
        })
    }

    fn compile(&self, source: &str, output: &Path) -> Result<Vec<PathBuf>, Error> {
        let temp = TempDir::new(parent(output)).map_err(io_error)?;
        let input = temp.path().join("program.c");
        let binary = temp.path().join("program");
        let depfile = temp.path().join("program.d");
        fs::write(&input, source).map_err(io_error)?;
        let result = Command::new(&self.executable)
            .args(FLAGS)
            .args(self.profile.flags())
            .args(["-MD", "-MT", "resin", "-MF"])
            .arg(&depfile)
            .arg("-I")
            .arg(&self.include)
            .arg(&input)
            .arg(&self.library)
            .arg("-o")
            .arg(&binary)
            .args(LIBRARIES)
            .output()
            .map_err(|error| Error(format!("cannot run {}: {error}", self.executable.display())))?;
        if !result.status.success() {
            return Err(Error(format!(
                "C compiler failed ({}):\n{}",
                result.status,
                String::from_utf8_lossy(&result.stderr)
            )));
        }
        let dependencies =
            dependencies::parse(&fs::read_to_string(depfile).map_err(io_error)?, &input);
        fs::rename(binary, output).map_err(io_error)?;
        Ok(dependencies)
    }

    fn fingerprint(&self, source: &str, dependencies: &[PathBuf]) -> Result<String, Error> {
        let mut hash = DefaultHasher::new();
        source.hash(&mut hash);
        FLAGS.hash(&mut hash);
        self.profile.flags().hash(&mut hash);
        LIBRARIES.hash(&mut hash);
        self.executable.hash(&mut hash);
        let mut environment: Vec<_> = std::env::vars_os().collect();
        environment.sort();
        environment.hash(&mut hash);
        metadata(&std::env::current_exe().map_err(io_error)?, &mut hash)?;
        metadata(&self.executable, &mut hash)?;
        contents(&self.library, &mut hash)?;
        headers(&self.include, &mut hash, &mut HashSet::new())?;
        for path in dependencies {
            path.hash(&mut hash);
            if let Err(error) = contents(path, &mut hash) {
                error.to_string().hash(&mut hash);
            }
        }
        Ok(format!("{:016x}", hash.finish()))
    }
}

pub(super) fn resolve(compiler: &OsStr) -> Result<PathBuf, Error> {
    let candidates = if Path::new(compiler).components().count() > 1 {
        vec![PathBuf::from(compiler)]
    } else {
        std::env::split_paths(&std::env::var_os("PATH").unwrap_or_default())
            .map(|path| path.join(compiler))
            .collect()
    };
    for path in candidates {
        if !path.is_file() {
            continue;
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            if fs::metadata(&path).map_err(io_error)?.permissions().mode() & 0o111 == 0 {
                continue;
            }
        }
        // Preserve the invocation name: compiler drivers may inspect argv[0].
        return std::path::absolute(path).map_err(io_error);
    }
    Err(Error(format!(
        "cannot run {}: executable not found",
        compiler.to_string_lossy()
    )))
}

pub(super) fn metadata(path: &Path, hash: &mut DefaultHasher) -> Result<(), Error> {
    let path = fs::canonicalize(path).map_err(io_error)?;
    let metadata = fs::metadata(&path).map_err(io_error)?;
    path.hash(hash);
    metadata.len().hash(hash);
    metadata.modified().map_err(io_error)?.hash(hash);
    Ok(())
}

fn contents(path: &Path, hash: &mut DefaultHasher) -> Result<(), Error> {
    metadata(path, hash)?;
    fs::read(path).map_err(io_error)?.hash(hash);
    Ok(())
}

fn headers(
    path: &Path,
    hash: &mut DefaultHasher,
    visited: &mut HashSet<PathBuf>,
) -> Result<(), Error> {
    path.hash(hash);
    if !visited.insert(fs::canonicalize(path).map_err(io_error)?) {
        return Ok(());
    }
    if path.is_dir() {
        let mut entries = fs::read_dir(path)
            .map_err(io_error)?
            .map(|entry| entry.map(|entry| entry.path()))
            .collect::<std::io::Result<Vec<_>>>()
            .map_err(io_error)?;
        entries.sort();
        for entry in entries {
            headers(&entry, hash, visited)?;
        }
    } else {
        contents(path, hash)?;
    }
    Ok(())
}

fn runtime_library() -> Result<PathBuf, Error> {
    if let Some(path) = std::env::var_os("RESIN_RUNTIME_LIB") {
        let path = PathBuf::from(path);
        if path.is_file() {
            return Ok(path);
        }
        return Err(Error(format!(
            "runtime library not found: {}",
            path.display()
        )));
    }
    let executable = std::env::current_exe().map_err(io_error)?;
    let directory = executable.parent().unwrap();
    for path in [
        directory.join("deps/libresin_runtime.a"),
        directory.join("libresin_runtime.a"),
    ] {
        if path.is_file() {
            return Ok(path);
        }
    }
    Err(Error(
        "cannot find libresin_runtime.a beside the compiler or in deps; set RESIN_RUNTIME_LIB"
            .into(),
    ))
}
