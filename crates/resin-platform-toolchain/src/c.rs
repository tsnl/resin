use resin_common::prelude::*;
use std::{
    collections::{HashSet, hash_map::DefaultHasher},
    ffi::OsStr,
    fs,
    hash::{Hash, Hasher},
    path::{Path, PathBuf},
    process::Command,
};

use crate::{CProfile, Error, Executable};

use super::platform::{C_FLAGS, LIBRARIES};
use super::{Settings, dependencies, io_error, parent, write_output};

const FLAGS: &[&str] = &[
    "-std=c11",
    "-fno-strict-aliasing",
    "-Wall",
    "-Wextra",
    "-Werror",
    "-pedantic",
];

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

pub(super) fn build_c(
    file: &Path,
    entry: &str,
    source: &str,
    settings: &Settings,
    profile: CProfile,
) -> Result<Executable, Error> {
    let build = lock_executable(settings, file, entry, profile)?;
    let compiler = Compiler::new(settings, profile)?;
    refresh_executable(&compiler, source, build.path())?;
    Ok(build)
}

fn lock_executable(
    settings: &Settings,
    file: &Path,
    entry: &str,
    profile: CProfile,
) -> Result<Executable, Error> {
    let directory = cache_directory(settings, file, entry);
    let lock = super::files::lock_directory(&directory)?;
    let directory = directory.join(profile.directory());
    fs::create_dir_all(&directory)?;
    Ok(Executable {
        executable: directory.join(format!("program{}", std::env::consts::EXE_SUFFIX)),
        _lock: lock,
    })
}

fn cache_directory(settings: &Settings, file: &Path, entry: &str) -> PathBuf {
    // The source identity may exist only in a compiler session's overlay.
    let file = settings.directory.join(file);
    let mut hash = DefaultHasher::new();
    file.hash(&mut hash);
    entry.hash(&mut hash);
    let mut name = file
        .file_stem()
        .unwrap_or(OsStr::new("program"))
        .to_os_string();
    name.push(format!("-{:016x}", hash.finish()));
    settings.cache.join(name)
}

fn refresh_executable(compiler: &Compiler<'_>, source: &str, output: &Path) -> Result<(), Error> {
    let directory = parent(output);
    let dependency_file = directory.join("dependencies");
    let dependencies = read_dependencies(&dependency_file);
    let fingerprint = compiler.fingerprint(source, &dependencies)?;
    let stamp = directory.join("fingerprint");
    if output.is_file() && fs::read_to_string(&stamp).ok().as_ref() == Some(&fingerprint) {
        return Ok(());
    }
    remove_stamp(&stamp)?;
    write_output(source.as_bytes(), &directory.join("program.c"))?;
    let started = std::time::SystemTime::now();
    let found = compiler.compile(source, output)?;
    write_dependencies(&dependency_file, &found)?;
    // Inputs may change while the compiler runs; such a build must not become a cache hit.
    if compiler.fingerprint(source, &dependencies)? == fingerprint
        && unchanged_since(&found, started)
    {
        write_output(compiler.fingerprint(source, &found)?.as_bytes(), &stamp)?;
    }
    Ok(())
}

fn read_dependencies(path: &Path) -> Vec<PathBuf> {
    fs::read_to_string(path)
        .unwrap_or_default()
        .split('\0')
        .filter(|path| !path.is_empty())
        .map(PathBuf::from)
        .collect()
}

fn write_dependencies(path: &Path, dependencies: &[PathBuf]) -> Result<(), Error> {
    let paths = dependencies
        .iter()
        .map(|path| path.to_string_lossy())
        .collect::<Vec<_>>()
        .join("\0");
    write_output(paths.as_bytes(), path)
}

fn remove_stamp(path: &Path) -> Result<(), Error> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
}

fn unchanged_since(paths: &[PathBuf], started: std::time::SystemTime) -> bool {
    paths.iter().all(|path| {
        fs::metadata(path)
            .and_then(|m| m.modified())
            .is_ok_and(|time| time <= started)
    })
}

pub(super) fn compile_c(source: &str, output: &Path, settings: &Settings) -> Result<(), Error> {
    Compiler::new(settings, CProfile::Release)?
        .compile(source, &settings.directory.join(output))
        .map(|_| ())
}

struct Compiler<'a> {
    profile: CProfile,
    executable: &'a Path,
    include: &'a Path,
    library: &'a Path,
    settings: &'a Settings,
}

impl<'a> Compiler<'a> {
    fn new(settings: &'a Settings, profile: CProfile) -> Result<Self, Error> {
        Ok(Self {
            profile,
            executable: settings.cc()?,
            include: &settings.runtime_include,
            library: settings.runtime_library()?,
            settings,
        })
    }

    fn compile(&self, source: &str, output: &Path) -> Result<Vec<PathBuf>, Error> {
        let temp = TempDir::new(parent(output)).map_err(io_error)?;
        let input = temp.path().join("program.c");
        let binary = temp
            .path()
            .join(format!("program{}", std::env::consts::EXE_SUFFIX));
        let depfile = temp.path().join("program.d");
        fs::write(&input, format!("{source}\n")).map_err(io_error)?;
        super::process::run(self.command(&input, &binary, &depfile), "C compiler")?;
        let dependencies =
            dependencies::parse(&fs::read_to_string(depfile).map_err(io_error)?, &input);
        fs::rename(binary, output).map_err(io_error)?;
        Ok(dependencies)
    }

    fn command(&self, input: &Path, binary: &Path, depfile: &Path) -> Command {
        let mut command = self.settings.command(self.executable);
        command
            .args(FLAGS)
            .args(C_FLAGS)
            .args(self.profile.flags())
            .args(["-MD", "-MT", "resin", "-MF"])
            .arg(depfile)
            .arg("-I")
            .arg(self.include)
            .arg(input)
            .arg(self.library)
            .arg("-o")
            .arg(binary)
            .args(LIBRARIES);
        command
    }

    fn fingerprint(&self, source: &str, dependencies: &[PathBuf]) -> Result<String, Error> {
        let mut hash = DefaultHasher::new();
        source.hash(&mut hash);
        FLAGS.hash(&mut hash);
        C_FLAGS.hash(&mut hash);
        self.profile.flags().hash(&mut hash);
        LIBRARIES.hash(&mut hash);
        self.executable.hash(&mut hash);
        self.settings.environment.hash(&mut hash);
        self.settings.directory.hash(&mut hash);
        metadata(&self.settings.executable, &mut hash)?;
        metadata(self.executable, &mut hash)?;
        contents(self.library, &mut hash)?;
        headers(self.include, &mut hash, &mut HashSet::new())?;
        for path in dependencies {
            path.hash(&mut hash);
            if let Err(error) = contents(path, &mut hash) {
                error.to_string().hash(&mut hash);
            }
        }
        Ok(format!("{:016x}", hash.finish()))
    }
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
