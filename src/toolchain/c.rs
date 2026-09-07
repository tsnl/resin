use std::{
    collections::{HashSet, hash_map::DefaultHasher},
    ffi::OsStr,
    fs,
    hash::{Hash, Hasher},
    path::{Path, PathBuf},
    process::Command,
};

use crate::backend::Error;

use super::platform::{C_FLAGS, LIBRARIES};
use super::{Settings, TempDir, dependencies, io_error, parent, write_output};

const FLAGS: &[&str] = &[
    "-std=c11",
    "-fno-strict-aliasing",
    "-Wall",
    "-Wextra",
    "-Werror",
    "-pedantic",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
    settings: &Settings,
    profile: CProfile,
) -> Result<CBuild, Error> {
    // The caller supplies the source identity, which may exist only in a session overlay.
    let file = settings.directory.join(file);
    let mut hash = DefaultHasher::new();
    file.hash(&mut hash);
    entry.hash(&mut hash);
    let mut name = file
        .file_stem()
        .unwrap_or(OsStr::new("program"))
        .to_os_string();
    name.push(format!("-{:016x}", hash.finish()));
    let directory = settings.cache.join(name);
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
    let compiler = Compiler::new(settings, profile)?;
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

pub fn compile_c(source: &str, output: &Path, settings: &Settings) -> Result<(), Error> {
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
        let result = Command::new(self.executable)
            .current_dir(&self.settings.directory)
            .env_clear()
            .envs(&self.settings.environment)
            .args(FLAGS)
            .args(C_FLAGS)
            .args(self.profile.flags())
            .args(["-MD", "-MT", "resin", "-MF"])
            .arg(&depfile)
            .arg("-I")
            .arg(self.include)
            .arg(&input)
            .arg(self.library)
            .arg("-o")
            .arg(&binary)
            .args(LIBRARIES)
            .output()
            .map_err(|error| Error(format!("cannot run {}: {error}", self.executable.display())))?;
        if !result.status.success() {
            return Err(Error(format!(
                "C compiler failed ({}):\n{}{}",
                result.status,
                String::from_utf8_lossy(&result.stdout),
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
