//! Explicit host settings supplied by the compiler's caller.
use crate::{CProfile, Error, files, platform};
use std::{
    collections::{BTreeMap, HashSet, hash_map::DefaultHasher},
    ffi::{OsStr, OsString},
    fs,
    hash::{Hash, Hasher},
    path::{Path, PathBuf},
    process::Command,
};

/// Environment and paths resolved before compilation. Discovery failures are retained
/// so unused tools (in particular glslc for host-only code) remain optional.
pub(super) struct Settings {
    pub(super) cc: PathBuf,
    pub(super) glslc: PathBuf,
    pub(super) ninja: PathBuf,
    pub(super) runtime_include: PathBuf,
    pub(super) runtime_library: PathBuf,
    pub(super) environment: BTreeMap<OsString, OsString>,
    pub(super) cache: PathBuf,
    pub(super) directory: PathBuf,
    pub(super) executable: PathBuf,
}

impl Settings {
    pub(super) fn command(&self, executable: &Path) -> Result<Command, Error> {
        let mut command = Command::new(executable);
        command.current_dir(&self.directory).env_clear();
        for (name, value) in &self.environment {
            command.env(name, self.environment_value(name, value)?);
        }
        Ok(command)
    }

    fn environment_value(&self, name: &OsStr, value: &OsStr) -> Result<OsString, Error> {
        if !search_path_variable(name) {
            return Ok(value.into());
        }
        // Ninja runs inside its cache; relative search entries still name the
        // caller's directory, including empty entries which mean that directory.
        std::env::join_paths(std::env::split_paths(value).map(|path| self.directory.join(path)))
            .map_err(|error| Error(format!("invalid {}: {error}", name.to_string_lossy())))
    }

    pub(super) fn configure(&self, profile: CProfile, directory: &Path) -> Result<(), Error> {
        for path in [
            &self.cc,
            &self.glslc,
            &self.executable,
            &self.runtime_include,
            &self.runtime_library,
        ] {
            if path
                .as_os_str()
                .as_encoded_bytes()
                .iter()
                .any(|byte| matches!(byte, b'\n' | b'\r'))
            {
                return Err(Error("Ninja tool paths cannot contain newlines".into()));
            }
        }
        let mut text = Vec::new();
        command_variable(&mut text, "cc", [self.cc.as_os_str()]);
        command_variable(&mut text, "glslc", [self.glslc.as_os_str()]);
        command_variable(&mut text, "resin", [self.executable.as_os_str()]);
        command_variable(
            &mut text,
            "cflags",
            self.cflags(profile).iter().map(OsString::as_os_str),
        );
        command_variable(
            &mut text,
            "ldflags",
            self.ldflags().iter().map(OsString::as_os_str),
        );
        text.extend_from_slice(b"runtime_library = ");
        escape_ninja(
            &mut text,
            self.runtime_library.as_os_str().as_encoded_bytes(),
            true,
        );
        text.push(b'\n');
        files::write_changed(&text, &directory.join("toolchain.ninja"))?;
        files::write_changed(
            self.fingerprint().as_bytes(),
            &directory.join("toolchain.state"),
        )
    }

    fn cflags(&self, profile: CProfile) -> Vec<OsString> {
        [
            "-std=c11",
            "-fno-strict-aliasing",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pedantic",
        ]
        .into_iter()
        .chain(platform::C_FLAGS.iter().copied())
        .chain([profile.optimization(), "-I"])
        .map(OsString::from)
        .chain([self.runtime_include.clone().into_os_string()])
        .collect()
    }

    fn ldflags(&self) -> Vec<OsString> {
        std::iter::once(self.runtime_library.clone().into_os_string())
            .chain(platform::LIBRARIES.iter().map(OsString::from))
            .collect()
    }

    fn fingerprint(&self) -> String {
        let mut hash = DefaultHasher::new();
        self.environment.hash(&mut hash);
        self.directory.hash(&mut hash);
        for path in [&self.executable, &self.cc, &self.glslc, &self.ninja] {
            hash_metadata(path, &mut hash);
        }
        hash_contents(&self.runtime_library, &mut hash, &mut HashSet::new());
        hash_contents(&self.runtime_include, &mut hash, &mut HashSet::new());
        format!("{:016x}\n", hash.finish())
    }
}

fn search_path_variable(name: &OsStr) -> bool {
    [
        "PATH",
        "CPATH",
        "C_INCLUDE_PATH",
        "CPLUS_INCLUDE_PATH",
        "OBJC_INCLUDE_PATH",
        "LIBRARY_PATH",
        "COMPILER_PATH",
    ]
    .into_iter()
    .any(|candidate| {
        if cfg!(windows) {
            name.as_encoded_bytes()
                .eq_ignore_ascii_case(candidate.as_bytes())
        } else {
            name == candidate
        }
    }) || cfg!(windows)
        && ["INCLUDE", "LIB", "LIBPATH"].into_iter().any(|candidate| {
            name.as_encoded_bytes()
                .eq_ignore_ascii_case(candidate.as_bytes())
        })
}

fn command_variable<'a>(out: &mut Vec<u8>, name: &str, args: impl IntoIterator<Item = &'a OsStr>) {
    out.extend_from_slice(name.as_bytes());
    out.extend_from_slice(b" =");
    for arg in args {
        out.push(b' ');
        escape_ninja(out, &quote(arg.as_encoded_bytes()), false);
    }
    out.push(b'\n');
}

fn escape_ninja(out: &mut Vec<u8>, bytes: &[u8], path: bool) {
    for &byte in bytes {
        if byte == b'$' || (path && matches!(byte, b' ' | b':')) {
            out.push(b'$');
        }
        out.push(byte);
    }
}

#[cfg(not(windows))]
fn quote(bytes: &[u8]) -> Vec<u8> {
    let mut out = vec![b'\''];
    for &byte in bytes {
        if byte == b'\'' {
            out.extend_from_slice(b"'\\''");
        } else {
            out.push(byte);
        }
    }
    out.push(b'\'');
    out
}

#[cfg(windows)]
fn quote(bytes: &[u8]) -> Vec<u8> {
    let mut out = vec![b'"'];
    let mut slashes = 0;
    for &byte in bytes {
        if byte == b'\\' {
            slashes += 1;
            continue;
        }
        out.extend(std::iter::repeat_n(
            b'\\',
            if byte == b'"' {
                2 * slashes + 1
            } else {
                slashes
            },
        ));
        slashes = 0;
        out.push(byte);
    }
    out.extend(std::iter::repeat_n(b'\\', 2 * slashes));
    out.push(b'"');
    out
}

fn hash_metadata(path: &Path, hash: &mut DefaultHasher) {
    path.hash(hash);
    fs::canonicalize(path).ok().hash(hash);
    match fs::metadata(path) {
        Ok(metadata) => {
            metadata.len().hash(hash);
            metadata.modified().ok().hash(hash);
        }
        Err(error) => error.to_string().hash(hash),
    }
}

fn hash_contents(path: &Path, hash: &mut DefaultHasher, seen: &mut HashSet<PathBuf>) {
    hash_metadata(path, hash);
    if !seen.insert(fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())) {
        return;
    }
    if let Ok(entries) = fs::read_dir(path) {
        let mut entries: Vec<_> = entries
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .collect();
        entries.sort();
        for entry in entries {
            hash_contents(&entry, hash, seen);
        }
    } else {
        match fs::read(path) {
            Ok(bytes) => bytes.hash(hash),
            Err(error) => error.to_string().hash(hash),
        }
    }
}
