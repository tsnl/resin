//! Explicit host settings supplied by the compiler's caller.
use crate::{CProfile, Error, NativeInputs, files, platform};
use resin_executor::Cancellation;
use std::{
    collections::{BTreeMap, HashSet},
    ffi::{OsStr, OsString},
    path::{Path, PathBuf},
};
use tokio::{fs, io::AsyncWriteExt, process::Command};

/// Environment and paths resolved before compilation. Discovery failures are retained
/// so unused tools (in particular spirv-opt for host-only code) remain optional.
#[derive(Clone)]
pub(super) struct Settings {
    pub(super) cc: PathBuf,
    pub(super) spirv_opt: PathBuf,
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
            .map_err(|error| Error::new(format!("invalid {}: {error}", name.to_string_lossy())))
    }

    pub(super) async fn configure(
        &self,
        profile: CProfile,
        directory: &Path,
        inputs: &NativeInputs,
        cancellation: &Cancellation,
    ) -> Result<(), Error> {
        for path in [
            &self.cc,
            &self.spirv_opt,
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
                return Err(Error::new(
                    "Ninja tool paths cannot contain newlines".into(),
                ));
            }
        }
        let mut text = Vec::new();
        command_variable(&mut text, "cc", [self.cc.as_os_str()]);
        command_variable(&mut text, "spirv_opt", [self.spirv_opt.as_os_str()]);
        command_variable(&mut text, "resin", [self.executable.as_os_str()]);
        command_variable(
            &mut text,
            "cflags",
            self.cflags(profile, inputs).iter().map(OsString::as_os_str),
        );
        command_variable(
            &mut text,
            "captured_cflags",
            self.compilation_flags(profile, inputs)
                .iter()
                .map(OsString::as_os_str),
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
        text.extend_from_slice(NATIVE_RULES.as_bytes());
        files::write_changed(&text, &directory.join("toolchain.ninja")).await?;
        files::write_changed(
            self.fingerprint(cancellation).await?.as_bytes(),
            &directory.join("toolchain.state"),
        )
        .await
    }

    pub(super) async fn preprocess(
        &self,
        profile: CProfile,
        directory: &Path,
        inputs: &NativeInputs,
        cancellation: &Cancellation,
    ) -> Result<(), Error> {
        if inputs.translation_units.is_empty() {
            return Ok(());
        }
        let temporary = files::temporary(directory).await?;
        let roots = if inputs.restrict_header_paths {
            Some(crate::headers::allowed_roots(self, directory, cancellation).await?)
        } else {
            None
        };
        let path = temporary.path().join("native-inputs.state");
        let mut output = fs::File::create(&path).await?;
        output.write_all(b"resin-native-inputs-v2\0").await?;
        output
            .write_u64_le(inputs.translation_units.len() as u64)
            .await?;
        for unit in &inputs.translation_units {
            cancellation.check()?;
            let mut command = self.command(&self.cc)?;
            command
                .current_dir(directory)
                .args(self.cflags(profile, inputs));
            // Keep line markers: they preserve system-header diagnostic classification
            // when the compiler subsequently reads this already-preprocessed unit.
            command.args(["-E", "-x", "c"]);
            let dependencies = temporary.path().join("dependencies.d");
            if roots.is_some() {
                command
                    .args(["-MD", "-MT", "resin-input", "-MF"])
                    .arg(&dependencies);
            }
            let source = if unit.source.as_os_str().as_encoded_bytes().starts_with(b"-") {
                Path::new(".").join(&unit.source)
            } else {
                unit.source.clone()
            };
            command.arg(source);
            let captured = temporary.path().join(&unit.preprocessed);
            fs::create_dir_all(captured.parent().expect("captured file parent")).await?;
            let mut captured_file = fs::File::create(&captured).await?;
            let length =
                crate::process::preprocess(command, &mut captured_file, cancellation).await?;
            captured_file.flush().await?;
            drop(captured_file);
            if let Some(roots) = &roots {
                crate::headers::validate(&dependencies, directory, roots, cancellation).await?;
            }
            for path in [&unit.source, &unit.preprocessed] {
                let name = path.to_str().expect("JSON paths are UTF-8").as_bytes();
                output.write_u64_le(name.len() as u64).await?;
                output.write_all(name).await?;
            }
            output.write_u64_le(length).await?;
            tokio::io::copy(&mut fs::File::open(&captured).await?, &mut output).await?;
        }
        output.flush().await?;
        drop(output);
        for unit in &inputs.translation_units {
            cancellation.check()?;
            files::install_changed(
                &temporary.path().join(&unit.preprocessed),
                &directory.join(&unit.preprocessed),
            )
            .await?;
        }
        cancellation.check()?;
        files::install_changed(&path, &directory.join("native-inputs.state")).await
    }

    fn cflags(&self, profile: CProfile, inputs: &NativeInputs) -> Vec<OsString> {
        self.compilation_flags(profile, inputs)
            .into_iter()
            .chain(self.preprocessing_flags(inputs))
            .collect()
    }

    fn compilation_flags(&self, profile: CProfile, inputs: &NativeInputs) -> Vec<OsString> {
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
        .chain([profile.optimization()])
        .map(OsString::from)
        .chain(inputs.c_flags.iter().map(OsString::from))
        .collect()
    }

    fn preprocessing_flags(&self, inputs: &NativeInputs) -> Vec<OsString> {
        [
            OsString::from("-I"),
            self.runtime_include.clone().into_os_string(),
        ]
        .into_iter()
        .chain(platform::PREPROCESSING_FLAGS.iter().map(OsString::from))
        .chain(inputs.preprocessing_flags.iter().map(OsString::from))
        .collect()
    }

    fn ldflags(&self) -> Vec<OsString> {
        std::iter::once(self.runtime_library.clone().into_os_string())
            .chain(platform::LINK_FLAGS.iter().map(OsString::from))
            .chain(platform::LIBRARIES.iter().map(OsString::from))
            .collect()
    }

    async fn fingerprint(&self, cancellation: &Cancellation) -> Result<String, Error> {
        let mut hash = blake3::Hasher::new();
        state_bytes(&mut hash, b"resin-toolchain-v1");
        state_bytes(&mut hash, std::env::consts::OS.as_bytes());
        state_bytes(&mut hash, std::env::consts::ARCH.as_bytes());
        hash.update(&(self.environment.len() as u64).to_le_bytes());
        for (name, value) in &self.environment {
            state_string(&mut hash, name);
            state_string(&mut hash, value);
        }
        state_string(&mut hash, self.directory.as_os_str());
        for path in [
            &self.executable,
            &self.cc,
            &self.spirv_opt,
            &self.ninja,
            &self.runtime_library,
            &self.runtime_include,
        ] {
            hash_contents(path, &mut hash, cancellation).await?;
        }
        Ok(format!("{}\n", hash.finalize().to_hex()))
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

// Persisted fingerprints use BLAKE3 and explicit u64 little-endian length framing,
// never Rust's unspecified Hash/DefaultHasher encodings. OS strings use native bytes
// on Unix and UTF-16LE on Windows; the state includes its target OS/architecture.
fn state_bytes(hash: &mut blake3::Hasher, bytes: &[u8]) {
    hash.update(&(bytes.len() as u64).to_le_bytes());
    hash.update(bytes);
}

#[cfg(unix)]
fn state_string(hash: &mut blake3::Hasher, value: &OsStr) {
    use std::os::unix::ffi::OsStrExt;
    state_bytes(hash, value.as_bytes());
}

#[cfg(windows)]
fn state_string(hash: &mut blake3::Hasher, value: &OsStr) {
    use std::os::windows::ffi::OsStrExt;
    let bytes: Vec<_> = value.encode_wide().flat_map(u16::to_le_bytes).collect();
    state_bytes(hash, &bytes);
}

async fn hash_contents(
    path: &Path,
    hash: &mut blake3::Hasher,
    cancellation: &Cancellation,
) -> Result<(), Error> {
    let mut pending = vec![path.to_path_buf()];
    let mut seen = HashSet::new();
    while let Some(path) = pending.pop() {
        cancellation.check()?;
        state_string(hash, path.as_os_str());
        let canonical = fs::canonicalize(&path)
            .await
            .unwrap_or_else(|_| path.clone());
        state_string(hash, canonical.as_os_str());
        if !seen.insert(canonical) {
            state_bytes(hash, b"seen");
            continue;
        }
        if let Ok(mut entries) = fs::read_dir(&path).await {
            state_bytes(hash, b"directory");
            let mut paths = Vec::new();
            while let Some(entry) = entries.next_entry().await? {
                paths.push(entry.path());
            }
            paths.sort();
            hash.update(&(paths.len() as u64).to_le_bytes());
            pending.extend(paths.into_iter().rev());
        } else {
            match fs::read(&path).await {
                Ok(bytes) => {
                    state_bytes(hash, b"file");
                    hash.update(&(bytes.len() as u64).to_le_bytes());
                    for chunk in bytes.chunks(64 * 1024) {
                        cancellation.check()?;
                        hash.update(chunk);
                        tokio::task::yield_now().await;
                    }
                }
                Err(error) => {
                    state_bytes(hash, b"unavailable");
                    hash.update(&error.raw_os_error().unwrap_or_default().to_le_bytes());
                }
            }
        }
    }
    Ok(())
}

// Native commands belong to the toolchain; generated projects describe only edges.
// Captured preprocessor output intentionally retains GNU line markers (including
// system-header flags). Clang must accept its own markers under -pedantic -Werror.
const NATIVE_RULES: &str = "\
rule optimize_shader
  command = $spirv_opt --target-env=vulkan1.3 -O $in -o $out
  description = SPIR-V $in

rule embed_shader
  command = $resin --embed $in --symbol $symbol --output $out
  description = EMBED $in
  restat = 1

rule compile_preprocessed_program
  command = $cc $captured_cflags -Wno-unused-command-line-argument -Wno-gnu-line-marker $in -o $out $ldflags
  description = C $in

rule compile_program
  command = $cc $cflags -MMD -MF $out.d -MT $out $in -o $out $ldflags
  description = C $in
  depfile = $out.d
  deps = gcc

";
