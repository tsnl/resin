//! Resolve argv defaults from one snapshot of the invoking process.
use crate::{self as toolchain, Environment, Settings, Toolchain};
use std::{
    ffi::OsStr,
    path::{Path, PathBuf},
};

impl Environment {
    pub(super) fn resolve_tools(&self, cc: Option<&OsStr>, glslc: Option<&OsStr>) -> Toolchain {
        let settings = Settings {
            cc: self.resolve(self.compiler(cc, "CC", toolchain::DEFAULT_C_COMPILER)),
            glslc: self.resolve(self.compiler(glslc, "GLSLC", "glslc")),
            runtime_include: self.path(
                "RESIN_RUNTIME_INCLUDE",
                concat!(env!("CARGO_MANIFEST_DIR"), "/../resin-runtime/include"),
            ),
            runtime_library: self.runtime_library(),
            environment: self.variables.clone(),
            cache: self.directory.join("build"),
            directory: self.directory.clone(),
            temporary: self.temporary.clone(),
            executable: self.executable.clone(),
        };
        Toolchain { settings }
    }

    fn compiler<'a>(
        &'a self,
        option: Option<&'a OsStr>,
        variable: &str,
        fallback: &'a str,
    ) -> &'a OsStr {
        option
            .or_else(|| self.variable(variable))
            .unwrap_or_else(|| OsStr::new(fallback))
    }

    fn runtime_library(&self) -> Result<PathBuf, String> {
        if let Some(path) = self.variable("RESIN_RUNTIME_LIB") {
            let path = self.directory.join(path);
            return if path.is_file() {
                Ok(path)
            } else {
                Err(format!("runtime library not found: {}", path.display()))
            };
        }
        let directory = self.executable.parent().unwrap_or(Path::new("."));
        for path in [
            directory.join("deps").join(toolchain::RUNTIME_ARCHIVE),
            directory.join(toolchain::RUNTIME_ARCHIVE),
        ] {
            if path.is_file() {
                return Ok(path);
            }
        }
        Err(format!(
            "cannot find {} beside the compiler or in deps; set RESIN_RUNTIME_LIB",
            toolchain::RUNTIME_ARCHIVE
        ))
    }

    fn resolve(&self, compiler: &OsStr) -> Result<PathBuf, String> {
        let candidates = if Path::new(compiler).components().count() > 1 {
            vec![self.directory.join(compiler)]
        } else {
            std::env::split_paths(self.variable("PATH").unwrap_or_default())
                .map(|path| self.directory.join(path).join(compiler))
                .collect()
        };
        for path in candidates {
            #[cfg(windows)]
            let path = if !path.is_file() && path.extension().is_none() {
                path.with_extension("exe")
            } else {
                path
            };
            if !path.is_file() {
                continue;
            }
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                if std::fs::metadata(&path)
                    .map_err(|error| error.to_string())?
                    .permissions()
                    .mode()
                    & 0o111
                    == 0
                {
                    continue;
                }
            }
            // Preserve the invocation name: compiler drivers may inspect argv[0].
            return Ok(path);
        }
        Err(format!(
            "cannot run {}: executable not found",
            compiler.to_string_lossy()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TempDir;
    use std::{collections::BTreeMap, fs};

    fn executable(directory: &Path, name: &str) -> PathBuf {
        let path = directory.join(format!("{name}{}", std::env::consts::EXE_SUFFIX));
        fs::write(&path, []).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).unwrap();
        }
        path
    }

    #[test]
    fn resolves_flags_environment_and_defaults_from_supplied_process_inputs() {
        let temp = TempDir::new(&std::env::temp_dir()).unwrap();
        let bin = temp.path().join("tools with spaces");
        fs::create_dir(&bin).unwrap();
        let default_cc = executable(&bin, toolchain::DEFAULT_C_COMPILER);
        let default_glslc = executable(&bin, "glslc");
        let custom = executable(&bin, "custom");
        let explicit = executable(&bin, "explicit");
        let runtime = bin.join(toolchain::RUNTIME_ARCHIVE);
        fs::write(&runtime, []).unwrap();
        let mut env = Environment {
            variables: BTreeMap::from([("PATH".into(), "tools with spaces".into())]),
            directory: temp.path().into(),
            executable: bin.join("resin"),
            temporary: temp.path().into(),
        };
        let settings = env.toolchain(None, None).settings;
        assert_eq!(settings.cc.unwrap(), default_cc);
        assert_eq!(settings.glslc.unwrap(), default_glslc);
        assert_eq!(settings.runtime_library.unwrap(), runtime);
        assert_eq!(settings.cache, temp.path().join("build"));
        assert_eq!(settings.temporary, temp.path());
        assert_eq!(
            settings.runtime_include,
            PathBuf::from(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../resin-runtime/include"
            ))
        );
        for (name, value) in [
            ("CC", "custom"),
            ("GLSLC", "custom"),
            ("RESIN_STDLIB", "std"),
            ("RESIN_RUNTIME_INCLUDE", "include"),
            ("RESIN_RUNTIME_LIB", "archive"),
        ] {
            env.variables.insert(name.into(), value.into());
        }
        fs::write(temp.path().join("archive"), []).unwrap();
        let settings = env.toolchain(None, None).settings;
        assert_eq!(settings.cc.unwrap(), custom);
        assert_eq!(settings.glslc.unwrap(), custom);
        assert_eq!(settings.runtime_include, temp.path().join("include"));
        assert_eq!(
            settings.runtime_library.unwrap(),
            temp.path().join("archive")
        );
        let settings = env
            .toolchain(Some(OsStr::new("explicit")), Some(explicit.as_os_str()))
            .settings;
        assert_eq!(settings.cc.as_ref().unwrap(), &explicit);
        assert_eq!(settings.glslc.as_ref().unwrap(), &explicit);
        env.variables.insert("CC".into(), "missing".into());
        assert_eq!(
            settings.environment.get(OsStr::new("CC")).unwrap(),
            "custom"
        );
        assert!(env.toolchain(None, None).settings.cc.is_err());
        // In particular, failure to resolve an explicit choice never falls back.
        assert!(
            env.toolchain(Some(OsStr::new("missing")), None)
                .settings
                .cc
                .is_err()
        );
    }

    #[test]
    #[cfg(windows)]
    fn windows_environment_names_and_executable_extensions_are_resolved() {
        let temp = TempDir::new(&std::env::temp_dir()).unwrap();
        let path = executable(temp.path(), "compiler with spaces");
        let env = Environment {
            variables: BTreeMap::from([
                ("Path".into(), temp.path().into()),
                ("cc".into(), "compiler with spaces".into()),
            ]),
            directory: temp.path().into(),
            executable: temp.path().join("resin.exe"),
            temporary: temp.path().into(),
        };
        assert_eq!(env.toolchain(None, None).settings.cc.unwrap(), path);
        assert_eq!(env.resolve(path.as_os_str()).unwrap(), path);
        assert_eq!(
            env.resolve(path.with_extension("").as_os_str()).unwrap(),
            path
        );
    }
}
