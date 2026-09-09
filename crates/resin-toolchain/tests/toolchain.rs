use resin_common::prelude::*;
use resin_toolchain::{CProfile, Environment};
use std::{ffi::OsStr, fs, process::Command};

fn native_environment(temp: &TempDir) -> Environment {
    let mut environment = Environment::capture().unwrap();
    environment.directory = temp.path().into();
    environment.temporary = temp.path().into();
    let include = temp.path().join("include");
    fs::create_dir(&include).unwrap();
    let library = temp.path().join("runtime.a");
    fs::write(&library, b"!<arch>\n").unwrap();
    environment
        .variables
        .insert("RESIN_RUNTIME_LIB".into(), library.into());
    environment
        .variables
        .insert("RESIN_RUNTIME_INCLUDE".into(), include.into());
    environment
}

#[test]
fn missing_tools_are_diagnosed_only_by_operations_that_need_them() {
    let temp = TempDir::new(&std::env::temp_dir()).unwrap();
    let environment = native_environment(&temp);
    let tools = environment.toolchain(
        Some(OsStr::new("missing-resin-cc")),
        Some(OsStr::new("missing-resin-glslc")),
    );
    let error = tools.compile_c("", &temp.path().join("out")).unwrap_err();
    assert!(error.to_string().contains("missing-resin-cc"), "{error}");
    let error = tools.compile_glsl("", Stage::Compute).unwrap_err();
    assert!(error.to_string().contains("missing-resin-glslc"), "{error}");
}

#[test]
fn cached_executable_runs_literal_arguments_and_copies_to_new_directories() {
    let temp = TempDir::new(&std::env::temp_dir()).unwrap();
    let environment = native_environment(&temp);
    let tools = environment.toolchain(None, Some(OsStr::new("missing-resin-glslc")));
    let source = r#"#include <string.h>
int main(int argc, char **argv) {
    return argc == 3 && strcmp(argv[1], "two words") == 0 && strcmp(argv[2], "$(literal)") == 0 ? 23 : 99;
}"#;
    let build = tools
        .build_c(
            OsStr::new("overlay.resin").as_ref(),
            "main",
            source,
            CProfile::Debug,
        )
        .unwrap();
    assert_eq!(
        build
            .run_with_args(&["two words".into(), "$(literal)".into()])
            .unwrap(),
        23
    );
    let output = temp
        .path()
        .join("new directory")
        .join(format!("copy{}", std::env::consts::EXE_SUFFIX));
    build.copy_to(&output).unwrap();
    assert_eq!(
        Command::new(output)
            .args(["two words", "$(literal)"])
            .status()
            .unwrap()
            .code(),
        Some(23)
    );
    let path = build.path().to_owned();
    let modified = fs::metadata(&path).unwrap().modified().unwrap();
    drop(build);
    let reused = tools
        .build_c(
            OsStr::new("overlay.resin").as_ref(),
            "main",
            source,
            CProfile::Debug,
        )
        .unwrap();
    assert_eq!(reused.path(), path);
    assert_eq!(
        fs::metadata(reused.path()).unwrap().modified().unwrap(),
        modified
    );
}

#[test]
fn direct_compilation_reports_errors_without_overwriting_existing_output() {
    let temp = TempDir::new(&std::env::temp_dir()).unwrap();
    let tools = native_environment(&temp).toolchain(None, None);
    let output = temp
        .path()
        .join(format!("program{}", std::env::consts::EXE_SUFFIX));
    tools
        .compile_c("int main(void) { return 7; }", &output)
        .unwrap();
    let before = fs::read(&output).unwrap();
    let error = tools.compile_c("this is not C", &output).unwrap_err();
    assert!(error.to_string().contains("C compiler failed"), "{error}");
    assert_eq!(fs::read(output).unwrap(), before);
}

#[cfg(unix)]
mod shader_processes {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    fn shader_compiler(temp: &TempDir, body: &str) -> std::path::PathBuf {
        let compiler = temp.path().join("shader compiler");
        fs::write(&compiler, format!("#!/bin/sh\n{body}\n")).unwrap();
        fs::set_permissions(&compiler, fs::Permissions::from_mode(0o755)).unwrap();
        compiler
    }

    #[test]
    fn invocation_uses_captured_environment_and_validates_compiler_output() {
        let temp = TempDir::new(&std::env::temp_dir()).unwrap();
        let compiler = shader_compiler(&temp, "printf '%s' \"$RESIN_TEST_MARKER\" >&2; exit 19");
        let mut environment = native_environment(&temp);
        environment
            .variables
            .insert("RESIN_TEST_MARKER".into(), "captured marker".into());
        let tools = environment.toolchain(None, Some(compiler.as_os_str()));
        environment
            .variables
            .insert("RESIN_TEST_MARKER".into(), "later marker".into());
        let error = tools
            .compile_glsl("", Stage::Fragment)
            .unwrap_err()
            .to_string();
        assert!(error.contains("captured marker"), "{error}");
        assert!(!error.contains("later marker"), "{error}");
        shader_compiler(
            &temp,
            "for output do :; done; printf 'invalid' > \"$output\"",
        );
        let error = tools
            .compile_glsl("", Stage::Fragment)
            .unwrap_err()
            .to_string();
        assert!(error.contains("invalid SPIR-V"), "{error}");
    }

    #[test]
    fn shader_cache_reuses_valid_outputs_and_rebuilds_corrupt_outputs() {
        let temp = TempDir::new(&std::env::temp_dir()).unwrap();
        let compiler = shader_compiler(
            &temp,
            r#"printf 'call\n' >> "$RESIN_TEST_CALLS"
for output do :; done
printf '\003\002\043\007\000\000\001\000\000\000\000\000\001\000\000\000\000\000\000\000' > "$output""#,
        );
        let mut environment = native_environment(&temp);
        let calls = temp.path().join("calls");
        environment
            .variables
            .insert("RESIN_TEST_CALLS".into(), calls.clone().into());
        let tools = environment.toolchain(None, Some(compiler.as_os_str()));
        let expected = tools.build_glsl("shader", Stage::Compute).unwrap();
        assert_eq!(
            tools.build_glsl("shader", Stage::Compute).unwrap(),
            expected
        );
        assert_eq!(fs::read_to_string(&calls).unwrap(), "call\n");
        let cache = fs::read_dir(temp.path().join("build/shaders"))
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        fs::write(cache.join("shader.spv"), b"corrupt").unwrap();
        assert_eq!(
            tools.build_glsl("shader", Stage::Compute).unwrap(),
            expected
        );
        assert_eq!(fs::read_to_string(calls).unwrap(), "call\ncall\n");
    }
}
