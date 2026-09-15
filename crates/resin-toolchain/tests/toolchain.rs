use resin_executor::{Cancellation, Execution};
use resin_toolchain::{BuiltProject, CProfile, Environment, Toolchain};
use std::{
    ffi::OsStr,
    fs,
    path::{Path, PathBuf},
    process::Command,
};
use tempfile::TempDir;

fn native_environment(temp: &TempDir) -> Environment {
    let mut environment = Environment::capture().unwrap();
    environment.directory = temp.path().into();
    environment.temporary = temp.path().into();
    let include = temp.path().join("include with spaces");
    fs::create_dir(&include).unwrap();
    let library = temp.path().join("runtime with spaces.a");
    fs::write(&library, b"!<arch>\n").unwrap();
    environment
        .variables
        .insert("RESIN_RUNTIME_LIB".into(), library.into());
    environment
        .variables
        .insert("RESIN_RUNTIME_INCLUDE".into(), include.into());
    environment
}

fn c_project(temp: &TempDir, source: &str) -> PathBuf {
    let project = temp.path().join("generated source");
    fs::create_dir_all(&project).unwrap();
    fs::write(project.join("main.c"), source).unwrap();
    fs::write(project.join("build.ninja"), format!(
        "include toolchain.ninja\nbuild program{}: compile_preprocessed_program main.i | $runtime_library toolchain.state native-inputs.state\n",
        std::env::consts::EXE_SUFFIX)).unwrap();
    fs::write(
        project.join("native-inputs.json"),
        br#"{"translation_units":[{"source":"main.c","preprocessed":"main.i"}]}"#,
    )
    .unwrap();
    project
}

async fn build(tools: &Toolchain, project: &Path) -> BuiltProject {
    tools
        .build(
            project,
            "generated/module",
            "main",
            CProfile::Debug,
            &Execution::default(),
            &Cancellation::new(),
        )
        .await
        .unwrap()
}

fn program() -> String {
    format!("program{}", std::env::consts::EXE_SUFFIX)
}

#[tokio::test]
async fn missing_tools_fail_only_when_the_graph_uses_them() {
    let temp = TempDir::new().unwrap();
    let mut environment = native_environment(&temp);
    let project = c_project(&temp, "int main(void) { return 0; }");
    let host = environment.toolchain(None, Some(OsStr::new("missing-resin-spirv-opt")));
    assert_eq!(
        build(&host, &project)
            .await
            .executable(program())
            .unwrap()
            .run(&Execution::default(), &Cancellation::new())
            .await
            .unwrap(),
        0
    );
    let missing = environment.toolchain(Some(OsStr::new("missing-resin-cc")), None);
    let error = missing
        .build(
            &project,
            "generated/module",
            "main",
            CProfile::Debug,
            &Execution::default(),
            &Cancellation::new(),
        )
        .await
        .unwrap_err();
    assert!(error.to_string().contains("missing-resin-cc"), "{error}");
    fs::remove_file(project.join("native-inputs.json")).unwrap();
    fs::write(project.join("shader.unoptimized.spv"), []).unwrap();
    fs::write(
        project.join("build.ninja"),
        "include toolchain.ninja\nbuild shader.spv: optimize_shader shader.unoptimized.spv | toolchain.state\n",
    )
    .unwrap();
    let error = host
        .build(
            &project,
            "generated/module",
            "main",
            CProfile::Debug,
            &Execution::default(),
            &Cancellation::new(),
        )
        .await
        .unwrap_err();
    assert!(
        error.to_string().contains("missing-resin-spirv-opt"),
        "{error}"
    );
    environment
        .variables
        .insert("NINJA".into(), "missing-resin-ninja".into());
    let error = environment
        .toolchain(None, None)
        .build(
            &project,
            "generated/module",
            "main",
            CProfile::Debug,
            &Execution::default(),
            &Cancellation::new(),
        )
        .await
        .unwrap_err();
    assert!(error.to_string().contains("missing-resin-ninja"), "{error}");
}

#[tokio::test]
async fn cached_executable_runs_literal_arguments_and_copies_to_new_directories() {
    let temp = TempDir::new().unwrap();
    let tools = native_environment(&temp).toolchain(None, None);
    let project = c_project(
        &temp,
        r#"#include <string.h>
int main(int argc, char **argv) {
    return argc == 3 && strcmp(argv[1], "two words") == 0 && strcmp(argv[2], "$(literal)") == 0 ? 23 : 99;
}"#,
    );
    let built = build(&tools, &project).await;
    let executable = built.executable(program()).unwrap();
    let args = ["two words".into(), "$(literal)".into()];
    assert_eq!(
        executable
            .run_with_args(&args, &Execution::default(), &Cancellation::new())
            .await
            .unwrap(),
        23
    );
    let output = temp.path().join("new directory").join(program());
    executable
        .copy_to(&output, &Execution::default(), &Cancellation::new())
        .await
        .unwrap();
    assert_eq!(
        Command::new(output).args(&args).status().unwrap().code(),
        Some(23)
    );
    let path = executable.path().to_path_buf();
    let modified = fs::metadata(&path).unwrap().modified().unwrap();
    drop(executable);
    drop(built);
    let reused = build(&tools, &project).await;
    assert!(
        !path.exists(),
        "the last artifact owner removes its generation"
    );
    assert_ne!(reused.path(program()), path);
    assert_eq!(
        fs::metadata(reused.path(program()))
            .unwrap()
            .modified()
            .unwrap(),
        modified
    );
    assert!(reused.path("main.c").is_file());
    assert!(reused.path("build.ninja").is_file());
}

#[tokio::test]
async fn failed_builds_leave_published_executables_unchanged() {
    let temp = TempDir::new().unwrap();
    let tools = native_environment(&temp).toolchain(None, None);
    let project = c_project(&temp, "int main(void) { return 7; }");
    let built = build(&tools, &project).await;
    let output = built.path(program());
    let before = fs::read(&output).unwrap();
    fs::write(project.join("main.c"), "this is not C").unwrap();
    for _ in 0..2 {
        let error = tools
            .build(
                &project,
                "generated/module",
                "main",
                CProfile::Debug,
                &Execution::default(),
                &Cancellation::new(),
            )
            .await
            .unwrap_err();
        assert!(error.to_string().contains("native build failed"), "{error}");
        assert_eq!(fs::read(&output).unwrap(), before);
    }
    fs::write(project.join("main.c"), "int main(void) { return 8; }").unwrap();
    let rebuilt = build(&tools, &project).await;
    assert_eq!(
        rebuilt
            .executable(program())
            .unwrap()
            .run(&Execution::default(), &Cancellation::new())
            .await
            .unwrap(),
        8
    );
    let published = fs::metadata(rebuilt.path(program())).unwrap();
    let reused = build(&tools, &project).await;
    let staged = fs::metadata(reused.path(program())).unwrap();
    assert_eq!(
        published.modified().unwrap(),
        staged.modified().unwrap(),
        "publication must preserve the timestamp used for incremental Ninja builds"
    );
}

#[tokio::test]
async fn nested_headers_refresh_and_removed_inputs_are_not_left_in_the_cache() {
    let temp = TempDir::new().unwrap();
    let tools = native_environment(&temp).toolchain(None, None);
    let project = c_project(
        &temp,
        "#include \"nested/value.h\"\nint main(void) { return VALUE; }",
    );
    fs::create_dir(project.join("nested")).unwrap();
    let header = project.join("nested/value.h");
    fs::write(&header, "#define VALUE 4\n").unwrap();
    assert_eq!(
        build(&tools, &project)
            .await
            .executable(program())
            .unwrap()
            .run(&Execution::default(), &Cancellation::new())
            .await
            .unwrap(),
        4
    );
    fs::write(&header, "#define VALUE 5\n").unwrap();
    assert_eq!(
        build(&tools, &project)
            .await
            .executable(program())
            .unwrap()
            .run(&Execution::default(), &Cancellation::new())
            .await
            .unwrap(),
        5
    );
    fs::remove_file(header).unwrap();
    assert!(
        tools
            .build(
                &project,
                "generated/module",
                "main",
                CProfile::Debug,
                &Execution::default(),
                &Cancellation::new()
            )
            .await
            .is_err()
    );
}

#[tokio::test]
async fn outputs_removed_from_the_graph_disappear_from_the_published_project() {
    let temp = TempDir::new().unwrap();
    let tools = native_environment(&temp).toolchain(None, None);
    let project = c_project(&temp, "int main(void) { return 0; }");
    let first = build(&tools, &project).await;
    assert!(first.path(program()).is_file());
    drop(first);
    let graph = project.join("build.ninja");
    let replacement = format!("replacement{}", std::env::consts::EXE_SUFFIX);
    fs::write(
        &graph,
        fs::read_to_string(&graph).unwrap().replace(
            &format!("build {}:", program()),
            &format!("build {replacement}:"),
        ),
    )
    .unwrap();
    let second = build(&tools, &project).await;
    assert!(!second.path(program()).exists());
    assert_eq!(
        second
            .executable(replacement)
            .unwrap()
            .run(&Execution::default(), &Cancellation::new())
            .await
            .unwrap(),
        0
    );
}

#[tokio::test]
async fn generated_sources_cannot_replace_toolchain_configuration() {
    let temp = TempDir::new().unwrap();
    let tools = native_environment(&temp).toolchain(None, None);
    let project = c_project(&temp, "int main(void) { return 0; }");
    fs::write(project.join("toolchain.ninja"), "cc = unexpected").unwrap();
    let error = tools
        .build(
            &project,
            "module",
            "main",
            CProfile::Debug,
            &Execution::default(),
            &Cancellation::new(),
        )
        .await
        .unwrap_err();
    assert!(
        error.to_string().contains("reserved toolchain filename"),
        "{error}"
    );
}

#[tokio::test]
async fn captured_outputs_cannot_replace_supplied_files() {
    let temp = TempDir::new().unwrap();
    let tools = native_environment(&temp).toolchain(None, None);
    let project = c_project(&temp, "int main(void) { return 0; }");
    fs::write(project.join("main.i"), "supplied source").unwrap();
    let error = tools
        .build(
            &project,
            "module",
            "main",
            CProfile::Debug,
            &Execution::default(),
            &Cancellation::new(),
        )
        .await
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("cannot replace supplied project files"),
        "{error}"
    );
    assert_eq!(
        fs::read_to_string(project.join("main.i")).unwrap(),
        "supplied source"
    );
}

#[tokio::test]
async fn earlier_cached_native_metadata_does_not_reject_a_current_project() {
    let temp = TempDir::new().unwrap();
    let tools = native_environment(&temp).toolchain(None, None);
    let project = c_project(&temp, "int main(void) { return 4; }");
    let first = build(&tools, &project).await;
    let first_manifest = fs::read(first.path("native-inputs.json")).unwrap();
    let slot = fs::read_dir(temp.path().join("build"))
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .find(|path| path.file_name().unwrap() != ".artifacts")
        .unwrap();
    let staged = slot.join("debug/.ninja-work/native-inputs.json");
    let old_manifest =
        br#"{"translation_units":["main.c"],"c_flags":[],"generated_prerequisites":[]}"#;
    fs::write(&staged, old_manifest).unwrap();
    fs::write(project.join("main.c"), "int main(void) { return 5; }").unwrap();
    let second = build(&tools, &project).await;
    assert_eq!(
        fs::read(staged).unwrap(),
        fs::read(project.join("native-inputs.json")).unwrap()
    );
    assert_eq!(
        fs::read(first.path("native-inputs.json")).unwrap(),
        first_manifest
    );
    let execution = Execution::default();
    let cancellation = Cancellation::new();
    assert_eq!(
        first
            .executable(program())
            .unwrap()
            .run(&execution, &cancellation)
            .await
            .unwrap(),
        4
    );
    assert_eq!(
        second
            .executable(program())
            .unwrap()
            .run(&execution, &cancellation)
            .await
            .unwrap(),
        5
    );

    // Compatibility is limited to discarded cache metadata, not supplied inputs.
    fs::write(project.join("native-inputs.json"), old_manifest).unwrap();
    let error = tools
        .build(
            &project,
            "generated/module",
            "main",
            CProfile::Debug,
            &execution,
            &cancellation,
        )
        .await
        .unwrap_err();
    assert!(
        error.to_string().contains("invalid native-inputs.json"),
        "{error}"
    );
}

#[tokio::test]
async fn declared_captures_and_raw_c_inputs_can_share_a_native_graph() {
    let temp = TempDir::new().unwrap();
    let tools = native_environment(&temp).toolchain(None, None);
    let project = c_project(&temp, "int main(void) { return EXTRA; }");
    fs::write(
        project.join("raw.c"),
        "#include <local.h>\nint main(void) { return VALUE + EXTRA; }",
    )
    .unwrap();
    fs::write(project.join("local.h"), "#define VALUE 4\n").unwrap();
    fs::write(project.join("native-inputs.json"), br#"{"translation_units":[{"source":"main.c","preprocessed":"main.i"}],"preprocessing_flags":["-I",".","-DEXTRA=2"]}"#).unwrap();
    let graph = fs::read_to_string(project.join("build.ninja")).unwrap();
    fs::write(
        project.join("build.ninja"),
        format!(
            "{graph}\nbuild raw{}: compile_program raw.c | toolchain.state $runtime_library\n",
            std::env::consts::EXE_SUFFIX
        ),
    )
    .unwrap();
    let built = build(&tools, &project).await;
    let execution = Execution::default();
    let cancellation = Cancellation::new();
    assert_eq!(
        built
            .executable(program())
            .unwrap()
            .run(&execution, &cancellation)
            .await
            .unwrap(),
        2
    );
    assert_eq!(
        built
            .executable(format!("raw{}", std::env::consts::EXE_SUFFIX))
            .unwrap()
            .run(&execution, &cancellation)
            .await
            .unwrap(),
        6
    );
}

#[tokio::test]
async fn obsolete_captured_units_are_removed_before_replacement_inputs_are_staged() {
    let temp = TempDir::new().unwrap();
    let tools = native_environment(&temp).toolchain(None, None);
    let project = c_project(&temp, "int main(void) { return 4; }");
    let first = build(&tools, &project).await;
    let captured = fs::read(first.path("main.i")).unwrap();
    fs::write(
        project.join("native-inputs.json"),
        br#"{"translation_units":[{"source":"main.c","preprocessed":"captured/next.i"}]}"#,
    )
    .unwrap();
    let graph = fs::read_to_string(project.join("build.ninja")).unwrap();
    fs::write(
        project.join("build.ninja"),
        graph.replace("main.i", "captured/next.i"),
    )
    .unwrap();
    let second = build(&tools, &project).await;
    assert!(!second.path("main.i").exists());
    assert!(second.path("captured/next.i").exists());
    assert_eq!(fs::read(first.path("main.i")).unwrap(), captured);

    fs::remove_file(project.join("native-inputs.json")).unwrap();
    fs::write(project.join("build.ninja"), "build remaining: phony\n").unwrap();
    fs::create_dir_all(project.join("captured")).unwrap();
    fs::write(project.join("captured/next.i"), "new supplied input").unwrap();
    let supplied = build(&tools, &project).await;
    assert_eq!(
        fs::read_to_string(supplied.path("captured/next.i")).unwrap(),
        "new supplied input"
    );
    assert!(!supplied.path("native-inputs.state").exists());
    assert!(second.path("native-inputs.state").exists());
    assert_ne!(
        fs::read_to_string(second.path("captured/next.i")).unwrap(),
        "new supplied input"
    );

    fs::remove_file(project.join("captured/next.i")).unwrap();
    let final_project = build(&tools, &project).await;
    assert!(!final_project.path("captured/next.i").exists());
}

#[cfg(unix)]
mod process_configuration {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    fn executable(path: &Path, body: &str) {
        fs::write(path, format!("#!/bin/sh\n{body}\n")).unwrap();
        fs::set_permissions(path, fs::Permissions::from_mode(0o755)).unwrap();
    }

    fn counted_compiler(temp: &TempDir, environment: &mut Environment) -> PathBuf {
        let compiler = temp.path().join("counted-compiler");
        environment.variables.insert(
            "RESIN_TEST_REAL_CC".into(),
            std::env::var_os("CC").unwrap_or_else(|| "cc".into()),
        );
        environment
            .variables
            .insert("RESIN_TEST_CALLS".into(), temp.path().join("calls").into());
        executable(
            &compiler,
            r#"case " $* " in *" -E "*) ;; *) printf 'compile\n' >> "$RESIN_TEST_CALLS" ;; esac
exec "$RESIN_TEST_REAL_CC" "$@""#,
        );
        compiler
    }

    #[tokio::test]
    async fn compilation_consumes_the_header_version_captured_before_the_compiler_starts() {
        let temp = TempDir::new().unwrap();
        let mut environment = native_environment(&temp);
        environment.variables.insert("CPATH".into(), ".".into());
        environment
            .variables
            .insert("RESIN_TEST_CONTROL".into(), temp.path().into());
        let compiler = counted_compiler(&temp, &mut environment);
        executable(
            &compiler,
            r#"case " $* " in *" -E "*) exec "$RESIN_TEST_REAL_CC" "$@" ;; esac
printf started > "$RESIN_TEST_CONTROL/started"
while [ -f "$RESIN_TEST_CONTROL/block" ]; do sleep 0.01; done
exec "$RESIN_TEST_REAL_CC" "$@""#,
        );
        let header = temp.path().join("foreign.h");
        fs::write(&header, "#define VALUE 4\n").unwrap();
        fs::write(temp.path().join("block"), "").unwrap();
        let project = c_project(
            &temp,
            "#include <foreign.h>\nint main(void) { return VALUE; }",
        );
        let tools = environment.toolchain(Some(compiler.as_os_str()), None);
        let (pending_tools, pending_project) = (tools.clone(), project.clone());
        let pending = tokio::spawn(async move { build(&pending_tools, &pending_project).await });
        tokio::time::timeout(std::time::Duration::from_secs(10), async {
            while !temp.path().join("started").exists() {
                tokio::time::sleep(std::time::Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("compiler did not reach the post-capture gate");
        fs::write(&header, "#define VALUE 5\n").unwrap();
        fs::remove_file(temp.path().join("block")).unwrap();
        let first = pending.await.unwrap();
        let first_capture = fs::read(first.path("main.i")).unwrap();
        assert_eq!(
            first
                .executable(program())
                .unwrap()
                .run(&Execution::default(), &Cancellation::new())
                .await
                .unwrap(),
            4
        );
        let second = build(&tools, &project).await;
        assert_eq!(
            second
                .executable(program())
                .unwrap()
                .run(&Execution::default(), &Cancellation::new())
                .await
                .unwrap(),
            5
        );
        assert_eq!(fs::read(first.path("main.i")).unwrap(), first_capture);
        assert_ne!(fs::read(second.path("main.i")).unwrap(), first_capture);
    }

    #[tokio::test]
    async fn clang_compiles_captured_system_headers_with_strict_flags() {
        let mut probe = tokio::process::Command::new("clang");
        probe.arg("--version");
        match probe.output().await {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return,
            result => assert!(result.unwrap().status.success(), "clang --version failed"),
        }
        let temp = TempDir::new().unwrap();
        let tools = native_environment(&temp).toolchain(Some(OsStr::new("clang")), None);
        let project = c_project(
            &temp,
            "#include <stdio.h>\n#include <string.h>\n#include <local.h>\nint main(void) { return strcmp(VALUE, \"captured\") + EXTRA; }",
        );
        fs::write(project.join("local.h"), "#define VALUE \"captured\"\n").unwrap();
        fs::write(project.join("native-inputs.json"), br#"{"translation_units":[{"source":"main.c","preprocessed":"main.i"}],"preprocessing_flags":["-I",".","-DEXTRA=6"]}"#).unwrap();
        let first = build(&tools, &project).await;
        assert_eq!(
            first
                .executable(program())
                .unwrap()
                .run(&Execution::default(), &Cancellation::new())
                .await
                .unwrap(),
            6
        );
        let modified = fs::metadata(first.path(program()))
            .unwrap()
            .modified()
            .unwrap();
        let second = build(&tools, &project).await;
        assert_eq!(
            fs::metadata(second.path(program()))
                .unwrap()
                .modified()
                .unwrap(),
            modified
        );
    }

    #[tokio::test]
    async fn external_header_contents_invalidate_work_even_when_mtime_is_preserved() {
        let temp = TempDir::new().unwrap();
        let mut environment = native_environment(&temp);
        environment.variables.insert("CPATH".into(), ".".into());
        let compiler = counted_compiler(&temp, &mut environment);
        let header = temp.path().join("foreign.h");
        fs::write(&header, "#define VALUE 4\n").unwrap();
        let modified = fs::metadata(&header).unwrap().modified().unwrap();
        let project = c_project(
            &temp,
            "#include <foreign.h>\nint main(void) { return VALUE; }",
        );
        let tools = environment.toolchain(Some(compiler.as_os_str()), None);
        for _ in 0..2 {
            assert_eq!(
                build(&tools, &project)
                    .await
                    .executable(program())
                    .unwrap()
                    .run(&Execution::default(), &Cancellation::new())
                    .await
                    .unwrap(),
                4
            );
        }
        assert_eq!(
            fs::read_to_string(temp.path().join("calls")).unwrap(),
            "compile\n",
            "warm preprocessing must not recompile or hash changing CPATH workspace logs"
        );
        fs::write(&header, "#define VALUE 5\n").unwrap();
        fs::OpenOptions::new()
            .write(true)
            .open(&header)
            .unwrap()
            .set_modified(modified)
            .unwrap();
        assert_eq!(
            build(&tools, &project)
                .await
                .executable(program())
                .unwrap()
                .run(&Execution::default(), &Cancellation::new())
                .await
                .unwrap(),
            5
        );
        assert_eq!(
            fs::read_to_string(temp.path().join("calls")).unwrap(),
            "compile\ncompile\n"
        );
    }

    #[tokio::test]
    async fn new_shadowing_and_optional_headers_invalidate_preprocessed_inputs() {
        let temp = TempDir::new().unwrap();
        let mut environment = native_environment(&temp);
        let first = temp.path().join("first");
        let second = temp.path().join("second");
        fs::create_dir(&first).unwrap();
        fs::create_dir(&second).unwrap();
        fs::write(second.join("foreign.h"), "#define VALUE 4\n").unwrap();
        environment.variables.insert(
            "CPATH".into(),
            std::env::join_paths([&first, &second]).unwrap(),
        );
        let project = c_project(
            &temp,
            "#include <foreign.h>\n#if __has_include(<optional.h>)\n#include <optional.h>\n#else\n#define EXTRA 0\n#endif\nint main(void) { return VALUE + EXTRA; }",
        );
        let tools = environment.toolchain(None, None);
        assert_eq!(
            build(&tools, &project)
                .await
                .executable(program())
                .unwrap()
                .run(&Execution::default(), &Cancellation::new())
                .await
                .unwrap(),
            4
        );
        fs::write(first.join("foreign.h"), "#define VALUE 5\n").unwrap();
        assert_eq!(
            build(&tools, &project)
                .await
                .executable(program())
                .unwrap()
                .run(&Execution::default(), &Cancellation::new())
                .await
                .unwrap(),
            5
        );
        fs::write(first.join("optional.h"), "#define EXTRA 2\n").unwrap();
        assert_eq!(
            build(&tools, &project)
                .await
                .executable(program())
                .unwrap()
                .run(&Execution::default(), &Cancellation::new())
                .await
                .unwrap(),
            7
        );
    }

    #[tokio::test]
    async fn declared_generated_headers_are_built_before_preprocessing_with_matching_flags() {
        let temp = TempDir::new().unwrap();
        let mut environment = native_environment(&temp);
        let compiler = counted_compiler(&temp, &mut environment);
        let project = c_project(
            &temp,
            "#include <generated.h>\nint main(void) { return VALUE + EXTRA; }",
        );
        fs::write(project.join("header-input"), "#define VALUE 4\n").unwrap();
        fs::write(project.join("build.ninja"), format!("include toolchain.ninja\nrule header\n  command = cp $in $out\nbuild generated.h: header header-input\nbuild {}: compile_preprocessed_program main.i | generated.h toolchain.state native-inputs.state $runtime_library\ndefault {}\n", program(), program())).unwrap();
        fs::write(project.join("native-inputs.json"), br#"{"translation_units":[{"source":"main.c","preprocessed":"main.i"}],"preprocessing_flags":["-I",".","-DEXTRA=2"],"generated_prerequisites":["generated.h"]}"#).unwrap();
        let tools = environment.toolchain(Some(compiler.as_os_str()), None);
        for _ in 0..2 {
            assert_eq!(
                build(&tools, &project)
                    .await
                    .executable(program())
                    .unwrap()
                    .run(&Execution::default(), &Cancellation::new())
                    .await
                    .unwrap(),
                6
            );
        }
        assert_eq!(
            fs::read_to_string(temp.path().join("calls")).unwrap(),
            "compile\n"
        );
        fs::write(project.join("header-input"), "#define VALUE 5\n").unwrap();
        assert_eq!(
            build(&tools, &project)
                .await
                .executable(program())
                .unwrap()
                .run(&Execution::default(), &Cancellation::new())
                .await
                .unwrap(),
            7
        );
        assert_eq!(
            fs::read_to_string(temp.path().join("calls")).unwrap(),
            "compile\ncompile\n"
        );
    }

    #[tokio::test]
    async fn compiler_bytes_invalidate_work_when_size_and_mtime_are_unchanged() {
        let temp = TempDir::new().unwrap();
        let mut environment = native_environment(&temp);
        let compiler = counted_compiler(&temp, &mut environment);
        let original = fs::read_to_string(&compiler).unwrap();
        let signed = original.replace(
            "exec \"$RESIN_TEST_REAL_CC\" \"$@\"",
            "exec \"$RESIN_TEST_REAL_CC\" -fsigned-char \"$@\"   #",
        );
        let unsigned = signed
            .replace("-fsigned-char", "-funsigned-char")
            .replace("   #", " #");
        assert_eq!(signed.len(), unsigned.len());
        fs::write(&compiler, signed).unwrap();
        let modified = fs::metadata(&compiler).unwrap().modified().unwrap();
        let project = c_project(
            &temp,
            "int main(void) { volatile char value = (char)-1; int expected = -1; return value == expected; }",
        );
        let tools = environment.toolchain(Some(compiler.as_os_str()), None);
        assert_eq!(
            build(&tools, &project)
                .await
                .executable(program())
                .unwrap()
                .run(&Execution::default(), &Cancellation::new())
                .await
                .unwrap(),
            1
        );
        fs::write(&compiler, unsigned).unwrap();
        fs::OpenOptions::new()
            .write(true)
            .open(&compiler)
            .unwrap()
            .set_modified(modified)
            .unwrap();
        assert_eq!(
            build(&tools, &project)
                .await
                .executable(program())
                .unwrap()
                .run(&Execution::default(), &Cancellation::new())
                .await
                .unwrap(),
            0
        );
        assert_eq!(
            fs::read_to_string(temp.path().join("calls")).unwrap(),
            "compile\ncompile\n"
        );
    }

    #[tokio::test]
    async fn source_symlinks_cannot_reach_outside_the_project() {
        let temp = TempDir::new().unwrap();
        let tools = native_environment(&temp).toolchain(None, None);
        let project = c_project(&temp, "int main(void) { return 0; }");
        let outside = temp.path().join("outside");
        fs::write(&outside, "retained").unwrap();
        std::os::unix::fs::symlink(&outside, project.join("linked")).unwrap();
        let error = tools
            .build(
                &project,
                "module",
                "main",
                CProfile::Debug,
                &Execution::default(),
                &Cancellation::new(),
            )
            .await
            .unwrap_err();
        assert!(
            error.to_string().contains("cannot follow symlinks"),
            "{error}"
        );
        assert_eq!(fs::read_to_string(outside).unwrap(), "retained");
    }

    #[tokio::test]
    async fn relative_and_empty_include_paths_use_the_captured_directory() {
        for value in [".", ""] {
            let temp = TempDir::new().unwrap();
            let mut environment = native_environment(&temp);
            environment.variables.insert("CPATH".into(), value.into());
            let project = c_project(
                &temp,
                "#include <foreign.h>\nint main(void) { return VALUE; }",
            );
            let header = temp.path().join("foreign.h");
            fs::write(&header, "#define VALUE 4\n").unwrap();
            let tools = environment.toolchain(None, None);
            let first = build(&tools, &project).await;
            let modified = fs::metadata(first.path(program()))
                .unwrap()
                .modified()
                .unwrap();
            assert_eq!(
                first
                    .executable(program())
                    .unwrap()
                    .run(&Execution::default(), &Cancellation::new())
                    .await
                    .unwrap(),
                4
            );
            drop(first);
            let reused = build(&tools, &project).await;
            assert_eq!(
                fs::metadata(reused.path(program()))
                    .unwrap()
                    .modified()
                    .unwrap(),
                modified
            );
            drop(reused);
            fs::write(header, "#define VALUE 5\n").unwrap();
            assert_eq!(
                build(&tools, &project)
                    .await
                    .executable(program())
                    .unwrap()
                    .run(&Execution::default(), &Cancellation::new())
                    .await
                    .unwrap(),
                5
            );
        }
    }

    #[tokio::test]
    async fn dependencies_changed_during_a_command_are_rebuilt_on_the_next_build() {
        let temp = TempDir::new().unwrap();
        let mut environment = native_environment(&temp);
        let header = temp.path().join("external.h");
        let calls = temp.path().join("calls");
        fs::write(&header, "old").unwrap();
        environment
            .variables
            .insert("RESIN_TEST_HEADER".into(), header.into());
        environment
            .variables
            .insert("RESIN_TEST_CALLS".into(), calls.clone().into());
        let compiler = temp.path().join("compiler");
        executable(
            &compiler,
            r#"value=$(cat "$RESIN_TEST_HEADER")
printf 'call\n' >> "$RESIN_TEST_CALLS"
if [ "$value" = old ]; then sleep 0.05; printf new > "$RESIN_TEST_HEADER"; fi
printf '%s' "$value" > "$1"
printf '%s: %s\n' "$1" "$RESIN_TEST_HEADER" > "$1.d""#,
        );
        let project = temp.path().join("sources");
        fs::create_dir(&project).unwrap();
        fs::write(project.join("build.ninja"), "include toolchain.ninja\nrule c\n  command = $cc $out\n  depfile = $out.d\n  deps = gcc\nbuild output: c | toolchain.state\n").unwrap();
        let tools = environment.toolchain(Some(compiler.as_os_str()), None);
        assert_eq!(
            fs::read(build(&tools, &project).await.path("output")).unwrap(),
            b"old"
        );
        assert_eq!(
            fs::read(build(&tools, &project).await.path("output")).unwrap(),
            b"new"
        );
        drop(build(&tools, &project).await);
        assert_eq!(fs::read_to_string(calls).unwrap(), "call\ncall\n");
    }

    #[tokio::test]
    async fn compiler_paths_and_environment_changes_invalidate_ninja_work() {
        let temp = TempDir::new().unwrap();
        let mut environment = native_environment(&temp);
        let compiler = temp.path().join("compiler 'with $shell' characters");
        let calls = temp.path().join("calls");
        environment.variables.insert(
            "RESIN_TEST_COMPILER".into(),
            std::env::var_os("CC").unwrap_or_else(|| "cc".into()),
        );
        environment
            .variables
            .insert("RESIN_TEST_CALLS".into(), calls.clone().into());
        let body = "case \" $* \" in *\" -E \"*) ;; *) printf 'call\\n' >> \"$RESIN_TEST_CALLS\" ;; esac\nexec \"$RESIN_TEST_COMPILER\" \"$@\"";
        executable(&compiler, body);
        let project = c_project(&temp, "int main(void) { return 0; }");
        let tools = environment.toolchain(Some(compiler.as_os_str()), None);
        drop(build(&tools, &project).await);
        drop(build(&tools, &project).await);
        assert_eq!(fs::read_to_string(&calls).unwrap(), "call\n");
        executable(&compiler, &format!("{body}\n# changed"));
        drop(build(&tools, &project).await);
        assert_eq!(fs::read_to_string(&calls).unwrap(), "call\ncall\n");
        environment
            .variables
            .insert("CFLAGS".into(), "changed environment".into());
        drop(
            build(
                &environment.toolchain(Some(compiler.as_os_str()), None),
                &project,
            )
            .await,
        );
        assert_eq!(fs::read_to_string(&calls).unwrap(), "call\ncall\ncall\n");
    }

    #[tokio::test]
    async fn shader_only_graph_needs_no_c_compiler_or_runtime_and_uses_captured_environment() {
        let temp = TempDir::new().unwrap();
        let mut environment = native_environment(&temp);
        let shader = temp.path().join("shader optimizer");
        executable(
            &shader,
            r#"test "$1" = --target-env=vulkan1.3 || exit 1
test "$2" = -O || exit 1
test "$3" = shader.unoptimized.spv || exit 1
test "$4" = -o || exit 1
printf '%s' "$RESIN_TEST_MARKER" > "$5""#,
        );
        environment
            .variables
            .insert("RESIN_TEST_MARKER".into(), "captured marker".into());
        environment
            .variables
            .insert("RESIN_RUNTIME_LIB".into(), "missing-runtime.a".into());
        let tools = environment.toolchain(Some(OsStr::new("missing-cc")), Some(shader.as_os_str()));
        environment
            .variables
            .insert("RESIN_TEST_MARKER".into(), "later marker".into());
        let project = temp.path().join("shader sources");
        fs::create_dir(&project).unwrap();
        fs::write(project.join("shader.unoptimized.spv"), "shader").unwrap();
        fs::write(project.join("build.ninja"), "include toolchain.ninja\nbuild shader.spv: optimize_shader shader.unoptimized.spv | toolchain.state\n").unwrap();
        let built = build(&tools, &project).await;
        assert_eq!(
            fs::read(built.path("shader.spv")).unwrap(),
            b"captured marker"
        );
        assert!(built.executable(program()).is_err());
    }
}
