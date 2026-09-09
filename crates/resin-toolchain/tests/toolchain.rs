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
        "include toolchain.ninja\nrule c\n  command = $cc $cflags -MD -MF $out.d -MT $out $in -o $out $ldflags\n  depfile = $out.d\n  deps = gcc\nbuild program{}: c main.c | $runtime_library toolchain.state\n",
        std::env::consts::EXE_SUFFIX)).unwrap();
    project
}

fn build(tools: &Toolchain, project: &Path) -> BuiltProject {
    tools
        .build(project, "generated/module", "main", CProfile::Debug)
        .unwrap()
}

fn program() -> String {
    format!("program{}", std::env::consts::EXE_SUFFIX)
}

#[test]
fn missing_tools_fail_only_when_the_graph_uses_them() {
    let temp = TempDir::new().unwrap();
    let mut environment = native_environment(&temp);
    let project = c_project(&temp, "int main(void) { return 0; }");
    let host = environment.toolchain(None, Some(OsStr::new("missing-resin-glslc")));
    assert_eq!(
        build(&host, &project)
            .executable(program())
            .unwrap()
            .run()
            .unwrap(),
        0
    );
    let missing = environment.toolchain(Some(OsStr::new("missing-resin-cc")), None);
    let error = missing
        .build(&project, "generated/module", "main", CProfile::Debug)
        .unwrap_err();
    assert!(error.to_string().contains("missing-resin-cc"), "{error}");
    environment
        .variables
        .insert("NINJA".into(), "missing-resin-ninja".into());
    let error = environment
        .toolchain(None, None)
        .build(&project, "generated/module", "main", CProfile::Debug)
        .unwrap_err();
    assert!(error.to_string().contains("missing-resin-ninja"), "{error}");
}

#[test]
fn cached_executable_runs_literal_arguments_and_copies_to_new_directories() {
    let temp = TempDir::new().unwrap();
    let tools = native_environment(&temp).toolchain(None, None);
    let project = c_project(
        &temp,
        r#"#include <string.h>
int main(int argc, char **argv) {
    return argc == 3 && strcmp(argv[1], "two words") == 0 && strcmp(argv[2], "$(literal)") == 0 ? 23 : 99;
}"#,
    );
    let built = build(&tools, &project);
    let executable = built.executable(program()).unwrap();
    let args = ["two words".into(), "$(literal)".into()];
    assert_eq!(executable.run_with_args(&args).unwrap(), 23);
    let output = temp.path().join("new directory").join(program());
    executable.copy_to(&output).unwrap();
    assert_eq!(
        Command::new(output).args(&args).status().unwrap().code(),
        Some(23)
    );
    let path = executable.path().to_path_buf();
    let modified = fs::metadata(&path).unwrap().modified().unwrap();
    drop(executable);
    drop(built);
    let reused = build(&tools, &project);
    assert_eq!(reused.path(program()), path);
    assert_eq!(fs::metadata(path).unwrap().modified().unwrap(), modified);
    assert!(reused.path("main.c").is_file());
    assert!(reused.path("build.ninja").is_file());
}

#[test]
fn failed_builds_leave_published_executables_unchanged() {
    let temp = TempDir::new().unwrap();
    let tools = native_environment(&temp).toolchain(None, None);
    let project = c_project(&temp, "int main(void) { return 7; }");
    let built = build(&tools, &project);
    let output = built.path(program());
    let before = fs::read(&output).unwrap();
    drop(built);
    fs::write(project.join("main.c"), "this is not C").unwrap();
    for _ in 0..2 {
        let error = tools
            .build(&project, "generated/module", "main", CProfile::Debug)
            .unwrap_err();
        assert!(error.to_string().contains("native build failed"), "{error}");
        assert_eq!(fs::read(&output).unwrap(), before);
    }
    fs::write(project.join("main.c"), "int main(void) { return 8; }").unwrap();
    let rebuilt = build(&tools, &project);
    assert_eq!(rebuilt.executable(program()).unwrap().run().unwrap(), 8);
    let published = fs::metadata(rebuilt.path(program())).unwrap();
    let staged = fs::metadata(rebuilt.path(".ninja-work").join(program())).unwrap();
    assert_eq!(
        published.modified().unwrap(),
        staged.modified().unwrap(),
        "publication must preserve the timestamp used for incremental Ninja builds"
    );
}

#[test]
fn nested_headers_refresh_and_removed_inputs_are_not_left_in_the_cache() {
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
            .executable(program())
            .unwrap()
            .run()
            .unwrap(),
        4
    );
    fs::write(&header, "#define VALUE 5\n").unwrap();
    assert_eq!(
        build(&tools, &project)
            .executable(program())
            .unwrap()
            .run()
            .unwrap(),
        5
    );
    fs::remove_file(header).unwrap();
    assert!(
        tools
            .build(&project, "generated/module", "main", CProfile::Debug)
            .is_err()
    );
}

#[test]
fn outputs_removed_from_the_graph_disappear_from_the_published_project() {
    let temp = TempDir::new().unwrap();
    let tools = native_environment(&temp).toolchain(None, None);
    let project = c_project(&temp, "int main(void) { return 0; }");
    let first = build(&tools, &project);
    assert!(first.path(program()).is_file());
    drop(first);
    let graph = project.join("build.ninja");
    let replacement = format!("replacement{}", std::env::consts::EXE_SUFFIX);
    fs::write(
        &graph,
        fs::read_to_string(&graph)
            .unwrap()
            .replace(&program(), &replacement),
    )
    .unwrap();
    let second = build(&tools, &project);
    assert!(!second.path(program()).exists());
    assert_eq!(second.executable(replacement).unwrap().run().unwrap(), 0);
}

#[test]
fn generated_sources_cannot_replace_toolchain_configuration() {
    let temp = TempDir::new().unwrap();
    let tools = native_environment(&temp).toolchain(None, None);
    let project = c_project(&temp, "int main(void) { return 0; }");
    fs::write(project.join("toolchain.ninja"), "cc = unexpected").unwrap();
    let error = tools
        .build(&project, "module", "main", CProfile::Debug)
        .unwrap_err();
    assert!(
        error.to_string().contains("reserved toolchain filename"),
        "{error}"
    );
}

#[cfg(unix)]
mod process_configuration {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    fn executable(path: &Path, body: &str) {
        fs::write(path, format!("#!/bin/sh\n{body}\n")).unwrap();
        fs::set_permissions(path, fs::Permissions::from_mode(0o755)).unwrap();
    }

    #[test]
    fn source_symlinks_cannot_reach_outside_the_project() {
        let temp = TempDir::new().unwrap();
        let tools = native_environment(&temp).toolchain(None, None);
        let project = c_project(&temp, "int main(void) { return 0; }");
        let outside = temp.path().join("outside");
        fs::write(&outside, "retained").unwrap();
        std::os::unix::fs::symlink(&outside, project.join("linked")).unwrap();
        let error = tools
            .build(&project, "module", "main", CProfile::Debug)
            .unwrap_err();
        assert!(
            error.to_string().contains("cannot follow symlinks"),
            "{error}"
        );
        assert_eq!(fs::read_to_string(outside).unwrap(), "retained");
    }

    #[test]
    fn relative_and_empty_include_paths_use_the_captured_directory() {
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
            let first = build(&tools, &project);
            let modified = fs::metadata(first.path(program()))
                .unwrap()
                .modified()
                .unwrap();
            assert_eq!(first.executable(program()).unwrap().run().unwrap(), 4);
            drop(first);
            let reused = build(&tools, &project);
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
                    .executable(program())
                    .unwrap()
                    .run()
                    .unwrap(),
                5
            );
        }
    }

    #[test]
    fn dependencies_changed_during_a_command_are_rebuilt_on_the_next_build() {
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
            fs::read(build(&tools, &project).path("output")).unwrap(),
            b"old"
        );
        assert_eq!(
            fs::read(build(&tools, &project).path("output")).unwrap(),
            b"new"
        );
        drop(build(&tools, &project));
        assert_eq!(fs::read_to_string(calls).unwrap(), "call\ncall\n");
    }

    #[test]
    fn compiler_paths_and_environment_changes_invalidate_ninja_work() {
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
        let body =
            "printf 'call\\n' >> \"$RESIN_TEST_CALLS\"\nexec \"$RESIN_TEST_COMPILER\" \"$@\"";
        executable(&compiler, body);
        let project = c_project(&temp, "int main(void) { return 0; }");
        let tools = environment.toolchain(Some(compiler.as_os_str()), None);
        drop(build(&tools, &project));
        drop(build(&tools, &project));
        assert_eq!(fs::read_to_string(&calls).unwrap(), "call\n");
        executable(&compiler, &format!("{body}\n# changed"));
        drop(build(&tools, &project));
        assert_eq!(fs::read_to_string(&calls).unwrap(), "call\ncall\n");
        environment
            .variables
            .insert("CFLAGS".into(), "changed environment".into());
        drop(build(
            &environment.toolchain(Some(compiler.as_os_str()), None),
            &project,
        ));
        assert_eq!(fs::read_to_string(&calls).unwrap(), "call\ncall\ncall\n");
    }

    #[test]
    fn shader_only_graph_needs_no_c_compiler_or_runtime_and_uses_captured_environment() {
        let temp = TempDir::new().unwrap();
        let mut environment = native_environment(&temp);
        let shader = temp.path().join("shader compiler");
        executable(
            &shader,
            "for output do :; done\nprintf '%s' \"$RESIN_TEST_MARKER\" > \"$output\"",
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
        fs::write(project.join("shader.glsl"), "shader").unwrap();
        fs::write(project.join("build.ninja"), "include toolchain.ninja\nrule shader\n  command = $glslc $in -o $out\nbuild shader.spv: shader shader.glsl | toolchain.state\n").unwrap();
        let built = build(&tools, &project);
        assert_eq!(
            fs::read(built.path("shader.spv")).unwrap(),
            b"captured marker"
        );
        assert!(built.executable(program()).is_err());
    }
}
