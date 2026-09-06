#![cfg(unix)]

use std::{
    fs,
    os::unix::fs::PermissionsExt,
    path::PathBuf,
    process::{Command, Output, Stdio},
};

use resin::toolchain::TempDir;

#[path = "support/shaders.rs"]
mod shaders;

const WRAPPER: &str = "#!/bin/sh\nprintf 'compile\\n' >> \"$RESIN_TEST_COUNT\"\nprintf '%s\\n' \"$*\" >> \"$RESIN_TEST_FLAGS\"\nexec \"$RESIN_TEST_COMPILER\" \"$@\"\n";

#[test]
fn foreign_header_changes_rebuild_including_nested_dependencies() {
    let project = Project::new();
    let header = project.temp.path().join("foreign.h");
    let nested = project.temp.path().join("value.h");
    fs::write(&nested, "#define VALUE 41\n").unwrap();
    fs::write(
        &header,
        "#include \"value.h\"\nstatic inline int value(void) { return VALUE; }\n",
    )
    .unwrap();
    fs::write(
        &project.input,
        format!(
            "export {{ main }}; extern \"{}\" value () -> int; main() -> () = {{ print(\"{{0}}\", (value(),)); }};",
            header.display()
        ),
    )
    .unwrap();
    printed(&project.run(), b"41");
    printed(&project.run(), b"41");
    assert_eq!(project.calls(), 1);
    fs::write(&nested, "#define VALUE 42\n").unwrap();
    printed(&project.run(), b"42");
    assert_eq!(project.calls(), 2);
    fs::remove_file(nested).unwrap();
    let output = project.run();
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert_eq!(project.calls(), 3);
    fs::write(
        &project.input,
        "export { main }; main() -> () = { print(\"no header\", ()); };",
    )
    .unwrap();
    printed(&project.run(), b"no header");
    printed(&project.run(), b"no header");
    assert_eq!(project.calls(), 4);
}

#[test]
fn shader_objects_are_deduplicated_cached_and_rebuilt_with_imported_helpers() {
    let Some(glslc) = shaders::compiler() else {
        return;
    };
    let project = Project::new();
    let shader_compiler = project.temp.path().join("shader-compiler");
    let count = project.temp.path().join("shader-calls");
    fs::write(&shader_compiler, "#!/bin/sh\nprintf 'compile\\n' >> \"$RESIN_TEST_SHADER_COUNT\"\nexec \"$RESIN_TEST_SHADER_COMPILER\" \"$@\"\n").unwrap();
    fs::set_permissions(&shader_compiler, fs::Permissions::from_mode(0o755)).unwrap();
    let helper = project.temp.path().join("helper.resin");
    fs::write(
        &helper,
        "export { pixel }; pixel (i: uint) -> uint = { i + uint (1) };",
    )
    .unwrap();
    fs::write(
        &project.input,
        r#"
        export { main };
        import { "helper.resin" };
        kernel (i: uint) -> uint = { pixel(i) };
        main() -> () = {
            a = shader(kernel, "compute");
            b = shader(kernel, "compute");
            print("{0}", (a.length > ulong (0) && ulong (a.data) == ulong (b.data),));
        };
        "#,
    )
    .unwrap();
    let run = || {
        project
            .command()
            .arg("--glslc")
            .arg(&shader_compiler)
            .env("RESIN_TEST_SHADER_COUNT", &count)
            .env("RESIN_TEST_SHADER_COMPILER", &glslc)
            .output()
            .unwrap()
    };
    let calls = || fs::read_to_string(&count).unwrap().lines().count();
    printed(&run(), b"true");
    printed(&run(), b"true");
    assert_eq!(calls(), 1);
    assert_eq!(project.calls(), 1);
    fs::write(
        &helper,
        "export { pixel }; pixel (i: uint) -> uint = { i + uint (2) };",
    )
    .unwrap();
    printed(&run(), b"true");
    assert_eq!(calls(), 2);
    assert_eq!(project.calls(), 2);
    fs::write(
        &helper,
        "export { pixel }; pixel (i: uint) -> uint = { i / uint (2) };",
    )
    .unwrap();
    let output = run();
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert_eq!(calls(), 2);
    assert_eq!(project.calls(), 2);
}

struct Project {
    temp: TempDir,
    input: PathBuf,
    compiler: PathBuf,
}

impl Project {
    fn new() -> Self {
        let temp = TempDir::new(&std::env::temp_dir()).unwrap();
        let input = temp.path().join("main.resin");
        let compiler = temp.path().join("compiler");
        fs::write(
            &input,
            r#"export { main }; main() -> () = { print("first", ()); };"#,
        )
        .unwrap();
        fs::write(&compiler, WRAPPER).unwrap();
        fs::set_permissions(&compiler, fs::Permissions::from_mode(0o755)).unwrap();
        Self {
            temp,
            input,
            compiler,
        }
    }

    fn command(&self) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_resin"));
        command
            .current_dir(self.temp.path())
            .arg(&self.input)
            .arg("--cc")
            .arg(&self.compiler)
            .env("RESIN_TEST_COUNT", self.temp.path().join("calls"))
            .env("RESIN_TEST_FLAGS", self.temp.path().join("flags"))
            .env(
                "RESIN_TEST_COMPILER",
                std::env::var_os("CC").unwrap_or_else(|| "cc".into()),
            );
        command
    }

    fn run(&self) -> Output {
        self.command().output().unwrap()
    }

    fn calls(&self) -> usize {
        fs::read_to_string(self.temp.path().join("calls"))
            .unwrap()
            .lines()
            .count()
    }

    fn executable(&self) -> PathBuf {
        self.profile_executable("debug")
    }

    fn profile_executable(&self, profile: &str) -> PathBuf {
        let directories: Vec<_> = fs::read_dir(self.temp.path().join("build"))
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .collect();
        assert_eq!(directories.len(), 1);
        directories[0].join(profile).join("program")
    }
}

fn printed(output: &Output, text: &[u8]) {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.stdout, text);
}

#[test]
fn unchanged_programs_reuse_the_executable_and_still_run() {
    let project = Project::new();
    printed(&project.run(), b"first");
    let executable = project.executable();
    let modified = fs::metadata(&executable).unwrap().modified().unwrap();
    printed(&project.run(), b"first");
    assert_eq!(project.calls(), 1);
    assert_eq!(
        fs::metadata(&executable).unwrap().modified().unwrap(),
        modified
    );
    fs::write(
        &project.input,
        "export { main };\n\n// comment only\n\nmain() -> () = {\n    print(\"first\", ());\n};",
    )
    .unwrap();
    printed(&project.run(), b"first");
    assert_eq!(project.calls(), 1);
}

#[test]
fn entry_points_have_separate_reusable_artifacts() {
    let mut project = Project::new();
    fs::write(
        &project.input,
        r#"
        export { main, second };
        main() -> () = { print("first", ()); };
        second() -> () = { print("second", ()); };
    "#,
    )
    .unwrap();
    let original = project.input.clone();
    printed(&project.run(), b"first");
    for (entry, expected) in [
        ("second", &b"second"[..]),
        ("main", &b"first"[..]),
        ("second", &b"second"[..]),
    ] {
        let mut input = original.as_os_str().to_os_string();
        input.push(format!(":{entry}"));
        project.input = input.into();
        printed(&project.run(), expected);
    }
    assert_eq!(project.calls(), 2);
    assert_eq!(
        fs::read_dir(project.temp.path().join("build"))
            .unwrap()
            .count(),
        2
    );
    let mut input = original.as_os_str().to_os_string();
    input.push(":missing");
    project.input = input.into();
    let output = project.run();
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert_eq!(project.calls(), 2);
}

#[test]
fn executable_output_optimizes_and_both_profiles_stay_cached() {
    let project = Project::new();
    printed(&project.run(), b"first");
    let debug = project.executable();
    let debug_modified = fs::metadata(&debug).unwrap().modified().unwrap();
    printed(
        &project.command().args(["-o", "dist/"]).output().unwrap(),
        b"first",
    );
    let release = project.profile_executable("release");
    let release_modified = fs::metadata(&release).unwrap().modified().unwrap();
    assert_eq!(
        fs::read(&release).unwrap(),
        fs::read(project.temp.path().join("dist/main")).unwrap()
    );
    assert_eq!(project.calls(), 2);

    printed(&project.run(), b"first");
    printed(
        &project.command().args(["--out", "copy"]).output().unwrap(),
        b"first",
    );
    let output = project
        .command()
        .args(["--output", "exe", "-o", "exported"])
        .output()
        .unwrap();
    printed(&output, b"");
    assert_eq!(
        fs::read(&release).unwrap(),
        fs::read(project.temp.path().join("exported")).unwrap()
    );
    assert_eq!(
        fs::metadata(debug).unwrap().modified().unwrap(),
        debug_modified
    );
    assert_eq!(
        fs::metadata(release).unwrap().modified().unwrap(),
        release_modified
    );
    assert_eq!(project.calls(), 2);

    let flags = fs::read_to_string(project.temp.path().join("flags")).unwrap();
    let optimizations: Vec<_> = flags
        .lines()
        .map(|line| {
            line.split_whitespace()
                .filter(|flag| flag.starts_with("-O"))
                .collect::<Vec<_>>()
        })
        .collect();
    assert_eq!(optimizations, [vec!["-O0"], vec!["-O3"]]);
}

#[test]
fn textual_output_does_not_compile_an_executable() {
    let project = Project::new();
    let output = project
        .command()
        .args(["--output", "c", "-o", "source.c"])
        .output()
        .unwrap();
    printed(&output, b"");
    assert!(
        fs::read_to_string(project.temp.path().join("source.c"))
            .unwrap()
            .contains("int main(void)")
    );
    assert!(!project.temp.path().join("calls").exists());
    assert!(!project.temp.path().join("build").exists());
}

#[test]
fn changed_source_rebuilds_in_the_same_directory() {
    let project = Project::new();
    printed(&project.run(), b"first");
    let executable = project.executable();
    fs::write(
        &project.input,
        r#"export { main }; main() -> () = { print("second", ()); };"#,
    )
    .unwrap();
    printed(&project.run(), b"second");
    assert_eq!(project.calls(), 2);
    assert_eq!(project.executable(), executable);
    printed(&project.run(), b"second");
    assert_eq!(project.calls(), 2);
}

#[test]
fn compiler_and_environment_changes_invalidate_the_cache() {
    let project = Project::new();
    printed(&project.run(), b"first");
    fs::write(&project.compiler, format!("{WRAPPER}# changed wrapper\n")).unwrap();
    printed(&project.run(), b"first");
    assert_eq!(project.calls(), 2);
    printed(
        &project.command().env("CFLAGS", "changed").output().unwrap(),
        b"first",
    );
    assert_eq!(project.calls(), 3);
}

#[test]
fn compiler_symlinks_preserve_the_invocation_name() {
    let mut project = Project::new();
    let alias = project.temp.path().join("driver");
    let wrapper = WRAPPER.replacen(
        "#!/bin/sh\n",
        "#!/bin/sh\ncase \"$0\" in */driver) ;; *) exit 9 ;; esac\n",
        1,
    );
    fs::write(&project.compiler, wrapper).unwrap();
    std::os::unix::fs::symlink(&project.compiler, &alias).unwrap();
    project.compiler = alias;
    printed(&project.run(), b"first");
    printed(&project.run(), b"first");
    assert_eq!(project.calls(), 1);
}

#[test]
fn runtime_headers_and_archive_changes_invalidate_the_cache() {
    let project = Project::new();
    let include = project.temp.path().join("include");
    fs::create_dir_all(include.join("resin_runtime")).unwrap();
    fs::copy(
        PathBuf::from(resin_runtime::INCLUDE_DIR).join("resin_runtime.h"),
        include.join("resin_runtime.h"),
    )
    .unwrap();
    for entry in
        fs::read_dir(PathBuf::from(resin_runtime::INCLUDE_DIR).join("resin_runtime")).unwrap()
    {
        let entry = entry.unwrap();
        fs::copy(
            entry.path(),
            include.join("resin_runtime").join(entry.file_name()),
        )
        .unwrap();
    }
    let library = project.temp.path().join("runtime.a");
    let original = std::env::current_exe()
        .unwrap()
        .parent()
        .unwrap()
        .join("libresin_runtime.a");
    fs::copy(&original, &library).unwrap();
    let run = || {
        project
            .command()
            .env("RESIN_RUNTIME_INCLUDE", &include)
            .env("RESIN_RUNTIME_LIB", &library)
            .output()
            .unwrap()
    };
    printed(&run(), b"first");
    printed(&run(), b"first");
    assert_eq!(project.calls(), 1);
    let header = include.join("resin_runtime/print.h");
    let mut text = fs::read_to_string(&header).unwrap();
    text.push_str("\n// changed header\n");
    fs::write(header, text).unwrap();
    printed(&run(), b"first");
    assert_eq!(project.calls(), 2);

    fs::write(&library, "not a library").unwrap();
    let output = run();
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert_eq!(project.calls(), 3);
    fs::copy(original, &library).unwrap();
    printed(&run(), b"first");
    assert_eq!(project.calls(), 4);
}

#[test]
fn failed_rebuilds_preserve_the_old_executable_but_never_run_it() {
    let project = Project::new();
    printed(&project.run(), b"first");
    let executable = project.executable();
    let original = fs::read(&executable).unwrap();
    fs::write(&project.compiler, "#!/bin/sh\nexit 9\n").unwrap();
    for _ in 0..2 {
        let output = project.run();
        assert!(!output.status.success());
        assert!(output.stdout.is_empty());
        assert_eq!(fs::read(&executable).unwrap(), original);
    }
    assert!(!executable.parent().unwrap().join("fingerprint").exists());
    fs::write(&project.compiler, WRAPPER).unwrap();
    printed(&project.run(), b"first");
    assert_eq!(project.calls(), 2);
}

#[test]
fn missing_artifacts_are_rebuilt() {
    let project = Project::new();
    printed(&project.run(), b"first");
    fs::remove_file(project.executable()).unwrap();
    printed(&project.run(), b"first");
    assert_eq!(project.calls(), 2);
}

#[test]
fn concurrent_runs_share_one_build() {
    let project = Project::new();
    let first = project
        .command()
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let second = project
        .command()
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    printed(&first.wait_with_output().unwrap(), b"first");
    printed(&second.wait_with_output().unwrap(), b"first");
    assert_eq!(project.calls(), 1);
    assert!(project.executable().is_file());
}
