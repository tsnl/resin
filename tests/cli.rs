use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
};

use resin::toolchain::TempDir;

#[path = "support/shaders.rs"]
mod shaders;
mod support;

fn cli(source: &str, args: &[&str]) -> Output {
    selected(source, None, args)
}

fn selected(source: &str, entry: Option<&str>, args: &[&str]) -> Output {
    let temp = TempDir::new(&std::env::temp_dir()).unwrap();
    let input = temp.path().join("source.resin");
    fs::write(&input, source).unwrap();
    let input = selector(&input, entry);
    invoke(temp.path(), &input, args)
}

fn selector(path: &Path, entry: Option<&str>) -> PathBuf {
    let mut input = path.as_os_str().to_os_string();
    if let Some(entry) = entry {
        input.push(format!(":{entry}"));
    }
    input.into()
}

fn invoke(cwd: &Path, input: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_resin"))
        .current_dir(cwd)
        .arg(input)
        .args(args)
        .output()
        .unwrap()
}

fn success(output: &Output) {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn artifact(cwd: &Path, profile: &str) -> PathBuf {
    let files: Vec<_> = fs::read_dir(cwd.join("build"))
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .collect();
    assert_eq!(files.len(), 1, "{files:?}");
    let executable = files[0]
        .join(profile)
        .join(format!("program{}", std::env::consts::EXE_SUFFIX));
    assert!(executable.is_file());
    executable
}

#[test]
fn omitted_unit_returns_run_and_reject_non_unit_tails() {
    let output = cli(
        r#"export { main }; def greet() = { print("hello\n", ()); }; def main() = { greet() };"#,
        &[],
    );
    success(&output);
    assert_eq!(output.stdout, b"hello\n");

    let output = cli("export { main }; def main() = { 42 };", &[]);
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8_lossy(&output.stderr).contains("TypeMismatch"));
}

#[test]
fn defer_runs_when_native_status_propagates_to_the_entry() {
    let output = cli(
        "export { main }; import { \"std/status.resin\" }; def main() -> Result<(), _> = { status(0)?; defer print(\"cleanup\\n\", ()); status(8)?; defer print(\"not reached\\n\", ()); ok(()) };",
        &[],
    );
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(
        String::from_utf8_lossy(&output.stdout).replace("\r\n", "\n"),
        "cleanup\n"
    );
    assert!(String::from_utf8_lossy(&output.stderr).contains("unhandled error: RuntimeError"));
}

#[test]
fn default_output_builds_in_cwd_and_runs() {
    let temp = TempDir::new(&std::env::temp_dir()).unwrap();
    let sources = temp.path().join("sources");
    fs::create_dir(&sources).unwrap();
    let input = sources.join("hello world.resin");
    fs::write(
        &input,
        r#"export { main }; def main () -> int = { print("hello\n", ()); 7 };"#,
    )
    .unwrap();
    let output = invoke(temp.path(), &input, &[]);
    assert_eq!(output.status.code(), Some(7));
    assert_eq!(output.stdout, b"hello\n");
    assert!(output.stderr.is_empty());
    assert!(!sources.join("build").exists());
    let executable = artifact(temp.path(), "debug");
    assert!(
        executable
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .file_name()
            .unwrap()
            .to_string_lossy()
            .starts_with("hello world-")
    );
    assert_eq!(
        Command::new(executable).output().unwrap().stdout,
        b"hello\n"
    );
}

#[test]
fn cached_programs_track_foreign_headers() {
    let temp = TempDir::new(&std::env::temp_dir()).unwrap();
    let input = temp.path().join("source.resin");
    let header = temp.path().join("value header.h");
    fs::write(&header, "static inline int value(void) { return 41; }\n").unwrap();
    fs::write(&input, format!(
        "export {{ main }}; extern \"{}\" def value() -> int; def main() -> int = {{ value() }};",
        header.to_string_lossy().replace('\\', "/")
    )).unwrap();
    assert_eq!(invoke(temp.path(), &input, &[]).status.code(), Some(41));
    let executable = artifact(temp.path(), "debug");
    let modified = fs::metadata(&executable).unwrap().modified().unwrap();
    assert!(executable.parent().unwrap().join("fingerprint").is_file());
    assert_eq!(invoke(temp.path(), &input, &[]).status.code(), Some(41));
    assert_eq!(
        fs::metadata(&executable).unwrap().modified().unwrap(),
        modified
    );

    fs::write(header, "static inline int value(void) { return 42; }\n").unwrap();
    assert_eq!(invoke(temp.path(), &input, &[]).status.code(), Some(42));
}

#[test]
fn strings_are_c_compatible_in_both_profiles() {
    let source = r#"
        export { main };
        extern "string.h" def strlen(text: Ptr<ubyte>) -> ulong;
        def main() -> int = {
            var path = "triangle.png";
            var text = "a\0b";
            if (strlen(Ptr<ubyte>(&path)) == ulong(12)
                && strlen(Ptr<ubyte>(&text)) == ulong(1)) {
                print("{0}:{1}", (path, text));
                0
            } else { 1 }
        };
    "#;
    let temp = TempDir::new(&std::env::temp_dir()).unwrap();
    let input = temp.path().join("source.resin");
    fs::write(&input, source).unwrap();
    let debug = invoke(temp.path(), &input, &[]);
    let name = format!("program{}", std::env::consts::EXE_SUFFIX);
    let build = invoke(temp.path(), &input, &["-o", &name]);
    success(&build);
    assert!(build.stdout.is_empty());
    let release = Command::new(temp.path().join(name)).output().unwrap();
    for output in [debug, release] {
        success(&output);
        assert_eq!(output.stdout, b"triangle.png:a\0b");
        assert!(output.stderr.is_empty());
    }
}

#[test]
fn executable_destination_builds_without_running() {
    let temp = TempDir::new(&std::env::temp_dir()).unwrap();
    let input = temp.path().join("source.resin");
    fs::write(
        &input,
        r#"export { main }; def main () -> int = { print("ran\n", ()); 7 };"#,
    )
    .unwrap();
    let destination = format!("dist/custom program{}", std::env::consts::EXE_SUFFIX);
    let output = invoke(temp.path(), &input, &["-o", &destination]);
    success(&output);
    assert!(output.stdout.is_empty());
    let executable = temp.path().join(destination);
    assert_eq!(
        fs::read(&executable).unwrap(),
        fs::read(artifact(temp.path(), "release")).unwrap()
    );
    let run = Command::new(executable).output().unwrap();
    assert_eq!(run.status.code(), Some(7));
    assert_eq!(run.stdout, b"ran\n");
}

#[test]
fn output_directories_receive_the_source_name() {
    for destination in ["existing", "new/nested/"] {
        let temp = TempDir::new(&std::env::temp_dir()).unwrap();
        fs::create_dir(temp.path().join("existing")).unwrap();
        let input = temp.path().join("hello.resin");
        fs::write(
            &input,
            r#"export { main }; def main() -> () = { print("hello\n", ()); };"#,
        )
        .unwrap();
        let output = invoke(temp.path(), &input, &["--output", "run", "-o", destination]);
        success(&output);
        assert!(output.stdout.is_empty());
        let executable = temp
            .path()
            .join(destination)
            .join(format!("hello{}", std::env::consts::EXE_SUFFIX));
        assert_eq!(
            fs::read(&executable).unwrap(),
            fs::read(artifact(temp.path(), "release")).unwrap()
        );
        let run = Command::new(executable).output().unwrap();
        success(&run);
        assert_eq!(run.stdout, b"hello\n");
    }
}

#[test]
fn explicit_exe_output_does_not_run() {
    let temp = TempDir::new(&std::env::temp_dir()).unwrap();
    let input = temp.path().join("source.resin");
    fs::write(
        &input,
        r#"export { main }; def main () -> int = { print("ran\n", ()); 7 };"#,
    )
    .unwrap();
    let output = invoke(temp.path(), &input, &["--output", "exe", "-o", "program"]);
    success(&output);
    assert!(output.stdout.is_empty());
    assert_eq!(
        fs::read(temp.path().join("program")).unwrap(),
        fs::read(artifact(temp.path(), "release")).unwrap()
    );
}

#[test]
fn sources_with_the_same_name_have_separate_caches() {
    let temp = TempDir::new(&std::env::temp_dir()).unwrap();
    for folder in ["first", "second"] {
        let directory = temp.path().join(folder);
        fs::create_dir(&directory).unwrap();
        let input = directory.join("source.resin");
        fs::write(
            &input,
            format!(r#"export {{ main }}; def main() -> () = {{ print("{folder}", ()); }};"#),
        )
        .unwrap();
        let output = invoke(temp.path(), &input, &[]);
        success(&output);
        assert_eq!(output.stdout, folder.as_bytes());
    }
    assert_eq!(fs::read_dir(temp.path().join("build")).unwrap().count(), 2);
}

#[test]
fn failed_copies_do_not_run_the_program() {
    let temp = TempDir::new(&std::env::temp_dir()).unwrap();
    let input = temp.path().join("source.resin");
    fs::write(
        &input,
        r#"export { main }; def main() -> () = { print("ran\n", ()); };"#,
    )
    .unwrap();
    fs::write(temp.path().join("not-a-directory"), "keep me").unwrap();
    let output = invoke(temp.path(), &input, &["-o", "not-a-directory/program"]);
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert_eq!(
        fs::read_to_string(temp.path().join("not-a-directory")).unwrap(),
        "keep me"
    );
    artifact(temp.path(), "release");
}

#[test]
fn directory_outputs_cannot_overwrite_the_source() {
    let temp = TempDir::new(&std::env::temp_dir()).unwrap();
    let input = temp
        .path()
        .join(format!("source{}", std::env::consts::EXE_SUFFIX));
    let source = r#"export { main }; def main() -> () = { print("must not run", ()); };"#;
    fs::write(&input, source).unwrap();
    let output = invoke(temp.path(), &input, &["-o", "."]);
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8_lossy(&output.stderr).contains("overwrite the source"));
    assert_eq!(fs::read_to_string(input).unwrap(), source);
    assert!(!temp.path().join("build").exists());
}

#[test]
fn explicit_ir_output_prints_verified_ir() {
    let source = "export { main }; def main () -> int = { 7 };";
    let output = cli(source, &["--output", "ir"]);
    success(&output);
    assert_eq!(
        String::from_utf8(output.stdout).unwrap(),
        format!("{}\n", resin::ir::format_module(&support::module(source)))
    );
}

#[test]
fn run_returns_the_program_exit_status() {
    let output = cli(
        "export { main }; def main () -> int = { 37 };",
        &["--output", "run"],
    );
    assert_eq!(
        output.status.code(),
        Some(37),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn run_prints_program_output() {
    let output = cli(
        r#"export { main }; def main() -> () = { var n = 42; print("x = {0}\n", (n,)); };"#,
        &["--output", "run"],
    );
    success(&output);
    assert_eq!(output.stdout, b"x = 42\n");
}

#[test]
fn one_file_can_have_multiple_exported_entry_points() {
    let source = r#"
        export { main, demo, status };
        def main() -> () = { print("main", ()); };
        def demo() -> () = { print("demo", ()); };
        def status() -> int = { 23 };
    "#;
    for (entry, expected, code) in [
        (None, &b"main"[..], 0),
        (Some("main"), &b"main"[..], 0),
        (Some("demo"), &b"demo"[..], 0),
        (Some("status"), &b""[..], 23),
    ] {
        let output = selected(source, entry, &[]);
        assert_eq!(
            output.status.code(),
            Some(code),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(output.stdout, expected);
    }
    let output = selected(
        "export { demo }; def demo() -> int = { 42 };",
        Some("demo"),
        &[],
    );
    assert_eq!(output.status.code(), Some(42));
}

#[test]
fn selected_entries_must_be_exported_resin_functions_with_the_right_signature() {
    for (source, entry, expected) in [
        ("def main() -> () = {};", None, "not exported"),
        (
            "export { main }; def main() -> () = {}; def hidden() -> () = {};",
            Some("hidden"),
            "not exported",
        ),
        (
            "export { main }; def main() -> () = {};",
            Some("missing"),
            "not exported",
        ),
        (
            "export { demo }; def demo(n: int) -> int = { n };",
            Some("demo"),
            "take (), and return int",
        ),
        (
            "export { demo }; def demo() -> uint = { uint(0) };",
            Some("demo"),
            "take (), and return int",
        ),
        (
            "export { rand }; extern \"stdlib.h\" def rand() -> int;",
            Some("rand"),
            "Resin function",
        ),
    ] {
        let output = selected(source, entry, &["--output", "c"]);
        assert!(!output.status.success());
        assert!(output.stdout.is_empty());
        assert!(
            String::from_utf8_lossy(&output.stderr).contains(expected),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    for stage in ["check", "ast", "ir"] {
        success(&cli(
            "type Item = int; def helper() -> int = { 1 };",
            &["--output", stage],
        ));
    }
}

#[test]
fn selectors_work_with_output_paths_and_source_protection() {
    let temp = TempDir::new(&std::env::temp_dir()).unwrap();
    let folder = temp.path().join(if cfg!(windows) {
        "directory with spaces"
    } else {
        "dir:with:colons"
    });
    fs::create_dir(&folder).unwrap();
    let input = folder.join("hello world.resin");
    let source = "export { demo }; def demo() -> int = { 19 };";
    fs::write(&input, source).unwrap();
    let selected = selector(&input, Some("demo"));
    let output = invoke(temp.path(), &selected, &["--output", "exe", "-o", "dist/"]);
    success(&output);
    let executable = temp.path().join(format!(
        "dist/hello world-demo{}",
        std::env::consts::EXE_SUFFIX
    ));
    assert_eq!(Command::new(executable).status().unwrap().code(), Some(19));
    let output = invoke(
        temp.path(),
        &selected,
        &["--output", "c", "-o", input.to_str().unwrap()],
    );
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("overwrite the source"));
    assert_eq!(fs::read_to_string(input).unwrap(), source);
}

#[test]
fn malformed_selectors_are_rejected() {
    for entry in ["", "1", "demo-name", "two words"] {
        let output = selected("export { main }; def main() -> () = {};", Some(entry), &[]);
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains("FILE[:ENTRY]"));
    }
}

#[test]
fn missing_runtime_preserves_existing_output() {
    let temp = TempDir::new(&std::env::temp_dir()).unwrap();
    let input = temp.path().join("input.resin");
    let output = temp.path().join("program");
    fs::write(&input, "export { main }; def main () -> int = { 0 };").unwrap();
    fs::write(&output, "keep me").unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_resin"))
        .current_dir(temp.path())
        .arg(&input)
        .arg("-o")
        .arg(&output)
        .env("RESIN_RUNTIME_LIB", temp.path().join("missing.a"))
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("runtime library not found"));
    assert_eq!(fs::read_to_string(output).unwrap(), "keep me");
}

#[test]
fn builds_and_executes_paths_with_spaces_and_shell_punctuation() {
    let temp = TempDir::new(&std::env::temp_dir()).unwrap();
    let executable = temp
        .path()
        .join(format!("program ; literal{}", std::env::consts::EXE_SUFFIX));
    let output = cli(
        "export { main }; def main () -> int = { 19 };",
        &["--output", "exe", "-o", executable.to_str().unwrap()],
    );
    success(&output);
    assert_eq!(Command::new(executable).status().unwrap().code(), Some(19));
}

#[test]
fn bad_destinations_and_missing_compilers_preserve_files() {
    let temp = TempDir::new(&std::env::temp_dir()).unwrap();
    let input = temp.path().join("input.resin");
    let source = "export { main }; def main () -> int = { 0 };";
    fs::write(&input, source).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_resin"))
        .current_dir(temp.path())
        .arg(&input)
        .args(["--output", "c", "-o"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("overwrite the source"));
    assert_eq!(fs::read_to_string(&input).unwrap(), source);

    let destination = temp.path().join("existing");
    let missing = temp.path().join("missing compiler");
    fs::write(&destination, b"keep me").unwrap();
    let output = cli(
        source,
        &[
            "--output",
            "exe",
            "-o",
            destination.to_str().unwrap(),
            "--cc",
            missing.to_str().unwrap(),
        ],
    );
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("cannot run"));
    assert_eq!(fs::read(destination).unwrap(), b"keep me");
    assert_eq!(fs::read_dir(temp.path()).unwrap().count(), 2);
}

#[test]
fn invalid_options_and_source_report_errors() {
    for args in [
        vec!["--output", "exe"],
        vec!["--output", "spirv"],
        vec!["--stage", "nonsense"],
    ] {
        assert!(
            !cli("export { main }; def main () -> int = { 0 };", &args)
                .status
                .success()
        );
    }
    assert!(!cli("main = ;", &["--output", "c"]).status.success());
}

#[test]
fn shader_output_selects_the_stage_and_entry() {
    success(&cli(
        "export { main }; def main(i: uint) -> uint = { i };",
        &["--output", "glsl"],
    ));
    let source = "export { paint }; def paint (i: uint) -> uint = { i };";
    let output = selected(source, Some("paint"), &["--output", "glsl"]);
    success(&output);
    assert!(String::from_utf8_lossy(&output.stdout).starts_with("#version 460\n"));

    let Some(compiler) = shaders::compiler() else {
        return;
    };
    let temp = TempDir::new(&std::env::temp_dir()).unwrap();
    let spirv = temp.path().join("shader.spv");
    let output = selected(
        source,
        Some("paint"),
        &[
            "--output",
            "spirv",
            "--stage",
            "compute",
            "--glslc",
            compiler.to_str().unwrap(),
            "-o",
            spirv.to_str().unwrap(),
        ],
    );
    success(&output);
    assert_eq!(&fs::read(&spirv).unwrap()[..4], &[3, 2, 35, 7]);
}

#[test]
fn graphics_execution_is_not_a_cli_output_mode() {
    let temp = TempDir::new(&std::env::temp_dir()).unwrap();
    let png = temp.path().join("image.png");
    let output = cli(
        "export { kernel }; def kernel (i: uint) -> uint = { i };",
        &["--output", "compute", "-o", png.to_str().unwrap()],
    );
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("invalid value"));
    assert!(!png.exists());
}
