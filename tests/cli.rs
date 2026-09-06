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
    let temp = TempDir::new(&std::env::temp_dir()).unwrap();
    let input = temp.path().join("source.resin");
    fs::write(&input, source).unwrap();
    invoke(temp.path(), &input, args)
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
fn default_output_builds_in_cwd_and_runs() {
    let temp = TempDir::new(&std::env::temp_dir()).unwrap();
    let sources = temp.path().join("sources");
    fs::create_dir(&sources).unwrap();
    let input = sources.join("hello world.resin");
    fs::write(&input, r#"main () -> int = { print("hello\n", ()); 7 };"#).unwrap();
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
fn strings_are_c_compatible_in_both_profiles() {
    let source = r#"
        extern "string.h" strlen(text: Ptr<ubyte>) -> ulong;
        path = "triangle.png";
        text = "a\0b";
        main() -> int = {
            if (strlen(Ptr<ubyte>(&path)) == ulong(12)
                && strlen(Ptr<ubyte>(&text)) == ulong(1)) {
                print("{0}:{1}", (path, text));
                0
            } else { 1 }
        };
    "#;
    for args in [&[][..], &["-o", "program"][..]] {
        let output = cli(source, args);
        success(&output);
        assert_eq!(output.stdout, b"triangle.png:a\0b");
        assert!(output.stderr.is_empty());
    }
}

#[test]
fn default_output_runs_then_copies_even_on_nonzero_exit() {
    let temp = TempDir::new(&std::env::temp_dir()).unwrap();
    let input = temp.path().join("source.resin");
    fs::write(&input, r#"main () -> int = { print("ran\n", ()); 7 };"#).unwrap();
    let output = invoke(temp.path(), &input, &["-o", "dist/custom program"]);
    assert_eq!(output.status.code(), Some(7));
    assert_eq!(output.stdout, b"ran\n");
    let executable = temp.path().join("dist/custom program");
    assert_eq!(
        fs::read(&executable).unwrap(),
        fs::read(artifact(temp.path(), "release")).unwrap()
    );
    assert_eq!(
        Command::new(executable).output().unwrap().status.code(),
        Some(7)
    );
}

#[test]
fn output_directories_receive_the_source_name() {
    for destination in ["existing", "new/nested/"] {
        let temp = TempDir::new(&std::env::temp_dir()).unwrap();
        fs::create_dir(temp.path().join("existing")).unwrap();
        let input = temp.path().join("hello.resin");
        fs::write(&input, r#"print("hello\n", ());"#).unwrap();
        let output = invoke(temp.path(), &input, &["--output", "run", "-o", destination]);
        success(&output);
        assert_eq!(output.stdout, b"hello\n");
        let executable = temp
            .path()
            .join(destination)
            .join(format!("hello{}", std::env::consts::EXE_SUFFIX));
        assert_eq!(
            fs::read(&executable).unwrap(),
            fs::read(artifact(temp.path(), "release")).unwrap()
        );
        assert!(Command::new(executable).output().unwrap().status.success());
    }
}

#[test]
fn explicit_exe_output_does_not_run() {
    let temp = TempDir::new(&std::env::temp_dir()).unwrap();
    let input = temp.path().join("source.resin");
    fs::write(&input, r#"main () -> int = { print("ran\n", ()); 7 };"#).unwrap();
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
        fs::write(&input, format!(r#"print("{folder}", ());"#)).unwrap();
        let output = invoke(temp.path(), &input, &[]);
        success(&output);
        assert_eq!(output.stdout, folder.as_bytes());
    }
    assert_eq!(fs::read_dir(temp.path().join("build")).unwrap().count(), 2);
}

#[test]
fn failed_copies_happen_after_execution() {
    let temp = TempDir::new(&std::env::temp_dir()).unwrap();
    let input = temp.path().join("source.resin");
    fs::write(&input, r#"print("ran\n", ());"#).unwrap();
    fs::write(temp.path().join("not-a-directory"), "keep me").unwrap();
    let output = invoke(temp.path(), &input, &["-o", "not-a-directory/program"]);
    assert!(!output.status.success());
    assert_eq!(output.stdout, b"ran\n");
    assert_eq!(
        fs::read_to_string(temp.path().join("not-a-directory")).unwrap(),
        "keep me"
    );
    artifact(temp.path(), "release");
}

#[test]
fn directory_outputs_cannot_overwrite_the_source() {
    let temp = TempDir::new(&std::env::temp_dir()).unwrap();
    let input = temp.path().join("source");
    let source = r#"print("must not run", ());"#;
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
    let source = "main () -> int = { 7 };";
    let output = cli(source, &["--output", "ir"]);
    success(&output);
    assert_eq!(
        String::from_utf8(output.stdout).unwrap(),
        format!("{}\n", resin::ir::format_module(&support::module(source)))
    );
}

#[test]
fn run_returns_the_program_exit_status() {
    let output = cli("main () -> int = { 37 };", &["--output", "run"]);
    assert_eq!(
        output.status.code(),
        Some(37),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn run_prints_program_output() {
    let output = cli(r#"n = 42; print("x = {0}\n", (n,));"#, &["--output", "run"]);
    success(&output);
    assert_eq!(output.stdout, b"x = 42\n");
}

#[test]
fn missing_runtime_preserves_existing_output() {
    let temp = TempDir::new(&std::env::temp_dir()).unwrap();
    let input = temp.path().join("input.resin");
    let output = temp.path().join("program");
    fs::write(&input, "main () -> int = { 0 };").unwrap();
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
    let executable = temp.path().join("program ; literal");
    let output = cli(
        "main () -> int = { 19 };",
        &["--output", "exe", "-o", executable.to_str().unwrap()],
    );
    success(&output);
    assert_eq!(Command::new(executable).status().unwrap().code(), Some(19));
}

#[test]
fn bad_destinations_and_missing_compilers_preserve_files() {
    let temp = TempDir::new(&std::env::temp_dir()).unwrap();
    let input = temp.path().join("input.resin");
    let source = "main () -> int = { 0 };";
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
        assert!(!cli("main () -> int = { 0 };", &args).status.success());
    }
    assert!(!cli("main = ;", &["--output", "c"]).status.success());
}

#[test]
fn shader_output_selects_the_stage_and_entry() {
    let source = "paint (i: uint) -> uint = { i };";
    let output = cli(source, &["--output", "glsl", "--entry", "paint"]);
    success(&output);
    assert!(String::from_utf8_lossy(&output.stdout).starts_with("#version 460\n"));

    let Some(compiler) = shaders::compiler() else {
        return;
    };
    let temp = TempDir::new(&std::env::temp_dir()).unwrap();
    let spirv = temp.path().join("shader.spv");
    let output = cli(
        source,
        &[
            "--output",
            "spirv",
            "--entry",
            "paint",
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
        "kernel (i: uint) -> uint = { i };",
        &["--output", "compute", "-o", png.to_str().unwrap()],
    );
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("invalid value"));
    assert!(!png.exists());
}
