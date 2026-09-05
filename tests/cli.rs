use std::{
    fs,
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
    Command::new(env!("CARGO_BIN_EXE_resin"))
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

#[test]
fn default_output_compiles_an_executable() {
    let temp = TempDir::new(&std::env::temp_dir()).unwrap();
    let executable = temp.path().join("program");
    let output = cli("main = () => 7;", &["-o", executable.to_str().unwrap()]);
    success(&output);
    assert_eq!(Command::new(executable).status().unwrap().code(), Some(7));
}

#[test]
fn explicit_ir_output_prints_verified_ir() {
    let source = "main = () => 7;";
    let output = cli(source, &["--output", "ir"]);
    success(&output);
    assert_eq!(
        String::from_utf8(output.stdout).unwrap(),
        format!("{}\n", resin::ir::format_module(&support::module(source)))
    );
}

#[test]
fn run_returns_the_program_exit_status() {
    let output = cli("main = () => 37;", &["--output", "run"]);
    assert_eq!(
        output.status.code(),
        Some(37),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn builds_and_executes_paths_with_spaces_and_shell_punctuation() {
    let temp = TempDir::new(&std::env::temp_dir()).unwrap();
    let executable = temp.path().join("program ; literal");
    let output = cli(
        "main = () => 19;",
        &["--output", "exe", "-o", executable.to_str().unwrap()],
    );
    success(&output);
    assert_eq!(Command::new(executable).status().unwrap().code(), Some(19));
}

#[test]
fn bad_destinations_and_missing_compilers_preserve_files() {
    let temp = TempDir::new(&std::env::temp_dir()).unwrap();
    let input = temp.path().join("input.resin");
    let source = "main = () => 0;";
    fs::write(&input, source).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_resin"))
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
        vec![],
        vec!["--output", "exe"],
        vec!["--output", "spirv"],
        vec!["--output", "run", "-o", "unused"],
        vec!["--stage", "nonsense"],
    ] {
        assert!(!cli("main = () => 0;", &args).status.success());
    }
    assert!(!cli("main = ;", &["--output", "c"]).status.success());
}

#[test]
fn shader_output_selects_the_stage_and_entry() {
    let source = "paint = (i: uint) => i;";
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

#[cfg(not(feature = "gpu"))]
#[test]
fn rendering_explains_the_optional_feature() {
    let temp = TempDir::new(&std::env::temp_dir()).unwrap();
    let png = temp.path().join("image.png");
    let output = cli(
        "kernel = (i: uint) => i;",
        &["--output", "compute", "-o", png.to_str().unwrap()],
    );
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("--features gpu"));
    assert!(!png.exists());
}
