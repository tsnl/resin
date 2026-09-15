#[allow(dead_code)]
mod support;

use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
};
use tempfile::TempDir;

use support::{service::Service, shaders};

fn cli(source: &str, args: &[&str]) -> Output {
    selected(source, None, args)
}

fn selected(source: &str, entry: Option<&str>, args: &[&str]) -> Output {
    let temp = TempDir::new_in(std::env::temp_dir()).unwrap();
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
    invoke_with(&Service::new(), cwd, input, args)
}

fn invoke_with(service: &Service, cwd: &Path, input: &Path, args: &[&str]) -> Output {
    service
        .command()
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

fn server_artifact(service: &Service, profile: &str) -> PathBuf {
    let files: Vec<_> = fs::read_dir(service.directory.path().join("build"))
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            path.file_name().unwrap() != "shaders" && path.file_name().unwrap() != ".artifacts"
        })
        .collect();
    assert_eq!(files.len(), 1, "{files:?}");
    let executable = files[0]
        .join(profile)
        .join(format!("program{}", std::env::consts::EXE_SUFFIX));
    assert!(executable.is_file());
    executable
}

#[test]
fn build_run_and_lsp_require_an_explicit_reachable_server() {
    let directory = TempDir::new().unwrap();
    let source = directory.path().join("main.resin");
    let destination = directory.path().join("program");
    let text = "export { main }; def main() -> int = { 37 };";
    fs::write(&source, text).unwrap();
    fs::write(&destination, "previous output").unwrap();
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let unreachable = format!("http://{}", listener.local_addr().unwrap());
    drop(listener);
    for setting in [None, Some("not a URL"), Some(unreachable.as_str())] {
        for mode in ["run", "build", "lsp"] {
            let mut command = Command::new(env!("CARGO_BIN_EXE_resin"));
            command
                .current_dir(directory.path())
                .env_remove("RESIN_SERVER");
            if let Some(setting) = setting {
                command.env("RESIN_SERVER", setting);
            }
            match mode {
                "run" => {
                    command.arg(&source);
                }
                "build" => {
                    command.arg(&source).arg("-o").arg(&destination);
                }
                "lsp" => {
                    command.arg("--lsp").arg(directory.path());
                }
                _ => unreachable!(),
            }
            let output = command.stdin(std::process::Stdio::null()).output().unwrap();
            assert!(!output.status.success(), "{setting:?}: {mode}");
            assert!(output.stdout.is_empty(), "{setting:?}: {mode}");
            let error = String::from_utf8_lossy(&output.stderr);
            assert!(
                error.contains(if setting == Some(unreachable.as_str()) {
                    "error sending request"
                } else {
                    "RESIN_SERVER"
                }),
                "{setting:?}: {mode}: {error}"
            );
            assert_eq!(fs::read_to_string(&source).unwrap(), text);
            assert_eq!(fs::read_to_string(&destination).unwrap(), "previous output");
        }
    }
    assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 2);
}

#[test]
fn omitted_unit_returns_run_and_reject_non_unit_tails() {
    let output = cli(
        r#"export { main }; import { "$/string.resin" }; def greet() = { print("hello\n"); }; def main() = { greet() };"#,
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
fn destruction_runs_when_native_status_propagates_to_the_entry() {
    let output = cli(
        "export { main }; import { \"$/string.resin\", \"$/status.resin\" }; struct Cleanup { def drop(self: Ptr<Cleanup>) = { print(fmt(\"cleanup\\n\", ())); }; };  def main() -> Result<(), _> = { RuntimeStatus.from_code(0)?; var cleanup = Cleanup {}; RuntimeStatus.from_code(8)?; ok(()) };",
        &[],
    );
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(
        String::from_utf8_lossy(&output.stdout).replace("\r\n", "\n"),
        "cleanup\n"
    );
    assert!(String::from_utf8_lossy(&output.stderr).contains("unhandled error: WindowUnavailable"));
}

#[test]
fn default_output_downloads_and_runs_without_a_client_build_directory() {
    let temp = TempDir::new_in(std::env::temp_dir()).unwrap();
    let sources = temp.path().join("sources");
    fs::create_dir(&sources).unwrap();
    let input = sources.join("hello world.resin");
    fs::write(
        &input,
        r#"export { main }; import { "$/string.resin" }; def main () -> int = { print("hello\n"); 7 };"#,
    )
    .unwrap();
    let output = invoke(temp.path(), &input, &[]);
    assert_eq!(output.status.code(), Some(7));
    assert_eq!(output.stdout, b"hello\n");
    assert!(output.stderr.is_empty());
    assert!(!sources.join("build").exists());
    assert!(!temp.path().join("build").exists());
    assert_eq!(fs::read_dir(temp.path()).unwrap().count(), 1);
}

#[test]
fn cached_programs_track_foreign_headers() {
    let service = Service::new();
    let temp = TempDir::new_in(std::env::temp_dir()).unwrap();
    let input = temp.path().join("source.resin");
    let header = temp.path().join("value header.h");
    fs::write(&header, "static inline int value(void) { return 41; }\n").unwrap();
    fs::write(&input, format!(
        "export {{ main }}; extern {{ \"{}\": {{ def value() -> int; }} }}; def main() -> int = {{ value() }};",
        header.to_string_lossy().replace('\\', "/")
    )).unwrap();
    assert_eq!(
        invoke_with(&service, temp.path(), &input, &[])
            .status
            .code(),
        Some(41)
    );
    let cold = service.server.counters();
    assert_eq!(
        invoke_with(&service, temp.path(), &input, &[])
            .status
            .code(),
        Some(41)
    );
    assert_eq!(service.server.counters(), cold);

    fs::write(header, "static inline int value(void) { return 42; }\n").unwrap();
    assert_eq!(
        invoke_with(&service, temp.path(), &input, &[])
            .status
            .code(),
        Some(42)
    );
    let edited = service.server.counters();
    assert_eq!(edited.hir_builds, cold.hir_builds);
    assert_eq!(edited.generated_builds, cold.generated_builds + 1);
}

#[test]
fn strings_are_c_compatible_in_both_profiles() {
    let source = r#"
        export { main };

        extern {
            "string.h": {
                def strlen(text: Ptr<ubyte>) -> ulong;
            },
        };
        import { "$/string.resin" };
        def main() -> int = {
            var path = "triangle.png";
            var text = "a\0b";
            if (strlen(path.data) == ulong(12)
                && strlen(text.data) == ulong(1)) {
                print(fmt("{0}:{1}", (path, text)));
                0
            } else { 1 }
        };"#;
    let temp = TempDir::new_in(std::env::temp_dir()).unwrap();
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
    let temp = TempDir::new_in(std::env::temp_dir()).unwrap();
    let input = temp.path().join("source.resin");
    fs::write(
        &input,
        r#"export { main }; import { "$/string.resin" }; def main () -> int = { print("ran\n"); 7 };"#,
    )
    .unwrap();
    let destination = format!("dist/custom program{}", std::env::consts::EXE_SUFFIX);
    let output = invoke(temp.path(), &input, &["-o", &destination]);
    success(&output);
    assert!(output.stdout.is_empty());
    let executable = temp.path().join(destination);
    assert!(executable.is_file());
    assert!(!temp.path().join("build").exists());
    let run = Command::new(executable).output().unwrap();
    assert_eq!(run.status.code(), Some(7));
    assert_eq!(run.stdout, b"ran\n");
}

#[test]
fn output_directories_receive_the_source_name() {
    for destination in ["existing", "new/nested/"] {
        let temp = TempDir::new_in(std::env::temp_dir()).unwrap();
        fs::create_dir(temp.path().join("existing")).unwrap();
        let input = temp.path().join("hello.resin");
        fs::write(
            &input,
            r#"export { main }; import { "$/string.resin" }; def main() -> () = { print("hello\n"); };"#,
        )
        .unwrap();
        let output = invoke(temp.path(), &input, &["-o", destination]);
        success(&output);
        assert!(output.stdout.is_empty());
        let executable = temp
            .path()
            .join(destination)
            .join(format!("hello{}", std::env::consts::EXE_SUFFIX));
        assert!(executable.is_file());
        assert!(!temp.path().join("build").exists());
        let run = Command::new(executable).output().unwrap();
        success(&run);
        assert_eq!(run.stdout, b"hello\n");
    }
}

#[test]
fn sources_with_the_same_name_keep_distinct_cached_results() {
    let service = Service::new();
    let temp = TempDir::new_in(std::env::temp_dir()).unwrap();
    for folder in ["first", "second"] {
        let directory = temp.path().join(folder);
        fs::create_dir(&directory).unwrap();
        let input = directory.join("source.resin");
        fs::write(
            &input,
            format!(r#"export {{ main }}; import {{ "$/string.resin" }}; def main() -> () = {{ print(fmt("{folder}", ())); }};"#),
        )
        .unwrap();
        let output = invoke_with(&service, temp.path(), &input, &[]);
        success(&output);
        assert_eq!(output.stdout, folder.as_bytes());
    }
    assert_eq!(service.server.counters().hir_builds, 2);
    assert_eq!(service.server.counters().generated_builds, 2);
    assert!(!temp.path().join("build").exists());
}

#[test]
fn invalid_destination_parents_fail_before_building_or_running() {
    let temp = TempDir::new_in(std::env::temp_dir()).unwrap();
    let input = temp.path().join("source.resin");
    fs::write(
        &input,
        r#"export { main }; import { "$/string.resin" }; def main() -> () = { print("ran\n"); };"#,
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
    assert!(!temp.path().join("build").exists());
}

#[test]
fn directory_outputs_cannot_overwrite_the_source() {
    let temp = TempDir::new_in(std::env::temp_dir()).unwrap();
    let input = temp
        .path()
        .join(format!("source{}", std::env::consts::EXE_SUFFIX));
    let source = r#"export { main }; import { "$/string.resin" }; def main() -> () = { print("must not run"); };"#;
    fs::write(&input, source).unwrap();
    let output = invoke(temp.path(), &input, &["-o", "."]);
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8_lossy(&output.stderr).contains("overwrite the source"));
    assert_eq!(fs::read_to_string(input).unwrap(), source);
    assert!(!temp.path().join("build").exists());
}

#[test]
fn run_returns_the_program_exit_status() {
    let output = cli("export { main }; def main () -> int = { 37 };", &[]);
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
        r#"export { main }; import { "$/string.resin" }; def main() -> () = { var n = 42; print(fmt("x = {0}\n", (n,))); };"#,
        &[],
    );
    success(&output);
    assert_eq!(output.stdout, b"x = 42\n");
}

#[test]
fn one_file_can_have_multiple_exported_entry_points() {
    let source = r#"
        export { main, demo, status }; import { "$/string.resin" };
        def main() -> () = { print("main"); };
        def demo() -> () = { print("demo"); };
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
            "take () or (int, Ptr<Ptr<ubyte>>, Ptr<Ptr<ubyte>>)",
        ),
        (
            "export { demo }; def demo() -> uint = { uint(0) };",
            Some("demo"),
            "take () or (int, Ptr<Ptr<ubyte>>, Ptr<Ptr<ubyte>>)",
        ),
        (
            "export { rand }; extern { \"stdlib.h\": { def rand() -> int; } };",
            Some("rand"),
            "Resin function",
        ),
    ] {
        let output = selected(source, entry, &[]);
        assert!(!output.status.success());
        assert!(output.stdout.is_empty());
        assert!(
            String::from_utf8_lossy(&output.stderr).contains(expected),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
fn selectors_work_with_output_paths_and_source_protection() {
    let temp = TempDir::new_in(std::env::temp_dir()).unwrap();
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
    let output = invoke(temp.path(), &selected, &["-o", "dist/"]);
    success(&output);
    let executable = temp.path().join(format!(
        "dist/hello world-demo{}",
        std::env::consts::EXE_SUFFIX
    ));
    assert_eq!(Command::new(executable).status().unwrap().code(), Some(19));
    let output = invoke(temp.path(), &selected, &["-o", input.to_str().unwrap()]);
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
    let temp = TempDir::new_in(std::env::temp_dir()).unwrap();
    let input = temp.path().join("input.resin");
    let output = temp.path().join("program");
    fs::write(&input, "export { main }; def main () -> int = { 0 };").unwrap();
    fs::write(&output, "keep me").unwrap();
    let service = Service::configured(|_, environment| {
        environment.variables.insert(
            "RESIN_RUNTIME_LIB".into(),
            temp.path().join("missing.a").into(),
        );
    });
    let result = service
        .command()
        .current_dir(temp.path())
        .arg(&input)
        .arg("-o")
        .arg(&output)
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("missing.a"));
    assert_eq!(fs::read_to_string(output).unwrap(), "keep me");
}

#[test]
fn builds_and_executes_paths_with_spaces_and_shell_punctuation() {
    let temp = TempDir::new_in(std::env::temp_dir()).unwrap();
    let executable = temp
        .path()
        .join(format!("program ; literal{}", std::env::consts::EXE_SUFFIX));
    let output = cli(
        "export { main }; def main () -> int = { 19 };",
        &["-o", executable.to_str().unwrap()],
    );
    success(&output);
    assert_eq!(Command::new(executable).status().unwrap().code(), Some(19));
}

#[test]
fn bad_destinations_and_missing_compilers_preserve_files() {
    let temp = TempDir::new_in(std::env::temp_dir()).unwrap();
    let input = temp.path().join("input.resin");
    let source = "export { main }; def main () -> int = { 0 };";
    fs::write(&input, source).unwrap();
    let service = Service::new();
    let output = service
        .command()
        .current_dir(temp.path())
        .arg(&input)
        .args(["-o"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("overwrite the source"));
    assert_eq!(fs::read_to_string(&input).unwrap(), source);

    let destination = temp.path().join("existing");
    let missing = temp.path().join("missing compiler");
    fs::write(&destination, b"keep me").unwrap();
    let service = Service::configured(|_, environment| {
        environment
            .variables
            .insert("CC".into(), missing.clone().into());
    });
    let output = invoke_with(
        &service,
        temp.path(),
        &input,
        &["-o", destination.to_str().unwrap()],
    );
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("missing compiler"));
    assert_eq!(fs::read(destination).unwrap(), b"keep me");
    assert_eq!(fs::read_dir(temp.path()).unwrap().count(), 2);
}

#[test]
fn invalid_options_and_source_report_errors() {
    for args in [
        vec!["-o"],
        vec!["--cc"],
        vec!["--spirv-opt"],
        vec!["--unknown"],
    ] {
        assert!(
            !cli("export { main }; def main() = {};", &args)
                .status
                .success()
        );
    }
    assert!(!cli("main = ;", &[]).status.success());
}

#[test]
fn decorated_host_calls_need_no_spirv_opt() {
    let service = Service::configured(|_, environment| {
        environment
            .variables
            .insert("SPIRV_OPT".into(), "/does/not/exist/spirv-opt".into());
    });
    let temp = TempDir::new().unwrap();
    let input = temp.path().join("host.resin");
    fs::write(&input, "export { main }; @compute_shader def kernel(invocation: ulong, output: Ptr<uint>) = { var i = uint(invocation); output.* := { i }; }; def main() -> int = { var output = 0_ui; kernel(7_ul, &output); if (output == 7_ui) { 0 } else { 1 } };").unwrap();
    success(&invoke_with(&service, temp.path(), &input, &[]));
}

#[test]
fn executable_build_retains_all_shader_stages_and_embeds_their_spirv() {
    let Some(spirv_opt) = shaders::optimizer() else {
        return;
    };
    let temp = TempDir::new_in(std::env::temp_dir()).unwrap();
    let input = temp.path().join("stages.resin");
    fs::write(&input, r#"
        export { main };
        import { "$/graphics.resin", "$/gpu.resin" };
        @compute_shader def kernel(invocation: ulong, output: Ptr<uint>) = { var i = uint(invocation); output.* := { i + 1_ui }; };
        @vertex_shader def vertex(i: int) -> Vertex = {
            Vertex {
                position = Position { x = 0.0_f, y = 0.0_f, z = 0.0_f, w = 1.0_f },
                color = Color { r = 1.0_f, g = 0.0_f, b = 0.0_f, a = 1.0_f },
            }
        };
        @fragment_shader def fragment(color: Color) -> Color = { color };
        def main() -> Result<(), _> = {
            if (0 == 1) {
                var gpu = Gpu.new()?;
                gpu.create_compute_pipeline(kernel)?;
                gpu.create_graphics_pipeline(vertex, fragment)?;
            };
            ok(())
        };
    "#).unwrap();
    let destination = temp
        .path()
        .join(format!("program{}", std::env::consts::EXE_SUFFIX));
    let service = Service::configured(|_, environment| {
        environment.variables.insert("SPIRV_OPT".into(), spirv_opt);
        environment
            .variables
            .insert("GLSLC".into(), "/missing/glslc".into());
    });
    let output = service
        .command()
        .current_dir(temp.path())
        .arg(&input)
        .arg("-o")
        .arg(&destination)
        .output()
        .unwrap();
    success(&output);
    assert!(output.stdout.is_empty());
    let executable = server_artifact(&service, "release");
    assert_eq!(
        fs::read(&executable).unwrap(),
        fs::read(&destination).unwrap()
    );
    let directory = executable.parent().unwrap();
    let c = fs::read_to_string(directory.join("main.c")).unwrap();
    let shaders = fs::read_dir(directory)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            path.extension().is_some_and(|extension| extension == "spv")
                && !path.to_string_lossy().ends_with(".unoptimized.spv")
        })
        .collect::<Vec<_>>();
    assert_eq!(shaders.len(), 3);
    assert!(directory.join("build.ninja").is_file());
    for shader in shaders {
        let raw = fs::read(shader.with_extension("unoptimized.spv")).unwrap();
        assert_eq!(&raw[..4], &[3, 2, 35, 7]);
        let bytes = fs::read(&shader).unwrap();
        assert_eq!(&bytes[..4], &[3, 2, 35, 7]);
        assert_eq!(bytes.len() % 4, 0);
        let header = fs::read_to_string(shader.with_extension("h")).unwrap();
        let embedded = header
            .split("0x")
            .skip(1)
            .map(|hex| u8::from_str_radix(&hex[..2], 16).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            embedded, bytes,
            "generated header must embed the exact SPIR-V bytes"
        );
        assert!(header.contains("_Alignas(4)"));
        assert!(
            c.contains(
                shader
                    .with_extension("h")
                    .file_name()
                    .unwrap()
                    .to_str()
                    .unwrap()
            )
        );
    }
    success(&Command::new(destination).output().unwrap());
}

#[test]
fn removed_artifact_switches_and_duplicate_destinations_are_rejected_before_building() {
    let service = Service::new();
    let temp = TempDir::new_in(std::env::temp_dir()).unwrap();
    let input = temp.path().join("source.resin");
    fs::write(&input, "export { main }; def main() = {};").unwrap();
    for args in [
        vec!["--target", "c"],
        vec!["--stage", "compute"],
        vec!["--cc", "cc"],
        vec!["--spirv-opt", "spirv-opt"],
        vec!["--output", "first", "-o", "second"],
    ] {
        let output = invoke_with(&service, temp.path(), &input, &args);
        assert_eq!(output.status.code(), Some(2));
        assert!(!output.stderr.is_empty());
    }
    assert!(!temp.path().join("first").exists());
    assert!(!temp.path().join("second").exists());
    assert!(!temp.path().join("build").exists());
    assert_eq!(service.server.counters(), resin_server::Counters::default());
}

#[test]
fn process_entries_receive_literal_arguments_in_run_and_compiled_modes() {
    let service = Service::new();
    let temp = TempDir::new_in(std::env::temp_dir()).unwrap();
    let input = temp.path().join("args.resin");
    fs::write(&input, r#"
        export { main }; import { "$/string.resin", "$/process.resin" };
        def main(argc: int, argv: Ptr<Ptr<ubyte>>, envp: Ptr<Ptr<ubyte>>) -> int = {
            var args = arguments(argc, argv);
            var with_sentinel = arguments(argc + 1, argv);
            var index = 1_ul;
            while (index < args.length) {
                print(fmt("[{0}]\n", (argument(args, index).bytes(),)));
                index := index + 1_ul;
            };
            if (argc == 6 && ulong(with_sentinel.at(ulong(argc))) == 0_ul && argument(args, 0_ul).length > 0_ul) { 0 } else { 1 }
        };
    "#).unwrap();
    let args = ["hello world", "", "--flag", "semi;$(literal)", "λ"];
    let expected = "[hello world]\n[]\n[--flag]\n[semi;$(literal)]\n[λ]\n";
    let output = service
        .command()
        .current_dir(temp.path())
        .arg(&input)
        .arg("--")
        .args(args)
        .output()
        .unwrap();
    success(&output);
    assert_eq!(
        String::from_utf8(output.stdout)
            .unwrap()
            .replace("\r\n", "\n"),
        expected
    );
    let executable = temp
        .path()
        .join(format!("args{}", std::env::consts::EXE_SUFFIX));
    success(&invoke(
        temp.path(),
        &input,
        &["-o", executable.to_str().unwrap()],
    ));
    let output = Command::new(&executable).args(args).output().unwrap();
    success(&output);
    assert_eq!(
        String::from_utf8(output.stdout)
            .unwrap()
            .replace("\r\n", "\n"),
        expected
    );
    let rejected = invoke(temp.path(), &input, &["-o", "unused", "--", "argument"]);
    assert!(!rejected.status.success());
    assert!(!temp.path().join("unused").exists());
}

#[test]
fn process_environment_is_frozen_and_distinguishes_empty_from_missing() {
    let service = Service::new();
    let temp = TempDir::new_in(std::env::temp_dir()).unwrap();
    let header = temp.path().join("mutate_environment.h");
    fs::write(
        &header,
        r#"
        #include <stdlib.h>
        #ifndef _WIN32
        extern int setenv(const char *, const char *, int);
        #endif
        static inline int mutate_environment(void) {
        #ifdef _WIN32
            return _putenv_s("RESIN_SNAPSHOT_TEST", "after");
        #else
            return setenv("RESIN_SNAPSHOT_TEST", "after", 1);
        #endif
        }
    "#,
    )
    .unwrap();
    let input = temp.path().join("environment.resin");
    let header_path = header.to_string_lossy().replace('\\', "/");
    fs::write(&input, format!(r#"
        export {{ main }};

        extern {{
            "{header_path}": {{
                def mutate_environment() -> int;
            }},
            "stdlib.h": {{
                def getenv(name: Ptr<ubyte>) -> Ptr<ubyte>;
            }},
        }};
        import {{ "$/string.resin", "$/process.resin", "$/span.resin" }};
        def main(argc: int, argv: Ptr<Ptr<ubyte>>, envp: Ptr<Ptr<ubyte>>) -> Result<int, _> = {{
            var name = "RESIN_SNAPSHOT_TEST";
            var empty = "RESIN_SNAPSHOT_EMPTY";
            var missing = "RESIN_SNAPSHOT_MISSING";
            var before = environment_get(envp, name.data)?;
            var status = mutate_environment();
            var after = environment_get(envp, name.data)?;
            var live = c_string(getenv(name.data));
            var absent = match (environment_get(envp, missing.data)) {{ ok(value) => {{ 1 == 0 }}, err(error) => {{ 1 == 1 }} }};
            var env = environment(envp);
            var with_sentinel = Span<Ptr<ubyte>> {{ data = envp, length = env.length + 1_ul }};
            print(fmt("{{0}}/{{1}}/{{2}}\n", (before.bytes(), after.bytes(), live.bytes())));
            ok(if (status == 0 && absent && environment_get(envp, empty.data)?.length == 0_ul
                && env.length >= 2_ul && ulong(with_sentinel.at(env.length)) == 0_ul) {{ 0 }} else {{ 1 }})
        }};"#)).unwrap();
    let output = service
        .command()
        .current_dir(temp.path())
        .arg(&input)
        .env("RESIN_SNAPSHOT_TEST", "before")
        .env("RESIN_SNAPSHOT_EMPTY", "")
        .env_remove("RESIN_SNAPSHOT_MISSING")
        .output()
        .unwrap();
    success(&output);
    assert_eq!(
        String::from_utf8(output.stdout)
            .unwrap()
            .replace("\r\n", "\n"),
        "before/before/after\n"
    );
    // Reusing the executable captures this invocation's environment, not build-time values.
    let output = service
        .command()
        .current_dir(temp.path())
        .arg(&input)
        .env("RESIN_SNAPSHOT_TEST", "fresh")
        .env("RESIN_SNAPSHOT_EMPTY", "")
        .env_remove("RESIN_SNAPSHOT_MISSING")
        .output()
        .unwrap();
    success(&output);
    assert_eq!(
        String::from_utf8(output.stdout)
            .unwrap()
            .replace("\r\n", "\n"),
        "fresh/fresh/after\n"
    );
}

#[test]
fn process_entry_signatures_results_and_argument_bounds_are_checked() {
    for signature in [
        "argc: uint, argv: Ptr<Ptr<ubyte>>, envp: Ptr<Ptr<ubyte>>",
        "argc: int, argv: Ptr<ubyte>, envp: Ptr<Ptr<ubyte>>",
        "argc: int, argv: Ptr<Ptr<ubyte>>",
    ] {
        let output = cli(
            &format!("export {{ main }}; def main({signature}) = {{}};"),
            &[],
        );
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains("take () or"));
    }
    for (result, body, code) in [
        ("()", "{}", 0),
        ("Result<int, E>", "ok(7)", 7),
        ("Result<(), E>", "err(E {})", 1),
    ] {
        let source = format!(
            "export {{ main }}; struct E {{}}; def main(argc: int, argv: Ptr<Ptr<ubyte>>, envp: Ptr<Ptr<ubyte>>) -> {result} = {{ {body} }};"
        );
        assert_eq!(cli(&source, &[]).status.code(), Some(code));
    }
    let output = cli(
        r#"export { main }; import { "$/process.resin" }; def main(argc: int, argv: Ptr<Ptr<ubyte>>, envp: Ptr<Ptr<ubyte>>) = { argument(arguments(argc, argv), ulong(argc)); };"#,
        &[],
    );
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("array index out of bounds"));
}

#[test]
#[cfg(unix)]
fn process_arguments_preserve_non_utf8_bytes() {
    use std::os::unix::ffi::OsStringExt;
    let service = Service::new();
    let temp = TempDir::new_in(std::env::temp_dir()).unwrap();
    let input = temp.path().join("bytes.resin");
    fs::write(&input, r#"export { main }; import { "$/string.resin", "$/process.resin" }; def main(argc: int, argv: Ptr<Ptr<ubyte>>, envp: Ptr<Ptr<ubyte>>) = { print(fmt("{0}", (argument(arguments(argc, argv), 1_ul).bytes(),))); };"#).unwrap();
    let output = service
        .command()
        .current_dir(temp.path())
        .arg(input)
        .arg("--")
        .arg(std::ffi::OsString::from_vec(vec![0xff, b'x']))
        .output()
        .unwrap();
    success(&output);
    assert_eq!(output.stdout, [0xff, b'x']);
}

#[test]
fn output_long_option_builds_without_running() {
    let temp = TempDir::new_in(std::env::temp_dir()).unwrap();
    let source = temp.path().join("main.resin");
    fs::write(&source, "export { main }; def main() -> int = { 42 };").unwrap();
    let destination = temp
        .path()
        .join(format!("published{}", std::env::consts::EXE_SUFFIX));
    let output = invoke(
        temp.path(),
        &source,
        &["--output", destination.to_str().unwrap()],
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(Command::new(destination).status().unwrap().code(), Some(42));
}

#[test]
fn unified_command_identifies_its_version_and_rejects_invalid_lsp_modes() {
    let temp = TempDir::new_in(std::env::temp_dir()).unwrap();
    let version = Command::new(env!("CARGO_BIN_EXE_resin"))
        .arg("--version")
        .output()
        .unwrap();
    assert!(version.status.success());
    assert_eq!(
        String::from_utf8_lossy(&version.stdout).trim(),
        concat!("resin ", env!("CARGO_PKG_VERSION"))
    );
    for args in [
        vec!["--lsp", "missing"],
        vec!["--lsp", ".", "."],
        vec!["--lsp", "--format", "."],
        vec!["--lsp", ".", "--output", "program"],
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_resin"))
            .args(&args)
            .current_dir(temp.path())
            .output()
            .unwrap();
        assert!(!output.status.success(), "{args:?}");
        assert!(output.stdout.is_empty(), "{args:?}");
    }
    assert!(!temp.path().join("build").exists());
}

#[test]
fn embedding_preserves_arbitrary_bytes_and_empty_length_without_tools() {
    let directory = TempDir::new_in(std::env::temp_dir()).unwrap();
    let input = directory.path().join("asset with spaces.bin");
    let output = directory.path().join("asset.h");
    for bytes in [vec![], vec![0, 1, 127, 128, 255, 0]] {
        fs::write(&input, &bytes).unwrap();
        let result = Command::new(env!("CARGO_BIN_EXE_resin"))
            .env("CC", "/missing/cc")
            .env("SPIRV_OPT", "/missing/spirv-opt")
            .env("NINJA", "/missing/ninja")
            .arg("--embed")
            .arg(&input)
            .args(["--symbol", "asset", "--output"])
            .arg(&output)
            .output()
            .unwrap();
        success(&result);
        let text = fs::read_to_string(&output).unwrap();
        assert!(text.contains(&format!("asset_length UINT64_C({})", bytes.len())));
        let encoded = text
            .split("0x")
            .skip(1)
            .map(|hex| u8::from_str_radix(&hex[..2], 16).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(encoded, bytes);
    }
}

#[test]
fn embedding_rejects_invalid_symbols_and_input_overwrites() {
    let directory = TempDir::new_in(std::env::temp_dir()).unwrap();
    let input = directory.path().join("asset.bin");
    let output = directory.path().join("asset.h");
    fs::write(&input, [1, 2, 3]).unwrap();
    fs::write(&output, "previous header").unwrap();
    for symbol in ["bad-name", "1start", "static", ""] {
        let result = Command::new(env!("CARGO_BIN_EXE_resin"))
            .arg("--embed")
            .arg(&input)
            .args(["--symbol", symbol, "--output"])
            .arg(&output)
            .output()
            .unwrap();
        assert!(!result.status.success());
        assert_eq!(fs::read_to_string(&output).unwrap(), "previous header");
    }
    let result = Command::new(env!("CARGO_BIN_EXE_resin"))
        .arg("--embed")
        .arg(&input)
        .args(["--symbol", "asset", "--output"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert_eq!(fs::read(input).unwrap(), [1, 2, 3]);
}
