//! Header analysis is cached independently of Resin bodies; foreign calls link directly.
#[allow(dead_code)]
mod support;
use std::{fs, path::Path, process::Command};
use support::service::Service;

fn build(service: &Service, source: &Path, output: &Path) -> i32 {
    let result = service
        .command()
        .arg(source)
        .arg("-o")
        .arg(output)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    Command::new(output)
        .output()
        .unwrap()
        .status
        .code()
        .unwrap()
}

#[test]
fn header_analysis_survives_body_edits_and_rechecks_transitive_headers() {
    let service = Service::new();
    let client = tempfile::tempdir().unwrap();
    let source = client.path().join("main.resin");
    let headers = client.path().join("include");
    fs::create_dir(&headers).unwrap();
    let detail = headers.join("declarations.h");
    let output = client
        .path()
        .join(format!("program{}", std::env::consts::EXE_SUFFIX));
    fs::write(headers.join("api.h"), "#include \"declarations.h\"\n").unwrap();
    fs::write(&detail, "int abs(int value);\n").unwrap();
    let write = |tail| {
        fs::write(&source, format!("export {{ main }}; extern {{ \"include/api.h\": {{ def abs(value: int) -> int; }} }}; def main() -> int = {{ abs(-40) + {tail} }};")).unwrap()
    };
    write(1);
    assert_eq!(build(&service, &source, &output), 41);
    let initial = service.server.counters();
    assert_eq!(initial.foreign_builds, 1);
    write(2);
    assert_eq!(build(&service, &source, &output), 42);
    let edited = service.server.counters();
    assert_eq!(edited.foreign_builds, initial.foreign_builds);
    assert_eq!(
        edited.native_object_builds,
        initial.native_object_builds + 1
    );
    fs::write(
        &detail,
        "int abs(int value); // new text, same symbol and ABI\n",
    )
    .unwrap();
    assert_eq!(build(&service, &source, &output), 42);
    let reanalyzed = service.server.counters();
    assert_eq!(reanalyzed.foreign_builds, edited.foreign_builds + 1);
    assert_eq!(reanalyzed.native_object_builds, edited.native_object_builds);
    assert_eq!(reanalyzed.executable_builds, edited.executable_builds);
}

#[test]
fn direct_libc_calls_preserve_pointer_integer_and_void_results() {
    let service = Service::new();
    let client = tempfile::tempdir().unwrap();
    let source = client.path().join("main.resin");
    fs::write(
        &source,
        r#"
        export { main };
        extern {
            "stdlib.h": { def abs(value: int) -> int; def srand(seed: uint); },
            "string.h": {
                def strlen(text: Ptr<ubyte>) -> ulong;
                def strchr(text: Ptr<ubyte>, byte: int) -> Ptr<ubyte>;
            },
        };
        def main() -> int = {
            srand(42_ui);
            var text = "linked";
            var suffix = strchr(text.data, 110_i);
            if (strlen(suffix) == 4_ul && abs(-42_i) == 42_i) { 42_i } else { 1_i }
        };
    "#,
    )
    .unwrap();
    let output = client
        .path()
        .join(format!("program{}", std::env::consts::EXE_SUFFIX));
    assert_eq!(build(&service, &source, &output), 42);
}

#[test]
fn macros_inline_functions_and_mismatched_declarations_do_not_publish_outputs() {
    let service = Service::new();
    let client = tempfile::tempdir().unwrap();
    let source = client.path().join("main.resin");
    let header = client.path().join("api.h");
    let output = client
        .path()
        .join(format!("program{}", std::env::consts::EXE_SUFFIX));
    fs::write(&source, "export { main }; extern { \"api.h\": { def requested(value: int) -> int; } }; def main() -> int = { requested(42) };").unwrap();
    for text in [
        "#define requested(value) (value)\n",
        "static inline int requested(int value) { return value; }\n",
        "static int requested(int value);\n",
        "unsigned int requested(unsigned int value);\n",
        "int requested(int value, ...);\n",
        "/* missing declaration */\n",
    ] {
        fs::write(&header, text).unwrap();
        fs::write(&output, b"previous artifact").unwrap();
        let result = service
            .command()
            .arg(&source)
            .arg("-o")
            .arg(&output)
            .output()
            .unwrap();
        assert!(!result.status.success(), "unexpectedly accepted {text}");
        assert!(!result.stderr.is_empty());
        assert_eq!(fs::read(&output).unwrap(), b"previous artifact");
    }
    assert_eq!(service.server.counters().native_object_builds, 0);
    assert_eq!(service.server.counters().executable_builds, 0);
}

#[test]
fn a_declaration_without_a_linked_definition_reports_a_link_failure() {
    let service = Service::new();
    let client = tempfile::tempdir().unwrap();
    let source = client.path().join("main.resin");
    let output = client
        .path()
        .join(format!("program{}", std::env::consts::EXE_SUFFIX));
    fs::write(
        client.path().join("api.h"),
        "int resin_test_missing_definition(void);\n",
    )
    .unwrap();
    fs::write(&source, "export { main }; extern { \"api.h\": { def resin_test_missing_definition() -> int; } }; def main() -> int = { resin_test_missing_definition() };\n").unwrap();
    fs::write(&output, b"previous artifact").unwrap();
    let result = service
        .command()
        .arg(&source)
        .arg("-o")
        .arg(&output)
        .output()
        .unwrap();
    assert!(!result.status.success());
    let message = String::from_utf8_lossy(&result.stderr);
    assert!(
        message.contains("resin_test_missing_definition"),
        "{message}"
    );
    assert_eq!(fs::read(&output).unwrap(), b"previous artifact");
    assert_eq!(service.server.counters().foreign_builds, 1);
    assert_eq!(service.server.counters().native_object_builds, 1);
}

#[test]
fn foreign_main_cannot_resolve_to_the_generated_startup_entry() {
    let service = Service::new();
    let client = tempfile::tempdir().unwrap();
    let source = client.path().join("main.resin");
    let output = client
        .path()
        .join(format!("program{}", std::env::consts::EXE_SUFFIX));
    fs::write(client.path().join("api.h"), "int main(void);\n").unwrap();
    fs::write(&source, "export { entry }; extern { \"api.h\": { def main() -> int; } }; def entry() -> int = { main() };\n").unwrap();
    fs::write(&output, b"previous artifact").unwrap();
    let result = service
        .command()
        .arg(format!("{}:entry", source.display()))
        .arg("-o")
        .arg(&output)
        .output()
        .unwrap();
    assert!(!result.status.success());
    let message = String::from_utf8_lossy(&result.stderr);
    assert!(
        message.contains("conflicts with the native startup entry"),
        "{message}"
    );
    assert_eq!(fs::read(&output).unwrap(), b"previous artifact");
}
