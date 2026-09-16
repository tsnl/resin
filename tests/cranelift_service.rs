//! Native compilation travels through the upload/download protocol.
#[allow(dead_code)]
mod support;

use std::{fs, path::Path, process::Command};
use support::service::Service;

fn service() -> Service {
    Service::configured(|_, environment| {
        // Direct object linking must not need a build graph runner.
        environment
            .variables
            .insert("NINJA".into(), "missing-ninja-for-native-test".into());
    })
}

fn build(service: &Service, source: &Path, destination: &Path) {
    let output = service
        .command()
        .arg(source)
        .arg("-o")
        .arg(destination)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn exit_code(path: &Path) -> Option<i32> {
    Command::new(path).output().unwrap().status.code()
}

#[test]
fn uploaded_edits_cache_objects_and_retain_downloaded_generations() {
    let service = service();
    let client = tempfile::tempdir().unwrap();
    let source = client.path().join("main.resin");
    let old_output = client
        .path()
        .join(format!("old{}", std::env::consts::EXE_SUFFIX));
    let new_output = client
        .path()
        .join(format!("new{}", std::env::consts::EXE_SUFFIX));
    fs::write(&source, "export { main }; def main() -> int = { 37 };").unwrap();
    build(&service, &source, &old_output);
    assert_eq!(exit_code(&old_output), Some(37));
    assert_eq!(service.server.counters().native_object_builds, 1);
    build(&service, &source, &new_output);
    assert_eq!(service.server.counters().native_object_builds, 1);

    // Different client roots with identical logical inputs share the same object.
    let another_client = tempfile::tempdir().unwrap();
    let another_source = another_client.path().join("main.resin");
    fs::copy(&source, &another_source).unwrap();
    build(&service, &another_source, &new_output);
    assert_eq!(service.server.counters().native_object_builds, 1);

    fs::write(&source, "export { main }; def main() -> int = { 42 };").unwrap();
    build(&service, &source, &new_output);
    assert_eq!(service.server.counters().native_object_builds, 2);
    assert_eq!(exit_code(&old_output), Some(37));
    assert_eq!(exit_code(&new_output), Some(42));
}

#[test]
fn run_and_build_use_distinct_optimization_keys() {
    let service = service();
    let client = tempfile::tempdir().unwrap();
    let source = client.path().join("main.resin");
    let destination = client
        .path()
        .join(format!("program{}", std::env::consts::EXE_SUFFIX));
    fs::write(&source, "export { main }; def main() -> int = { 42 };").unwrap();
    let output = service.command().arg(&source).output().unwrap();
    assert_eq!(
        output.status.code(),
        Some(42),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(service.server.counters().native_object_builds, 1);
    build(&service, &source, &destination);
    assert_eq!(service.server.counters().native_object_builds, 2);
    // Optimization belongs to code generation; frontend and verified LIR are reused.
    assert_eq!(service.server.counters().verified_builds, 1);
    build(&service, &source, &destination);
    assert_eq!(service.server.counters().native_object_builds, 2);
}

#[test]
fn invalid_program_preserves_existing_output() {
    let service = service();
    let client = tempfile::tempdir().unwrap();
    let source = client.path().join("main.resin");
    let destination = client.path().join("previous-output");
    fs::write(
        &source,
        "export { main }; def main() -> int = { missing_definition() };",
    )
    .unwrap();
    fs::write(&destination, b"previous output").unwrap();
    let output = service
        .command()
        .arg(&source)
        .arg("-o")
        .arg(&destination)
        .output()
        .unwrap();
    assert!(!output.status.success());
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(error.contains("missing_definition"), "{error}");
    assert_eq!(fs::read(&destination).unwrap(), b"previous output");
}
