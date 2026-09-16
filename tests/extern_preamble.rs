use std::fs;
use tempfile::TempDir;

mod support;

#[test]
fn empty_header_groups_remain_native_dependencies() {
    let directory = TempDir::new().unwrap();
    let header = directory.path().join("empty.h");
    fs::write(
        &header,
        "/* A declared dependency without foreign functions. */\n",
    )
    .unwrap();
    let source = format!(
        "export {{ main }}; extern {{ {:?}: {{}} }}; def main() -> int = {{ 42 }};",
        header.to_string_lossy().replace('\\', "/")
    );
    let module = support::module(&source);
    let project = support::project::Project::new(&module, Some("main")).unwrap();
    let mut environment = resin_toolchain::Environment::capture().unwrap();
    environment.directory = directory.path().into();
    let toolchain = environment.toolchain(None, None);

    drop(project.build(&toolchain).expect("declared header exists"));
    fs::remove_file(header).unwrap();
    assert!(
        project.build(&toolchain).is_err(),
        "removing an empty group's header must invalidate the native build"
    );
}

#[test]
fn foreign_signatures_use_imported_types_and_keep_module_visibility() {
    let directory = TempDir::new().unwrap();
    let header = directory.path().join("api.h");
    fs::write(&header, "int answer(int value);\n").unwrap();
    fs::write(
        directory.path().join("types.resin"),
        "export { CInt }; type CInt = int;",
    )
    .unwrap();
    fs::write(
        directory.path().join("api.resin"),
        format!(
            r#"export {{ call }};
            extern {{ {:?}: {{ def answer(value: CInt) -> CInt; }}, }};
            import {{ "types.resin" }};
            def call() -> int = {{ answer(35) }};
            "#,
            header.to_string_lossy().replace('\\', "/")
        ),
    )
    .unwrap();
    let entry = directory.path().join("main.resin");
    fs::write(
        &entry,
        "export { main }; import { \"api.resin\" }; def main() -> int = { call() };",
    )
    .unwrap();
    let module = support::pipeline::file_module(&entry).unwrap();
    let output = support::project::Project::new(&module, Some("main"))
        .unwrap()
        .with_native("int answer(int value);\nint answer(int value) { return value + 7; }\n")
        .run();
    assert_eq!(output.status.code(), Some(42));

    fs::write(
        &entry,
        "export { main }; import { \"api.resin\" }; def main() -> int = { answer(35) };",
    )
    .unwrap();
    let error = support::pipeline::file_module(&entry).unwrap_err();
    assert!(error.to_string().contains("UnboundValue"), "{error}");
}
