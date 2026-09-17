#[allow(dead_code)]
mod support;

use std::process::Command;

#[test]
fn local_documentation_renders_exports_overloads_fields_and_markdown_without_a_service() {
    let directory = tempfile::TempDir::new().unwrap();
    let path = directory.path().join("library.resin");
    std::fs::write(
        &path,
        r#"//! Module **introduction**.
export { Item, read, marker };
/// Public item.
struct Item {
    /// Value in units.
    value: int,
}
/// Read an item.
fn read(item: Ref<Item>) -> int { item.value }
/// Read an integer.
fn read(value: int) -> int { value }
/// Fenced text stays in the signature.
const marker = "```";
/// Private documentation.
fn hidden() {}
"#,
    )
    .unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_resin"))
        .env_remove("RESIN_SERVER")
        .args(["--doc"])
        .arg(&path)
        .output()
        .unwrap();
    assert!(output.status.success(), "{output:?}");
    let markdown = String::from_utf8(output.stdout).unwrap();
    for text in [
        "# library.resin",
        "Module **introduction**.",
        "## `Item`",
        "### `value`",
        "Value in units.",
        "Read an item.",
        "Read an integer.",
        "````resin\nconst marker",
    ] {
        assert!(markdown.contains(text), "missing {text}: {markdown}");
    }
    assert_eq!(markdown.matches("## `read`").count(), 2);
    assert!(!markdown.contains("hidden") && !markdown.contains("Private documentation"));
    assert!(
        !markdown.contains("{ item.value }"),
        "function bodies are not API signatures"
    );
}

#[test]
fn documentation_errors_preserve_existing_output_and_compilation_rejects_orphans() {
    let directory = tempfile::TempDir::new().unwrap();
    let path = directory.path().join("bad.resin");
    let output_path = directory.path().join("api.md");
    std::fs::write(&output_path, "previous documentation").unwrap();
    for source in ["/// Orphan\n", "fn broken(\n"] {
        std::fs::write(&path, source).unwrap();
        let output = Command::new(env!("CARGO_BIN_EXE_resin"))
            .env_remove("RESIN_SERVER")
            .arg("--doc")
            .arg(&path)
            .arg("-o")
            .arg(&output_path)
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert_eq!(
            std::fs::read_to_string(&output_path).unwrap(),
            "previous documentation"
        );
    }
    let error = support::pipeline::source_module("fn main() {}\n/// Orphan\n").unwrap_err();
    assert!(
        error
            .to_string()
            .contains("documentation comment must precede"),
        "{error}"
    );
}
