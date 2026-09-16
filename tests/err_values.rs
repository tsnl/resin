mod support;
use support::{module, pipeline};

#[test]
fn error_wrappers_preserve_identity_payloads_and_ownership() {
    let module = module(
        r#"export { main }; import { "$/string.resin" };
        def boxed<T>(value: T) -> Err<T> = { Err(value) };
        def main() = {
            var a: Err<int> = Err(42);
            var b = Err(Err("nested"));
            var c: int | Err<String> = boxed(String.from_str("owned"));
            var copy = c;
            print(fmt("{0} {1} {2} {3}", (a, b, c, copy)));
        };"#,
    );
    let output = support::project::Project::new(&module, Some("main"))
        .unwrap()
        .run();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        output.stdout,
        b"Err(42) Err(Err(\"nested\")) Err(\"owned\") Err(\"owned\")"
    );
}

#[test]
fn error_wrappers_cannot_hide_reference_payloads() {
    let error =
        pipeline::source_module("struct Invalid { error: Err<Ref<int>> }; def main() = {};")
            .unwrap_err()
            .to_string();
    assert!(
        error.contains("Ref") || error.contains("reference"),
        "{error}"
    );
}

#[test]
fn error_wrappers_have_distinct_union_tags_and_payload_layouts() {
    let module = module(
        r#"export { main }; import { "$/string.resin" };
        def show(value: int | Err<int>) = { match (value) {
            int(value) => { print(repr(value)) },
            Err<int>(error) => { print(repr(error)) },
        } };
        def main() = { show(7); print(" "); show(Err(7)); };"#,
    );
    let output = support::project::Project::new(&module, Some("main"))
        .unwrap()
        .run();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.stdout, b"7 Err(7)");
}

#[test]
fn shaders_preserve_error_wrapper_identity() {
    let module = module(
        r#"export { kernel };
        def error(value: int) -> Err<int> = { Err(value) };
        @compute_shader def kernel(index: ulong, output: Ptr<int>) = {
            var value: int | Err<int> = error(7);
            match (value) {
                int(value) => { output.* := value; },
                Err<int>(value) => { output.* := int(value); },
            }
        };"#,
    );
    let project = support::project::Project::new(&module, None).unwrap();
    support::shaders::validate(project.generated.shaders()[0].unoptimized_spirv());
}
