mod support;

#[test]
fn assertions_evaluate_once_and_remain_enabled_in_optimized_programs() {
    let module = support::module(
        "export { main }; def main() = { var n = 0; assert((n := n + 1) == 1); assert(n == 1); };",
    );
    let output = support::project::Project::new(&module, Some("main"))
        .unwrap()
        .run();
    assert!(output.status.success());
    let module = support::module("export { main }; def main() = { assert(false); };");
    let output = support::project::Project::new(&module, Some("main"))
        .unwrap()
        .run();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("assertion failed"));
}

#[test]
fn assertions_require_boolean_conditions_even_in_unused_functions() {
    let error = support::pipeline::source_module("def unused() = { assert(1); };")
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("TypeMismatch") && error.contains("expected: Bool"),
        "{error}"
    );
}

#[test]
fn shader_assertions_use_the_invocation_failure_path() {
    let module = support::module(
        "export { kernel }; @compute_shader def kernel(i: ulong, output: Ptr<uint>) = { assert(i == 0_ul); output.* := 42_ui; };",
    );
    let project = support::project::Project::new(&module, None).unwrap();
    let path = project.generated.shaders()[0].unoptimized_spirv();
    support::shaders::validate(path);
    let bytes = std::fs::read(path).unwrap();
    assert!(support::shaders::instructions(&bytes, 250).next().is_some()); // conditional failure exit
}
