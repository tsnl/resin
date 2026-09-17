mod support;

#[test]
fn integer_arithmetic_matches_host_semantics_and_compiles_to_valid_shaders() {
    for ty in ["u8", "i32", "u32", "i64", "u64"] {
        let mut source = support::integer_arithmetic::source(ty)
            .replace("export { kernel };", "export { kernel, main };");
        source.push_str("fn main() { let mut output = Output { value = 0, progress = 0 }; ");
        for (left, right, operation, expected) in support::integer_arithmetic::cases(ty) {
            if let Some(expected) = expected {
                source.push_str(&format!(
                    "evaluate(Input {{ left = {left}, right = {right}, operation = {operation} }}, output); assert(output.value == {expected} && output.progress == 3);"
                ));
            }
        }
        source.push('}');
        let module = support::module(&source);
        let project = support::project::Project::new(&module, Some("main")).unwrap();
        for shader in project.generated.shaders() {
            support::shaders::validate(shader.unoptimized_spirv());
        }
        let output = project.run();
        assert!(
            output.status.success(),
            "{ty}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
