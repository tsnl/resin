mod support;

#[test]
fn math_functions_preserve_float_width_and_use_radians() {
    let module = support::module(
        r#"export { main }; import { "$/math.resin" };
        fn main() -> int  {
            let mut root: float32 = sqrt(4.0);
            let mut sine = sin(1.5707963267948966_d);
            let mut cosine = cos(0.0_f);
            let mut invalid = sqrt(-1.0_d);
            if (root == 2.0_f && sine > 0.999999_d && cosine == 1.0_f && invalid != invalid) { 0 } else { 1 }
        }"#,
    );
    let output = support::project::Project::new(&module, Some("main"))
        .unwrap()
        .run();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn math_rejects_non_float_arguments_during_specialization() {
    for name in ["sqrt", "sin", "cos"] {
        let source = format!(
            r#"export {{ main }}; import {{ "$/math.resin" }}; fn main()  {{ {name}(4); }}"#
        );
        let error = support::pipeline::source_module(&source)
            .unwrap_err()
            .to_string();
        assert!(error.contains("UnsupportedBuiltin"), "{error}");
        assert!(!error.contains("invalid IR"), "{error}");
    }
}

#[test]
fn shader_math_emits_glsl_operations() {
    let module = support::module(
        r#"export { kernel }; import { "$/math.resin" };
        @compute_shader fn kernel(index: ulong, value: Ptr<float32>)  {
            value.* = sqrt(value.*) + sin(value.*) + cos(value.*);
        }"#,
    );
    let project = support::project::Project::new(&module, None).unwrap();
    let path = project.generated.shaders()[0].unoptimized_spirv();
    support::shaders::validate(path);
    let bytes = std::fs::read(path).unwrap();
    for operation in [13, 14, 31] {
        // GLSL.std.450 Sin, Cos, Sqrt
        assert!(
            support::shaders::instructions(&bytes, 12).any(|operands| operands[3] == operation)
        );
    }
}
