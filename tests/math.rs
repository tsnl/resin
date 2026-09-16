mod support;

#[test]
fn complex_arithmetic_borrows_operands_and_preserves_float_width() {
    let module = support::module(
        r#"export { main }; import { "$/math.resin" };
        fn main() {
            let a = complex(3.0_f, 4.0_f);
            let b = complex(1.0_f, -2.0_f);
            let sum = a:add(b);
            assert(sum.real == 4.0_f && sum.imag == 2.0_f);
            let product = a:mul(b);
            assert(product.real == 11.0_f && product.imag == -2.0_f);
            let square = a:squared();
            assert(square.real == -7.0_f && square.imag == 24.0_f);
            assert(a:magnitude_squared() == 25.0_f);
            let i = complex(0.0_d, 1.0_d);
            let minus_one: Complex<float64> = i:squared();
            assert(minus_one.real == -1.0_d && minus_one.imag == 0.0_d);
            assert(i:magnitude_squared() == 1.0_d);
        }"#,
    );
    let output = support::project::Project::new(&module, Some("main"))
        .unwrap()
        .run();
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
}

#[test]
fn complex_arithmetic_accepts_device_storage_in_shaders() {
    let module = support::module(
        r#"export { kernel }; import { "$/math.resin" };
        @compute_shader fn kernel(index: ulong, value: Ptr<Complex<float32>>) {
            let norm = value.*:magnitude_squared();
            let square = value.*:squared();
            value.real = square.real + norm;
            value.imag = square.imag;
        }"#,
    );
    let project = support::project::Project::new(&module, None).unwrap();
    support::shaders::validate(project.generated.shaders()[0].unoptimized_spirv());
}

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
