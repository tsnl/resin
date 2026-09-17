mod support;

#[test]
fn complex_value_arithmetic_composes_and_preserves_float_width() {
    let module = support::module(
        r#"export { main }; import { "$/math.resin" };
        fn main() {
            let a = complex(f32(3.0), f32(4.0));
            let b = complex(f32(1.0), f32(-2.0));
            let sum = a:add(b);
            assert(sum.real == f32(4.0) && sum.imag == f32(2.0));
            let product = a:mul(b);
            assert(product.real == f32(11.0) && product.imag == f32(-2.0));
            let square = a:squared();
            assert(square.real == f32(-7.0) && square.imag == f32(24.0));
            assert(a:magnitude_squared() == f32(25.0));
            assert(a.real == f32(3.0) && a.imag == f32(4.0));
            assert(b.real == f32(1.0) && b.imag == f32(-2.0));
            let chained = a:squared():add(b);
            assert(chained.real == f32(-6.0) && chained.imag == f32(22.0));
            let direct = complex(f64(0.0), f64(1.0)):squared();
            assert(direct.real == f64(-1.0));
            let i = complex(f64(0.0), f64(1.0));
            let minus_one: Complex<f64> = i:squared();
            assert(minus_one.real == f64(-1.0) && minus_one.imag == f64(0.0));
            assert(i:magnitude_squared() == f64(1.0));
            assert(i.real == f64(0.0) && i.imag == f64(1.0));
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
fn complex_arithmetic_accepts_device_storage_in_shaders() {
    let module = support::module(
        r#"export { kernel }; import { "$/math.resin" };
        @compute_shader fn kernel(index: u64, value: Ptr<Complex<f32>>) {
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
        fn main() -> i32  {
            let mut root: f32 = sqrt(4.0);
            let mut sine = sin(f64(1.5707963267948966));
            let mut cosine = cos(f32(0.0));
            let mut invalid = sqrt(f64(-1.0));
            if (root == f32(2.0) && sine > f64(0.999999) && cosine == f32(1.0) && invalid != invalid) { 0 } else { 1 }
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
        @compute_shader fn kernel(index: u64, value: Ptr<f32>)  {
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
