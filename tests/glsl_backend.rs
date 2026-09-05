use resin::{
    backend::glsl::{self, Stage},
    ir, toolchain,
};

#[path = "support/shaders.rs"]
mod shaders;
mod support;
use support::module;

fn example(name: &str) -> ir::Module {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join(name);
    ir::generate(&resin::ast::load(&path).unwrap()).unwrap()
}

#[test]
fn all_example_stages_emit_deterministically() {
    for (name, stage) in [
        ("gradient.resin", Stage::Compute),
        ("triangle.resin", Stage::Vertex),
        ("triangle.resin", Stage::Fragment),
    ] {
        let m = example(name);
        let first = glsl::emit(&m, stage.entry(), stage).unwrap();
        assert!(first.starts_with("#version 460\n"));
        assert!(first.contains("void main()"));
        assert_eq!(first, glsl::emit(&m, stage.entry(), stage).unwrap());
    }
}

#[test]
fn examples_helpers_and_control_flow_compile_to_spirv() {
    let Some(compiler) = shaders::compiler() else {
        return;
    };
    let modules = [
        (example("gradient.resin"), Stage::Compute),
        (example("triangle.resin"), Stage::Vertex),
        (example("triangle.resin"), Stage::Fragment),
        (
            module(
                "kernel (i: uint) -> uint = { x = i; x := x + uint (2); if (x < uint (4)) { x } else { x * uint (2) } };",
            ),
            Stage::Compute,
        ),
        (
            module("Pixel = uint; kernel (i: Pixel) -> Pixel = { i + Pixel (uint (1)) };"),
            Stage::Compute,
        ),
        (
            module(
                "kernel (i: uint) -> uint = { twice(i) }; twice (i: uint) -> uint = { add(i, i) }; add (a: uint, b: uint) -> uint = { a + b };",
            ),
            Stage::Compute,
        ),
    ];
    for (m, stage) in modules {
        let glsl = glsl::emit(&m, stage.entry(), stage).unwrap();
        let bytes = toolchain::compile_glsl(&glsl, stage, &compiler)
            .unwrap_or_else(|error| panic!("{error}\n{glsl}"));
        assert_eq!(&bytes[..4], &[3, 2, 35, 7]);
        assert_eq!(bytes.len() % 4, 0);
    }
}

#[test]
fn unsupported_shader_features_are_diagnosed() {
    for (source, expected) in [
        (
            "bias = uint (1); kernel (i: uint) -> uint = { i + bias };",
            "GlobalAddress",
        ),
        (
            "kernel (i: uint) -> uint = { helper(i) }; helper (i: uint) -> uint = { kernel(i) };",
            "recursive shader call graph",
        ),
        (
            "kernel (i: uint) -> uint = { i / uint (2) };",
            "unsupported shader builtin",
        ),
        (
            "kernel (i: uint) -> uint = { x = i; x := if (i == uint (0)) { uint (1) } else { uint (2) }; x };",
            "addresses or functions across block edges",
        ),
        ("kernel (i: long) -> long = { i };", "does not support type"),
        (
            "extern \"stdlib.h\" abs (i: int) -> int; kernel (i: uint) -> uint = { abs(1); i };",
            "foreign",
        ),
        (
            "helper (i: uint) -> uint = { print(\"hello\", ()); i }; kernel (i: uint) -> uint = { helper(i) };",
            "host programs",
        ),
        (
            "helper (i: uint) -> uint = { i }; kernel (i: uint) -> uint = { f = helper; f(i) };",
            "does not support type",
        ),
    ] {
        let error = glsl::emit(&module(source), "kernel", Stage::Compute).unwrap_err();
        assert!(error.to_string().contains(expected), "{source}\n{error}");
    }
}

#[test]
fn entry_interfaces_are_checked() {
    for (m, entry, stage, expected) in [
        (
            example("gradient.resin"),
            "absent",
            Stage::Compute,
            "expected one",
        ),
        (
            module("kernel (i: int) -> int = { i };"),
            "kernel",
            Stage::Compute,
            "map uint to uint",
        ),
        (
            module("vertex (i: int) -> int = { i };"),
            "vertex",
            Stage::Vertex,
            "position/color",
        ),
        (
            module("fragment (i: uint) -> uint = { i };"),
            "fragment",
            Stage::Fragment,
            "float32 fields",
        ),
    ] {
        let error = glsl::emit(&m, entry, stage).unwrap_err();
        assert!(error.to_string().contains(expected), "{error}");
    }
}

#[test]
fn compiler_errors_are_reported() {
    let Some(compiler) = shaders::compiler() else {
        return;
    };
    let error = toolchain::compile_glsl("not GLSL", Stage::Compute, &compiler).unwrap_err();
    assert!(error.to_string().contains("shader compiler failed"));
}
