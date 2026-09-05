use resin::{
    backend::glsl::{self, Stage},
    toolchain,
};

#[path = "support/shaders.rs"]
mod shaders;
mod support;
use support::module;

const GRADIENT: &str = include_str!("../examples/shaders/gradient.resin");
const TRIANGLE: &str = include_str!("../examples/shaders/triangle.resin");

#[test]
fn all_example_stages_emit_deterministically() {
    for (source, stage) in [
        (GRADIENT, Stage::Compute),
        (TRIANGLE, Stage::Vertex),
        (TRIANGLE, Stage::Fragment),
    ] {
        let m = module(source);
        let first = glsl::emit(&m, stage.entry(), stage).unwrap();
        assert!(first.starts_with("#version 460\n"));
        assert!(first.contains("void main()"));
        assert_eq!(first, glsl::emit(&m, stage.entry(), stage).unwrap());
    }
}

#[test]
fn examples_and_control_flow_compile_to_spirv() {
    let Some(compiler) = shaders::compiler() else {
        return;
    };
    let sources = [
        (GRADIENT, Stage::Compute),
        (TRIANGLE, Stage::Vertex),
        (TRIANGLE, Stage::Fragment),
        (
            "kernel = (i: uint) => { x = i; x := x + uint (2); if (x < uint (4)) { x } else { x * uint (2) } };",
            Stage::Compute,
        ),
        (
            "Pixel = uint; kernel = (i: Pixel) => { i + Pixel (uint (1)) };",
            Stage::Compute,
        ),
    ];
    for (source, stage) in sources {
        let glsl = glsl::emit(&module(source), stage.entry(), stage).unwrap();
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
            "bias = uint (1); kernel = (i: uint) => { i + bias };",
            "GlobalAddress",
        ),
        (
            "make = (bias: uint) => { kernel = (i: uint) => { i + bias }; kernel };",
            "capture",
        ),
        (
            "kernel = (i: uint) => { i / uint (2) };",
            "unsupported shader builtin",
        ),
        (
            "kernel = (i: uint) => { x = i; x := if (i == uint (0)) { uint (1) } else { uint (2) }; x };",
            "addresses across block edges",
        ),
        ("kernel = (i: long) => i;", "does not support type"),
    ] {
        let error = glsl::emit(&module(source), "kernel", Stage::Compute).unwrap_err();
        assert!(error.to_string().contains(expected), "{source}\n{error}");
    }
}

#[test]
fn entry_interfaces_are_checked() {
    for (source, entry, stage, expected) in [
        (GRADIENT, "absent", Stage::Compute, "expected one"),
        (
            "kernel = (i: int) => i;",
            "kernel",
            Stage::Compute,
            "map uint to uint",
        ),
        (
            "vertex = (i: int) => i;",
            "vertex",
            Stage::Vertex,
            "position/color",
        ),
        (
            "fragment = (i: uint) => i;",
            "fragment",
            Stage::Fragment,
            "float32 fields",
        ),
    ] {
        let error = glsl::emit(&module(source), entry, stage).unwrap_err();
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
