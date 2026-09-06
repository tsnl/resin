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
    ir::generate_program(&resin::ast::load(&path).unwrap()).unwrap()
}

#[test]
fn all_example_stages_emit_deterministically() {
    for (name, stage) in [
        ("gradient.resin", Stage::Compute),
        ("triangle.resin", Stage::Vertex),
        ("triangle.resin", Stage::Fragment),
        ("particles.resin", Stage::Compute),
        ("particles.resin", Stage::Vertex),
        ("particles.resin", Stage::Fragment),
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
        (example("particles.resin"), Stage::Compute),
        (example("particles.resin"), Stage::Vertex),
        (example("particles.resin"), Stage::Fragment),
        (
            module(
                "export { kernel }; kernel (i: uint) -> uint = { x = i; x := x + uint (2); if (x < uint (4)) { x } else { x * uint (2) } };",
            ),
            Stage::Compute,
        ),
        (
            module(
                "export { kernel }; Pixel = uint; kernel (i: Pixel) -> Pixel = { i + Pixel (uint (1)) };",
            ),
            Stage::Compute,
        ),
        (
            module(
                "export { kernel }; kernel (i: uint) -> uint = { twice(i) }; twice (i: uint) -> uint = { add(i, i) }; add (a: uint, b: uint) -> uint = { a + b };",
            ),
            Stage::Compute,
        ),
        (
            module(
                "export { kernel }; kernel (i: uint) -> uint = { x = i; n = uint (0); while (n < uint (3)) { j = uint (0); while (j < n) { x := x + j; j := j + uint (1); }; n := n + uint (1); }; x };",
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
fn device_pointers_and_shared_roots_compile() {
    let Some(compiler) = shaders::compiler() else {
        return;
    };
    for (source, stage) in [
        (
            "export { kernel }; Node = { value: uint, next: Ptr<Node> }; select (a: Ptr<Node>, b: Ptr<Node>, i: uint) -> Ptr<Node> = { if (i == uint (0)) { a } else { b } }; kernel (i: uint, root: Ptr<Node>) -> () = { p = select(root, root.next, i); p.value := uint (7); };",
            Stage::Compute,
        ),
        (
            "export { kernel }; Data = { wide: ulong, values: Ptr<uint> }; kernel (i: uint, root: Ptr<Data>) -> () = { p = Ptr<uint> (ulong (root.values)); q = p + i; q.* := uint (3); root.wide := ulong (4294967297); };",
            Stage::Compute,
        ),
        (
            "export { fragment }; Color = { r: float32, g: float32, b: float32, a: float32 }; Params = { scale: float32 }; fragment (color: Color, root: Ptr<Params>) -> Color = { Color { r = color.r * root.scale, g = color.g, b = color.b, a = color.a } };",
            Stage::Fragment,
        ),
    ] {
        let glsl = glsl::emit(&module(source), stage.entry(), stage).unwrap();
        toolchain::compile_glsl(&glsl, stage, &compiler)
            .unwrap_or_else(|error| panic!("{error}\n{glsl}"));
    }
}

#[test]
fn shader_addresses_cannot_hide_unsupported_layouts_or_escape_locals() {
    for (source, expected) in [
        (
            "export { kernel }; Data = { flag: bool }; kernel (i: uint, root: Ptr<Data>) -> () = { () };",
            "no shared host/device layout",
        ),
        (
            "export { kernel }; kernel (i: uint, root: Ptr<()>) -> () = { () };",
            "no shared host/device layout",
        ),
        (
            "export { kernel }; kernel (i: uint) -> uint = { x = i; p = &x; p.* };",
            "shader-local addresses cannot escape",
        ),
        (
            "export { kernel }; kernel (i: uint) -> uint = { x = i; ulong (&x); i };",
            "shader-local addresses cannot escape",
        ),
        (
            "export { kernel }; helper (i: uint) -> Ptr<uint> = { x = i; &x }; kernel (i: uint) -> uint = { helper(i).* };",
            "cannot return a local address",
        ),
    ] {
        let error = glsl::emit(&module(source), "kernel", Stage::Compute).unwrap_err();
        assert!(error.to_string().contains(expected), "{source}\n{error}");
    }
}

#[test]
fn unsupported_shader_features_are_diagnosed() {
    for (source, expected) in [
        (
            "export { kernel }; kernel (i: uint) -> uint = { helper(i) }; helper (i: uint) -> uint = { kernel(i) };",
            "recursive shader call graph",
        ),
        (
            "export { kernel }; kernel (i: uint) -> uint = { i / uint (2) };",
            "unsupported shader builtin",
        ),
        (
            "export { kernel }; kernel (i: uint) -> uint = { x = i; x := if (i == uint (0)) { uint (1) } else { uint (2) }; x };",
            "addresses or functions across block edges",
        ),
        (
            "export { kernel }; kernel (i: long) -> long = { i };",
            "does not support type",
        ),
        (
            "export { kernel }; extern \"stdlib.h\" abs (i: int) -> int; kernel (i: uint) -> uint = { abs(1); i };",
            "foreign",
        ),
        (
            "export { kernel }; helper (i: uint) -> uint = { print(\"hello\", ()); i }; kernel (i: uint) -> uint = { helper(i) };",
            "host programs",
        ),
        (
            "export { kernel }; helper (i: uint) -> uint = { i }; kernel (i: uint) -> uint = { f = helper; f(i) };",
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
            "expected an exported shader function",
        ),
        (
            module("export { kernel }; kernel (i: int) -> int = { i };"),
            "kernel",
            Stage::Compute,
            "map uint to uint",
        ),
        (
            module("export { vertex }; vertex (i: int) -> int = { i };"),
            "vertex",
            Stage::Vertex,
            "position/color",
        ),
        (
            module("export { fragment }; fragment (i: uint) -> uint = { i };"),
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
