#[path = "support/toolchain.rs"]
mod config;
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
fn defer_lowers_to_shader_control_flow_including_error_exits() {
    let m = module(
        "export { kernel }; struct E {}; struct Root { trace: uint }; def fail() -> Result<uint, E> = { err(E {}) }; def work(i: uint, root: Ptr<Root>) -> Result<uint, _> = { defer { root.trace := root.trace + uint(1); }; var n = fail()?; ok(n + i) }; def kernel(i: uint, root: Ptr<Root>) = { match (work(i, root)) { ok(n) => { root.trace := n; }, err(e) => {} }; };",
    );
    let source = glsl::emit(&m, "kernel", Stage::Compute).unwrap();
    if let Some(compiler) = shaders::compiler() {
        toolchain::compile_glsl(&source, Stage::Compute, &config::glsl(&compiler))
            .unwrap_or_else(|error| panic!("{error}\n{source}"));
    }
}

#[test]
fn shader_indexing_emits_no_bounds_checks() {
    for indexing in ["values(i)", "values.at(i)", "view(i)", "view.at(i)"] {
        let m = module(&format!(
            "export {{ kernel }}; def kernel(i: uint, output: Ptr<uint>) = {{ var values = [1I, 2I]; var view = Span<uint> {{ data = output, length = 2L }}; output.* := {indexing}.*; }};"
        ));
        let source = glsl::emit(&m, "kernel", Stage::Compute).unwrap();
        assert!(!source.contains("r_failed = true"), "{source}");
        if let Some(compiler) = shaders::compiler() {
            toolchain::compile_glsl(&source, Stage::Compute, &config::glsl(&compiler)).unwrap();
        }
    }
}

#[test]
fn shader_helpers_can_propagate_and_handle_results() {
    let m = module(
        "export { kernel }; struct Bad { index: uint }; def checked(i: uint) -> Result<uint, Bad> = { if (i == uint(0)) { err(Bad { index = i }) } else { ok(i) } }; def helper(i: uint) -> Result<uint, _> = { var value = checked(i)?; ok(value + uint(1)) }; def kernel(i: uint, output: Ptr<uint>) = { output.* := { match (helper(i)) { ok(value) => { value }, err(error) => { error.index } } }; };",
    );
    let source = glsl::emit(&m, "kernel", Stage::Compute).unwrap();
    if let Some(compiler) = shaders::compiler() {
        toolchain::compile_glsl(&source, Stage::Compute, &config::glsl(&compiler))
            .unwrap_or_else(|error| panic!("{error}\n{source}"));
    }
}

#[test]
fn inferred_shader_results_lower_without_backend_inference() {
    let m = module(
        "export { kernel }; def kernel(i: uint, output: Ptr<uint>) -> _ = { output.* := { var value: _; value := i + 1; value }; };",
    );
    assert_eq!(m.functions[0].result, ir::Ty::Unit);
    let source = glsl::emit(&m, "kernel", Stage::Compute).unwrap();
    if let Some(compiler) = shaders::compiler() {
        toolchain::compile_glsl(&source, Stage::Compute, &config::glsl(&compiler)).unwrap();
    }
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
                "export { kernel }; def kernel(i: uint, output: Ptr<uint>) = { output.* := { var x = i; x := x + uint (2); if (x < uint (4)) { x } else { x * uint (2) } }; };",
            ),
            Stage::Compute,
        ),
        (
            module(
                "export { kernel }; type Pixel = uint; def kernel(i: Pixel, output: Ptr<uint>) = { output.* := { i + Pixel (uint (1)) }; };",
            ),
            Stage::Compute,
        ),
        (
            module(
                "export { kernel }; def kernel(i: uint, output: Ptr<uint>) = { output.* := { twice(i) }; }; def twice (i: uint) -> uint = { add(i, i) }; def add (a: uint, b: uint) -> uint = { a + b };",
            ),
            Stage::Compute,
        ),
        (
            module(
                "export { kernel }; def kernel(i: uint, output: Ptr<uint>) = { output.* := { var x = i; var n = uint (0); while (n < uint (3)) { var j = uint (0); while (j < n) { x := x + j; j := j + uint (1); }; n := n + uint (1); }; x }; };",
            ),
            Stage::Compute,
        ),
    ];
    for (m, stage) in modules {
        let glsl = glsl::emit(&m, stage.entry(), stage).unwrap();
        let bytes = toolchain::compile_glsl(&glsl, stage, &config::glsl(&compiler))
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
            "export { kernel }; struct Node { value: uint, next: Ptr<Node> }; def select (a: Ptr<Node>, b: Ptr<Node>, i: uint) -> Ptr<Node> = { if (i == uint (0)) { a } else { b } }; def kernel (i: uint, root: Ptr<Node>) -> () = { var p = select(root, root.next, i); p.value := uint (7); };",
            Stage::Compute,
        ),
        (
            "export { kernel }; struct Data { wide: ulong, values: Ptr<uint> }; def kernel (i: uint, root: Ptr<Data>) -> () = { var p = Ptr<uint> (ulong (root.values)); var q = (Span<uint> { data = p, length = ulong(64) })(i); q.* := uint (3); root.wide := ulong (4294967297); };",
            Stage::Compute,
        ),
        (
            "export { fragment }; struct Color { r: float32, g: float32, b: float32, a: float32 }; struct Params { scale: float32 }; def fragment (color: Color, root: Ptr<Params>) -> Color = { Color { r = color.r * root.scale, g = color.g, b = color.b, a = color.a } };",
            Stage::Fragment,
        ),
    ] {
        let glsl = glsl::emit(&module(source), stage.entry(), stage).unwrap();
        toolchain::compile_glsl(&glsl, stage, &config::glsl(&compiler))
            .unwrap_or_else(|error| panic!("{error}\n{glsl}"));
    }
}

#[test]
fn shader_addresses_cannot_hide_unsupported_layouts_or_escape_locals() {
    for (source, expected) in [
        (
            "export { kernel }; def read(p: Ptr<uint>) -> uint = { p.* }; def kernel(i: uint, output: Ptr<uint>) = { var local = i; output.* := read(&local); };",
            "shader-local addresses cannot escape",
        ),
        (
            "export { kernel }; struct Data { flag: bool }; def kernel (i: uint, root: Ptr<Data>) -> () = { () };",
            "no shared host/device layout",
        ),
        (
            "export { kernel }; def kernel (i: uint, root: Ptr<()>) -> () = { () };",
            "no shared host/device layout",
        ),
        (
            "export { kernel }; def kernel(i: uint, output: Ptr<uint>) = { output.* := { var x = i; var p = &x; p.* }; };",
            "shader-local addresses cannot escape",
        ),
        (
            "export { kernel }; def kernel(i: uint, output: Ptr<uint>) = { output.* := { var x = i; ulong (&x); i }; };",
            "shader-local addresses cannot escape",
        ),
        (
            "export { kernel }; def helper (i: uint) -> Ptr<uint> = { var x = i; &x }; def kernel(i: uint, output: Ptr<uint>) = { output.* := { helper(i).* }; };",
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
            "export { kernel }; def kernel(i: uint, output: Ptr<uint>) = { output.* := { helper(i) }; }; def helper (i: uint) -> uint = { var output = 0I; kernel(i, &output); output };",
            "recursive shader call graph",
        ),
        (
            "export { kernel }; def kernel(i: uint, output: Ptr<uint>) = { output.* := { i / uint (2) }; };",
            "unsupported shader builtin",
        ),
        (
            "export { kernel }; def kernel (i: long) -> long = { i };",
            "does not support type",
        ),
        (
            "export { kernel }; extern \"stdlib.h\" def abs (i: int) -> int; def kernel(i: uint, output: Ptr<uint>) = { output.* := { abs(1); i }; };",
            "foreign",
        ),
        (
            "export { kernel }; def helper (i: uint) -> uint = { print(\"hello\", ()); i }; def kernel(i: uint, output: Ptr<uint>) = { output.* := { helper(i) }; };",
            "host programs",
        ),
        (
            "export { kernel }; def helper (i: uint) -> uint = { i }; def kernel(i: uint, output: Ptr<uint>) = { output.* := { var f = helper; f(i) }; };",
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
            module("export { kernel }; def kernel (i: int) -> int = { i };"),
            "kernel",
            Stage::Compute,
            "expected (uint, Ptr<T>)",
        ),
        (
            module("export { vertex }; def vertex (i: int) -> int = { i };"),
            "vertex",
            Stage::Vertex,
            "position/color",
        ),
        (
            module("export { fragment }; def fragment (i: uint) -> uint = { i };"),
            "fragment",
            Stage::Fragment,
            "float32 r/g/b/a fields",
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
    let error =
        toolchain::compile_glsl("not GLSL", Stage::Compute, &config::glsl(&compiler)).unwrap_err();
    assert!(error.to_string().contains("shader compiler failed"));
}

#[test]
fn compound_control_flow_compiles_to_spirv() {
    let Some(compiler) = shaders::compiler() else {
        return;
    };
    let m = module(include_str!("fixtures/compound_control.resin"));
    let source = glsl::emit(&m, "kernel", Stage::Compute).unwrap();
    toolchain::compile_glsl(&source, Stage::Compute, &config::glsl(&compiler))
        .unwrap_or_else(|error| panic!("{error}\n{source}"));
}

#[test]
fn imported_deferred_backend_errors_retain_expression_origins() {
    let temp = toolchain::TempDir::new(&std::env::temp_dir()).unwrap();
    let helper = temp.path().join("helper.resin");
    let entry = temp.path().join("main.resin");
    let mut session = resin::compiler::Session::default();
    session.set_overlay(&helper, "export { helper }; struct E {}; def helper(n: uint) -> Result<uint, E> = { defer n / 2I; var r: Result<(), E>; r := if (n == 0I) { err(E {}) } else { ok(()) }; r?; ok(n) };".into()).unwrap();
    session.set_overlay(&entry, "export { kernel }; import { \"helper.resin\" }; def kernel(i: uint, p: Ptr<uint>) = { match (helper(i)) { ok(n) => { p.* := n; }, err(e) => {} }; };".into()).unwrap();
    let snapshot = session.analyze(&entry).unwrap();
    let m = snapshot.module().unwrap();
    // Origins use canonical paths, including macOS temp aliases and Windows prefixes.
    let helper = resin::analysis::normalize_path(&helper).unwrap();
    let origins: Vec<_> = m
        .origins
        .instructions
        .values()
        .filter(|o| {
            o.path == helper
                && m.origins.sources[&o.path].get(o.span.start..o.span.end) == Some("n / 2I")
        })
        .collect();
    assert!(
        origins.len() >= 2,
        "both cleanup exits retain their original expression"
    );
    let error = glsl::emit(m, "kernel", Stage::Compute)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains(&format!("{}:1:", helper.display())),
        "{error}"
    );
    assert!(
        error.contains("n / 2I") && error.contains("function helper"),
        "{error}"
    );
    assert!(error.contains("unsupported shader builtin"), "{error}");
    let mut without = m.clone();
    without.origins = Default::default();
    let error = glsl::emit(&without, "kernel", Stage::Compute)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("block") && error.contains("instruction"),
        "{error}"
    );
}
