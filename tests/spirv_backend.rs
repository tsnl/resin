use resin_source::prelude::*;
use resin_types::prelude::*;
#[path = "support/toolchain.rs"]
mod toolchain;
use support::pipeline;

use support::shaders::{self, instructions};
mod support;
use support::module;

fn example(name: &str) -> resin_lir::Module {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join(name);
    pipeline::generate_program(&pipeline::load(&path).unwrap()).unwrap()
}

#[test]
fn shader_indexing_emits_no_bounds_checks() {
    for indexing in ["values(i)", "values.at(i)", "view(i)", "view.at(i)"] {
        let m = module(&format!(
            "export {{ kernel }}; @compute_shader def kernel(i: ulong, output: Ptr<uint>) = {{ var values = [1_ui, 2_ui]; var view = Span<uint> {{ data = output, length = 2_ul }}; output.* := {indexing}.*; }};"
        ));
        let project = support::project::Project::new(&m, None).unwrap();
        let source = std::fs::read(project.generated.shaders()[0].unoptimized_spirv()).unwrap();
        assert_eq!(
            instructions(&source, 250).count(),
            0,
            "indexing must not branch"
        );
        if let Some(compiler) = shaders::optimizer() {
            project.build(&toolchain::spirv(&compiler)).unwrap();
        }
    }
}

#[test]
fn shader_helpers_can_propagate_and_handle_results() {
    let m = module(
        "export { kernel }; struct Bad { index: uint }; def checked(i: uint) -> Result<uint, Bad> = { if (i == uint(0)) { err(Bad { index = i }) } else { ok(i) } }; def helper(i: uint) -> Result<uint, _> = { var value = checked(i)?; ok(value + uint(1)) }; @compute_shader def kernel(invocation: ulong, output: Ptr<uint>) = { var i = uint(invocation); output.* := { match (helper(i)) { ok(value) => { value }, err(error) => { error.index } } }; };",
    );
    let project = support::project::Project::new(&m, None).unwrap();
    if let Some(compiler) = shaders::optimizer() {
        project
            .build(&toolchain::spirv(&compiler))
            .unwrap_or_else(|error| panic!("{error}"));
    }
}

#[test]
fn inferred_shader_results_lower_without_backend_inference() {
    let m = module(
        "export { kernel }; @compute_shader def kernel(invocation: ulong, output: Ptr<uint>) -> _ = { var i = uint(invocation); output.* := { var value: _; value := i + 1; value }; };",
    );
    assert_eq!(m.functions[0].result, Ty::Unit);
    let project = support::project::Project::new(&m, None).unwrap();
    if let Some(compiler) = shaders::optimizer() {
        project.build(&toolchain::spirv(&compiler)).unwrap();
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
        let first_project = support::project::Project::new(&m, None).unwrap();
        let first_shader = first_project
            .generated
            .shaders()
            .iter()
            .find(|shader| shader.stage() == stage)
            .unwrap();
        let first = std::fs::read(first_shader.unoptimized_spirv()).unwrap();
        assert_eq!(&first[..4], &[3, 2, 35, 7]);
        assert_eq!(instructions(&first, 15).count(), 1, "one OpEntryPoint");
        let second_project = support::project::Project::new(&m, None).unwrap();
        let second_shader = second_project
            .generated
            .shaders()
            .iter()
            .find(|shader| shader.stage() == stage)
            .unwrap();
        assert_eq!(
            first,
            std::fs::read(second_shader.unoptimized_spirv()).unwrap()
        );
    }
}

#[test]
fn examples_helpers_and_control_flow_compile_to_spirv() {
    let Some(compiler) = shaders::optimizer() else {
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
                "export { kernel }; @compute_shader def kernel(invocation: ulong, output: Ptr<uint>) = { var i = uint(invocation); output.* := { var x = i; x := x + uint (2); if (x < uint (4)) { x } else { x * uint (2) } }; };",
            ),
            Stage::Compute,
        ),
        (
            module(
                "export { kernel }; type Pixel = ulong; @compute_shader def kernel(i: Pixel, output: Ptr<uint>) = { output.* := { uint(i + Pixel (1_ul)) }; };",
            ),
            Stage::Compute,
        ),
        (
            module(
                "export { kernel }; @compute_shader def kernel(invocation: ulong, output: Ptr<uint>) = { var i = uint(invocation); output.* := { twice(i) }; }; def twice (i: uint) -> uint = { add(i, i) }; def add (a: uint, b: uint) -> uint = { a + b };",
            ),
            Stage::Compute,
        ),
        (
            module(
                "export { kernel }; @compute_shader def kernel(invocation: ulong, output: Ptr<uint>) = { var i = uint(invocation); output.* := { var x = i; var n = uint (0); while (n < uint (3)) { var j = uint (0); while (j < n) { x := x + j; j := j + uint (1); }; n := n + uint (1); }; x }; };",
            ),
            Stage::Compute,
        ),
    ];
    for (m, stage) in modules {
        let project = support::project::Project::new(&m, None).unwrap();
        let shader = project
            .generated
            .shaders()
            .iter()
            .find(|shader| shader.stage() == stage)
            .unwrap();
        let built = project
            .build(&toolchain::spirv(&compiler))
            .unwrap_or_else(|error| panic!("{error}"));
        let bytes = std::fs::read(built.path(shader.spirv().file_name().unwrap())).unwrap();
        assert_eq!(&bytes[..4], &[3, 2, 35, 7]);
        assert_eq!(bytes.len() % 4, 0);
    }
}

#[test]
fn device_pointers_and_shared_roots_compile() {
    let Some(compiler) = shaders::optimizer() else {
        return;
    };
    for (source, stage) in [
        (
            "export { kernel }; struct Node { value: uint, next: Ptr<Node> }; def select (a: Ptr<Node>, b: Ptr<Node>, i: uint) -> Ptr<Node> = { if (i == uint (0)) { a } else { b } }; @compute_shader def kernel (invocation: ulong, root: Ptr<Node>) -> () = { var i = uint(invocation); var p = select(root, root.next, i); p.value := uint (7); };",
            Stage::Compute,
        ),
        (
            "export { kernel }; struct Data { wide: ulong, values: Ptr<uint> }; @compute_shader def kernel (invocation: ulong, root: Ptr<Data>) -> () = { var i = uint(invocation); var p = Ptr<uint> (ulong (root.values)); var q = (Span<uint> { data = p, length = ulong(64) })(i); q.* := uint (3); root.wide := ulong (4294967297); };",
            Stage::Compute,
        ),
        (
            "export { fragment }; struct Color { r: float32, g: float32, b: float32, a: float32 }; struct Params { scale: float32 }; @fragment_shader def fragment (color: Color, root: Ptr<Params>) -> Color = { Color { r = color.r * root.scale, g = color.g, b = color.b, a = color.a } };",
            Stage::Fragment,
        ),
    ] {
        let project = support::project::Project::new(&module(source), None).unwrap();
        assert!(
            project
                .generated
                .shaders()
                .iter()
                .any(|shader| shader.stage() == stage)
        );
        project
            .build(&toolchain::spirv(&compiler))
            .unwrap_or_else(|error| panic!("{error}"));
    }
}

#[test]
fn shader_addresses_cannot_hide_unsupported_layouts_or_escape_locals() {
    for (source, expected) in [
        (
            "export { kernel }; def read(p: Ptr<uint>) -> uint = { p.* }; @compute_shader def kernel(invocation: ulong, output: Ptr<uint>) = { var i = uint(invocation); var local = i; output.* := read(&local); };",
            "shader-local addresses cannot escape",
        ),
        (
            "export { kernel }; struct Data { flag: bool }; @compute_shader def kernel (invocation: ulong, root: Ptr<Data>) -> () = { var i = uint(invocation); () };",
            "no shared host/device layout",
        ),
        (
            "export { kernel }; @compute_shader def kernel (invocation: ulong, root: Ptr<()>) -> () = { var i = uint(invocation); () };",
            "no shared host/device layout",
        ),
        (
            "export { kernel }; @compute_shader def kernel(invocation: ulong, output: Ptr<uint>) = { var i = uint(invocation); output.* := { var x = i; var p = &x; p.* }; };",
            "shader-local addresses cannot escape",
        ),
        (
            "export { kernel }; @compute_shader def kernel(invocation: ulong, output: Ptr<uint>) = { var i = uint(invocation); output.* := { var x = i; ulong (&x); i }; };",
            "shader-local addresses cannot escape",
        ),
        (
            "export { kernel }; def helper (i: uint) -> Ptr<uint> = { var x = i; &x }; @compute_shader def kernel(invocation: ulong, output: Ptr<uint>) = { var i = uint(invocation); output.* := { helper(i).* }; };",
            "cannot return a local address",
        ),
    ] {
        let error = support::project::Project::new(&module(source), None).unwrap_err();
        assert!(error.to_string().contains(expected), "{source}\n{error}");
    }
}

#[test]
fn unsupported_shader_features_are_diagnosed() {
    for (source, expected) in [
        (
            "export { kernel }; @compute_shader def kernel(invocation: ulong, output: Ptr<uint>) = { var i = uint(invocation); output.* := { helper(i) }; }; def helper (i: uint) -> uint = { var output = 0_ui; kernel(ulong(i), &output); output };",
            "recursive shader call graph",
        ),
        (
            "export { kernel }; @compute_shader def kernel(invocation: ulong, output: Ptr<uint>) = { var i = uint(invocation); output.* := { i / uint (2) }; };",
            "unsupported shader builtin",
        ),
        (
            "export { kernel }; extern \"stdlib.h\" def abs (i: int) -> int; @compute_shader def kernel(invocation: ulong, output: Ptr<uint>) = { var i = uint(invocation); output.* := { abs(1); i }; };",
            "foreign",
        ),
        (
            "export { kernel }; def helper (i: uint) -> uint = { print(fmt(\"hello\", ())); i }; @compute_shader def kernel(invocation: ulong, output: Ptr<uint>) = { var i = uint(invocation); output.* := { helper(i) }; };",
            "host programs",
        ),
        (
            "export { kernel }; def helper (i: uint) -> uint = { i }; @compute_shader def kernel(invocation: ulong, output: Ptr<uint>) = { var i = uint(invocation); output.* := { var f = helper; f(i) }; };",
            "does not support type",
        ),
    ] {
        let error = support::project::Project::new(&module(source), None).unwrap_err();
        assert!(error.to_string().contains(expected), "{source}\n{error}");
    }
}

#[test]
fn entry_interfaces_are_checked_before_codegen() {
    for (source, expected) in [
        (
            "@compute_shader def kernel(i: int) -> int = { i };",
            "expected (ulong, Ptr<T>)",
        ),
        (
            "@compute_shader def kernel(i: long) -> long = { i };",
            "expected (ulong, Ptr<T>)",
        ),
        (
            "@vertex_shader def vertex(i: int) -> int = { i };",
            "position/color",
        ),
        (
            "@fragment_shader def fragment(i: uint) -> uint = { i };",
            "float32 r/g/b/a fields",
        ),
    ] {
        let error = pipeline::generate(&support::parse(source)).unwrap_err();
        assert!(error.to_string().contains(expected), "{error}");
    }
}

#[test]
fn optimizer_errors_are_reported() {
    let Some(compiler) = shaders::optimizer() else {
        return;
    };
    let error = toolchain::optimize_spirv(b"not SPIR-V", &compiler).unwrap_err();
    assert!(error.to_string().contains("failed"));
}

#[test]
fn compound_control_flow_compiles_to_spirv() {
    let Some(compiler) = shaders::optimizer() else {
        return;
    };
    let m = module(include_str!("fixtures/compound_control.resin"));
    let project = support::project::Project::new(&m, None).unwrap();
    project
        .build(&toolchain::spirv(&compiler))
        .unwrap_or_else(|error| panic!("{error}"));
}

#[test]
fn imported_backend_errors_retain_expression_origins() {
    let helper = Source::new(
        "helper.resin",
        "export { helper }; struct E {}; def helper(n: uint) -> Result<uint, E> = { n / 2_ui; var r: Result<(), E>; r := if (n == 0_ui) { err(E {}) } else { ok(()) }; r?; ok(n) };",
    );
    let entry = Source::new(
        "main.resin",
        "export { kernel }; import { \"helper.resin\" }; @compute_shader def kernel(invocation: ulong, p: Ptr<uint>) = { var i = uint(invocation); match (helper(i)) { ok(n) => { p.* := n; }, err(e) => {} }; };",
    );
    let mut loader = resin_source::Loader::new(Default::default());
    loader
        .set_import(&entry, "helper.resin", helper.clone())
        .unwrap();
    let compilation = resin_compiler::Compiler::new().compile(entry, &mut loader);
    let m = compilation.module().unwrap();
    let origins: Vec<_> = m
        .origins
        .instructions
        .values()
        .filter(|o| {
            o.source == helper && o.source.text().get(o.span.start..o.span.end) == Some("n / 2_ui")
        })
        .collect();
    assert!(
        !origins.is_empty(),
        "the operation retains its original expression"
    );
    let error = support::project::Project::new(m, None)
        .unwrap_err()
        .to_string();
    assert!(error.contains(&format!("{}:1:", helper.name())), "{error}");
    assert!(
        error.contains("n / 2_ui") && error.contains("function helper"),
        "{error}"
    );
    assert!(error.contains("unsupported shader builtin"), "{error}");
    let mut without = m.clone();
    without.origins = Default::default();
    let error = support::project::Project::new(&without, None)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("block") && error.contains("instruction"),
        "{error}"
    );
}

#[test]
fn managed_fields_are_opaque_until_consumed_by_a_shader() {
    let prefix = "export { kernel }; struct Host { value: float64 }; struct Root { owner: Arc<Host>, weak: Weak<Host>, result: uint };";
    let m = module(&format!(
        "{prefix} @compute_shader def kernel(invocation: ulong, root: Ptr<Root>) = {{ var i = uint(invocation); root.result := i; var address = &root.owner; }};"
    ));
    let project = support::project::Project::new(&m, None).unwrap();
    if let Some(compiler) = shaders::optimizer() {
        project.build(&toolchain::spirv(&compiler)).unwrap();
    }
    for body in [
        "var value = root.owner;",
        "root.owner := root.owner;",
        "var result = root.weak.upgrade();",
    ] {
        let m = module(&format!(
            "{prefix} @compute_shader def kernel(invocation: ulong, root: Ptr<Root>) = {{ var i = uint(invocation); {body} }};"
        ));
        let error = support::project::Project::new(&m, None)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("shader cannot consume a managed value"),
            "{error}"
        );
    }
}

#[test]
fn options_of_plain_values_work_in_shaders() {
    let m = module(
        "export { kernel }; @compute_shader def kernel(invocation: ulong, output: Ptr<uint>) = { var i = uint(invocation); var value: uint | None; value := if (i == 0_ui) { 42_ui } else { None }; output.* := match (value) { uint(n) => { n }, None => { 0_ui } }; };",
    );
    let project = support::project::Project::new(&m, None).unwrap();
    if let Some(compiler) = shaders::optimizer() {
        project.build(&toolchain::spirv(&compiler)).unwrap();
    }
}

#[test]
fn literal_strings_report_the_missing_shader_storage_support() {
    let m = module(
        r#"export { kernel }; @compute_shader def kernel(invocation: ulong, output: Ptr<uint>) = { var i = uint(invocation); var text = "abc"; output.* := uint(text.length); };"#,
    );
    let error = support::project::Project::new(&m, None).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("shader string literals need device-backed storage"),
        "{error}"
    );
}

#[test]
fn compute_index_uses_wide_arithmetic_and_indexes_spans_directly() {
    let m = module(
        "export { kernel }; @compute_shader def kernel(index: ulong, output: Ptr<Span<ulong>>) = { if (index < output.length) { output.at(index).* := index; }; };",
    );
    let project = support::project::Project::new(&m, None).unwrap();
    let source = std::fs::read(project.generated.shaders()[0].unoptimized_spirv()).unwrap();
    // OpDecorate BuiltIn: group and local IDs preserve 64-bit index arithmetic.
    let builtins = instructions(&source, 71)
        .filter(|args| args.get(1) == Some(&11))
        .map(|args| args[2])
        .collect::<Vec<_>>();
    assert!(builtins.contains(&26) && builtins.contains(&27));
    assert!(
        !builtins.contains(&28),
        "GlobalInvocationId would overflow at 32 bits"
    );
    let wide = instructions(&source, 21)
        .find(|args| args[1..] == [64, 0])
        .unwrap()[0];
    assert!(
        instructions(&source, 132).any(|args| args[0] == wide),
        "wide OpIMul"
    );
    if let Some(compiler) = shaders::optimizer() {
        project.build(&toolchain::spirv(&compiler)).unwrap();
    }
}

#[test]
fn branch_only_shaders_do_not_acquire_dispatch_loops() {
    let m = example("triangle.resin");
    let project = support::project::Project::new(&m, None).unwrap();
    for shader in project.generated.shaders() {
        let source = std::fs::read(shader.unoptimized_spirv()).unwrap();
        assert_eq!(instructions(&source, 246).count(), 0, "no OpLoopMerge");
    }
    let Some(compiler) = shaders::optimizer() else {
        return;
    };
    let built = project.build(&toolchain::spirv(&compiler)).unwrap();
    for shader in project.generated.shaders() {
        let bytes = std::fs::read(built.path(shader.spirv().file_name().unwrap())).unwrap();
        let words: Vec<_> = bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        let mut instructions = &words[5..];
        while let Some(&first) = instructions.first() {
            let length = (first >> 16) as usize;
            assert!(length > 0);
            // OpLoopMerge declares a structured SPIR-V loop.
            assert_ne!(first & 0xffff, 246, "a branch-only shader gained a loop");
            instructions = &instructions[length..];
        }
    }
}

#[test]
fn structured_loop_conditions_and_early_returns_compile_to_spirv() {
    let m = module(include_str!("fixtures/structured_control.resin"));
    let project = support::project::Project::new(&m, None).unwrap();
    let source = std::fs::read(project.generated.shaders()[0].unoptimized_spirv()).unwrap();
    assert_eq!(
        instructions(&source, 246).count(),
        2,
        "two OpLoopMerge loops"
    );
    if let Some(compiler) = shaders::optimizer() {
        project
            .build(&toolchain::spirv(&compiler))
            .unwrap_or_else(|error| panic!("{error}"));
    }
}

#[test]
fn sequential_conditionals_and_error_propagation_preserve_structured_control() {
    let mut source = String::from(
        "export { kernel }; struct Failed {}; def step() -> Result<(), Failed> = { ok(()) }; def helper(value: uint) -> Result<uint, Failed> = { var result = value; ",
    );
    for _ in 0..512 {
        source.push_str(
            "if (result == 0_ui) { result := 1_ui; } else { result := 0_ui; }; step()?; ",
        );
    }
    source.push_str("ok(result) }; @compute_shader def kernel(i: ulong, output: Ptr<uint>) = { output.* := match (helper(uint(i))) { ok(value) => { value }, err(error) => { 99_ui } }; };");
    let project = support::project::Project::new(&module(&source), None).unwrap();
    let source = std::fs::read(project.generated.shaders()[0].unoptimized_spirv()).unwrap();
    assert_eq!(
        instructions(&source, 246).count(),
        0,
        "sequential branches need no loops"
    );
    if let Some(compiler) = shaders::optimizer() {
        project.build(&toolchain::spirv(&compiler)).unwrap();
    }
}
