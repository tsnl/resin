use resin_source::prelude::*;
use resin_types::prelude::*;
use support::pipeline;
use support::toolchain;

use support::shaders::{self, instructions};
mod support;

#[test]
fn compute_workgroup_size_specializes_execution_mode_and_wide_index_arithmetic() {
    let m = module(
        "export { kernel }; @compute_shader fn kernel(index: ulong, output: Ptr<ulong>)  { output.* = index; }",
    );
    let project = support::project::Project::new(&m, None).unwrap();
    let source = std::fs::read(project.generated.shaders()[0].unoptimized_spirv()).unwrap();
    // OpExecutionModeId LocalSizeId must use the runtime ABI's specialization ID.
    let dimensions = instructions(&source, 331)
        .find(|args| args[1] == 38)
        .unwrap();
    let width = dimensions[2];
    assert!(instructions(&source, 71).any(|args| args
        == [
            width,
            1,
            resin_runtime::RESIN_COMPUTE_WORKGROUP_SIZE_SPEC_ID
        ]));
    assert!(instructions(&source, 50).any(|args| args[1..] == [width, 1]));
    for id in &dimensions[3..] {
        assert!(instructions(&source, 43).any(|args| args[1..] == [*id, 1]));
    }
    assert!(!instructions(&source, 16).any(|args| args[1] == 17));
    let wide = instructions(&source, 21)
        .find(|args| args[1..] == [64, 0])
        .unwrap()[0];
    let widened_width = instructions(&source, 113)
        .find(|args| args[0] == wide && args[2] == width)
        .unwrap()[1];
    assert!(instructions(&source, 132).any(|args| args[0] == wide && args[3] == widened_width));
    if let Some(frontend) = shaders::optimizer() {
        let built = project.build(&toolchain::spirv(&frontend)).unwrap();
        let optimized =
            std::fs::read(built.path(project.generated.shaders()[0].spirv().file_name().unwrap()))
                .unwrap();
        assert!(
            instructions(&optimized, 71)
                .any(|args| args[1..] == [1, resin_runtime::RESIN_COMPUTE_WORKGROUP_SIZE_SPEC_ID])
        );
    }
}

fn example(name: &str) -> resin_lir::Module {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join(name);
    pipeline::file_module(&path).unwrap()
}

#[test]
fn shader_indexing_emits_no_bounds_checks() {
    for indexing in [
        "values(i)",
        "values:at(i)",
        "device_index(view.data, view.length, i).*",
    ] {
        let m = module(&format!(
            "export {{ kernel }}; import {{ \"$/span.resin\" }}; @compute_shader fn kernel(i: ulong, output: Ptr<uint>)  {{ let mut values = [1_ui, 2_ui]; let mut view = Span<uint> {{ data = output, length = 2_ul }}; output.* = {indexing}; }}"
        ));
        let project = support::project::Project::new(&m, None).unwrap();
        let source = std::fs::read(project.generated.shaders()[0].unoptimized_spirv()).unwrap();
        // Unchecked indexing and its infallible source wrappers need no guards,
        // even before optimization.
        for comparison in 172..=179 {
            assert_eq!(instructions(&source, comparison).count(), 0);
        }
        assert_eq!(instructions(&source, 250).count(), 0);
        assert_eq!(failure_loads(&source), 0);
        if let Some(frontend) = shaders::optimizer() {
            let built = project.build(&toolchain::spirv(&frontend)).unwrap();
            let artifact = &project.generated.shaders()[0];
            let optimized =
                std::fs::read(built.path(artifact.spirv().file_name().unwrap())).unwrap();
            assert_eq!(
                instructions(&optimized, 250).count(),
                0,
                "unchecked indexing needs no branches after optimization"
            );
        }
    }
}

fn failure_loads(bytes: &[u8]) -> usize {
    // The invocation failure flag is the module's sole Private variable.
    let private = instructions(bytes, 59)
        .filter(|operands| operands[2] == 6)
        .collect::<Vec<_>>();
    assert_eq!(private.len(), 1);
    let flag = private[0][1];
    instructions(bytes, 61)
        .filter(|operands| operands[2] == flag)
        .count()
}

#[test]
fn shader_calls_guard_only_helpers_with_emitted_failure_exits() {
    for (helpers, expression, checks, loads) in [
        (
            "fn outer(x: ubyte) -> ulong  { inner(x) } fn inner(x: ubyte) -> ulong  { ulong(x) + 1_ul }",
            "outer(7_ub)",
            0,
            0,
        ),
        (
            "fn pure(x: ubyte) -> ulong  { ulong(x) + 1_ul } fn outer(x: ulong) -> uint  { inner(x) } fn inner(x: ulong) -> uint  { uint(x) }",
            "pure(7_ub) + ulong(outer(i))",
            3,
            2,
        ),
        (
            "fn outer() -> ulong  { inner() } fn inner() -> ulong  { let mut value: None; value = None; let mut result: ulong; result = value!; result }",
            "outer()",
            3,
            2,
        ),
    ] {
        let source = format!(
            "export {{ kernel }}; {helpers} @compute_shader fn kernel(i: ulong, output: Ptr<ulong>)  {{ output.* = {expression}; }}"
        );
        let module = module(&source);
        let project = support::project::Project::new(&module, None).unwrap();
        let shader = &project.generated.shaders()[0];
        let bytes = std::fs::read(shader.unoptimized_spirv()).unwrap();
        assert_eq!(instructions(&bytes, 250).count(), checks, "{source}");
        assert_eq!(failure_loads(&bytes), loads, "{source}");
        shaders::validate(shader.unoptimized_spirv());
    }
}

#[test]
fn graphics_output_guards_follow_the_emitted_entry_fallibility() {
    let types = "struct Position { x: float32, y: float32, z: float32, w: float32, } struct Color { r: float32, g: float32, b: float32, a: float32, } struct Vertex { position: Position, color: Color, }";
    for checked in [false, true] {
        for (entry, expression, body) in [
            (
                "@vertex_shader fn vertex(index: int) -> Vertex",
                "ubyte(index)",
                "Vertex { position = Position { x = 0.0_f, y = 0.0_f, z = 0.0_f, w = 1.0_f }, color = Color { r = 1.0_f, g = 0.0_f, b = 0.0_f, a = 1.0_f } }",
            ),
            (
                "@fragment_shader fn fragment(color: Color) -> Color",
                "ubyte(color.r)",
                "color",
            ),
        ] {
            let name = if entry.starts_with("@vertex") {
                "vertex"
            } else {
                "fragment"
            };
            let check = if checked {
                format!("{expression};")
            } else {
                String::new()
            };
            let source = format!("export {{ {name} }}; {types} {entry} {{ {check} {body} }}");
            let module = module(&source);
            let project = support::project::Project::new(&module, None).unwrap();
            let shader = &project.generated.shaders()[0];
            let bytes = std::fs::read(shader.unoptimized_spirv()).unwrap();
            assert_eq!(instructions(&bytes, 250).count(), usize::from(checked) * 2);
            assert_eq!(failure_loads(&bytes), usize::from(checked));
            shaders::validate(shader.unoptimized_spirv());
        }
    }
}

#[test]
fn shader_helpers_can_propagate_and_handle_results() {
    let m = module(
        "export { kernel }; struct Bad { index: uint, } fn checked(i: uint) -> (uint | Err<Bad>)  { if (i == uint(0)) { Err(Bad { index = i }) } else { (i) } } fn helper(i: uint) -> (uint | Err<_>)  { let mut value = checked(i)?; (value + uint(1)) } @compute_shader fn kernel(invocation: ulong, output: Ptr<uint>)  { let mut i = uint(invocation); output.* = { match (helper(i)) { uint(value) => { value }, Err(error) => { error.index } } }; }",
    );
    let project = support::project::Project::new(&m, None).unwrap();
    if let Some(frontend) = shaders::optimizer() {
        project
            .build(&toolchain::spirv(&frontend))
            .unwrap_or_else(|error| panic!("{error}"));
    }
}

#[test]
fn inferred_shader_results_lower_without_backend_inference() {
    let m = module(
        "export { kernel }; @compute_shader fn kernel(invocation: ulong, output: Ptr<uint>) -> _  { let mut i = uint(invocation); output.* = { let mut value: _; value = i + 1; value }; }",
    );
    assert_eq!(m.functions[0].result, Ty::Unit);
    let project = support::project::Project::new(&m, None).unwrap();
    if let Some(frontend) = shaders::optimizer() {
        project.build(&toolchain::spirv(&frontend)).unwrap();
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
    let Some(frontend) = shaders::optimizer() else {
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
                "export { kernel }; @compute_shader fn kernel(invocation: ulong, output: Ptr<uint>)  { let mut i = uint(invocation); output.* = { let mut x = i; x = x + uint(2); if (x < uint(4)) { x } else { x * uint(2) } }; }",
            ),
            Stage::Compute,
        ),
        (
            module(
                "export { kernel }; type Pixel = ulong; @compute_shader fn kernel(i: Pixel, output: Ptr<uint>)  { output.* = { uint(i + Pixel(1_ul)) }; }",
            ),
            Stage::Compute,
        ),
        (
            module(
                "export { kernel }; @compute_shader fn kernel(invocation: ulong, output: Ptr<uint>)  { let mut i = uint(invocation); output.* = { twice(i) }; } fn twice (i: uint) -> uint  { add(i, i) } fn add (a: uint, b: uint) -> uint  { a + b }",
            ),
            Stage::Compute,
        ),
        (
            module(
                "export { kernel }; @compute_shader fn kernel(invocation: ulong, output: Ptr<uint>)  { let mut i = uint(invocation); output.* = { let mut x = i; let mut n = uint(0); while (n < uint(3)) { let mut j = uint(0); while (j < n) { x = x + j; j = j + uint(1); }; n = n + uint(1); }; x }; }",
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
            .build(&toolchain::spirv(&frontend))
            .unwrap_or_else(|error| panic!("{error}"));
        let bytes = std::fs::read(built.path(shader.spirv().file_name().unwrap())).unwrap();
        assert_eq!(&bytes[..4], &[3, 2, 35, 7]);
        assert_eq!(bytes.len() % 4, 0);
    }
}

#[test]
fn device_pointers_and_shared_roots_compile() {
    let Some(frontend) = shaders::optimizer() else {
        return;
    };
    for (source, stage) in [
        (
            "export { kernel }; struct Node { value: uint, next: Ptr<Node>, } fn select (a: Ptr<Node>, b: Ptr<Node>, i: uint) -> Ptr<Node>  { if (i == uint(0)) { a } else { b } } @compute_shader fn kernel (invocation: ulong, root: Ptr<Node>) -> ()  { let mut i = uint(invocation); let mut p = select(root, root.next, i); p.value = uint(7); }",
            Stage::Compute,
        ),
        (
            "export { kernel }; import { \"$/span.resin\" }; struct Data { wide: ulong, values: Ptr<uint>, } @compute_shader fn kernel (invocation: ulong, root: Ptr<Data>) -> ()  { let mut i = uint(invocation); let mut p = root.values; let mut q: Ref<uint> = device_index(p, 64_ul, ulong(i)).*; q = uint(3); root.wide = ulong(4294967297); }",
            Stage::Compute,
        ),
        (
            "export { fragment }; struct Color { r: float32, g: float32, b: float32, a: float32, } struct Params { scale: float32, } @fragment_shader fn fragment (color: Color, root: Ptr<Params>) -> Color  { Color { r = color.r * root.scale, g = color.g, b = color.b, a = color.a } }",
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
            .build(&toolchain::spirv(&frontend))
            .unwrap_or_else(|error| panic!("{error}"));
    }
}

#[test]
fn shader_addresses_cannot_hide_unsupported_layouts_or_escape_locals() {
    for (source, expected) in [
        (
            "export { kernel }; fn read(p: Ptr<uint>) -> uint  { p.* } @compute_shader fn kernel(invocation: ulong, output: Ptr<uint>)  { let mut i = uint(invocation); let mut local = i; output.* = read(&local); }",
            "shader-local addresses cannot escape",
        ),
        (
            "export { kernel }; struct Data { flag: bool, } @compute_shader fn kernel (invocation: ulong, root: Ptr<Data>) -> ()  { let mut i = uint(invocation); () }",
            "no shared host/device layout",
        ),
        (
            "export { kernel }; @compute_shader fn kernel (invocation: ulong, root: Ptr<()>) -> ()  { let mut i = uint(invocation); () }",
            "no shared host/device layout",
        ),
        (
            "export { kernel }; @compute_shader fn kernel(invocation: ulong, output: Ptr<uint>)  { let mut i = uint(invocation); output.* = { let mut x = i; let mut p = &x; p.* }; }",
            "shader-local addresses cannot escape",
        ),
        (
            "export { kernel }; @compute_shader fn kernel(invocation: ulong, output: Ptr<uint>)  { let mut i = uint(invocation); output.* = { let mut x = i; ulong(&x); i }; }",
            "shader pointer casts are unsupported",
        ),
        (
            "export { kernel }; fn helper (i: uint) -> Ptr<uint>  { let mut x = i; &x } @compute_shader fn kernel(invocation: ulong, output: Ptr<uint>)  { let mut i = uint(invocation); output.* = { helper(i).* }; }",
            "cannot return a local address",
        ),
    ] {
        let error = pipeline::shader_error(source);
        assert!(error.to_string().contains(expected), "{source}\n{error}");
    }
}

#[test]
fn unsupported_shader_features_are_diagnosed() {
    for (source, expected) in [
        (
            "export { kernel }; @compute_shader fn kernel(invocation: ulong, output: Ptr<uint>)  { let mut i = uint(invocation); output.* = { helper(i) }; } fn helper (i: uint) -> uint  { let mut output = 0_ui; kernel(ulong(i), &output); output }",
            "recursive shader call graph",
        ),
        (
            "export { kernel }; @compute_shader fn kernel(invocation: ulong, output: Ptr<uint>)  { let mut i = uint(invocation); output.* = { i / uint(2) }; }",
            "unsupported shader builtin",
        ),
        (
            "export { kernel }; extern { \"stdlib.h\": { fn abs (i: int) -> int; } }; @compute_shader fn kernel(invocation: ulong, output: Ptr<uint>)  { let mut i = uint(invocation); output.* = { abs(1); i }; }",
            "foreign",
        ),
        (
            "export { kernel }; intrinsic \"format_bytes\" fn render<A>(data: Ptr<ubyte>, length: ulong, args: A) -> StrongOwner; struct Root { data: Ptr<ubyte>, length: ulong, } @compute_shader fn kernel(invocation: ulong, root: Ptr<Root>)  { let mut text = render(root.data, root.length, ()); }",
            "shader cannot consume managed values",
        ),
        (
            "export { kernel }; fn helper (i: uint) -> uint  { i } @compute_shader fn kernel(invocation: ulong, output: Ptr<uint>)  { let mut i = uint(invocation); output.* = { let mut f = helper; f(i) }; }",
            "does not support type",
        ),
    ] {
        let error = pipeline::shader_error(source);
        assert!(error.to_string().contains(expected), "{source}\n{error}");
    }
}

#[test]
fn entry_interfaces_are_checked_before_codegen() {
    for (source, expected) in [
        (
            "@compute_shader fn kernel(i: int) -> int  { i }",
            "expected (ulong, Ptr<T>)",
        ),
        (
            "@compute_shader fn kernel(i: long) -> long  { i }",
            "expected (ulong, Ptr<T>)",
        ),
        (
            "@vertex_shader fn vertex(i: int) -> int  { i }",
            "position/color",
        ),
        (
            "@fragment_shader fn fragment(i: uint) -> uint  { i }",
            "float32 r/g/b/a fields",
        ),
    ] {
        let error = pipeline::generate(&support::parse(source)).unwrap_err();
        assert!(error.to_string().contains(expected), "{error}");
    }
}

#[test]
fn optimizer_errors_are_reported() {
    let Some(frontend) = shaders::optimizer() else {
        return;
    };
    let error = toolchain::optimize_spirv(b"not SPIR-V", &frontend).unwrap_err();
    assert!(error.to_string().contains("failed"));
}

#[test]
fn compound_control_flow_compiles_to_spirv() {
    let Some(frontend) = shaders::optimizer() else {
        return;
    };
    let m = module(include_str!("fixtures/compound_control.resin"));
    let project = support::project::Project::new(&m, None).unwrap();
    project
        .build(&toolchain::spirv(&frontend))
        .unwrap_or_else(|error| panic!("{error}"));
}

#[test]
fn imported_backend_errors_retain_expression_origins() {
    let helper = Source::new(
        "helper.resin",
        "export { helper }; fn helper(n: uint) -> Ptr<uint>  { let mut value = n; &value }",
    );
    let entry = Source::new(
        "main.resin",
        "export { kernel }; import { \"helper.resin\" }; @compute_shader fn kernel(invocation: ulong, p: Ptr<uint>)  { p.* = helper(uint(invocation)).*; }",
    );
    let mut loader = resin_source::Loader::new(Default::default());
    loader
        .set_import(&entry, "helper.resin", helper.clone())
        .unwrap();
    let output = support::frontend::analyze(entry, &mut loader, None);
    let m = support::pipeline::verified_lir(&output, "kernel", resin_lir::Profile::Shader)
        .unwrap()
        .into_module();
    let origins: Vec<_> = m
        .origins
        .instructions
        .values()
        .filter(|o| {
            o.source == helper && o.source.text().get(o.span.start..o.span.end) == Some("&value")
        })
        .collect();
    assert!(
        !origins.is_empty(),
        "the operation retains its original expression"
    );
    let error = support::project::Project::new(&m, None)
        .unwrap_err()
        .to_string();
    assert!(error.contains(&format!("{}:1:", helper.name())), "{error}");
    assert!(
        error.contains("helper.resin:1:") && error.contains("function helper"),
        "{error}"
    );
    assert!(error.contains("cannot return a local address"), "{error}");
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
    let prefix = "export { kernel }; import { \"$/shared.resin\" }; struct Host { value: float64, } struct Root { owner: ArcPtr<Host>, weak: WeakPtr<Host>, result: uint, }";
    let m = module(&format!(
        "{prefix} @compute_shader fn kernel(invocation: ulong, root: Ptr<Root>)  {{ let mut i = uint(invocation); root.result = i; let mut address = &root.owner; }}"
    ));
    let project = support::project::Project::new(&m, None).unwrap();
    if let Some(frontend) = shaders::optimizer() {
        project.build(&toolchain::spirv(&frontend)).unwrap();
    }
    for body in [
        "let value = root.owner:clone();",
        "root.owner = root.owner:clone();",
        "let mut result = root.weak:upgrade();",
    ] {
        let error = pipeline::shader_error(&format!(
            "{prefix} @compute_shader fn kernel(invocation: ulong, root: Ptr<Root>)  {{ let mut i = uint(invocation); {body} }}"
        ));
        assert!(
            error.contains("shader cannot consume managed values"),
            "{error}"
        );
    }
}

#[test]
fn options_of_plain_values_work_in_shaders() {
    let m = module(
        "export { kernel }; @compute_shader fn kernel(invocation: ulong, output: Ptr<uint>)  { let mut i = uint(invocation); let mut value: uint | None; value = if (i == 0_ui) { 42_ui } else { None }; output.* = match (value) { uint(n) => { n }, None => { 0_ui } }; }",
    );
    let project = support::project::Project::new(&m, None).unwrap();
    if let Some(frontend) = shaders::optimizer() {
        project.build(&toolchain::spirv(&frontend)).unwrap();
    }
}

#[test]
fn literal_strings_report_the_missing_shader_storage_support() {
    let error = pipeline::shader_error(
        r#"export { kernel }; @compute_shader fn kernel(invocation: ulong, output: Ptr<uint>)  { let mut i = uint(invocation); let mut text = "abc"; output.* = uint(text.length); }"#,
    );
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
        "export { kernel }; import { \"$/span.resin\" }; @compute_shader fn kernel(index: ulong, output: Ptr<Span<ulong>>)  { if (index < output.length) { output.*:at(index) = index; }; }",
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
    if let Some(frontend) = shaders::optimizer() {
        project.build(&toolchain::spirv(&frontend)).unwrap();
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
    let Some(frontend) = shaders::optimizer() else {
        return;
    };
    let built = project.build(&toolchain::spirv(&frontend)).unwrap();
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
    if let Some(frontend) = shaders::optimizer() {
        project
            .build(&toolchain::spirv(&frontend))
            .unwrap_or_else(|error| panic!("{error}"));
    }
}

#[test]
fn sequential_conditionals_and_error_propagation_preserve_structured_control() {
    let mut source = String::from(
        "export { kernel }; struct Failed {} fn step() -> (() | Err<Failed>)  { (()) } fn helper(value: uint) -> (uint | Err<Failed>) { let mut result = value; ",
    );
    for _ in 0..512 {
        source
            .push_str("if (result == 0_ui) { result = 1_ui; } else { result = 0_ui; }; step()?; ");
    }
    source.push_str("(result) } @compute_shader fn kernel(i: ulong, output: Ptr<uint>)  { output.* = match (helper(uint(i))) { uint(value) => { value }, Err(error) => { 99_ui } }; }");
    let project = support::project::Project::new(&module(&source), None).unwrap();
    let source = std::fs::read(project.generated.shaders()[0].unoptimized_spirv()).unwrap();
    assert_eq!(
        instructions(&source, 246).count(),
        0,
        "sequential branches need no loops"
    );
    if let Some(frontend) = shaders::optimizer() {
        project.build(&toolchain::spirv(&frontend)).unwrap();
    }
}

fn module(source: &str) -> resin_lir::Module {
    support::module(&format!(
        r#"{source} intrinsic "pointer_index" fn device_index<T>(data: Ptr<T>, length: ulong, index: ulong) -> Ptr<T>;"#
    ))
}
