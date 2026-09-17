use resin_source::prelude::*;
use resin_types::prelude::*;
use support::pipeline;
use support::toolchain;

use support::shaders::{self, instructions};
mod support;

#[test]
fn compute_workgroup_size_specializes_execution_mode_and_wide_index_arithmetic() {
    let m = module(
        "export { kernel }; @compute_shader fn kernel(index: u64, output: Ptr<u64>)  { output.* = index; }",
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
            "export {{ kernel }}; import {{ \"$/span.resin\" }}; @compute_shader fn kernel(i: u64, output: Ptr<u32>)  {{ let mut values = [u32(1), u32(2)]; let mut view = Span<u32> {{ data = output, length = u64(2) }}; output.* = {indexing}; }}"
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
            "fn outer(x: u8) -> u64  { inner(x) } fn inner(x: u8) -> u64  { u64(x) + u64(1) }",
            "outer(u8(7))",
            0,
            0,
        ),
        (
            "fn pure(x: u8) -> u64  { u64(x) + u64(1) } fn outer(x: u64) -> u32  { inner(x) } fn inner(x: u64) -> u32  { u32(x) }",
            "pure(u8(7)) + u64(outer(i))",
            3,
            2,
        ),
        (
            "fn outer() -> u64  { inner() } fn inner() -> u64  { let mut value: None; value = None; let mut result: u64; result = value!; result }",
            "outer()",
            3,
            2,
        ),
    ] {
        let source = format!(
            "export {{ kernel }}; {helpers} @compute_shader fn kernel(i: u64, output: Ptr<u64>)  {{ output.* = {expression}; }}"
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
    let types = "struct Position { x: f32, y: f32, z: f32, w: f32, } struct Color { r: f32, g: f32, b: f32, a: f32, } struct Vertex { position: Position, color: Color, }";
    for checked in [false, true] {
        for (entry, expression, body) in [
            (
                "@vertex_shader fn vertex(index: i32) -> Vertex",
                "u8(index)",
                "Vertex { position = Position { x = f32(0.0), y = f32(0.0), z = f32(0.0), w = f32(1.0) }, color = Color { r = f32(1.0), g = f32(0.0), b = f32(0.0), a = f32(1.0) } }",
            ),
            (
                "@fragment_shader fn fragment(color: Color) -> Color",
                "u8(color.r)",
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
        "export { kernel }; struct Bad { index: u32, } fn checked(i: u32) -> (u32 | Err<Bad>)  { if (i == u32(0)) { Err(Bad { index = i }) } else { (i) } } fn helper(i: u32) -> (u32 | Err<_>)  { let mut value = checked(i)?; (value + u32(1)) } @compute_shader fn kernel(invocation: u64, output: Ptr<u32>)  { let mut i = u32(invocation); output.* = { match (helper(i)) { u32(value) => { value }, Err(error) => { error.index } } }; }",
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
        "export { kernel }; @compute_shader fn kernel(invocation: u64, output: Ptr<u32>) -> _  { let mut i = u32(invocation); output.* = { let mut value: _; value = i + 1; value }; }",
    );
    assert_eq!(m.functions[0].result, Ty::Unit);
    let project = support::project::Project::new(&m, None).unwrap();
    if let Some(frontend) = shaders::optimizer() {
        project.build(&toolchain::spirv(&frontend)).unwrap();
    }
}

#[test]
fn all_example_stages_emit_deterministically_and_compile_to_spirv() {
    let optimizer = shaders::optimizer();
    for (name, stages) in [
        ("gradient.resin", &[Stage::Compute][..]),
        ("triangle.resin", &[Stage::Vertex, Stage::Fragment][..]),
        (
            "particles.resin",
            &[Stage::Compute, Stage::Vertex, Stage::Fragment][..],
        ),
    ] {
        let m = example(name);
        // Generation and native builds already include every stage in the module.
        let first_project = support::project::Project::new(&m, None).unwrap();
        let second_project = support::project::Project::new(&m, None).unwrap();
        assert_eq!(first_project.generated.shaders().len(), stages.len());
        assert_eq!(second_project.generated.shaders().len(), stages.len());
        for &stage in stages {
            let first_shader = first_project
                .generated
                .shaders()
                .iter()
                .find(|shader| shader.stage() == stage)
                .unwrap();
            let second_shader = second_project
                .generated
                .shaders()
                .iter()
                .find(|shader| shader.stage() == stage)
                .unwrap();
            let first = std::fs::read(first_shader.unoptimized_spirv()).unwrap();
            assert_eq!(&first[..4], &[3, 2, 35, 7]);
            assert_eq!(instructions(&first, 15).count(), 1, "one OpEntryPoint");
            assert_eq!(
                first,
                std::fs::read(second_shader.unoptimized_spirv()).unwrap(),
                "{name}: {stage:?}"
            );
        }
        if let Some(optimizer) = &optimizer {
            let built = first_project.build(&toolchain::spirv(optimizer)).unwrap();
            for shader in first_project.generated.shaders() {
                let bytes = std::fs::read(built.path(shader.spirv().file_name().unwrap())).unwrap();
                assert_eq!(&bytes[..4], &[3, 2, 35, 7]);
                assert_eq!(bytes.len() % 4, 0);
            }
        }
    }
}

#[test]
fn helpers_and_control_flow_compile_to_spirv() {
    let Some(frontend) = shaders::optimizer() else {
        return;
    };
    let modules = [
        (
            module(
                "export { kernel }; @compute_shader fn kernel(invocation: u64, output: Ptr<u32>)  { let mut i = u32(invocation); output.* = { let mut x = i; x = x + u32(2); if (x < u32(4)) { x } else { x * u32(2) } }; }",
            ),
            Stage::Compute,
        ),
        (
            module(
                "export { kernel }; type Pixel = u64; @compute_shader fn kernel(i: Pixel, output: Ptr<u32>)  { output.* = { u32(i + Pixel(u64(1))) }; }",
            ),
            Stage::Compute,
        ),
        (
            module(
                "export { kernel }; @compute_shader fn kernel(invocation: u64, output: Ptr<u32>)  { let mut i = u32(invocation); output.* = { twice(i) }; } fn twice (i: u32) -> u32  { add(i, i) } fn add (a: u32, b: u32) -> u32  { a + b }",
            ),
            Stage::Compute,
        ),
        (
            module(
                "export { kernel }; @compute_shader fn kernel(invocation: u64, output: Ptr<u32>)  { let mut i = u32(invocation); output.* = { let mut x = i; let mut n: u32 = 0; while (n < u32(3)) { let mut j: u32 = 0; while (j < n) { x = x + j; j = j + u32(1); }; n = n + u32(1); }; x }; }",
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
            "export { kernel }; struct Node { value: u32, next: Ptr<Node>, } fn select (a: Ptr<Node>, b: Ptr<Node>, i: u32) -> Ptr<Node>  { if (i == u32(0)) { a } else { b } } @compute_shader fn kernel (invocation: u64, root: Ptr<Node>) -> ()  { let mut i = u32(invocation); let mut p = select(root, root.next, i); p.value = u32(7); }",
            Stage::Compute,
        ),
        (
            "export { kernel }; import { \"$/span.resin\" }; struct Data { wide: u64, values: Ptr<u32>, } @compute_shader fn kernel (invocation: u64, root: Ptr<Data>) -> ()  { let mut i = u32(invocation); let mut p = root.values; let mut q: RefMut<u32> = device_index(p, u64(64), u64(i)).*; q = u32(3); root.wide = u64(4294967297); }",
            Stage::Compute,
        ),
        (
            "export { fragment }; struct Color { r: f32, g: f32, b: f32, a: f32, } struct Params { scale: f32, } @fragment_shader fn fragment (color: Color, root: Ptr<Params>) -> Color  { Color { r = color.r * root.scale, g = color.g, b = color.b, a = color.a } }",
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
            "export { kernel }; fn read(p: Ptr<u32>) -> u32  { p.* } @compute_shader fn kernel(invocation: u64, output: Ptr<u32>)  { let mut i = u32(invocation); let mut local = i; output.* = read(&local); }",
            "cannot take the address of a local value",
        ),
        (
            "export { kernel }; struct Data { flag: bool, } @compute_shader fn kernel (invocation: u64, root: Ptr<Data>) -> ()  { let mut i = u32(invocation); () }",
            "no shared host/device layout",
        ),
        (
            "export { kernel }; @compute_shader fn kernel (invocation: u64, root: Ptr<()>) -> ()  { let mut i = u32(invocation); () }",
            "no shared host/device layout",
        ),
        (
            "export { kernel }; @compute_shader fn kernel(invocation: u64, output: Ptr<u32>)  { let mut i = u32(invocation); output.* = { let mut x = i; let mut p = &x; p.* }; }",
            "cannot take the address of a local value",
        ),
        (
            "export { kernel }; @compute_shader fn kernel(invocation: u64, output: Ptr<u32>)  { let mut i = u32(invocation); output.* = { let mut x = i; u64(&x); i }; }",
            "cannot take the address of a local value",
        ),
        (
            "export { kernel }; fn helper (i: u32) -> Ref<u32>  { let mut x = i; x } @compute_shader fn kernel(invocation: u64, output: Ptr<u32>)  { let mut i = u32(invocation); output.* = { helper(i) }; }",
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
            "export { kernel }; @compute_shader fn kernel(invocation: u64, output: Ptr<u32>)  { let mut i = u32(invocation); output.* = { helper(i, output) }; } fn helper (i: u32, output: Ptr<u32>) -> u32  { kernel(u64(i), output); output.* }",
            "recursive shader call graph",
        ),
        (
            "export { kernel }; extern { \"stdlib.h\": { fn abs (i: i32) -> i32; } }; @compute_shader fn kernel(invocation: u64, output: Ptr<u32>)  { let mut i = u32(invocation); output.* = { abs(1); i }; }",
            "foreign",
        ),
        (
            "export { kernel }; intrinsic \"format_bytes\" fn render<A>(data: Ptr<u8>, length: u64, args: A) -> StrongOwner; struct Root { data: Ptr<u8>, length: u64, } @compute_shader fn kernel(invocation: u64, root: Ptr<Root>)  { let mut text = render(root.data, root.length, ()); }",
            "shader cannot consume managed values",
        ),
        (
            "export { kernel }; fn helper (i: u32) -> u32  { i } @compute_shader fn kernel(invocation: u64, output: Ptr<u32>)  { let mut i = u32(invocation); output.* = { let mut f = helper; f(i) }; }",
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
            "@compute_shader fn kernel(i: i32) -> i32  { i }",
            "expected (u64, Ptr<T>)",
        ),
        (
            "@compute_shader fn kernel(i: i64) -> i64  { i }",
            "expected (u64, Ptr<T>)",
        ),
        (
            "@vertex_shader fn vertex(i: i32) -> i32  { i }",
            "position/color",
        ),
        (
            "@fragment_shader fn fragment(i: u32) -> u32  { i }",
            "f32 r/g/b/a fields",
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
        "export { helper }; fn helper(n: u32) -> Ref<u32>  { let mut value = n; value }",
    );
    let entry = Source::new(
        "main.resin",
        "export { kernel }; import { \"helper.resin\" }; @compute_shader fn kernel(invocation: u64, p: Ptr<u32>)  { p.* = helper(u32(invocation)); }",
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
            o.source == helper && o.source.text().get(o.span.start..o.span.end) == Some("value")
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
    let prefix = "export { kernel }; import { \"$/shared.resin\" }; struct Host { value: f64, } struct Root { owner: ArcPtr<Host>, weak: WeakPtr<Host>, result: u32, }";
    let m = module(&format!(
        "{prefix} @compute_shader fn kernel(invocation: u64, root: Ptr<Root>)  {{ let mut i = u32(invocation); root.result = i; let mut address = &root.owner; }}"
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
            "{prefix} @compute_shader fn kernel(invocation: u64, root: Ptr<Root>)  {{ let mut i = u32(invocation); {body} }}"
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
        "export { kernel }; @compute_shader fn kernel(invocation: u64, output: Ptr<u32>)  { let mut i = u32(invocation); let mut value: u32 | None; value = if (i == u32(0)) { u32(42) } else { None }; output.* = match (value) { u32(n) => { n }, None => { u32(0) } }; }",
    );
    let project = support::project::Project::new(&m, None).unwrap();
    if let Some(frontend) = shaders::optimizer() {
        project.build(&toolchain::spirv(&frontend)).unwrap();
    }
}

#[test]
fn literal_strings_report_the_missing_shader_storage_support() {
    let error = pipeline::shader_error(
        r#"export { kernel }; @compute_shader fn kernel(invocation: u64, output: Ptr<u32>)  { let mut i = u32(invocation); let mut text = "abc"; output.* = u32(text.length); }"#,
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
        "export { kernel }; import { \"$/span.resin\" }; @compute_shader fn kernel(index: u64, output: Ptr<Span<u64>>)  { if (index < output.length) { output.*:at_mut(index) = index; }; }",
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
    // Exercise repeated selection merges without making spirv-opt a stress test.
    let source = support::control::shader_source(32);
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
        r#"{source} intrinsic "pointer_index" fn device_index<T>(data: Ptr<T>, length: u64, index: u64) -> Ptr<T>;"#
    ))
}

#[test]
fn local_reference_specialization_bounds_nested_emission() {
    let mut source = String::from("export { kernel }; ");
    for index in 0..130 {
        source.push_str(&format!(
            "fn helper_{index}(value: RefMut<u32>) {{ helper_{}(value); }} ",
            index + 1
        ));
    }
    source.push_str("fn helper_130(value: RefMut<u32>) { value = u32(42); } ");
    source.push_str("@compute_shader fn kernel(index: u64, output: Ptr<u32>) { let mut value: u32 = 0; helper_0(value); output.* = value; }");
    let error = pipeline::shader_error(&source);
    assert!(
        error
            .to_string()
            .contains("local-reference specialization limit"),
        "{error}"
    );
}
