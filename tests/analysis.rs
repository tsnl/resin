use resin_hir::GenerateErrorKind;
use resin_hir::Hir;
use resin_source::normalize_path;
use resin_source::prelude::*;
use std::{collections::BTreeMap, path::Path, sync::Arc};
use tempfile::TempDir;

#[test]
fn generic_method_calls_show_substituted_signatures_and_original_definitions() {
    let library = "export { Cell }; struct Cell<T> { value: T, def read(self: Ptr<Cell<T>>) -> T = { self.value }; def choose<U>(self: Ptr<Cell<T>>, value: U) -> U = { value }; };";
    let source = "import { \"library.resin\" }; type IntCell = Cell<int>; def use(cell: Ptr<IntCell>) -> ulong = { cell.read(); cell.choose::<ulong>(42) };";
    let project = Project::new(&[("main.resin", source), ("library.resin", library)]);
    let analysis = project.checked();
    let input = project.source("main.resin");
    for (name, expected) in [
        ("read", "read: () -> int"),
        ("choose", "choose: (ulong) -> ulong"),
    ] {
        let offset = source.find(&format!("cell.{name}")).unwrap() + 5;
        assert_eq!(analysis.hover(&input, offset).unwrap().text, expected);
        let items = analysis.completions(&input, offset);
        assert_eq!(
            items.iter().find(|item| item.name == name).unwrap().detail,
            expected
        );
        let origin = analysis.definition(&input, offset).unwrap();
        assert_eq!(origin.source, project.source("library.resin"));
        assert_eq!(
            origin.span.start,
            library.find(&format!("def {name}")).unwrap() + 4
        );
    }
}

#[test]
fn unfinished_generic_method_access_keeps_owner_substitution_and_method_binders() {
    let library = "export { Cell }; struct Cell<T> { value: T, def read(self: Ptr<Cell<T>>) -> T = { self.value }; def choose<U>(self: Ptr<Cell<T>>, value: U) -> U = { value }; def make(value: T) -> Cell<T> = { Cell<T> { value = value } }; };";
    for receiver in ["cell", "Cell<int>"] {
        let source = format!(
            "import {{ \"library.resin\" }}; def use(cell: Ptr<Cell<int>>) = {{ {receiver}.; }};"
        );
        let project = Project::new(&[("main.resin", &source), ("library.resin", library)]);
        let analysis = project.build_hir();
        assert!(!analysis.diagnostics().is_empty());
        let input = project.source("main.resin");
        let items = analysis.completions(&input, source.find(".;").unwrap() + 1);
        let choose = items.iter().find(|item| item.name == "choose").unwrap();
        assert!(choose.detail.contains("<U>"), "{choose:?}");
        assert!(!choose.detail.contains("Cell<T>"), "{choose:?}");
        let read = items.iter().find(|item| item.name == "read").unwrap();
        if receiver == "cell" {
            assert_eq!(read.detail, "read: () -> int");
            assert_eq!(choose.detail, "choose: <U> (U) -> U");
            assert!(!items.iter().any(|item| item.name == "make"), "{items:?}");
        } else {
            assert!(read.detail.contains("Ptr<Cell<int>>"), "{read:?}");
            assert!(items.iter().any(|item| item.name == "make"), "{items:?}");
        }
    }
}

#[test]
fn generic_method_references_show_the_expected_function_instantiation() {
    let source = "struct Factory { def create<T>() -> T = { 42 }; }; def main() -> int = { var create: () -> int; create := Factory.create; create() };";
    let project = Project::new(&[("main.resin", source)]);
    let analysis = project.checked();
    let input = project.source("main.resin");
    let offset = source.find("Factory.create").unwrap() + 8;
    assert_eq!(
        analysis.hover(&input, offset).unwrap().text,
        "create: () -> int"
    );
    let origin = analysis.definition(&input, offset).unwrap();
    assert_eq!(origin.span.start, source.find("def create").unwrap() + 4);
}

#[test]
fn generic_method_editor_snapshots_keep_completed_imported_schemes() {
    let input = Source::new(
        "main.resin",
        "import { \"library.resin\" }; def use(cell: Ptr<Cell<int>>) -> _ = { cell.read() };",
    );
    let original = Source::new(
        "library.resin",
        "export { Cell }; struct Cell<T> { value: T, def read(self: Ptr<Cell<T>>) -> T = { self.value }; };",
    );
    let changed = original.with_text("export { Cell }; struct Cell<T> { padding: ubyte, value: T, def read(self: Ptr<Cell<T>>) -> long = { 42 }; };");
    let mut loader = resin_source::Loader::new(Default::default());
    loader
        .set_import(&input, "library.resin", original.clone())
        .unwrap();
    let before = Hir::build(input.clone(), &mut loader, None);
    loader
        .set_import(&input, "library.resin", changed.clone())
        .unwrap();
    let after = Hir::build(input.clone(), &mut loader, Some(&before));
    let offset = input.text().find("cell.read").unwrap() + 5;
    for (analysis, library, expected) in [
        (&before, &original, "read: () -> int"),
        (&after, &changed, "read: () -> long"),
        (&before, &original, "read: () -> int"),
    ] {
        assert!(
            analysis.diagnostics().is_empty(),
            "{:?}",
            analysis.diagnostics()
        );
        assert_eq!(analysis.hover(&input, offset).unwrap().text, expected);
        let origin = analysis.definition(&input, offset).unwrap();
        assert_eq!(&origin.source, library);
        assert_eq!(
            origin.span.start,
            library.text().find("def read").unwrap() + 4
        );
    }
}

#[test]
fn generic_nominal_fields_retain_substitution_and_declaration_navigation() {
    let library = "export { Cell }; struct Cell<T> { value: T };";
    for (parameters, receiver, result) in [
        ("", "Cell<int>", "int"),
        ("", "Ptr<Ptr<Cell<int>>>", "int"),
        ("<T>", "Ptr<Ptr<Cell<T>>>", "T"),
    ] {
        let source = format!(
            "import {{ \"library.resin\" }}; def read{parameters}(cell: {receiver}) -> {result} = {{ cell.value }};"
        );
        let project = Project::new(&[("main.resin", &source), ("library.resin", library)]);
        let analysis = project.checked();
        let input = project.source("main.resin");
        let field = source.rfind("value").unwrap();
        let items = analysis.completions(&input, field);
        let fields = items
            .iter()
            .filter(|item| item.kind == resin_hir::DefinitionKind::Field)
            .collect::<Vec<_>>();
        assert_eq!(fields.len(), 1, "{items:?}");
        assert_eq!(fields[0].detail, format!("value: {result}"));
        assert_eq!(
            analysis.hover(&input, field).unwrap().text,
            fields[0].detail
        );
        let origin = analysis.definition(&input, field).unwrap();
        assert_eq!(origin.source, project.source("library.resin"));
        assert_eq!(origin.span.start, library.find("value").unwrap());
        assert_eq!(&library[origin.span.start..origin.span.end], "value");
    }
}

#[test]
fn nominal_wrappers_of_generic_fields_keep_navigation_and_method_completion() {
    let library = "export { Outer }; struct Cell<T> { value: T }; struct Wrapped { cell: Cell<int> }; struct Outer { wrapped: Wrapped, read: int, def read(self: Ptr<Outer>) -> int = { self.wrapped.cell.value }; };";
    let source = "import { \"library.resin\" }; def use(outer: Ptr<Outer>) -> int = { Outer.read(outer); outer.read(); outer.wrapped.cell.value };";
    let project = Project::new(&[("main.resin", source), ("library.resin", library)]);
    let analysis = project.checked();
    let input = project.source("main.resin");
    for (name, expected) in [
        ("wrapped", "wrapped: Wrapped"),
        ("cell", "cell: Cell<int>"),
        ("value", "value: int"),
    ] {
        let field = source.rfind(name).unwrap();
        let items = analysis.completions(&input, field);
        let item = items.iter().find(|item| item.name == name).unwrap();
        assert_eq!(item.detail, expected);
        assert_eq!(analysis.hover(&input, field).unwrap().text, expected);
        let origin = analysis.definition(&input, field).unwrap();
        assert_eq!(origin.source, project.source("library.resin"));
        assert_eq!(
            origin.span.start,
            library.find(&format!("{name}:")).unwrap()
        );
    }
    let items = analysis.completions(&input, source.find("outer.wrapped").unwrap() + 6);
    let method = items.iter().find(|item| item.name == "read").unwrap();
    assert_eq!(method.kind, resin_hir::DefinitionKind::Function);
    assert_eq!(items.iter().filter(|item| item.name == "read").count(), 1);
    let associated = analysis.completions(&input, source.find("Outer.read").unwrap() + 6);
    assert_eq!(associated.len(), 1, "{associated:?}");
    assert_eq!(associated[0].name, "read");
    assert_eq!(associated[0].kind, resin_hir::DefinitionKind::Function);
}

#[test]
fn generic_field_completion_survives_an_unfinished_access() {
    let source = "struct Cell<T> { value: T }; def read(cell: Ptr<Cell<int>>) = { cell.; };";
    let project = Project::new(&[("main.resin", source)]);
    let analysis = project.build_hir();
    assert!(!analysis.diagnostics().is_empty());
    let items = analysis.completions(
        &project.source("main.resin"),
        source.find("cell.;").unwrap() + 5,
    );
    let fields = items
        .iter()
        .filter(|item| item.kind == resin_hir::DefinitionKind::Field)
        .collect::<Vec<_>>();
    assert_eq!(fields.len(), 1, "{items:?}");
    assert_eq!(fields[0].detail, "value: int");
    assert!(
        items
            .iter()
            .any(|item| item.detail == "replace: (Cell<int>) -> Cell<int>")
    );
}

#[test]
fn generic_field_editor_snapshots_keep_original_imported_declarations() {
    let input = Source::new(
        "main.resin",
        "import { \"library.resin\" }; def read(cell: Cell<int>) -> _ = { cell.value };",
    );
    let original = Source::new(
        "library.resin",
        "export { Cell }; struct Cell<T> { value: T };",
    );
    let changed =
        original.with_text("export { Cell }; struct Cell<T> { padding: ubyte, value: long };");
    let mut loader = resin_source::Loader::new(Default::default());
    loader
        .set_import(&input, "library.resin", original.clone())
        .unwrap();
    let before = Hir::build(input.clone(), &mut loader, None);
    loader
        .set_import(&input, "library.resin", changed.clone())
        .unwrap();
    let after = Hir::build(input.clone(), &mut loader, Some(&before));
    let field = input.text().rfind("value").unwrap();
    for (analysis, library, expected) in [
        (&before, &original, "value: int"),
        (&after, &changed, "value: long"),
        (&before, &original, "value: int"),
    ] {
        assert!(
            analysis.diagnostics().is_empty(),
            "{:?}",
            analysis.diagnostics()
        );
        assert_eq!(analysis.hover(&input, field).unwrap().text, expected);
        let origin = analysis.definition(&input, field).unwrap();
        assert_eq!(&origin.source, library);
        assert_eq!(origin.span.start, library.text().find("value").unwrap());
    }
}

#[test]
fn tuple_members_and_function_hovers_use_source_syntax() {
    let source = "def zero() -> int = { 0 }; def pair(value: (int, bool)) = {}; def add(a: int, b: int) -> int = { a + b }; def main() = { var empty = zero; var tuple = pair; var binary = add; var values = (1_i, 1 == 1); values.0; values.; };";
    let project = Project::new(&[("main.resin", source)]);
    let input = project.source("main.resin");
    let analysis = project.build_hir();
    let members = analysis.completions(&input, source.rfind("values.").unwrap() + 7);
    assert!(
        members.iter().any(|member| member.detail == "0: int"),
        "{members:?}"
    );
    assert!(
        members.iter().any(|member| member.detail == "1: bool"),
        "{members:?}"
    );
    assert_eq!(
        analysis
            .hover(&input, source.find("values.0").unwrap() + 7)
            .unwrap()
            .text,
        "0: int"
    );
    for (name, expected) in [
        ("empty", "empty: () -> int"),
        ("tuple", "tuple: ((int, bool)) -> ()"),
        ("binary", "binary: (int, int) -> int"),
    ] {
        assert_eq!(
            analysis
                .hover(&input, source.find(&format!("var {name}")).unwrap() + 4)
                .unwrap()
                .text,
            expected
        );
    }
}

#[test]
fn imported_generic_aliases_keep_binder_navigation_and_concrete_hover() {
    let library = "export { View }; type View<T> = Ptr<T>;";
    let source = "import { \"library.resin\" }; def use_view<T>(view: View<T>) -> T = { view.* }; def main() -> int = { var value = 42; var pointer: View<int>; pointer := &value; use_view(pointer) };";
    let project = Project::new(&[("main.resin", source), ("library.resin", library)]);
    let analysis = project.checked();
    let input = project.source("main.resin");
    let origin = analysis
        .definition(&input, source.find("View<T>").unwrap())
        .unwrap();
    assert_eq!(origin.source, project.source("library.resin"));
    let hover = analysis
        .hover(&input, source.rfind("pointer").unwrap())
        .unwrap();
    assert_eq!(hover.text, "pointer: Ptr<int>");
    let library_source = project.source("library.resin");
    let origin = analysis
        .definition(&library_source, library.find("Ptr<T>").unwrap() + 4)
        .unwrap();
    assert_eq!(origin.span.start, library.find("View<T>").unwrap() + 5);
}

#[test]
fn template_scopes_retain_named_types_and_imported_schemes() {
    let library = "export { identity }; def identity<T>(value: T) -> _ = { value };";
    let source = "import { \"library.resin\" }; def forward<U>(input: U) -> U = { var copy = identity(input); copy }; def main() -> int = { forward(42) };";
    let project = Project::new(&[("main.resin", source), ("library.resin", library)]);
    let analysis = project.checked();
    let input = project.source("main.resin");
    let hover = analysis
        .hover(&input, source.rfind("copy").unwrap())
        .unwrap();
    assert_eq!(hover.text, "copy: U");
    let origin = analysis
        .definition(&input, source.find("input: U").unwrap() + 7)
        .unwrap();
    assert_eq!(origin.source, input);
    assert_eq!(origin.span.start, source.find("<U>").unwrap() + 1);
    let origin = analysis
        .definition(&input, source.find("identity(input)").unwrap())
        .unwrap();
    assert_eq!(origin.source, project.source("library.resin"));
    let completions = analysis.completions(&input, source.rfind("copy").unwrap() + 4);
    assert!(
        completions.iter().any(|item| item.detail == "copy: U"),
        "{completions:?}"
    );
}

#[test]
fn option_payload_fields_remain_available_in_incomplete_code() {
    let source = "struct Item { count: int }; def f(value: Item | None) = { value!.; };";
    let project = Project::new(&[("main.resin", source)]);
    let items = project.build_hir().completions(
        &project.source("main.resin"),
        source.find("value!.").unwrap() + 7,
    );
    assert_eq!(items.len(), 1);
    assert_eq!(items[0].detail, "count: int");
}

#[test]
fn weak_upgrade_recovery_distinguishes_wrapper_and_payload_members() {
    for (receiver, expected, absent) in [
        ("weak.upgrade()!", "get", "count"),
        ("weak.upgrade()!.get()", "count", "downgrade"),
    ] {
        let source = format!(
            "import {{ \"$/shared.resin\" }}; struct Item {{ count: int }}; def f(weak: WeakPtr<Item>) = {{ {receiver}.; }};"
        );
        let project = Project::new(&[("main.resin", &source)]);
        let items = project.build_hir().completions(
            &project.source("main.resin"),
            source.rfind(".;").unwrap() + 1,
        );
        assert!(items.iter().any(|item| item.name == expected), "{items:?}");
        assert!(!items.iter().any(|item| item.name == absent), "{items:?}");
    }
}

#[test]
fn standard_library_resource_methods_support_editor_navigation_and_recovery() {
    for tail in ["ok(()) };", "buffer."] {
        let source = format!(
            r#"import {{ "$/gpu.resin" }};
            def f() -> Result<(), _> = {{
                var gpu = Gpu.new()?;
                var bytes = gpu.alloc_in::<ubyte>(4_ul, Memory.default())?;
                var commands = gpu.start_command_recording()?;
                var buffer = gpu.alloc::<int>(4)?;
                buffer.at(0).store(42);
                var readable = buffer.read_only();
                {tail}"#
        );
        let mut loader = resin_source::Loader::new(resin_source::library_root());
        let input = loader
            .source_from_text(Path::new("main.resin"), source.clone())
            .unwrap();
        let analysis = Hir::build(input.clone(), &mut loader, None);
        if tail != "buffer." {
            assert!(
                analysis.diagnostics().is_empty(),
                "{:?}",
                analysis.diagnostics()
            );
        }
        for (method, result) in [
            ("alloc_in", "-> Result<GpuSpan<ubyte>,"),
            ("start_command_recording", "-> Result<GpuCommands,"),
        ] {
            let call = source.find(&format!(".{method}")).unwrap() + 1;
            let definition = analysis
                .definition(&input, call)
                .unwrap_or_else(|| panic!("tail: {tail}\n{:?}", analysis.diagnostics()));
            assert!(
                loader
                    .path(&definition.source)
                    .unwrap()
                    .ends_with("resin/gpu.resin")
            );
            let hover = analysis.hover(&input, call).unwrap().text;
            assert!(hover.contains(method), "{hover}");
            assert!(hover.contains(result), "{hover}");
        }
        let offset = if tail == "buffer." {
            source.len()
        } else {
            source.find("at(0)").unwrap()
        };
        let items = analysis.completions(&input, offset);
        for name in [
            "at",
            "slice",
            "read_only",
            "write_only",
            "copy_to",
            "data",
            "length",
        ] {
            assert!(
                items.iter().any(|item| item.name == name),
                "missing {name}: {items:?}"
            );
        }
        assert!(
            items
                .iter()
                .all(|item| !matches!(item.name.as_str(), "host_pointer" | "device_pointer"))
        );
    }
}

#[test]
fn gpu_commands_check_pipeline_stages_and_host_arguments() {
    for (call, valid) in [
        (
            "dispatch(compute, { value = buffer }, 1_ui, 1_ui, 1_ui)",
            true,
        ),
        ("draw(graphics, { value = buffer }, 3_ui)", true),
        ("draw(empty, None, 3_ui)", true),
        (
            "dispatch(graphics, { value = buffer }, 1_ui, 1_ui, 1_ui)",
            false,
        ),
        ("draw(compute, { value = buffer }, 3_ui)", false),
        ("dispatch(compute, None, 1_ui, 1_ui, 1_ui)", false),
        ("dispatch(compute, 0_ul, 1_ui, 1_ui, 1_ui)", false),
        (
            "dispatch(compute, { value = pointer }, 1_ui, 1_ui, 1_ui)",
            false,
        ),
        (
            "dispatch(compute, { value = bytes }, 1_ui, 1_ui, 1_ui)",
            false,
        ),
        (
            "dispatch(compute, { other = buffer }, 1_ui, 1_ui, 1_ui)",
            false,
        ),
        ("dispatch(compute, buffer, 1_ui, 1_ui, 1_ui)", false),
        ("draw(graphics, None, 3_ui)", false),
        ("draw(graphics, { value = pointer }, 3_ui)", false),
        ("draw(empty, { value = buffer }, 3_ui)", false),
        ("set_pipeline(compute)", false),
    ] {
        let source = format!(
            r#"import {{ "$/gpu.resin", "$/graphics.resin" }};
            struct Root {{ value: Ptr<int> }};
            @compute_shader
            def kernel(index: ulong, root: Ptr<Root>) = {{}};
            @vertex_shader
            def vertex(index: int, root: Ptr<Root>) -> Vertex = {{ rootless(index) }};
            @vertex_shader
            def rootless(index: int) -> Vertex = {{
                Vertex {{ position = Position {{ x = 0_f, y = 0_f, z = 0_f, w = 1_f }},
                    color = Color {{ r = 1_f, g = 0_f, b = 0_f, a = 1_f }} }}
            }};
            @fragment_shader
            def fragment(color: Color) -> Color = {{ color }};
            def f(commands: GpuCommands, gpu: Gpu, buffer: GpuPtr<int>, bytes: GpuPtr<ubyte>, pointer: Ptr<int>) -> Result<(), _> = {{
                var compute = gpu.create_compute_pipeline(kernel)?;
                var graphics = gpu.create_graphics_pipeline(vertex, fragment)?;
                var empty = gpu.create_graphics_pipeline(rootless, fragment)?;
                commands.{call}?;
                ok(())
            }};"#
        );
        let mut loader = resin_source::Loader::new(resin_source::library_root());
        let input = loader
            .source_from_text(Path::new("main.resin"), source)
            .unwrap();
        let analysis = Hir::build(input.clone(), &mut loader, None);
        assert_eq!(
            analysis.diagnostics().is_empty(),
            valid,
            "{call}: {:?}",
            analysis.diagnostics()
        );
        assert!(
            analysis
                .diagnostics()
                .iter()
                .all(|diagnostic| diagnostic.location.source == input),
            "{call}: {:?}",
            analysis.diagnostics()
        );
    }
}

#[test]
fn typed_pipeline_calls_show_shader_contracts_in_editor_signatures() {
    let source = r#"import { "$/gpu.resin", "$/span.resin" };
        struct Root { values: Span<int>, scale: int };
        @compute_shader
        def kernel(index: ulong, root: Ptr<Root>) = {};
        def f(gpu: Gpu, commands: GpuCommands, values: GpuSpan<int>) -> Result<(), _> = {
            var pipeline = gpu.create_compute_pipeline(kernel)?;
            commands.dispatch(pipeline, { values = values, scale = 2 }, 1, 1, 1)?;
            ok(())
        };"#;
    let mut loader = resin_source::Loader::new(resin_source::library_root());
    let input = loader
        .source_from_text(Path::new("main.resin"), source)
        .unwrap();
    let analysis = Hir::build(input.clone(), &mut loader, None);
    assert!(
        analysis.diagnostics().is_empty(),
        "{:?}",
        analysis.diagnostics()
    );
    for (method, contract) in [
        ("create_compute_pipeline", "GpuComputePipeline<Root,"),
        ("dispatch", "GpuSpan<int>"),
    ] {
        let call = source.find(&format!(".{method}")).unwrap() + 1;
        let definition = analysis.definition(&input, call).unwrap();
        assert!(
            loader
                .path(&definition.source)
                .unwrap()
                .ends_with("resin/gpu.resin")
        );
        let hover = analysis.hover(&input, call).unwrap().text;
        assert!(hover.contains(contract), "{method}: {hover}");
        assert!(!hover.contains("GpuArguments"), "{method}: {hover}");
        let completions = analysis.completions(&input, call);
        let item = completions.iter().find(|item| item.name == method).unwrap();
        assert!(item.detail.contains(contract), "{method}: {}", item.detail);
    }
}

#[test]
fn pointer_hover_and_completion_use_angle_bracket_types() {
    let source = "import { \"$/span.resin\" }; def main (value: Ptr<Span<int>>) -> Ptr<Span<int>> = { value };";
    let project = Project::new(&[("main.resin", source)]);
    let analysis = project.checked();
    assert_eq!(
        analysis
            .hover(
                &project.source("main.resin"),
                source.rfind("value").unwrap()
            )
            .unwrap()
            .text,
        "value: Ptr<Span<int>>"
    );

    let source = "type Number = int; def main () -> () = { var value: Ptr<Num>; };";
    let project = Project::new(&[("main.resin", source)]);
    let items = project.build_hir().completions(
        &project.source("main.resin"),
        source.rfind("Num").unwrap() + 3,
    );
    assert_eq!(
        items
            .iter()
            .map(|item| item.name.as_str())
            .collect::<Vec<_>>(),
        ["Number"]
    );
}

struct Project {
    sources: BTreeMap<String, Source>,
}

#[test]
fn inherent_methods_have_navigation_hover_and_member_completion() {
    let library = "export { Counter }; struct Counter { count: int, def new() -> Counter = { Counter { count = 7 } }; def read(self: Ptr<Counter>) -> int = { self.count }; }; ";
    for incomplete in [None, Some("c"), Some("Counter")] {
        let tail = incomplete
            .map(|base| format!("{base}.;"))
            .unwrap_or_default();
        let source = format!(
            "import {{ \"lib.resin\" }}; def main() = {{ var c = Counter.new(); c.read(); {tail} }};"
        );
        let project = Project::new(&[("main.resin", &source), ("lib.resin", library)]);
        let analysis = project.build_hir();
        let input = project.source("main.resin");
        assert_eq!(
            analysis.diagnostics().is_empty(),
            incomplete.is_none(),
            "{:?}",
            analysis.diagnostics()
        );
        for method in ["new", "read"] {
            let call = source.find(&format!(".{method}()")).unwrap() + 1;
            let definition = analysis.definition(&input, call).unwrap();
            assert_eq!(definition.source, project.source("lib.resin"));
            assert_eq!(
                definition.span.start,
                library.find(&format!("{method}(")).unwrap()
            );
            assert!(
                analysis
                    .hover(&input, call)
                    .unwrap()
                    .text
                    .starts_with(&format!("def {method}("))
            );
            let items = analysis.completions(&input, call);
            assert!(items.iter().any(
                |item| item.name == method && item.kind == resin_hir::DefinitionKind::Function
            ));
        }
        let global = analysis.completions(&input, source.find("var c").unwrap());
        assert!(
            global
                .iter()
                .all(|item| item.name != "read" && item.name != "new")
        );
        if let Some(base) = incomplete {
            let items = analysis.completions(
                &input,
                source.rfind(&format!("{base}.;")).unwrap() + base.len() + 1,
            );
            assert_eq!(
                items
                    .iter()
                    .map(|item| item.name.as_str())
                    .collect::<Vec<_>>(),
                if base == "c" {
                    vec!["count", "read"]
                } else {
                    vec!["new", "read"]
                }
            );
        }
    }
    let project = Project::new(&[("main.resin", library)]);
    let items = project.build_hir().completions(
        &project.source("main.resin"),
        library.find("self.count").unwrap(),
    );
    assert!(
        items
            .iter()
            .all(|item| item.name != "read" && item.name != "new")
    );
}

#[test]
fn at_indexing_has_hover_and_completion_in_valid_and_incomplete_code() {
    for receiver in ["values", "holder.values"] {
        for tail in ["", " values.;", " holder.values.;", " holder.values.at(; "] {
            let source = format!(
                "import {{ \"$/span.resin\" }}; def main() = {{ var values = [1_i, 2_i]; var holder = {{ values = Span<int> {{ data = Ptr<int>(&values), length = 2_ul }} }}; {receiver}.at(0).* := 3;{tail} }};"
            );
            let project = Project::new(&[("main.resin", &source)]);
            let analysis = project.build_hir();
            let input = project.source("main.resin");
            assert_eq!(
                analysis.diagnostics().is_empty(),
                tail.is_empty(),
                "{:?}",
                analysis.diagnostics()
            );
            let offset = source.find(".at(0)").unwrap() + 1;
            assert_eq!(
                analysis.hover(&input, offset).unwrap().text,
                "at: (ulong) -> Ptr<int>"
            );
            let items = analysis.completions(&input, offset);
            assert!(
                items
                    .iter()
                    .any(|item| item.name == "at"
                        && item.kind == resin_hir::DefinitionKind::Function)
            );
            if !tail.is_empty() {
                let offset = source.rfind(".;").or_else(|| source.rfind(".at(")).unwrap() + 1;
                let items = analysis.completions(&input, offset);
                assert!(
                    items.iter().any(|item| item.name == "at"),
                    "{source}\n{items:?}"
                );
            }
        }
    }
}

#[test]
fn shared_receiver_completion_and_navigation_include_ordinary_drop_methods() {
    let library = "export { Counter }; struct Counter { count: int, def drop(self: Ptr<Counter>) = {}; def read(self: Ptr<Counter>) -> int = { self.count }; }; ";
    for tail in ["", "c.get().;"] {
        let source = format!(
            "import {{ \"lib.resin\", \"$/shared.resin\" }}; def f(c: ArcPtr<Counter>) = {{ c.get().read(); {tail} }};"
        );
        let project = Project::new(&[("main.resin", &source), ("lib.resin", library)]);
        let analysis = project.build_hir();
        let input = project.source("main.resin");
        let call = source.find(".read").unwrap() + 1;
        assert_eq!(
            analysis
                .definition(&input, call)
                .unwrap_or_else(|| panic!(
                    "{:?}\n{:?}",
                    analysis.diagnostics(),
                    analysis.recovered_file(&input)
                ))
                .source,
            project.source("lib.resin")
        );
        assert!(
            analysis
                .hover(&input, call)
                .unwrap()
                .text
                .starts_with("def read(")
        );
        let offset = if tail.is_empty() {
            call
        } else {
            source.rfind(".;").unwrap() + 1
        };
        let items = analysis.completions(&input, offset);
        assert!(items.iter().any(|item| item.name == "read"));
        assert!(items.iter().any(|item| item.name == "drop"));
    }
}

#[test]
fn inferred_errors_and_match_payloads_have_editor_types() {
    let source = "struct Broken { code: int }; def fail() -> Result<int, _> = { err(Broken { code = 7 }) }; def main() = { var result = fail(); match (result) { ok(value) => { value; }, err(error) => { error.code; } }; };";
    let project = Project::new(&[("main.resin", source)]);
    let analysis = project.checked();
    let input = project.source("main.resin");
    assert_eq!(
        analysis
            .hover(&input, source.find("match (result").unwrap() + 7)
            .unwrap()
            .text,
        "result: Result<int, Broken>"
    );
    assert_eq!(
        analysis
            .hover(&input, source.find("error.code").unwrap())
            .unwrap()
            .text,
        "error: Broken"
    );
    let fields = analysis.completions(&input, source.find("error.code").unwrap() + 6);
    assert!(
        fields
            .iter()
            .any(|field| field.name == "code" && field.detail == "code: int")
    );
    let origin = analysis
        .definition(&input, source.find("error.code").unwrap())
        .unwrap();
    assert_eq!(origin.span.start, source.find("err(error)").unwrap() + 4);
    assert!(
        analysis
            .completions(&input, source.find("var result").unwrap())
            .iter()
            .all(|item| item.name != "error")
    );
}

#[test]
fn inferred_imported_results_and_local_annotations_support_editor_queries() {
    let source =
        "import { \"lib.resin\" }; def main() = { var value: _; value := make(); value.count; };";
    let library = "export { make }; struct Counter { count: int }; def make() -> _ = { Counter { count = 42 } };";
    let project = Project::new(&[("main.resin", source), ("lib.resin", library)]);
    let analysis = project.checked();
    let input = project.source("main.resin");
    assert_eq!(
        analysis
            .hover(&input, source.rfind("value").unwrap())
            .unwrap()
            .text,
        "value: Counter"
    );
    let fields = analysis.completions(&input, source.rfind("count").unwrap());
    assert!(
        fields
            .iter()
            .any(|field| field.name == "count" && field.detail == "count: int")
    );
    let definition = analysis
        .definition(&input, source.find("make()").unwrap())
        .unwrap();
    assert_eq!(definition.source, project.source("lib.resin"));
}

#[test]
fn inference_does_not_publish_speculative_type_references() {
    let source = "type Value = int; def f() -> _ = { type Value = bool; type Wrapper = Value; Wrapper(Value(1 == 1)) };";
    let project = Project::new(&[("main.resin", source)]);
    let analysis = project.checked();
    let reference = source.find("Wrapper = Value").unwrap() + "Wrapper = ".len();
    let definition = analysis
        .definition(&project.source("main.resin"), reference)
        .unwrap();
    assert_eq!(
        definition.span.start,
        source.rfind("type Value").unwrap() + 5
    );
}

#[test]
fn field_completion_before_existing_statements() {
    for (receiver, fields) in [
        ("root", vec!["height", "pixels", "width"]),
        ("(root)", vec!["height", "pixels", "width"]),
        ("root.pixels", vec!["data", "length"]),
    ] {
        for following in [
            "if (index < ulong(root.width)) { root.pixels.at(index).* := 0_ui; };",
            "var later = root.width; later;",
            "while (false) { root.width; };",
            "root.width;",
            "",
        ] {
            for preceding in ["", "var earlier = root.height;"] {
                let source = format!(
                    "// é🌲\nimport {{ \"$/span.resin\" }}; struct Root {{ width: uint, height: uint, pixels: Span<uint> }};\n\
                    @compute_shader def kernel(index: ulong, root: Ptr<Root>) = {{\n\
                    {preceding}\n{receiver}.\n{following}\n}};"
                );
                let project = Project::new(&[("main.resin", &source)]);
                let analysis = project.build_hir();
                let input = project.source("main.resin");
                let offset = source.find(&format!("{receiver}.\n")).unwrap() + receiver.len() + 1;
                let items = analysis.completions(&input, offset);
                let names = items
                    .iter()
                    .filter(|item| item.kind == resin_hir::DefinitionKind::Field)
                    .map(|item| item.name.as_str())
                    .collect::<Vec<_>>();
                assert_eq!(names, fields, "{source}");
                for item in &items {
                    assert_eq!(
                        item.replace,
                        Span {
                            start: offset,
                            end: offset
                        }
                    );
                }
                if following.starts_with("var later") {
                    assert_eq!(
                        analysis
                            .hover(&input, source.rfind("later;").unwrap())
                            .unwrap()
                            .text,
                        "later: uint"
                    );
                }
                if following != "root.width;" {
                    assert!(
                        analysis.hir().is_err(),
                        "unfinished access must remain invalid"
                    );
                }
            }
        }
    }
}

#[test]
fn field_completion_uses_receiver_types_and_replaces_only_the_field() {
    for (setup, receiver) in [
        ("var value = { count = 1, label = 2 };", "value"),
        (
            "var record = { count = 1, label = 2 }; var value = &record;",
            "value",
        ),
        (
            "var value = { inner = { count = 1, label = 2 } };",
            "value.inner",
        ),
        ("var value = { count = 1, label = 2 };", "(value)"),
    ] {
        for field in ["", "co", "count"] {
            let source = format!("def main () = {{ {setup} {receiver}.{field}; }};");
            let project = Project::new(&[("main.resin", &source)]);
            let start = source.rfind('.').unwrap() + 1;
            let offset = start + field.len().min(2);
            let items = project
                .build_hir()
                .completions(&project.source("main.resin"), offset);
            let names = items
                .iter()
                .filter(|item| item.kind == resin_hir::DefinitionKind::Field)
                .map(|item| item.name.as_str())
                .collect::<Vec<_>>();
            assert_eq!(
                names,
                if field.is_empty() {
                    vec!["count", "label"]
                } else {
                    vec!["count"]
                },
                "{source}"
            );
            assert_eq!(items[0].detail, "count: long");
            assert_eq!(items[0].kind, resin_hir::DefinitionKind::Field);
            assert_eq!(items[0].replace.start, start);
            assert_eq!(items[0].replace.end, start + field.len());
        }
    }
}

#[test]
fn field_completion_resolves_imported_nominal_function_results() {
    let source = "import { \"lib.resin\" }; def main () = { make().; };";
    let project = Project::new(&[
        ("main.resin", source),
        (
            "lib.resin",
            "export { make }; struct Counter { count: int }; def make () -> Counter = { Counter { count = 0 } };",
        ),
    ]);
    let items = project.build_hir().completions(
        &project.source("main.resin"),
        source.rfind('.').unwrap() + 1,
    );
    assert_eq!(
        items
            .iter()
            .map(|item| item.name.as_str())
            .collect::<Vec<_>>(),
        ["count"]
    );
}

#[test]
fn field_completion_does_not_offer_unrelated_names() {
    for expression in ["1.", "missing.", "\"text.\"", "// value."] {
        let source = format!("def main () = {{ var value = {{ count = 1 }}; {expression}\n }};");
        let project = Project::new(&[("main.resin", &source)]);
        assert!(
            project
                .build_hir()
                .completions(
                    &project.source("main.resin"),
                    source.rfind('.').unwrap() + 1
                )
                .is_empty(),
            "{source}"
        );
    }
}

#[test]
fn field_completion_recovers_unfinished_functions_and_uninitialized_locals() {
    for source in [
        "struct Point { x: float32, y: float32 }; def main () = { var point: Point; point.; };",
        "def main (point: { x: float32, y: float32 }) = { point.",
        "def main () = { var point = { x = 1, y = 2 }; point.",
    ] {
        let project = Project::new(&[("main.resin", source)]);
        let items = project.build_hir().completions(
            &project.source("main.resin"),
            source.rfind('.').unwrap() + 1,
        );
        assert_eq!(
            items
                .iter()
                .map(|item| item.name.as_str())
                .collect::<Vec<_>>(),
            ["x", "y"],
            "{source}"
        );
    }
}
impl Project {
    fn new(files: &[(&str, &str)]) -> Self {
        let sources = files
            .iter()
            .map(|(name, text)| (name.to_string(), Source::new(*name, *text)))
            .collect();
        Self { sources }
    }
    fn build_hir(&self) -> Hir {
        let mut loader = resin_source::Loader::new(resin_source::library_root());
        for importer in self.sources.values() {
            for (reference, target) in &self.sources {
                loader
                    .set_import(importer, reference, target.clone())
                    .unwrap();
            }
        }
        Hir::build(self.source("main.resin"), &mut loader, None)
    }
    #[track_caller]
    fn checked(&self) -> Hir {
        let analysis = self.build_hir();
        assert!(
            analysis.diagnostics().is_empty(),
            "{:?}",
            analysis.diagnostics()
        );
        analysis
    }
    fn source(&self, name: &str) -> Source {
        self.sources[name].clone()
    }
}

#[test]
fn inferred_types_and_parameter_definitions_come_from_compilation() {
    let source = "def main (argument: int) -> int = { var value = argument + 1; value };";
    let project = Project::new(&[("main.resin", source)]);
    let analysis = project.checked();
    let input = project.source("main.resin");
    assert_eq!(
        analysis
            .hover(&input, source.rfind("value").unwrap())
            .unwrap()
            .text,
        "value: int"
    );
    let location = analysis
        .definition(&input, source.rfind("argument").unwrap())
        .unwrap();
    assert_eq!(location.source, input);
    assert_eq!(&source[location.span.start..location.span.end], "argument");
    assert_eq!(location.span.start, source.find("argument").unwrap());
}

#[test]
fn reexports_keep_original_definitions_through_diamond_imports() {
    let source = "import { \"left.resin\", \"right.resin\" }; def main () -> Number = { make() };";
    let project = Project::new(&[
        ("main.resin", source),
        (
            "base.resin",
            "export { Number, make }; type Number = int; def make () -> Number = { Number(42) }; def secret () -> int = { 1 };",
        ),
        (
            "left.resin",
            "export { Number, make }; import { \"base.resin\" };",
        ),
        (
            "right.resin",
            "export { Number, make }; import { \"base.resin\" };",
        ),
    ]);
    let analysis = project.checked();
    let input = project.source("main.resin");
    for word in ["Number", "make"] {
        assert_eq!(
            analysis
                .definition(&input, source.rfind(word).unwrap())
                .unwrap()
                .source,
            project.source("base.resin")
        );
    }
    let completions = analysis.completions(&input, source.rfind("make").unwrap());
    assert!(completions.iter().any(|c| c.name == "make"));
    assert!(!completions.iter().any(|c| c.name == "secret"));
    assert_eq!(
        analysis
            .definition(&input, source.find("left.resin").unwrap())
            .unwrap()
            .source,
        project.source("left.resin")
    );
}

#[test]
fn nested_bindings_shadow_without_leaking_out_of_their_scope() {
    let source =
        "def main () -> int = { var value = 1; var inner = { var value = 2; value }; value };";
    let project = Project::new(&[("main.resin", source)]);
    let analysis = project.checked();
    let input = project.source("main.resin");
    let inside = source.find("value };").unwrap();
    assert_eq!(
        analysis.definition(&input, inside).unwrap().span.start,
        source.find("value = 2").unwrap()
    );
    assert_eq!(
        analysis
            .definition(&input, source.rfind("value").unwrap())
            .unwrap()
            .span
            .start,
        source.find("value = 1").unwrap()
    );
}

#[test]
fn incomplete_code_keeps_parameters_and_prior_locals_available() {
    for source in [
        "def main (argument: int) -> int = { var local = 1; arg };",
        "def main (argument: int) -> int = { var local = 1; arg",
        "def main (argument: int) -> int = { var local = 1; print(arg };",
    ] {
        let project = Project::new(&[("main.resin", source)]);
        let analysis = project.build_hir();
        let at = source.rfind("arg").unwrap() + 3;
        let completions = analysis.completions(&project.source("main.resin"), at);
        assert!(
            completions.iter().any(|c| c.name == "argument"),
            "{source}: {completions:?}"
        );
        let local = analysis.completions(&project.source("main.resin"), at - 3);
        assert!(
            local.iter().any(|c| c.name == "local"),
            "{source}: {local:?}"
        );
    }
}

#[test]
fn conflicting_imports_are_ambiguous_and_report_related_locations() {
    let source = "import { \"a.resin\", \"b.resin\" }; def main () -> int = { answer() };";
    let project = Project::new(&[
        ("main.resin", source),
        (
            "a.resin",
            "export { answer }; def answer () -> int = { 1 };",
        ),
        (
            "b.resin",
            "export { answer }; def answer () -> int = { 2 };",
        ),
    ]);
    let analysis = project.build_hir();
    assert!(analysis.diagnostics().iter().any(|d| d.related.len() == 2));
    assert!(
        analysis
            .definition(
                &project.source("main.resin"),
                source.rfind("answer").unwrap()
            )
            .is_none()
    );
    assert!(
        !analysis
            .completions(
                &project.source("main.resin"),
                source.rfind("answer").unwrap()
            )
            .iter()
            .any(|c| c.name == "answer")
    );
}

#[test]
fn missing_imports_and_cycles_have_source_ranges() {
    let source = "import { \"missing.resin\", \"available.resin\" }; def main() = { var point = make(); point.x; };";
    let project = Project::new(&[
        ("main.resin", source),
        (
            "available.resin",
            "export { make }; struct Point { x: int }; def make() -> Point = { Point { x = 1 } };",
        ),
    ]);
    let analysis = project.build_hir();
    assert_eq!(analysis.diagnostics().len(), 1);
    let error = &analysis.diagnostics()[0];
    assert_eq!(error.location.source, project.source("main.resin"));
    assert_eq!(
        &source[error.location.span.start..error.location.span.end],
        "\"missing.resin\""
    );
    let fields = analysis.completions(
        &project.source("main.resin"),
        source.find("point.x").unwrap() + 6,
    );
    assert_eq!(
        fields[0].detail, "x: int",
        "a failed import must preserve later dependencies"
    );
    let project = Project::new(&[
        ("main.resin", "import { \"a.resin\" };"),
        ("a.resin", "import { \"main.resin\" };"),
    ]);
    assert!(
        project
            .build_hir()
            .diagnostics()
            .iter()
            .any(|d| d.message.contains("cyclic"))
    );
}

#[test]
fn imported_syntax_errors_stay_at_the_dependency() {
    let project = Project::new(&[
        ("main.resin", "import { \"a.resin\" };"),
        ("a.resin", "export { value }; def value () -> int = { + };"),
    ]);
    let analysis = project.build_hir();
    let error = analysis.program().unwrap_err();
    assert_eq!(error.source, project.source("a.resin"));
    assert!(error.span.is_some());
    assert_eq!(
        error.related[0].location.source,
        project.source("main.resin")
    );
    assert!(
        project
            .build_hir()
            .diagnostics()
            .iter()
            .any(|d| d.location.source == project.source("a.resin"))
    );
}

#[test]
fn malformed_foreign_headers_are_diagnostics_not_panics() {
    for source in [
        r#"extern "bad\q" def release();"#,
        "extern \"unfinished def release();",
        "extern def release();",
    ] {
        let project = Project::new(&[("main.resin", source)]);
        let input = project.source("main.resin");
        let analysis = project.build_hir();
        let error = analysis.program().unwrap_err();
        assert_eq!(error.source, input);
        assert!(error.span.is_some(), "{source}: {error}");
        assert!(!analysis.diagnostics().is_empty(), "{source}");
        assert!(analysis.hir().is_err(), "{source}");
    }
}

#[test]
fn malformed_function_names_do_not_create_editor_definitions() {
    let source = "extern \"native.h\" def releasex: int); def main() = {};";
    let project = Project::new(&[("main.resin", source)]);
    let analysis = project.build_hir();
    let input = project.source("main.resin");
    assert!(analysis.hir().is_err());
    assert!(
        analysis
            .definition(&input, source.find("releasex").unwrap())
            .is_none()
    );
    assert!(
        analysis
            .completions(&input, source.len())
            .iter()
            .all(|item| item.name != "releasex")
    );
}

#[test]
#[cfg(unix)]
fn new_paths_resolve_through_symlinks() {
    let temp = TempDir::new_in(std::env::temp_dir()).unwrap();
    let root = std::fs::canonicalize(temp.path()).unwrap();
    std::fs::create_dir(root.join("real")).unwrap();
    std::os::unix::fs::symlink(root.join("real"), root.join("alias")).unwrap();
    assert_eq!(
        normalize_path(&root.join("alias/new.resin")).unwrap(),
        root.join("real/new.resin")
    );
}

#[test]
fn source_buffers_need_no_disk_write() {
    let temp = TempDir::new_in(std::env::temp_dir()).unwrap();
    let path = temp.path().join("lib.resin");
    std::fs::write(&path, "disk").unwrap();
    let mut loader = resin_source::Loader::new(temp.path().to_path_buf());
    let source = loader.source_from_text(&path, "buffer").unwrap();
    assert_eq!(source.text(), "buffer");
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "disk");
}

#[test]
fn normalized_paths_are_stable_for_existing_and_unsaved_files() {
    let temp = TempDir::new_in(std::env::temp_dir()).unwrap();
    std::fs::write(temp.path().join("saved.resin"), "").unwrap();
    let root = std::fs::canonicalize(temp.path()).unwrap();
    for name in ["saved.resin", "unsaved.resin"] {
        let expected = root.join(name);
        assert_eq!(normalize_path(&temp.path().join(name)).unwrap(), expected);
        assert_eq!(normalize_path(&expected).unwrap(), expected);
    }
}

#[test]
fn completion_respects_type_context_and_ignores_strings_and_comments() {
    let source = "type Number = int; def main () -> () = { var value = 1; var other: Num; };";
    let project = Project::new(&[("main.resin", source)]);
    let analysis = project.build_hir();
    let suggestions = analysis.completions(
        &project.source("main.resin"),
        source.rfind("Num").unwrap() + 3,
    );
    assert_eq!(
        suggestions
            .iter()
            .map(|c| c.name.as_str())
            .collect::<Vec<_>>(),
        ["Number"]
    );
    for source in ["// val", "def main () -> () = { var value = \"val\"; };"] {
        let project = Project::new(&[("main.resin", source)]);
        let at = if source.contains('"') {
            source.rfind("val").unwrap() + 2
        } else {
            source.len() - 1
        };
        assert!(
            project
                .build_hir()
                .completions(&project.source("main.resin"), at)
                .is_empty(),
            "{source}"
        );
    }
}

#[test]
fn completion_obeys_parameter_shadowing_and_module_type_order() {
    let source = "def main (value: int) -> int = { var value = 2; val };";
    let project = Project::new(&[("main.resin", source)]);
    let analysis = project.build_hir();
    let items = analysis.completions(
        &project.source("main.resin"),
        source.rfind("val").unwrap() + 3,
    );
    let value = items.iter().find(|item| item.name == "value").unwrap();
    assert_eq!(value.kind, resin_hir::DefinitionKind::Variable);

    // Module type definitions run before function signatures, but run in source
    // order relative to each other. Completion must match those frontend phases.
    for (source, expected) in [
        ("def main (value: Num) -> () = {}; type Number = int;", true),
        ("type Earlier = Num; type Number = int;", false),
    ] {
        let project = Project::new(&[("main.resin", source)]);
        let analysis = project.build_hir();
        let items = analysis.completions(
            &project.source("main.resin"),
            source.find("Num").unwrap() + 2,
        );
        assert_eq!(
            items.iter().any(|item| item.name == "Number"),
            expected,
            "{source}: {items:?}"
        );
    }
    let project = Project::new(&[("main.resin", "// comment")]);
    assert!(
        project
            .build_hir()
            .completions(&project.source("main.resin"), 10)
            .is_empty()
    );
}

#[test]
fn module_analysis_needs_no_entry_and_rejects_runtime_globals() {
    let source = "export { run }; def run () -> int = { helper() }; def helper () -> int = { 1 };";
    let project = Project::new(&[("main.resin", source)]);
    let analysis = project.checked();
    assert_eq!(
        analysis
            .hir()
            .unwrap()
            .entries
            .keys()
            .map(|name| name.as_ref())
            .collect::<Vec<_>>(),
        ["run"]
    );
    assert_eq!(
        analysis
            .definition(&project.source("main.resin"), source.find("run").unwrap())
            .unwrap()
            .span
            .start,
        source.find("run ()").unwrap()
    );
    let project = Project::new(&[("main.resin", "export {}; def helper () -> int = { 1 };")]);
    assert!(project.build_hir().hir().unwrap().entries.is_empty());

    for source in [
        "var global = 1; def main () -> int = { glo };",
        "var global: int; def main () -> int = { glo };",
    ] {
        let project = Project::new(&[("main.resin", source)]);
        let analysis = project.build_hir();
        assert!(!analysis.diagnostics().is_empty(), "{source}");
        let at = source.rfind("glo").unwrap();
        assert!(
            analysis
                .definition(&project.source("main.resin"), at)
                .is_none()
        );
        assert!(
            !analysis
                .completions(&project.source("main.resin"), at + 3)
                .iter()
                .any(|item| item.name == "global")
        );
    }
}

#[test]
fn implicit_unit_signatures_and_declaration_keywords_support_editor_features() {
    let source = "extern \"native.h\" def release(value: int); def run() = { release(1); };";
    let project = Project::new(&[("main.resin", source)]);
    let analysis = project.checked();
    let input = project.source("main.resin");
    for (name, signature) in [
        (
            "release",
            "extern \"native.h\" def release(value: int) -> ()",
        ),
        ("run", "def run() -> ()"),
    ] {
        let offset = source.rfind(name).unwrap();
        assert_eq!(analysis.hover(&input, offset).unwrap().text, signature);
        let items = analysis.completions(&input, offset + name.len());
        assert_eq!(
            items.iter().find(|item| item.name == name).unwrap().detail,
            signature
        );
        assert_eq!(
            analysis.definition(&input, offset).unwrap().span.start,
            source.find(name).unwrap()
        );
    }
    let project = Project::new(&[("main.resin", "def run() = { var value = 1; ")]);
    let items = project
        .build_hir()
        .completions(&project.source("main.resin"), 28);
    for keyword in ["def", "var", "type"] {
        assert!(items.iter().any(|item| item.name == keyword), "{items:?}");
    }
    let project = Project::new(&[("main.resin", "def run() = { 1 };")]);
    assert!(
        !project.build_hir().diagnostics().is_empty(),
        "unit returns are not inferred from bodies"
    );
}

#[test]
fn holes_preserve_later_locals_and_functions_without_producing_ir() {
    for broken in [
        "var broken = ;",
        "var broken: ;",
        "var broken = 1 + ;",
        "unknown_name;",
        "var broken = missing(1);",
    ] {
        let source = format!(
            "def first() = {{ {broken} var point = {{ x = 1, y = 2 }}; point.; }}; def later(arg: int) -> int = {{ var result = arg; result }};"
        );
        let project = Project::new(&[("main.resin", &source)]);
        let analysis = project.build_hir();
        let input = project.source("main.resin");
        assert!(analysis.hir().is_err(), "{source}");
        assert!(!analysis.diagnostics().is_empty(), "{source}");
        let items = analysis.completions(&input, source.find("point.").unwrap() + 6);
        assert_eq!(
            items.iter().map(|i| i.name.as_str()).collect::<Vec<_>>(),
            ["x", "y"],
            "{source}"
        );
        let hover = analysis
            .hover(&input, source.rfind("result").unwrap())
            .unwrap();
        assert_eq!(hover.text, "result: int", "{source}");
    }
}

#[test]
fn unknown_bindings_shadow_outer_values_without_fabricating_types() {
    for initializer in ["", "missing(1)", "1 + (1 == 1)"] {
        let source = format!(
            "def main(point: {{ x: int }}) = {{ var point = {initializer}; var alias = point; alias.; var healthy = {{ count = 42 }}; healthy.count; }};"
        );
        let project = Project::new(&[("main.resin", &source)]);
        let analysis = project.build_hir();
        let input = project.source("main.resin");
        assert_eq!(
            analysis
                .hover(&input, source.rfind("alias").unwrap())
                .unwrap()
                .text,
            "alias: ?"
        );
        assert!(
            analysis
                .completions(&input, source.find("alias.").unwrap() + 6)
                .is_empty()
        );
        assert_eq!(
            analysis
                .definition(&input, source.rfind("point").unwrap())
                .unwrap()
                .span
                .start,
            source.find("var point").unwrap() + 4
        );
        assert!(analysis.hir().is_err());
        let fields = analysis.completions(&input, source.rfind("count").unwrap());
        assert_eq!(
            fields
                .iter()
                .map(|item| item.detail.as_str())
                .collect::<Vec<_>>(),
            ["count: long"],
            "{source}"
        );
    }
}

#[test]
fn unrelated_errors_preserve_expression_types_and_field_completion() {
    let mut failures = Vec::new();
    for (setup, expected) in [
        ("var value = -128b;", "sbyte"),
        ("var value = float32(42);", "float32"),
        ("var unused: int; var value = size_of(unused);", "ulong"),
        (
            "var value: Result<int, Never>; value := ok(42);",
            "Result<int, Never>",
        ),
    ] {
        for broken in ["", "def broken() = { missing; };"] {
            let source = format!(
                "{broken} def main() = {{ {setup} value; var record = {{ payload = value }}; record.payload; }};"
            );
            let project = Project::new(&[("main.resin", &source)]);
            let analysis = project.build_hir();
            let input = project.source("main.resin");
            assert_eq!(
                analysis.diagnostics().is_empty(),
                broken.is_empty(),
                "{source}: {:?}",
                analysis.diagnostics()
            );
            assert_eq!(analysis.hir().is_ok(), broken.is_empty(), "{source}");
            let hover = analysis
                .hover(&input, source.find("value;").unwrap())
                .unwrap();
            let fields = analysis.completions(&input, source.rfind("payload").unwrap());
            let details: Vec<_> = fields.iter().map(|item| item.detail.as_str()).collect();
            if hover.text != format!("value: {expected}")
                || details != [format!("payload: {expected}")]
            {
                failures.push(format!(
                    "{source}\nexpected {expected}, got {} and {details:?}",
                    hover.text
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn failed_compound_constraints_do_not_poison_independent_inference() {
    let source = "def main() = { var value: _; var broken: { first: int, second: bool }; broken := { first = value, second = 0 }; value := 1.5f; value; };";
    let project = Project::new(&[("main.resin", source)]);
    let analysis = project.build_hir();
    assert!(!analysis.diagnostics().is_empty());
    assert!(analysis.hir().is_err());
    let input = project.source("main.resin");
    assert_eq!(
        analysis
            .hover(&input, source.rfind("value").unwrap())
            .unwrap()
            .text,
        "value: float32"
    );
    assert_eq!(
        analysis
            .definition(&input, source.rfind("value").unwrap())
            .unwrap()
            .span
            .start,
        source.find("value").unwrap()
    );
}

#[test]
fn recovery_uses_unsaved_imports_and_keeps_nominal_field_types() {
    let source =
        "import { \"lib.resin\" }; def main() = { var broken = ; var point = make(); point.; };";
    let project = Project::new(&[
        ("main.resin", source),
        (
            "lib.resin",
            "export { make }; struct Point { x: float32 }; def broken() = { var hole = ; }; def make() -> Point = { Point { x = 1 } };",
        ),
    ]);
    let analysis = project.build_hir();
    let items = analysis.completions(
        &project.source("main.resin"),
        source.find("point.").unwrap() + 6,
    );
    assert_eq!(items.len(), 1);
    assert_eq!(items[0].detail, "x: float32");
    assert!(analysis.hir().is_err());
}

#[test]
fn recovered_ast_contains_expression_type_and_field_holes() {
    let source = "def main() = { var value = ; var typed: ; var point = { x = 1 }; point.; };";
    let project = Project::new(&[("main.resin", source)]);
    let analysis = project.build_hir();
    let ast = resin_ast::format_source(
        analysis
            .recovered_file(&project.source("main.resin"))
            .unwrap(),
    );
    assert!(ast.contains("(hole "), "{ast}");
    assert!(ast.contains("(type-hole "), "{ast}");
    assert!(ast.contains("(field-hole "), "{ast}");
    assert!(analysis.program().is_err());
}

#[test]
fn editor_analysis_tolerates_truncation_and_deleted_tokens() {
    for source in [
        "export { main }; struct Point { x: int }; def main(arg: Ptr<Point>) = { var value = arg.x + 1; print(fmt(\"{}\", value)); };",
        "def main(arg: int) -> int = { var pair = { left = arg, right = 1 }; if (arg == 0) (pair.left) else (pair.right) };",
        "def main() = { var values = [1, 2]; while (1 == 1) { var missing: Ptr<int>; }; };",
        "struct Cleanup { value: Ptr<int>, def drop(self: Ptr<Cleanup>) = { self.value.* := 42; }; };  def main() = { var n = 0; var cleanup = Cleanup { value = &n }; };",
        "struct Item { value: int }; def main() = { var owner = ArcPtr<Item> { value = 42 }; var weak = owner.downgrade(); match (weak.upgrade()) { ArcPtr<Item>(item) => { item.value; }, None => {} }; };",
    ] {
        for end in 0..=source.len() {
            let project = Project::new(&[("main.resin", &source[..end])]);
            let analysis = project.build_hir();
            analysis.completions(&project.source("main.resin"), end);
        }
        for index in 0..source.len() {
            let mut edited = source.to_owned();
            edited.remove(index);
            let project = Project::new(&[("main.resin", &edited)]);
            project.build_hir();
        }
    }
}

#[test]
fn checking_rejects_holes_even_when_given_a_recovered_ast() {
    for source in [
        "def main() = { var value = ; };",
        "def main() = { var value: ; };",
        "def main(point: { x: int }) = { point.; };",
    ] {
        let project = Project::new(&[("main.resin", source)]);
        let analysis = project.build_hir();
        let file = analysis
            .recovered_file(&project.source("main.resin"))
            .unwrap();
        let error = resin_hir::generate(file).unwrap_err();
        assert_eq!(
            error.kind,
            GenerateErrorKind::IncompleteSyntax,
            "{source}: {error}"
        );
        assert!(error.span.start <= error.span.end && error.span.end <= source.len());
    }
}

#[test]
fn indexing_and_shader_artifacts_keep_editor_types_and_completions() {
    let source = "@compute_shader def kernel(invocation: ulong, output: Ptr<uint>) = { var i = uint(invocation); output.* := { i }; }; def main() = { var xs = [1, 2]; var p = xs(0); var code = kernel.spirv; code.length; };";
    let project = Project::new(&[("main.resin", source)]);
    let analysis = project.checked();
    let input = project.source("main.resin");
    assert_eq!(
        analysis
            .hover(&input, source.find("p =").unwrap())
            .unwrap()
            .text,
        "p: Ptr<long>"
    );
    assert_eq!(
        analysis
            .hover(&input, source.rfind("code.length").unwrap())
            .unwrap()
            .text,
        "code: { data: Ptr<ubyte>, length: ulong }"
    );
    let source = "@compute_shader def kernel(invocation: ulong, output: Ptr<uint>) = { var i = uint(invocation); output.* := { i }; }; def main() = { kernel. };";
    let project = Project::new(&[("main.resin", source)]);
    let items = project.build_hir().completions(
        &project.source("main.resin"),
        source.find("kernel. }").unwrap() + 7,
    );
    assert!(
        items
            .iter()
            .any(|i| i.name == "spirv" && i.detail.contains("data: Ptr<ubyte>")),
        "{items:?}"
    );
    let source = "@compute_shader def kernel(invocation: ulong, output: Ptr<uint>) = { var i = uint(invocation); output.* := { i }; }; def main() = { var alias = kernel; alias. };";
    let project = Project::new(&[("main.resin", source)]);
    let items = project.build_hir().completions(
        &project.source("main.resin"),
        source.find("alias. }").unwrap() + 6,
    );
    assert!(items.iter().all(|i| i.name != "spirv"));
}

#[test]
fn suffixes_and_one_armed_if_have_editor_types() {
    let source = "def main() = { var count = 42_ul; if (count > 0_ul) { var speed = 1.5_f; speed; }; count; };";
    let project = Project::new(&[("main.resin", source)]);
    let analysis = project.checked();
    let input = project.source("main.resin");
    assert_eq!(
        analysis
            .hover(&input, source.rfind("count").unwrap())
            .unwrap()
            .text,
        "count: ulong"
    );
    assert_eq!(
        analysis
            .hover(&input, source.rfind("speed").unwrap())
            .unwrap()
            .text,
        "speed: float32"
    );
}

#[test]
fn failed_children_invalidate_composites_without_hiding_later_bindings() {
    for expression in [
        "if (1 == 1) { missing } else { 42 }",
        "[missing, 42]",
        "ok(missing)",
        "place := missing",
    ] {
        let source = format!(
            "def f() = {{ var place = 1; var bad = {expression}; var alias = bad; var healthy = 1.5f; alias; healthy; }};"
        );
        let project = Project::new(&[("main.resin", &source)]);
        let analysis = project.build_hir();
        let input = project.source("main.resin");
        assert!(!analysis.diagnostics().is_empty(), "{source}");
        assert!(analysis.hir().is_err());
        for (name, expected) in [
            ("bad", "bad: ?"),
            ("alias", "alias: ?"),
            ("healthy", "healthy: float32"),
        ] {
            assert_eq!(
                analysis
                    .hover(&input, source.rfind(name).unwrap())
                    .unwrap()
                    .text,
                expected,
                "{source}"
            );
        }
    }
}

#[test]
fn broken_annotations_and_duplicate_declarations_retain_recognizable_children() {
    for (source, name, expected) in [
        (
            "def f() -> { a: _, b: _, c: _, d: Missing } = {}; def later() = { var healthy = 1.5f; healthy; };",
            "healthy",
            "healthy: float32",
        ),
        (
            "def f() = { var duplicate = 1; var duplicate = { var healthy = 1.5f; healthy }; };",
            "healthy",
            "healthy: float32",
        ),
        (
            "type T = int; def f() = { type T = Missing; var value: T; value; };",
            "value",
            "value: ?",
        ),
    ] {
        let project = Project::new(&[("main.resin", source)]);
        let analysis = project.build_hir();
        assert!(!analysis.diagnostics().is_empty());
        assert!(analysis.hir().is_err());
        assert_eq!(
            analysis
                .hover(&project.source("main.resin"), source.rfind(name).unwrap())
                .unwrap()
                .text,
            expected,
            "{source}"
        );
    }
}

#[test]
fn unknown_exports_retain_identity_and_poison_consumers_through_reexports() {
    let library =
        "export { Broken, broken }; type Broken = Missing; def broken() -> _ = { missing };";
    let source = "import { \"left.resin\", \"right.resin\" }; def main() = { var value: Broken; var result = broken(); value; result; };";
    let project = Project::new(&[
        ("main.resin", source),
        ("base.resin", library),
        (
            "left.resin",
            "export { Broken, broken }; import { \"base.resin\" };",
        ),
        (
            "right.resin",
            "export { Broken, broken }; import { \"base.resin\" };",
        ),
    ]);
    let analysis = project.build_hir();
    assert!(analysis.hir().is_err());
    let input = project.source("main.resin");
    for name in ["Broken", "broken"] {
        let offset = source.rfind(name).unwrap();
        let definition = analysis
            .definition(&input, offset)
            .unwrap_or_else(|| panic!("missing {name}: {:?}", analysis.diagnostics()));
        assert_eq!(definition.source, project.source("base.resin"));
        assert_eq!(&library[definition.span.start..definition.span.end], name);
        assert!(
            analysis
                .completions(&input, offset)
                .iter()
                .any(|item| item.name == name)
        );
    }
    assert_eq!(
        analysis
            .hover(&input, source.rfind("result").unwrap())
            .unwrap()
            .text,
        "result: ?"
    );
    assert!(
        analysis
            .diagnostics()
            .iter()
            .all(|d| d.location.source != input || !d.message.contains("Unbound")),
        "{:?}",
        analysis.diagnostics()
    );
}

#[test]
fn callers_cannot_resurrect_failed_result_inference() {
    for (result, body, caller_result, expected) in [
        ("_", "missing", "int", "alias: ?"),
        (
            "_",
            "if (1 == 1) { caller() } else { missing }",
            "int",
            "alias: ?",
        ),
        (
            "_",
            "if (1 == 1) { caller() } else { 1 + () }",
            "int",
            "alias: ?",
        ),
        (
            "Result<int, _>",
            "missing",
            "Result<int, Never>",
            "alias: ?",
        ),
        ("int", "missing", "int", "alias: () -> int"),
    ] {
        let source = format!(
            "def broken() -> {result} = {{ var healthy = 1.5f; healthy; {body} }}; def caller() -> {caller_result} = {{ broken() }}; def observer() = {{ var alias = broken; alias; }};"
        );
        let project = Project::new(&[("main.resin", &source)]);
        let analysis = project.build_hir();
        let input = project.source("main.resin");
        assert!(analysis.program().is_ok(), "{source}");
        assert!(!analysis.diagnostics().is_empty());
        assert!(analysis.hir().is_err());
        assert_eq!(
            analysis
                .hover(&input, source.rfind("alias").unwrap())
                .unwrap()
                .text,
            expected,
            "{source}"
        );
        assert_eq!(
            analysis
                .hover(&input, source.rfind("healthy").unwrap())
                .unwrap()
                .text,
            "healthy: float32"
        );
    }
}

#[test]
fn recursive_failure_discards_copied_caller_result_facts() {
    let source = "def broken() -> _ = { if (1 == 0) { caller() } else { 1 + () } }; def caller() -> _ = { broken() }; def observer() = { var forced = int(caller()); var alias = caller; var failed = broken; alias; failed; };";
    let project = Project::new(&[("main.resin", source)]);
    let analysis = project.build_hir();
    assert!(analysis.program().is_ok());
    assert!(!analysis.diagnostics().is_empty());
    assert!(analysis.hir().is_err());
    let input = project.source("main.resin");
    assert_eq!(
        analysis
            .hover(&input, source.rfind("alias").unwrap())
            .unwrap()
            .text,
        "alias: ?"
    );
    assert_eq!(
        analysis
            .hover(&input, source.rfind("failed").unwrap())
            .unwrap()
            .text,
        "failed: ?"
    );
}

#[test]
fn incomplete_impls_and_method_arguments_keep_editor_recovery() {
    let source = "struct Counter { count: int, def add(counter: Counter, amount: int) -> int = { counter.count + amount }; };  def f(c: Counter) -> int = { c.add(1) };";
    for end in source
        .char_indices()
        .map(|(index, _)| index)
        .chain([source.len()])
    {
        let project = Project::new(&[("main.resin", &source[..end])]);
        project.build_hir();
    }
    let source = "struct Counter { count: int, def read(counter: Counter) -> int = { counter.count }; };  def f(c: Counter) = { c.read(; };";
    let project = Project::new(&[("main.resin", source)]);
    let analysis = project.build_hir();
    let offset = source.rfind("read").unwrap();
    assert!(
        analysis
            .definition(&project.source("main.resin"), offset)
            .is_some()
    );
}

#[test]
fn pointer_replace_has_ordinary_method_hover_and_recovery() {
    for tail in ["", "p.;"] {
        let source = format!("def f(p: Ptr<int>) = {{ p.replace(3); {tail} }};");
        let project = Project::new(&[("main.resin", &source)]);
        let analysis = project.build_hir();
        let input = project.source("main.resin");
        assert_eq!(
            analysis.diagnostics().is_empty(),
            tail.is_empty(),
            "{:?}",
            analysis.diagnostics()
        );
        let offset = source.find("replace").unwrap();
        assert_eq!(
            analysis.hover(&input, offset).unwrap().text,
            "replace: (int) -> int"
        );
        let items = analysis.completions(
            &input,
            if tail.is_empty() {
                offset
            } else {
                source.rfind("p.;").unwrap() + 2
            },
        );
        assert!(
            items
                .iter()
                .any(|item| item.name == "replace"
                    && item.kind == resin_hir::DefinitionKind::Function),
            "{items:?}"
        );
    }
}

#[test]
fn formatted_string_and_literal_string_types_survive_editor_recovery() {
    for (source, members) in [
        (
            "import { \"$/string.resin\" }; def main() = { var text = fmt(\"{0}\", (42,)); text.storage.; };",
            ["get", "downgrade"],
        ),
        (
            "import { \"$/string.resin\" }; def main() = { var text = fmt(\"{0}\", (42,)); text.get().; };",
            ["data", "length"],
        ),
        (
            "def main() = { var text = \"literal\"; text.; };",
            ["data", "length"],
        ),
    ] {
        let project = Project::new(&[("main.resin", source)]);
        let offset = source.rfind(".;").unwrap() + 1;
        let items = project
            .build_hir()
            .completions(&project.source("main.resin"), offset);
        for member in members {
            assert!(items.iter().any(|item| item.name == member), "{items:?}");
        }
    }
}

#[test]
fn invalid_method_arguments_preserve_receiver_facts_and_later_bindings() {
    for duplicate in ["", "def read(self: Missing) -> int = { 0 };"] {
        let source = format!(
            "struct Item {{ count: int, def read(self: Ptr<Item>) -> int = {{ self.count }}; {duplicate} }};  def f(c: Item) = {{ var bad = c.read(missing); var alias = bad; var healthy = 1.5f; alias; healthy; c.; }};"
        );
        let project = Project::new(&[("main.resin", &source)]);
        let analysis = project.build_hir();
        let input = project.source("main.resin");
        assert!(analysis.hir().is_err());
        for (name, expected) in [("alias", "alias: ?"), ("healthy", "healthy: float32")] {
            assert_eq!(
                analysis
                    .hover(&input, source.rfind(name).unwrap())
                    .unwrap()
                    .text,
                expected
            );
        }
        assert_eq!(
            analysis
                .definition(&input, source.find("c.read").unwrap() + 2)
                .unwrap()
                .span
                .start,
            source.find("read(self").unwrap()
        );
        let members = analysis.completions(&input, source.rfind("c.;").unwrap() + 2);
        assert!(members.iter().any(|member| member.name == "read"));
        assert!(members.iter().any(|member| member.detail == "count: int"));
    }
}

#[test]
fn string_constructor_is_an_ordinary_discoverable_static_method() {
    for source in [
        "import { \"$/string.resin\" }; def main() = { var text = String.from_str(\"title\"); text.get().; };",
        "import { \"$/string.resin\" }; def main() = { String.; };",
    ] {
        let project = Project::new(&[("main.resin", source)]);
        let analysis = project.build_hir();
        let offset = source.rfind(".;").unwrap() + 1;
        let items = analysis.completions(&project.source("main.resin"), offset);
        let member = if source.contains("from_str") {
            "length"
        } else {
            "from_str"
        };
        assert!(items.iter().any(|item| item.name == member), "{items:?}");
        if member == "from_str" {
            assert!(
                items.iter().any(|item| item.name == "from_bytes"),
                "{items:?}"
            );
        }
    }
}

#[test]
fn literal_string_hover_preserves_its_distinct_primitive_type() {
    let source = "def main() = { var text = \"literal\"; text; };";
    let project = Project::new(&[("main.resin", source)]);
    let analysis = project.build_hir();
    let hover = analysis
        .hover(&project.source("main.resin"), source.rfind("text").unwrap())
        .unwrap();
    assert_eq!(hover.text, "text: str");
}

#[test]
fn imported_intrinsics_keep_generic_navigation_and_declaration_signatures() {
    let library = r#"export { at }; intrinsic "pointer_index" def at<T>(data: Ptr<T>, length: ulong, index: ulong) -> Ptr<T>;"#;
    let source =
        "import { \"library.resin\" }; def use(data: Ptr<uint>) -> Ptr<uint> = { at(data, 4, 2) };";
    let project = Project::new(&[("main.resin", source), ("library.resin", library)]);
    let analysis = project.checked();
    let input = project.source("main.resin");
    let offset = source.find("at(data").unwrap();
    let hover = analysis.hover(&input, offset).unwrap();
    assert!(hover.text.contains("def at<T>"), "{hover:?}");
    assert!(hover.text.contains("data: Ptr<T>"), "{hover:?}");
    let origin = analysis.definition(&input, offset).unwrap();
    assert_eq!(origin.source, project.source("library.resin"));
    assert_eq!(origin.span.start, library.find("def at").unwrap() + 4);
}
