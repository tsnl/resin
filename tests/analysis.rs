#[allow(dead_code)]
mod support;

use resin_hir::Hir;
use resin_source::normalize_path;
use resin_source::prelude::*;
use std::{collections::BTreeMap, path::Path};
use tempfile::TempDir;

#[test]
fn colon_completion_uses_visible_free_signatures_and_dot_completion_uses_fields() {
    let library = "export { Item, read, unrelated }; struct Item { value: i32 } fn read(value: Ref<Item>) -> i32 { value.value } fn unrelated(value: bool) -> bool { value } fn hidden(value: Ref<Item>) {}";
    for suffix in [":", ":re", ":read()", "."] {
        let source = format!(
            "import {{ \"library.resin\" }}; fn main() -> _ {{ let item = Item {{ value = 42 }}; item{suffix} }}"
        );
        let project = Project::new(&[("main.resin", &source), ("library.resin", library)]);
        let analysis = project.build_hir();
        let input = project.source("main.resin");
        let start = source.rfind("item").unwrap() + 4;
        let offset = start + suffix.find('(').unwrap_or(suffix.len());
        let items = analysis.completions(&input, offset);
        if suffix == "." {
            assert_eq!(
                items
                    .iter()
                    .map(|item| item.name.as_str())
                    .collect::<Vec<_>>(),
                ["value"],
                "{source}\n{items:?}"
            );
        } else {
            assert!(
                items.iter().any(|item| item.name == "read"),
                "{source}\n{items:?}"
            );
            assert!(
                !items
                    .iter()
                    .any(|item| matches!(item.name.as_str(), "value" | "unrelated" | "hidden")),
                "{items:?}"
            );
        }
        if suffix == ":read()" {
            let definition = analysis.definition(&input, start + 1).unwrap();
            assert_eq!(definition.source, project.source("library.resin"));
            assert_eq!(definition.span.start, library.find("fn read").unwrap() + 3);
        }
    }
}

#[test]
fn operator_symbols_navigate_to_the_selected_overload() {
    let library = "export { Number , __add__ }; struct Number<T> { value: T,  }\nfn __add__<T>(a: Number<T>, b: T) -> T  { a.value + b }\n";
    let source = "import { \"library.resin\" }; fn main() -> i32  { Number<i32> { value = 40 } + 2 } fn named(value: Number<i32>) -> i32  { value:__add__(2) }";
    let project = Project::new(&[("main.resin", source), ("library.resin", library)]);
    let analysis = project.checked();
    let input = project.source("main.resin");
    let offset = source.rfind('+').unwrap();
    let definition = analysis.definition(&input, offset).unwrap();
    assert_eq!(definition.source, project.source("library.resin"));
    assert_eq!(definition.span.start, library.rfind("__add__").unwrap());
    assert!(
        analysis
            .hover(&input, offset)
            .unwrap()
            .text
            .contains("(Number<i32>, i32) -> i32")
    );
    let named = source.find("__add__").unwrap();
    assert_eq!(analysis.definition(&input, named).unwrap(), definition);
    assert!(
        analysis
            .completions(&input, named + 3)
            .iter()
            .any(|item| item.name == "__add__")
    );
}

#[test]
fn exported_constants_have_hover_completion_and_definition() {
    let library = "export { answer }; const ( answer: i32 = 42; );";
    let source = "import { \"library.resin\" }; fn main() -> i32  { answer }";
    let project = Project::new(&[("main.resin", source), ("library.resin", library)]);
    let analysis = project.checked();
    let input = project.source("main.resin");
    let offset = source.rfind("answer").unwrap();
    assert_eq!(
        analysis.hover(&input, offset).unwrap().text,
        "const answer: i32"
    );
    let definition = analysis.definition(&input, offset).unwrap();
    assert_eq!(definition.source, project.source("library.resin"));
    assert_eq!(definition.span.start, library.rfind("answer").unwrap());
    let completion = analysis
        .completions(&input, offset)
        .into_iter()
        .find(|item| item.name == "answer")
        .unwrap();
    assert_eq!(completion.kind, resin_hir::DefinitionKind::Constant);
    assert_eq!(completion.detail, "const answer: i32");
}

#[test]
fn sizeof_and_iota_have_builtin_help() {
    let source = "const index = iota; fn size() -> u64  { sizeof(i32) }";
    let project = Project::new(&[("main.resin", source)]);
    let analysis = project.checked();
    let input = project.source("main.resin");
    for name in ["sizeof", "iota"] {
        assert!(
            analysis
                .hover(&input, source.find(name).unwrap())
                .unwrap()
                .text
                .starts_with(name)
        );
    }
}

#[test]
fn gpu_workgroup_size_has_source_method_hover_completion_and_navigation() {
    let source =
        r#"import { "$/gpu.resin" }; fn size(gpu: Gpu) -> u64  { gpu:compute_workgroup_size() }"#;
    let project = Project::new(&[("main.resin", source)]);
    let analysis = project.checked();
    let input = project.source("main.resin");
    let offset = source.find("compute_workgroup_size").unwrap();
    let hover = analysis.hover(&input, offset).unwrap();
    assert_eq!(
        hover.text,
        "fn compute_workgroup_size(self: Ref<Gpu>) -> u64"
    );
    let completions = analysis.completions(&input, offset + "compute_".len());
    let method = completions
        .iter()
        .find(|item| item.name == "compute_workgroup_size")
        .unwrap();
    assert_eq!(method.detail, "compute_workgroup_size: (Ref<Gpu>) -> u64");
    assert_eq!(method.kind, resin_hir::DefinitionKind::Function);
    let origin = analysis.definition(&input, offset).unwrap();
    assert!(origin.source.name().ends_with("gpu.resin"));
    assert_eq!(
        &origin.source.text()[origin.span.start..origin.span.end],
        "compute_workgroup_size"
    );
}

#[test]
fn generic_method_calls_show_substituted_signatures_and_original_definitions() {
    let library = "export { Cell , read, choose }; struct Cell<T> { value: T,   }\nfn read<T>(self: Ptr<Cell<T>>) -> T  { self.value }\n\nfn choose<T, U>(self: Ptr<Cell<T>>, value: U) -> U  { value }\n";
    let source = "import { \"library.resin\" }; type IntCell = Cell<i32>; fn use(cell: Ptr<IntCell>) -> u64  { cell:read(); cell:choose::<_, u64>(42) }";
    let project = Project::new(&[("main.resin", source), ("library.resin", library)]);
    let analysis = project.checked();
    let input = project.source("main.resin");
    for (name, expected) in [
        ("read", "read: (Ptr<Cell<i32>>) -> i32"),
        ("choose", "choose: (Ptr<Cell<i32>>, u64) -> u64"),
    ] {
        let offset = source.find(&format!("cell:{name}")).unwrap() + 5;
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
            library.find(&format!("fn {name}")).unwrap() + 3
        );
    }
}

#[test]
fn unfinished_generic_operation_access_keeps_receiver_substitution_and_other_binders() {
    let library = "export { Cell, read, choose, cell_make }; struct Cell<T> { value: T } fn read<T>(self: Ptr<Cell<T>>) -> T { self.value } fn choose<T, U>(self: Ptr<Cell<T>>, value: U) -> U { value } fn cell_make<T>(value: T) -> Cell<T> { Cell<T> { value = value } }";
    let source = "import { \"library.resin\" }; fn use(cell: Ptr<Cell<i32>>) { cell:; }";
    let project = Project::new(&[("main.resin", source), ("library.resin", library)]);
    let analysis = project.build_hir();
    assert!(!analysis.diagnostics().is_empty());
    let items = analysis.completions(
        &project.source("main.resin"),
        source.find(":;").unwrap() + 1,
    );
    let choose = items.iter().find(|item| item.name == "choose").unwrap();
    assert_eq!(choose.detail, "choose: (Ptr<Cell<i32>>, U) -> U");
    let read = items.iter().find(|item| item.name == "read").unwrap();
    assert_eq!(read.detail, "read: (Ptr<Cell<i32>>) -> i32");
}

#[test]
fn generic_function_references_keep_their_original_declaration() {
    let source = "fn create<T>() -> T { 42 } fn main() -> i32 { let create_int: () -> i32 = create; create_int() }";
    let project = Project::new(&[("main.resin", source)]);
    let analysis = project.checked();
    let input = project.source("main.resin");
    let offset = source.find("= create").unwrap() + 2;
    assert!(
        analysis
            .hover(&input, offset)
            .unwrap()
            .text
            .contains("fn create<T>")
    );
    let origin = analysis.definition(&input, offset).unwrap();
    assert_eq!(origin.span.start, source.find("fn create").unwrap() + 3);
}

#[test]
fn generic_method_editor_snapshots_keep_completed_imported_schemes() {
    let input = Source::new(
        "main.resin",
        "import { \"library.resin\" }; fn use(cell: Ptr<Cell<i32>>) -> _  { cell:read() }",
    );
    let original = Source::new(
        "library.resin",
        "export { Cell , read }; struct Cell<T> { value: T,  }\nfn read<T>(self: Ptr<Cell<T>>) -> T  { self.value }\n",
    );
    let changed = original.with_text("export { Cell , read }; struct Cell<T> { padding: u8, value: T,  }\nfn read<T>(self: Ptr<Cell<T>>) -> i64  { 42 }\n");
    let mut loader = resin_source::Loader::new(Default::default());
    loader
        .set_import(&input, "library.resin", original.clone())
        .unwrap();
    let before = support::frontend::analyze(input.clone(), &mut loader, None);
    loader
        .set_import(&input, "library.resin", changed.clone())
        .unwrap();
    let after = support::frontend::analyze(input.clone(), &mut loader, Some(&before));
    let offset = input.text().find("cell:read").unwrap() + 5;
    for (analysis, library, expected) in [
        (&before, &original, "read: (Ptr<Cell<i32>>) -> i32"),
        (&after, &changed, "read: (Ptr<Cell<i32>>) -> i64"),
        (&before, &original, "read: (Ptr<Cell<i32>>) -> i32"),
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
            library.text().find("fn read").unwrap() + 3
        );
    }
}

#[test]
fn generic_nominal_fields_retain_substitution_and_declaration_navigation() {
    let library = "export { Cell }; struct Cell<T> { value: T, }";
    for (parameters, receiver, result) in [
        ("", "Cell<i32>", "i32"),
        ("", "Ptr<Ptr<Cell<i32>>>", "i32"),
        ("<T>", "Ptr<Ptr<Cell<T>>>", "T"),
    ] {
        let source = format!(
            "import {{ \"library.resin\" }}; fn read{parameters}(cell: {receiver}) -> {result} {{ cell.value }}"
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
    let library = "export { Outer , read }; struct Cell<T> { value: T, } struct Wrapped { cell: Cell<i32>, } struct Outer { wrapped: Wrapped, read: i32,  }\nfn read(self: Ptr<Outer>) -> i32  { self.wrapped.cell.value }\n";
    let source = "import { \"library.resin\" }; fn use(outer: Ptr<Outer>) -> i32  { read(outer); outer:read(); outer.wrapped.cell.value }";
    let project = Project::new(&[("main.resin", source), ("library.resin", library)]);
    let analysis = project.checked();
    let input = project.source("main.resin");
    for (name, expected) in [
        ("wrapped", "wrapped: Wrapped"),
        ("cell", "cell: Cell<i32>"),
        ("value", "value: i32"),
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
    let fields = analysis.completions(&input, source.find("outer.wrapped").unwrap() + 6);
    assert_eq!(
        fields.iter().find(|item| item.name == "read").unwrap().kind,
        resin_hir::DefinitionKind::Field
    );
    let operations = analysis.completions(&input, source.find("outer:read").unwrap() + 6);
    assert_eq!(
        operations
            .iter()
            .find(|item| item.name == "read")
            .unwrap()
            .kind,
        resin_hir::DefinitionKind::Function
    );
}

#[test]
fn generic_field_completion_survives_an_unfinished_access() {
    let source = "struct Cell<T> { value: T, } fn read(cell: Ptr<Cell<i32>>)  { cell.; }";
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
    assert_eq!(fields[0].detail, "value: i32");
    assert!(
        items
            .iter()
            .all(|item| item.kind == resin_hir::DefinitionKind::Field)
    );
}

#[test]
fn generic_field_editor_snapshots_keep_original_imported_declarations() {
    let input = Source::new(
        "main.resin",
        "import { \"library.resin\" }; fn read(cell: Cell<i32>) -> _  { cell.value }",
    );
    let original = Source::new(
        "library.resin",
        "export { Cell }; struct Cell<T> { value: T, }",
    );
    let changed =
        original.with_text("export { Cell }; struct Cell<T> { padding: u8, value: i64, }");
    let mut loader = resin_source::Loader::new(Default::default());
    loader
        .set_import(&input, "library.resin", original.clone())
        .unwrap();
    let before = support::frontend::analyze(input.clone(), &mut loader, None);
    loader
        .set_import(&input, "library.resin", changed.clone())
        .unwrap();
    let after = support::frontend::analyze(input.clone(), &mut loader, Some(&before));
    let field = input.text().rfind("value").unwrap();
    for (analysis, library, expected) in [
        (&before, &original, "value: i32"),
        (&after, &changed, "value: i64"),
        (&before, &original, "value: i32"),
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
    let source = "fn zero() -> i32  { 0 } fn pair(value: (i32, bool))  {} fn add(a: i32, b: i32) -> i32  { a + b } fn main()  { let mut empty = zero; let mut tuple = pair; let mut binary = add; let mut values = (i32(1), 1 == 1); values.0; values.; }";
    let project = Project::new(&[("main.resin", source)]);
    let input = project.source("main.resin");
    let analysis = project.build_hir();
    let members = analysis.completions(&input, source.rfind("values.").unwrap() + 7);
    assert!(
        members.iter().any(|member| member.detail == "0: i32"),
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
        "0: i32"
    );
    for (name, expected) in [
        ("empty", "empty: () -> i32"),
        ("tuple", "tuple: ((i32, bool)) -> ()"),
        ("binary", "binary: (i32, i32) -> i32"),
    ] {
        assert_eq!(
            analysis
                .hover(&input, source.find(&format!("let mut {name}")).unwrap() + 8)
                .unwrap()
                .text,
            expected
        );
    }
}

#[test]
fn imported_generic_aliases_keep_binder_navigation_and_concrete_hover() {
    let library = "export { View }; type View<T> = Ptr<T>;";
    let source = "import { \"library.resin\" }; fn use_view<T>(view: View<T>) -> T  { view.* } fn main() -> i32  { let mut value = 42; let mut pointer: View<i32>; pointer = View<i32>(u64(0)); use_view(pointer) }";
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
    assert_eq!(hover.text, "pointer: Ptr<i32>");
    let library_source = project.source("library.resin");
    let origin = analysis
        .definition(&library_source, library.find("Ptr<T>").unwrap() + 4)
        .unwrap();
    assert_eq!(origin.span.start, library.find("View<T>").unwrap() + 5);
}

#[test]
fn template_scopes_retain_named_types_and_imported_schemes() {
    let library = "export { identity }; fn identity<T>(value: T) -> _  { value }";
    let source = "import { \"library.resin\" }; fn forward<U>(input: U) -> U  { let mut copy = identity(input); copy } fn main() -> i32  { forward(42) }";
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
    let source = "struct Item { count: i32, } fn f(value: Item | None)  { value!.; }";
    let project = Project::new(&[("main.resin", source)]);
    let items = project.build_hir().completions(
        &project.source("main.resin"),
        source.find("value!.").unwrap() + 7,
    );
    assert_eq!(items.len(), 1);
    assert_eq!(items[0].detail, "count: i32");
}

#[test]
fn weak_upgrade_recovery_distinguishes_wrapper_and_payload_members() {
    for (receiver, separator, expected, absent) in [
        ("weak:upgrade()!", ":", "get", "count"),
        ("weak:upgrade()!:get()", ".", "count", "downgrade"),
    ] {
        let source = format!(
            "import {{ \"$/shared.resin\" }}; struct Item {{ count: i32, }} fn f(weak: WeakPtr<Item>)  {{ {receiver}{separator}; }}"
        );
        let project = Project::new(&[("main.resin", &source)]);
        let items = project.build_hir().completions(
            &project.source("main.resin"),
            source.rfind(&format!("{separator};")).unwrap() + 1,
        );
        assert!(items.iter().any(|item| item.name == expected), "{items:?}");
        assert!(!items.iter().any(|item| item.name == absent), "{items:?}");
    }
}

#[test]
fn standard_library_resource_methods_support_editor_navigation_and_recovery() {
    for tail in ["(()) }", "buffer:"] {
        let source = format!(
            r#"import {{ "$/gpu.resin" }};
            fn f() -> (() | Err<_>) {{
                let mut gpu = gpu_new()?;
                let mut bytes = gpu:alloc_in::<u8>(u64(4), memory_default)?;
                let mut commands = gpu:start_command_recording()?;
                let mut buffer = gpu:alloc::<i32>(4)?;
                let element = buffer:at(0);
                element:store(42);
                let mut readable = buffer:read_only();
                {tail}"#
        );
        let mut loader = resin_source::Loader::new(resin_source::library_root());
        let input = loader
            .source_from_text(Path::new("main.resin"), source.clone())
            .unwrap();
        let analysis = support::frontend::analyze(input.clone(), &mut loader, None);
        if tail != "buffer:" {
            assert!(
                analysis.diagnostics().is_empty(),
                "{:?}",
                analysis.diagnostics()
            );
        }
        for (method, result) in [
            ("alloc_in", "-> GpuSpan<u8> | Err<"),
            ("start_command_recording", "GpuCommands | Err<"),
        ] {
            let call = source.find(&format!(":{method}")).unwrap() + 1;
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
        let offset = if tail == "buffer:" {
            source.len()
        } else {
            source.find("at(0)").unwrap()
        };
        let items = analysis.completions(&input, offset);
        for name in ["at", "slice", "read_only", "write_only", "copy_to"] {
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
            "dispatch(compute, HostRoot<_> { value = buffer }, u32(1), u32(1), u32(1))",
            true,
        ),
        (
            "draw(graphics, HostRoot<_> { value = buffer }, u32(3))",
            true,
        ),
        ("draw(empty, None, u32(3))", true),
        (
            "dispatch(graphics, HostRoot<_> { value = buffer }, u32(1), u32(1), u32(1))",
            false,
        ),
        (
            "draw(compute, HostRoot<_> { value = buffer }, u32(3))",
            false,
        ),
        ("dispatch(compute, None, u32(1), u32(1), u32(1))", false),
        ("dispatch(compute, u64(0), u32(1), u32(1), u32(1))", false),
        (
            "dispatch(compute, HostRoot<_> { value = pointer }, u32(1), u32(1), u32(1))",
            false,
        ),
        (
            "dispatch(compute, HostRoot<_> { value = bytes }, u32(1), u32(1), u32(1))",
            false,
        ),
        (
            "dispatch(compute, WrongRoot<_> { other = buffer }, u32(1), u32(1), u32(1))",
            false,
        ),
        ("dispatch(compute, buffer, u32(1), u32(1), u32(1))", false),
        ("draw(graphics, None, u32(3))", false),
        (
            "draw(graphics, HostRoot<_> { value = pointer }, u32(3))",
            false,
        ),
        ("draw(empty, HostRoot<_> { value = buffer }, u32(3))", false),
        ("set_pipeline(compute)", false),
    ] {
        let source = format!(
            r#"import {{ "$/gpu.resin", "$/graphics.resin" }};
            struct Root {{ value: Ptr<i32>, }}
            struct HostRoot<T> {{ value: T, }}
            struct WrongRoot<T> {{ other: T, }}
            @compute_shader
            fn kernel(index: u64, root: Ptr<Root>)  {{}}
            @vertex_shader
            fn vertex(index: i32, root: Ptr<Root>) -> Vertex  {{ rootless(index) }}
            @vertex_shader
            fn rootless(index: i32) -> Vertex  {{
                Vertex {{ position = Position {{ x = f32(0), y = f32(0), z = f32(0), w = f32(1) }},
                    color = Color {{ r = f32(1), g = f32(0), b = f32(0), a = f32(1) }} }}
            }}
            @fragment_shader
            fn fragment(color: Color) -> Color  {{ color }}
            fn f(commands: GpuCommands, gpu: Gpu, buffer: GpuPtr<i32>, bytes: GpuPtr<u8>, pointer: Ptr<i32>) -> (() | Err<_>)  {{
                let mut compute = gpu:create_compute_pipeline(kernel)?;
                let mut graphics = gpu:create_graphics_pipeline(vertex, fragment)?;
                let mut empty = gpu:create_graphics_pipeline(rootless, fragment)?;
                commands:{call}?;
                (())
            }}"#
        );
        let mut loader = resin_source::Loader::new(resin_source::library_root());
        let input = loader
            .source_from_text(Path::new("main.resin"), source)
            .unwrap();
        let analysis = support::frontend::analyze(input.clone(), &mut loader, None);
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
        struct FieldsValuesScale<T0, T1> { values: T0, scale: T1, }
struct Root { values: Span<i32>, scale: i32, }
        @compute_shader
        fn kernel(index: u64, root: Ptr<Root>)  {}
        fn f(gpu: Gpu, commands: GpuCommands, values: GpuSpan<i32>) -> (() | Err<_>)  {
            let mut pipeline = gpu:create_compute_pipeline(kernel)?;
            commands:dispatch(pipeline, FieldsValuesScale<_, _> { values = values, scale = 2 }, 1, 1, 1)?;
            (())
        }"#;
    let mut loader = resin_source::Loader::new(resin_source::library_root());
    let input = loader
        .source_from_text(Path::new("main.resin"), source)
        .unwrap();
    let analysis = support::frontend::analyze(input.clone(), &mut loader, None);
    assert!(
        analysis.diagnostics().is_empty(),
        "{:?}",
        analysis.diagnostics()
    );
    for (method, contract) in [
        ("create_compute_pipeline", "GpuComputePipeline<Root,"),
        ("dispatch", "GpuSpan<i32>"),
    ] {
        let call = source.find(&format!(":{method}")).unwrap() + 1;
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
    let source =
        "import { \"$/span.resin\" }; fn main (value: Ptr<Span<i32>>) -> Ptr<Span<i32>>  { value }";
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
        "value: Ptr<Span<i32>>"
    );

    let source = "type Number = i32; fn main () -> ()  { let mut value: Ptr<Num>; }";
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
fn free_operations_have_navigation_hover_and_receiver_completion() {
    let library = "export { Counter, counter_new, read }; struct Counter { count: i32 } fn counter_new() -> Counter { Counter { count = 7 } } fn read(self: Ref<Counter>) -> i32 { self.count }";
    for tail in ["", "c:;", "c.;"] {
        let source = format!(
            "import {{ \"lib.resin\" }}; fn main() {{ let c = counter_new(); c:read(); {tail} }}"
        );
        let project = Project::new(&[("main.resin", &source), ("lib.resin", library)]);
        let analysis = project.build_hir();
        let input = project.source("main.resin");
        assert_eq!(
            analysis.diagnostics().is_empty(),
            tail.is_empty(),
            "{:?}",
            analysis.diagnostics()
        );
        for name in ["counter_new", "read"] {
            let call = source.find(&format!("{name}()")).unwrap();
            let definition = analysis.definition(&input, call).unwrap();
            assert_eq!(definition.source, project.source("lib.resin"));
            assert_eq!(
                definition.span.start,
                library.find(&format!("fn {name}")).unwrap() + 3
            );
            assert!(
                analysis
                    .hover(&input, call)
                    .unwrap()
                    .text
                    .starts_with(&format!("fn {name}("))
            );
            assert!(
                analysis
                    .completions(&input, call)
                    .iter()
                    .any(|item| item.name == name)
            );
        }
        let global = analysis.completions(&input, source.find("let c").unwrap());
        assert!(global.iter().any(|item| item.name == "read"));
        assert!(global.iter().any(|item| item.name == "counter_new"));
        if !tail.is_empty() {
            let items = analysis.completions(&input, source.rfind(tail).unwrap() + 2);
            assert!(
                items
                    .iter()
                    .any(|item| item.name == if tail == "c:;" { "read" } else { "count" }),
                "{items:?}"
            );
            assert!(!items.iter().any(|item| item.name == "counter_new"));
        }
    }
}

#[test]
fn at_indexing_has_hover_and_completion_in_valid_and_incomplete_code() {
    for receiver in ["values", "holder.values"] {
        for tail in ["", " values:;", " holder.values:;", " holder.values:at(; "] {
            let source = format!(
                "import {{ \"$/span.resin\" }}; struct FieldsValues<T0> {{ values: T0, }}\nfn main()  {{ let mut values = [i32(1), i32(2)]; let mut holder = FieldsValues<_> {{ values = Span<i32> {{ data = Ptr<i32>(u64(0)), length = u64(2) }} }}; {receiver}:at_mut(0) = 3;{tail} }}"
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
            let offset = source.find(":at(0)").unwrap() + 1;
            assert_eq!(
                analysis.hover(&input, offset).unwrap().text,
                if receiver == "values" {
                    "at: (Ref<[i32; 2]>, u64) -> Ref<i32>"
                } else {
                    "at: (Ref<Span<i32>>, u64) -> Ref<i32>"
                }
            );
            let items = analysis.completions(&input, offset);
            assert!(
                items
                    .iter()
                    .any(|item| item.name == "at"
                        && item.kind == resin_hir::DefinitionKind::Function)
            );
            if !tail.is_empty() {
                let offset = source.rfind(":;").or_else(|| source.rfind(":at(")).unwrap() + 1;
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
    let library = "export { Counter , drop, read }; struct Counter { count: i32,   }\nfn drop(self: RefMut<Counter>)  {}\n\nfn read(self: Ref<Counter>) -> i32  { self.count }\n ";
    for tail in ["", "c:get().*:;"] {
        let source = format!(
            "import {{ \"lib.resin\", \"$/shared.resin\" }}; fn f(c: ArcPtr<Counter>)  {{ c:get().*:read(); {tail} }}"
        );
        let project = Project::new(&[("main.resin", &source), ("lib.resin", library)]);
        let analysis = project.build_hir();
        let input = project.source("main.resin");
        let call = source.find(":read").unwrap() + 1;
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
                .starts_with("fn read(")
        );
        let offset = if tail.is_empty() {
            call
        } else {
            source.rfind(":;").unwrap() + 1
        };
        let items = analysis.completions(&input, offset);
        assert!(items.iter().any(|item| item.name == "read"));
        assert!(items.iter().any(|item| item.name == "drop"));
    }
}

#[test]
fn inferred_errors_and_match_payloads_have_editor_types() {
    let source = "struct Broken { code: i32, } fn fail() -> (i32 | Err<_>)  { Err(Broken { code = 7 }) } fn main()  { let mut result = fail(); match (result) { i32(value) => { value; }, Err(error) => { error.code; } }; }";
    let project = Project::new(&[("main.resin", source)]);
    let analysis = project.checked();
    let input = project.source("main.resin");
    assert_eq!(
        analysis
            .hover(&input, source.find("match (result").unwrap() + 7)
            .unwrap()
            .text,
        "result: i32 | Err<Broken>"
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
            .any(|field| field.name == "code" && field.detail == "code: i32")
    );
    let origin = analysis
        .definition(&input, source.find("error.code").unwrap())
        .unwrap();
    assert_eq!(origin.span.start, source.find("Err(error)").unwrap() + 4);
    assert!(
        analysis
            .completions(&input, source.find("let mut result").unwrap())
            .iter()
            .all(|item| item.name != "error")
    );
}

#[test]
fn inferred_imported_results_and_local_annotations_support_editor_queries() {
    let source =
        "import { \"lib.resin\" }; fn main()  { let mut value: _; value = make(); value.count; }";
    let library = "export { make }; struct Counter { count: i32, } fn make() -> _  { Counter { count = 42 } }";
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
            .any(|field| field.name == "count" && field.detail == "count: i32")
    );
    let definition = analysis
        .definition(&input, source.find("make()").unwrap())
        .unwrap();
    assert_eq!(definition.source, project.source("lib.resin"));
}

#[test]
fn inference_does_not_publish_speculative_type_references() {
    let source = "type Value = i32; fn f() -> _  { type Value = bool; type Wrapper = Value; Wrapper(Value(1 == 1)) }";
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
            "if (index < u64(root.width)) { root.pixels:at_mut(index) = u32(0); };",
            "let mut later = root.width; later;",
            "while (false) { root.width; };",
            "root.width;",
            "",
        ] {
            for preceding in ["", "let mut earlier = root.height;"] {
                let source = format!(
                    "// é🌲\nimport {{ \"$/span.resin\" }}; struct Root {{ width: u32, height: u32, pixels: Span<u32>, }}\n@compute_shader fn kernel(index: u64, root: Ptr<Root>)  {{\n{preceding}\n{receiver}.\n{following}\n}}"
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
                if following.starts_with("let mut later") {
                    assert_eq!(
                        analysis
                            .hover(&input, source.rfind("later;").unwrap())
                            .unwrap_or_else(|| panic!(
                                "{source}\n{}\n{:#?}",
                                support::frontend::cst(&source, None)
                                    .tree()
                                    .root_node()
                                    .to_sexp(),
                                analysis.recovered_file(&input)
                            ))
                            .text,
                        "later: u32"
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
        ("let mut value = Fields { count = 1, label = 2 };", "value"),
        (
            "let mut record = Fields { count = 1, label = 2 }; let mut value = &record;",
            "value",
        ),
        (
            "let mut value = Outer { inner = Fields { count = 1, label = 2 } };",
            "value.inner",
        ),
        (
            "let mut value = Fields { count = 1, label = 2 };",
            "(value)",
        ),
    ] {
        for field in ["", "co", "count"] {
            let source = format!(
                "struct Fields {{ count: i64, label: i64, }} struct Outer {{ inner: Fields, }} fn main ()  {{ {setup} {receiver}.{field}; }}"
            );
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
            assert_eq!(items[0].detail, "count: i64");
            assert_eq!(items[0].kind, resin_hir::DefinitionKind::Field);
            assert_eq!(items[0].replace.start, start);
            assert_eq!(items[0].replace.end, start + field.len());
        }
    }
}

#[test]
fn field_completion_resolves_imported_nominal_function_results() {
    let source = "import { \"lib.resin\" }; fn main ()  { make().; }";
    let project = Project::new(&[
        ("main.resin", source),
        (
            "lib.resin",
            "export { make }; struct Counter { count: i32, } fn make () -> Counter  { Counter { count = 0 } }",
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
        let source = format!(
            "struct FieldsCount<T0> {{ count: T0, }}\nfn main ()  {{ let mut value = FieldsCount<_> {{ count = 1 }}; {expression}\n }}"
        );
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
        "struct Point { x: f32, y: f32, } fn main ()  { let mut point: Point; point.; }",
        "struct FieldsXY<T0, T1> { x: T0, y: T1, }\nfn main (point: FieldsXY<f32, f32>) { point.",
        "struct FieldsXY<T0, T1> { x: T0, y: T1, }\nfn main () { let mut point = FieldsXY<_, _> { x = 1, y = 2 }; point.",
    ] {
        let project = Project::new(&[("main.resin", source)]);
        let analysis = project.build_hir();
        let items = analysis.completions(
            &project.source("main.resin"),
            source.rfind('.').unwrap() + 1,
        );
        assert_eq!(
            items
                .iter()
                .map(|item| item.name.as_str())
                .collect::<Vec<_>>(),
            ["x", "y"],
            "{source}\n{:#?}",
            analysis.recovered_file(&project.source("main.resin"))
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
        support::frontend::analyze(self.source("main.resin"), &mut loader, None)
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
    let source = "fn main (argument: i32) -> i32  { let mut value = argument + 1; value }";
    let project = Project::new(&[("main.resin", source)]);
    let analysis = project.checked();
    let input = project.source("main.resin");
    assert_eq!(
        analysis
            .hover(&input, source.rfind("value").unwrap())
            .unwrap()
            .text,
        "value: i32"
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
    let source = "import { \"left.resin\", \"right.resin\" }; fn main () -> Number  { make() }";
    let project = Project::new(&[
        ("main.resin", source),
        (
            "base.resin",
            "export { Number, make }; type Number = i32; fn make () -> Number  { Number(42) } fn secret () -> i32  { 1 }",
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
    let source = "fn main () -> i32  { let mut value = 1; let mut inner = { let mut value = 2; value }; value }";
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
        "fn main (argument: i32) -> i32  { let mut local = 1; arg }",
        "fn main (argument: i32) -> i32 { let mut local = 1; arg",
        "fn main (argument: i32) -> i32  { let mut local = 1; print(arg }",
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
    let source = "import { \"a.resin\", \"b.resin\" }; fn main () -> i32  { answer }";
    let project = Project::new(&[
        ("main.resin", source),
        ("a.resin", "export { answer }; const answer: i32 = 1;"),
        ("b.resin", "export { answer }; const answer: i32 = 2;"),
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
    let source = "import { \"missing.resin\", \"available.resin\" }; fn main()  { let mut point = make(); point.x; }";
    let project = Project::new(&[
        ("main.resin", source),
        (
            "available.resin",
            "export { make }; struct Point { x: i32, } fn make() -> Point  { Point { x = 1 } }",
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
        fields[0].detail, "x: i32",
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
        ("a.resin", "export { value }; fn value () -> i32  { + }"),
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
        r#"extern { "bad\q": { fn release(); } };"#,
        "extern { \"unfinished: { fn release(); } };",
        "extern { { fn release(); } };",
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
    let source = "extern { \"native.h\": { fn releasex: i32); } }; fn main()  {}";
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
    let source = "type Number = i32; fn main () -> ()  { let mut value = 1; let mut other: Num; }";
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
    for source in ["// val", "fn main () -> ()  { let mut value = \"val\"; }"] {
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
    let source = "fn main (value: i32) -> i32 { let value = 2; val }";
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
        ("fn main (value: Num) -> () {} type Number = i32;", true),
        ("type Earlier = Num; type Number = i32;", false),
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
    let source = "export { run }; fn run () -> i32  { helper() } fn helper () -> i32  { 1 }";
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
    let project = Project::new(&[("main.resin", "export {}; fn helper () -> i32  { 1 }")]);
    assert!(project.build_hir().hir().unwrap().entries.is_empty());

    for source in [
        "let mut global = 1; fn main () -> i32  { glo }",
        "let mut global: i32; fn main () -> i32  { glo }",
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
    let source = "extern { \"native.h\": { fn release(value: i32); } }; fn run()  { release(1); }";
    let project = Project::new(&[("main.resin", source)]);
    let analysis = project.checked();
    let input = project.source("main.resin");
    for (name, signature) in [
        ("release", "fn release(value: i32) -> ()"),
        ("run", "fn run() -> ()"),
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
    let project = Project::new(&[("main.resin", "fn run() { let mut value = 1; ")]);
    let items = project.build_hir().completions(
        &project.source("main.resin"),
        "fn run() { let mut value = 1; ".len(),
    );
    for keyword in ["fn", "let", "type"] {
        assert!(items.iter().any(|item| item.name == keyword), "{items:?}");
    }
    let project = Project::new(&[("main.resin", "fn run()  { 1 }")]);
    assert!(
        !project.build_hir().diagnostics().is_empty(),
        "unit returns are not inferred from bodies"
    );
}

#[test]
fn holes_preserve_later_locals_and_functions_without_producing_ir() {
    for broken in [
        "let mut broken = ;",
        "let mut broken: ;",
        "let mut broken = 1 + ;",
        "unknown_name;",
        "let mut broken = missing(1);",
    ] {
        let source = format!(
            "struct FieldsXY<T0, T1> {{ x: T0, y: T1, }}\nfn first()  {{ {broken} let mut point = FieldsXY<_, _> {{ x = 1, y = 2 }}; point.; }} fn later(arg: i32) -> i32  {{ let mut result = arg; result }}"
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
        assert_eq!(hover.text, "result: i32", "{source}");
    }
}

#[test]
fn unknown_bindings_shadow_outer_values_without_fabricating_types() {
    for initializer in ["", "missing(1)", "1 + (1 == 1)"] {
        let source = format!(
            "struct FieldsX<T0> {{ x: T0, }}\nstruct FieldsCount<T0> {{ count: T0; }}\nfn main(point: FieldsX<i32>)  {{ let mut point = {initializer}; let mut alias = point; alias.; let mut healthy = FieldsCount<_> {{ count = 42 }}; healthy.count; }}"
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
            source.find("let mut point").unwrap() + 8
        );
        assert!(analysis.hir().is_err());
        let fields = analysis.completions(&input, source.rfind("count").unwrap());
        assert_eq!(
            fields
                .iter()
                .map(|item| item.detail.as_str())
                .collect::<Vec<_>>(),
            ["count: i64"],
            "{source}"
        );
    }
}

#[test]
fn unrelated_errors_preserve_expression_types_and_field_completion() {
    let mut failures = Vec::new();
    for (setup, expected) in [
        ("let mut value: i8 = -128;", "i8"),
        ("let mut value: f32 = 42;", "f32"),
        (
            "let mut unused: i32; let mut value = size_of(unused);",
            "u64",
        ),
        (
            "let mut value: (i32 | Err<Never>); value = (42);",
            "i32 | Err<Never>",
        ),
    ] {
        for broken in ["", "fn broken()  { missing; }"] {
            let source = format!(
                "{broken} struct FieldsPayload<T0> {{ payload: T0, }}\nfn main()  {{ {setup} value; let mut record = FieldsPayload<_> {{ payload = value }}; record.payload; }}"
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
    let source = "struct FieldsFirstSecond<T0, T1> { first: T0, second: T1, }\nfn main()  { let mut value: _; let mut broken: FieldsFirstSecond<i32, bool>; broken = FieldsFirstSecond<_, _> { first = value, second = 0 }; value = f32(1.5); value; }";
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
        "value: f32"
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
    let source = "import { \"lib.resin\" }; fn main()  { let mut broken = ; let mut point = make(); point.; }";
    let project = Project::new(&[
        ("main.resin", source),
        (
            "lib.resin",
            "export { make }; struct Point { x: f32, } fn broken()  { let mut hole = ; } fn make() -> Point  { Point { x = 1 } }",
        ),
    ]);
    let analysis = project.build_hir();
    let items = analysis.completions(
        &project.source("main.resin"),
        source.find("point.").unwrap() + 6,
    );
    assert_eq!(items.len(), 1);
    assert_eq!(items[0].detail, "x: f32");
    assert!(analysis.hir().is_err());
}

#[test]
fn recovered_ast_contains_expression_type_and_field_holes() {
    let source = "struct FieldsX<T0> { x: T0, }\nfn main()  { let mut value = ; let mut typed: ; let mut point = FieldsX<_> { x = 1 }; point.; }";
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
        "export { main }; struct Point { x: i32, } fn main(arg: Ptr<Point>)  { let mut value = arg.x + 1; print(fmt(\"{}\", value)); }",
        "struct FieldsLeftRight<T0, T1> { left: T0, right: T1, }\nfn main(arg: i32) -> i32  { let mut pair = FieldsLeftRight<_, _> { left = arg, right = 1 }; if (arg == 0) (pair.left) else (pair.right) }",
        "fn main()  { let mut values = [1, 2]; while (1 == 1) { let mut missing: Ptr<i32>; }; }",
        "struct Cleanup { value: Ptr<i32>,  }\nfn drop(self: RefMut<Cleanup>)  { self.value.* = 42; }\n  fn main()  { let mut n = 0; let mut cleanup = Cleanup { value = &n }; }",
        "struct Item { value: i32, } fn main()  { let mut owner = ArcPtr<Item> { value = 42 }; let mut weak = owner:downgrade(); match (weak:upgrade()) { ArcPtr<Item>(item) => { item.value; }, None => {} }; }",
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
        "fn main()  { let mut value = ; }",
        "fn main()  { let mut value: ; }",
        "struct FieldsX<T0> { x: T0, }\nfn main(point: FieldsX<i32>)  { point.; }",
    ] {
        let project = Project::new(&[("main.resin", source)]);
        let analysis = project.build_hir();
        let file = analysis
            .recovered_file(&project.source("main.resin"))
            .unwrap();
        let error = support::frontend::check_hir(&resin_ast::Program {
            modules: vec![resin_ast::SourceModule {
                source: project.source("main.resin"),
                file: std::sync::Arc::new(file.clone()),
                imports: vec![],
            }],
        })
        .into_module()
        .unwrap_err();
        assert!(
            error.diagnostic.contains("IncompleteSyntax"),
            "{source}: {error}"
        );
        let span = error.span.expect("hole has a span");
        assert!(span.start <= span.end && span.end <= source.len());
    }
}

#[test]
fn indexing_keeps_editor_types_and_shader_functions_have_no_bytecode_property() {
    let source = "fn main()  { let mut xs = [1, 2]; let mut p = xs(0); }";
    let project = Project::new(&[("main.resin", source)]);
    let analysis = project.checked();
    let input = project.source("main.resin");
    assert_eq!(
        analysis
            .hover(&input, source.find("p =").unwrap())
            .unwrap()
            .text,
        "p: i64"
    );
    for body in ["kernel.", "let mut alias = kernel; alias."] {
        let source = format!(
            "@compute_shader fn kernel(index: u64, output: Ptr<u32>)  {{}} fn main()  {{ {body} }}"
        );
        let project = Project::new(&[("main.resin", &source)]);
        let items = project.build_hir().completions(
            &project.source("main.resin"),
            source.rfind('.').unwrap() + 1,
        );
        assert!(items.iter().all(|item| item.name != "spirv"), "{items:?}");
    }
}

#[test]
fn contextual_literals_and_one_armed_if_have_editor_types() {
    let source = "fn main()  { let mut count: u64 = 42; if (count > u64(0)) { let mut speed: f32 = 1.5; speed; }; count; }";
    let project = Project::new(&[("main.resin", source)]);
    let analysis = project.checked();
    let input = project.source("main.resin");
    assert_eq!(
        analysis
            .hover(&input, source.rfind("count").unwrap())
            .unwrap()
            .text,
        "count: u64"
    );
    assert_eq!(
        analysis
            .hover(&input, source.rfind("speed").unwrap())
            .unwrap()
            .text,
        "speed: f32"
    );
}

#[test]
fn failed_children_invalidate_composites_without_hiding_later_bindings() {
    for expression in [
        "if (1 == 1) { missing } else { 42 }",
        "[missing, 42]",
        "(missing)",
        "place = missing",
    ] {
        let source = format!(
            "fn f()  {{ let mut place = 1; let mut bad = {expression}; let mut alias = bad; let mut healthy: f32 = 1.5; alias; healthy; }}"
        );
        let project = Project::new(&[("main.resin", &source)]);
        let analysis = project.build_hir();
        let input = project.source("main.resin");
        assert!(!analysis.diagnostics().is_empty(), "{source}");
        assert!(analysis.hir().is_err());
        for (name, expected) in [
            ("bad", "bad: ?"),
            ("alias", "alias: ?"),
            ("healthy", "healthy: f32"),
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
            "struct FieldsABCD<T0, T1, T2, T3> { a: T0, b: T1, c: T2, d: T3, }\nfn f() -> FieldsABCD<_, _, _, Missing>  {} fn later()  { let mut healthy: f32 = 1.5; healthy; }",
            "healthy",
            "healthy: f32",
        ),
        (
            "fn f()  { let mut duplicate = 1; let mut duplicate = { let mut healthy: f32 = 1.5; healthy }; }",
            "healthy",
            "healthy: f32",
        ),
        (
            "type T = i32; fn f()  { type T = Missing; let mut value: T; value; }",
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
    let library = "export { Broken, broken }; type Broken = Missing; fn broken() -> _  { missing }";
    let source = "import { \"left.resin\", \"right.resin\" }; fn main()  { let mut value: Broken; let mut result = broken(); value; result; }";
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
        ("_", "missing", "i32", "alias: ?"),
        (
            "_",
            "if (1 == 1) { caller() } else { missing }",
            "i32",
            "alias: ?",
        ),
        (
            "_",
            "if (1 == 1) { caller() } else { 1 + () }",
            "i32",
            "alias: ?",
        ),
        ("(i32 | Err<_>)", "missing", "i32 | Err<Never>", "alias: ?"),
        ("i32", "missing", "i32", "alias: () -> i32"),
    ] {
        let source = format!(
            "fn broken() -> {result}  {{ let mut healthy: f32 = 1.5; healthy; {body} }} fn caller() -> {caller_result}  {{ broken() }} fn observer()  {{ let mut alias = broken; alias; }}"
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
            "healthy: f32"
        );
    }
}

#[test]
fn recursive_failure_discards_copied_caller_result_facts() {
    let source = "fn broken() -> _  { if (1 == 0) { caller() } else { 1 + () } } fn caller() -> _  { broken() } fn observer()  { let mut forced = i32(caller()); let mut alias = caller; let mut failed = broken; alias; failed; }";
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
    let source = "struct Counter { count: i32,  }\nfn add(counter: Counter, amount: i32) -> i32  { counter.count + amount }\n  fn f(c: Counter) -> i32  { c:add(1) }";
    for end in source
        .char_indices()
        .map(|(index, _)| index)
        .chain([source.len()])
    {
        let project = Project::new(&[("main.resin", &source[..end])]);
        project.build_hir();
    }
    let source = "struct Counter { count: i32,  }\nfn read(counter: Counter) -> i32  { counter.count }\n  fn f(c: Counter)  { c:read(; }";
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
    for tail in ["", "p:;"] {
        let source = format!("fn f(p: Ptr<i32>)  {{ p:replace(3); {tail} }}");
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
            "replace: (Ptr<i32>, i32) -> i32"
        );
        let items = analysis.completions(
            &input,
            if tail.is_empty() {
                offset
            } else {
                source.rfind("p:;").unwrap() + 2
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
            "import { \"$/string.resin\" }; fn main()  { let mut text = fmt(\"{0}\", (42,)); text.storage:; }",
            ["get", "clone"],
        ),
        (
            "import { \"$/string.resin\" }; fn main()  { let mut text = fmt(\"{0}\", (42,)); text:get().; }",
            ["data", "length"],
        ),
        (
            "fn main()  { let mut text = \"literal\"; text.; }",
            ["data", "length"],
        ),
    ] {
        let project = Project::new(&[("main.resin", source)]);
        let offset = source.rfind(".;").or_else(|| source.rfind(":;")).unwrap() + 1;
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
    for duplicate in ["", "fn read(self: Missing) -> i32  { 0 }"] {
        let source = format!(
            "struct Item {{ count: i32 }} fn read(self: Ref<Item>) -> i32 {{ self.count }} {duplicate} fn f(c: Item) {{ let bad = c:read(missing); let alias = bad; let healthy: f32 = 1.5; alias; healthy; c.; c:; }}"
        );
        let project = Project::new(&[("main.resin", &source)]);
        let analysis = project.build_hir();
        let input = project.source("main.resin");
        assert!(analysis.hir().is_err());
        for (name, expected) in [("alias", "alias: ?"), ("healthy", "healthy: f32")] {
            assert_eq!(
                analysis
                    .hover(&input, source.rfind(name).unwrap())
                    .unwrap()
                    .text,
                expected
            );
        }
        let fields = analysis.completions(&input, source.rfind("c.;").unwrap() + 2);
        assert!(fields.iter().any(|member| member.detail == "count: i32"));
        let operations = analysis.completions(&input, source.rfind("c:;").unwrap() + 2);
        assert!(operations.iter().any(|member| member.name == "read"));
    }
}

#[test]
fn string_constructor_is_an_ordinary_discoverable_free_function() {
    for source in [
        "import { \"$/string.resin\" }; fn main()  { let mut text = string_from_str(\"title\"); text:get().; }",
        "import { \"$/string.resin\" }; fn main() { string_; }",
    ] {
        let project = Project::new(&[("main.resin", source)]);
        let analysis = project.build_hir();
        let offset = source
            .rfind(".;")
            .map(|at| at + 1)
            .unwrap_or_else(|| source.find("string_;").unwrap() + 7);
        let items = analysis.completions(&project.source("main.resin"), offset);
        let member = if source.contains("string_from_str") {
            "length"
        } else {
            "string_from_str"
        };
        assert!(items.iter().any(|item| item.name == member), "{items:?}");
        if member == "string_from_str" {
            assert!(
                items.iter().any(|item| item.name == "string_from_bytes"),
                "{items:?}"
            );
        }
    }
}

#[test]
fn literal_string_hover_preserves_its_distinct_primitive_type() {
    let source = "fn main()  { let mut text = \"literal\"; text; }";
    let project = Project::new(&[("main.resin", source)]);
    let analysis = project.build_hir();
    let hover = analysis
        .hover(&project.source("main.resin"), source.rfind("text").unwrap())
        .unwrap();
    assert_eq!(hover.text, "text: str");
}

#[test]
fn imported_intrinsics_keep_generic_navigation_and_declaration_signatures() {
    let library = r#"export { at }; intrinsic "pointer_index" fn at<T>(data: Ptr<T>, length: u64, index: u64) -> Ptr<T>;"#;
    let source =
        "import { \"library.resin\" }; fn use(data: Ptr<u32>) -> Ptr<u32>  { at(data, 4, 2) }";
    let project = Project::new(&[("main.resin", source), ("library.resin", library)]);
    let analysis = project.checked();
    let input = project.source("main.resin");
    let offset = source.find("at(data").unwrap();
    let hover = analysis.hover(&input, offset).unwrap();
    assert_eq!(hover.text, "at: (Ptr<u32>, u64, u64) -> Ptr<u32>");
    let origin = analysis.definition(&input, offset).unwrap();
    assert_eq!(origin.source, project.source("library.resin"));
    assert_eq!(origin.span.start, library.find("fn at").unwrap() + 3);
}

#[test]
fn lea_completion_requires_addressable_element_storage() {
    let source = r#"import { "$/shared.resin", "$/span.resin" };
        fn main() -> () | Err<_> {
            let values = [i32(1), i32(2)];
            let owner = arc_ptr_alloc(values)?;
            let pointer = owner:get();
            values:at(u64(0));
            pointer:lea(u64(0));
        }
    "#;
    let project = Project::new(&[("main.resin", source)]);
    let analysis = project.checked();
    let input = project.source("main.resin");
    for (receiver, available) in [("values:at", false), ("pointer:lea", true)] {
        let offset = source.find(receiver).unwrap() + receiver.find(':').unwrap() + 1;
        assert_eq!(
            analysis
                .completions(&input, offset)
                .iter()
                .any(|item| item.name == "lea"),
            available
        );
    }
    let offset = source.find("pointer:lea").unwrap() + "pointer:".len();
    assert_eq!(
        analysis.hover(&input, offset).unwrap().text,
        "lea: (Ptr<[i32; 2]>, u64) -> Ptr<i32>"
    );
}

#[test]
fn imported_declarations_fields_and_colon_calls_keep_documentation() {
    let library = "//! Library docs.\nexport { Item, read };\n/// An item.\nstruct Item {\n/// Its value.\nvalue: i32, }\n/// Read **without moving**.\nfn read(item: Ref<Item>) -> i32 { item.value }";
    let source = "import { \"library.resin\" }; fn main() -> i32 { let item = Item { value = 42 }; item:read() + item.value }";
    let project = Project::new(&[("main.resin", source), ("library.resin", library)]);
    let hir = project.build_hir();
    hir.hir().unwrap();
    let input = project.source("main.resin");
    for (needle, expected) in [
        ("Item {", "An item."),
        ("read()", "Read **without moving**."),
        ("value }", "Its value."),
    ] {
        let hover = hir.hover(&input, source.rfind(needle).unwrap()).unwrap();
        assert_eq!(hover.documentation, expected, "{needle}: {hover:?}");
    }
    let library_source = project.source("library.resin");
    let declaration = hir
        .hover(&library_source, library.find("fn read").unwrap() + 3)
        .unwrap();
    assert_eq!(declaration.documentation, "Read **without moving**.");
}
