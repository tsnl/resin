use resin_hir::GenerateErrorKind;
use resin_source::prelude::*;
use resin_types::prelude::*;
use std::{fs, process::Output};
use support::pipeline;
use tempfile::TempDir;

mod support;

struct Project(TempDir);

impl Project {
    fn new(files: &[(&str, &str)]) -> Self {
        let project = Self(TempDir::new_in(std::env::temp_dir()).unwrap());
        for (name, source) in files {
            let path = project.0.path().join(name);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, source).unwrap();
        }
        project
    }

    fn compile(&self) -> Result<resin_lir::Module, SourceError> {
        pipeline::file_module(&self.0.path().join("main.resin"))
    }

    fn run(&self) -> Output {
        let module = self.compile().unwrap();
        support::project::Project::new(&module, Some("main"))
            .unwrap()
            .run()
    }

    fn error(&self, expected: &str) -> String {
        let error = self.compile().unwrap_err().to_string();
        assert!(error.contains(expected), "{error}");
        error
    }
}

#[test]
fn ast_preserves_exports_imports_and_their_spans() {
    let source =
        "export { answer, Box, }; import { \"a.resin\", \"b.resin\", }; fn answer() -> int  { 42 }";
    let file = support::parse(source);
    assert_eq!(
        file.exports
            .iter()
            .map(|n| n.val.as_ref())
            .collect::<Vec<_>>(),
        ["answer", "Box"]
    );
    assert_eq!(
        file.imports
            .iter()
            .map(|p| p.val.as_ref())
            .collect::<Vec<_>>(),
        ["a.resin", "b.resin"]
    );
    assert_eq!(
        &source[file.exports[0].span.start..file.exports[0].span.end],
        "answer"
    );
    assert_eq!(file.stmts.len(), 1);
    assert!(resin_ast::format_source(&file).contains("(export answer Box)"));
    assert!(
        Project::new(&[("main.resin", source)])
            .compile()
            .unwrap_err()
            .to_string()
            .contains("a.resin")
    );
    let project = Project::new(&[("main.resin", "export { value }; fn value() -> int  { 1 }")]);
    let program = pipeline::load(&project.0.path().join("main.resin")).unwrap();
    let output = resin_ast::format_program(&program);
    assert!(output.starts_with("(program"), "{output}");
    assert!(output.contains("main.resin\""), "{output}");
}

#[test]
fn private_helpers_and_types_are_resolved_in_their_own_files() {
    let project = Project::new(&[
        (
            "left.resin",
            "export { left }; type Item = int; fn helper(value: Item) -> int  { int(value) } fn left() -> int  { helper(Item(20)) }",
        ),
        (
            "right.resin",
            "export { right }; type Item = int; fn helper(value: Item) -> int  { int(value) } fn right() -> int  { helper(Item(22)) }",
        ),
        (
            "main.resin",
            "export { main }; import { \"left.resin\", \"right.resin\" }; fn helper () -> int  { 99 } fn main () -> int  { left() + right() }",
        ),
    ]);
    assert_eq!(project.run().status.code(), Some(42));
}

#[test]
fn private_names_are_not_visible_to_consumers() {
    for (exports, declaration, use_site, error) in [
        (
            "",
            "fn hidden() -> int  { 1 }",
            "fn main() -> ()  { hidden(); }",
            "UnboundValue",
        ),
        (
            "export {};",
            "fn hidden () -> int  { 1 }",
            "fn main() -> ()  { hidden(); }",
            "UnboundValue",
        ),
        (
            "export {};",
            "type Hidden = int;",
            "fn main() -> ()  { let mut x: Hidden; }",
            "UnboundType",
        ),
        (
            "export {};",
            "extern type Hidden;",
            "fn main() -> ()  { let mut x = Ptr<Hidden>(ulong(0)); }",
            "UnboundType",
        ),
        (
            "export {};",
            "extern { \"stdlib.h\": { fn abs (n: int) -> int; } };",
            "fn main() -> ()  { abs(-1); }",
            "UnboundValue",
        ),
    ] {
        Project::new(&[
            ("library.resin", &format!("{exports} {declaration}")),
            (
                "main.resin",
                &format!("import {{ \"library.resin\" }}; {use_site}"),
            ),
        ])
        .error(error);
    }
}

#[test]
fn a_dependency_cannot_see_its_consumers_names() {
    let error = Project::new(&[
        (
            "library.resin",
            "export { read }; fn read () -> int  { secret }",
        ),
        (
            "main.resin",
            "export { main }; import { \"library.resin\" }; fn main() -> ()  { let mut secret = 42; read(); }",
        ),
    ])
    .error("UnboundValue");
    assert!(error.contains("library.resin:1:"), "{error}");
}

#[test]
fn reexports_keep_binding_identity_through_diamond_imports() {
    let project = Project::new(&[
        (
            "base.resin",
            "export { Number, make }; struct Number { value: int, } fn make(counter: Ptr<int>) -> Number  { counter.* = counter.* + 1; Number { value = 42 } }",
        ),
        (
            "left.resin",
            "export { Number, make }; import { \"base.resin\" };",
        ),
        (
            "right.resin",
            "export { Number, make }; import { \"base.resin\" };",
        ),
        (
            "main.resin",
            "export { main }; import { \"left.resin\", \"right.resin\", \"./base.resin\" }; fn main() -> int  { let mut counter = 0; let mut n = make(&counter); n.value + counter - 1 }",
        ),
    ]);
    assert_eq!(project.run().status.code(), Some(42));
    let module = project.compile().unwrap();
    assert_eq!(
        module
            .types
            .iter()
            .filter_map(|d| d.name())
            .map(|name| name.as_ref())
            .collect::<Vec<&str>>(),
        ["Number"]
    );
    assert_eq!(module.functions.len(), 2);
}

#[test]
fn dependencies_are_not_implicitly_reexported() {
    Project::new(&[
        (
            "base.resin",
            "export { secret }; fn secret() -> int  { 42 }",
        ),
        (
            "library.resin",
            "export { read }; import { \"base.resin\" }; fn read() -> int  { secret() }",
        ),
        (
            "main.resin",
            "export { main }; import { \"library.resin\" }; fn main() -> ()  { secret; }",
        ),
    ])
    .error("UnboundValue");
}

#[test]
fn conflicting_imports_report_both_definition_locations() {
    for definition in ["type Shared = int;", "extern type Shared;"] {
        let name = if definition.contains("Shared") {
            "Shared"
        } else {
            "shared"
        };
        let source = format!("export {{ {name} }};\n{definition}");
        let error = Project::new(&[
            ("left.resin", &source),
            ("right.resin", &source),
            ("main.resin", "import { \"left.resin\", \"right.resin\" };"),
        ])
        .error("conflicting binding");
        for path in ["main.resin:1:", "left.resin:2:", "right.resin:2:"] {
            assert!(error.contains(path), "{error}");
        }
    }
}

#[test]
fn imported_function_overloads_conflict_only_at_ambiguous_calls_and_allow_nested_shadowing() {
    for source in [
        "import { \"library.resin\" }; fn shared () -> int  { 1 }",
        "extern { \"x.h\": { fn shared () -> int; } }; import { \"library.resin\" };",
    ] {
        Project::new(&[
            (
                "library.resin",
                "export { shared }; fn shared() -> int  { 42 }",
            ),
            (
                "main.resin",
                &format!("{source} fn main() -> int {{ shared() }}"),
            ),
        ])
        .error("ambiguous overload");
    }
    let project = Project::new(&[
        (
            "library.resin",
            "export { shared }; fn shared() -> int  { 99 }",
        ),
        (
            "main.resin",
            "export { main }; import { \"library.resin\" }; fn main () -> int  { let mut shared = 42; shared }",
        ),
    ]);
    assert_eq!(project.run().status.code(), Some(42));
}

#[test]
fn invalid_exports_are_rejected_even_in_the_entry_file() {
    for (source, error) in [
        ("export { missing };", "UnknownExport"),
        ("export { Missing };", "UnknownExport"),
        (
            "export { value, value }; fn value() -> int  { 1 }",
            "DuplicateExport",
        ),
        (
            "export { Value, Value }; type Value = int;",
            "DuplicateExport",
        ),
        (
            "export { value, main }; fn main() -> ()  { let mut value = 1; }",
            "UnknownExport",
        ),
    ] {
        Project::new(&[("main.resin", source)]).error(error);
        assert!(
            pipeline::generate(&support::parse(source))
                .unwrap_err()
                .to_string()
                .contains(error)
        );
    }
}

#[test]
fn exported_functions_can_return_private_types() {
    let project = Project::new(&[
        (
            "library.resin",
            "export { make, read }; struct Hidden { value: int, } fn make () -> Hidden  { Hidden { value = 42 } } fn read (n: Hidden) -> int  { n.value }",
        ),
        (
            "main.resin",
            "export { main }; import { \"library.resin\" }; fn main () -> int  { read(make()) }",
        ),
    ]);
    assert_eq!(project.run().status.code(), Some(42));
}

#[test]
fn private_nominal_types_keep_distinct_identities() {
    Project::new(&[
        (
            "left.resin",
            "export { make }; struct Hidden { value: int, } fn make () -> Hidden  { Hidden { value = 42 } }",
        ),
        (
            "right.resin",
            "export { read }; struct Hidden { value: int, } fn read (n: Hidden) -> int  { n.value }",
        ),
        (
            "main.resin",
            "export { main }; import { \"left.resin\", \"right.resin\" }; fn main() -> ()  { read(make()); }",
        ),
    ])
    .error("TypeMismatch");
}

#[test]
fn importing_modules_does_not_execute_their_functions() {
    let project = Project::new(&[
        (
            "base.resin",
            "import { \"$/string.resin\" }; fn main() -> ()  { print(\"A\"); }",
        ),
        (
            "left.resin",
            "import { \"base.resin\", \"$/string.resin\" }; fn main() -> ()  { print(\"B\"); }",
        ),
        (
            "right.resin",
            "import { \"base.resin\", \"$/string.resin\" }; fn main() -> ()  { print(\"C\"); }",
        ),
        (
            "main.resin",
            "export { main }; import { \"left.resin\", \"right.resin\", \"./base.resin\", \"$/string.resin\" }; fn main() -> ()  { print(\"D\"); }",
        ),
    ]);
    let output = project.run();
    assert!(output.status.success());
    assert_eq!(output.stdout, b"D");
}

#[test]
fn mutually_recursive_functions_still_work_within_a_module() {
    let project = Project::new(&[
        (
            "library.resin",
            "export { even }; fn even (n: int) -> int  { if (n == 0) { 42 } else { odd(n - 1) } } fn odd (n: int) -> int  { if (n == 0) { 0 } else { even(n - 1) } }",
        ),
        (
            "main.resin",
            "export { main }; import { \"library.resin\" }; fn main () -> int  { even(10) }",
        ),
    ]);
    assert_eq!(project.run().status.code(), Some(42));
}

#[test]
fn private_main_is_not_an_entry_point() {
    for root in [
        "import { \"library.resin\" };",
        "export { main }; import { \"library.resin\" }; fn main() -> ()  {}",
    ] {
        let project = Project::new(&[
            ("library.resin", "fn main () -> int  { 42 }"),
            ("main.resin", root),
        ]);
        if !root.starts_with("export") {
            assert!(
                support::project::Project::new(&project.compile().unwrap(), Some("main"))
                    .unwrap_err()
                    .to_string()
                    .contains("export { main }")
            );
        } else {
            assert!(project.run().status.success());
        }
    }
}

#[test]
fn an_imported_main_must_be_reexported_to_be_an_entry_point() {
    for exports in ["", "export { main };"] {
        let project = Project::new(&[
            ("library.resin", "export { main }; fn main() -> int  { 42 }"),
            (
                "main.resin",
                &format!("{exports} import {{ \"library.resin\" }};"),
            ),
        ]);
        if exports.is_empty() {
            assert!(
                support::project::Project::new(&project.compile().unwrap(), Some("main"))
                    .unwrap_err()
                    .to_string()
                    .contains("export { main }")
            );
        } else {
            assert_eq!(project.run().status.code(), Some(42));
        }
    }
}

#[test]
fn shader_declarations_preserve_the_entry_files_export_scope() {
    let project = Project::new(&[
        (
            "left.resin",
            "export { left }; @compute_shader fn kernel(invocation: ulong, output: Ptr<uint>)  { let mut i = uint(invocation); output.* = { i + uint(1) }; } fn left (i: uint, output: Ptr<uint>)  { kernel(ulong(i), output); }",
        ),
        (
            "right.resin",
            "export { right }; @compute_shader fn kernel(invocation: ulong, output: Ptr<uint>)  { let mut i = uint(invocation); output.* = { i + uint(2) }; } fn right (i: uint, output: Ptr<uint>)  { kernel(ulong(i), output); }",
        ),
        (
            "main.resin",
            "export { kernel, main }; import { \"left.resin\", \"right.resin\", \"$/gpu.resin\" }; @compute_shader fn kernel(invocation: ulong, output: Ptr<uint>)  { let mut i = uint(invocation); left(i, output); right(i, output); } fn main() -> (() | Err<_>)  { if (0 == 1) { gpu_new()?:create_compute_pipeline(kernel)?; }; (()) }",
        ),
    ]);
    let module = project.compile().unwrap();
    let kernel = module.entries["kernel"];
    assert_eq!(
        module.functions[kernel.index()].profile,
        resin_lir::Profile::Host
    );
    let (&shader, _) = module
        .shaders
        .iter()
        .find(|(_, shader)| shader.embedded)
        .unwrap();
    assert_ne!(kernel, shader);
    assert_eq!(
        module.origins.functions[&kernel],
        module.origins.functions[&shader]
    );
    assert_eq!(module.shaders.len(), 3);
    assert_eq!(
        module
            .shaders
            .values()
            .filter(|shader| shader.embedded)
            .count(),
        1
    );
    let project = Project::new(&[
        (
            "library.resin",
            "export { kernel }; @compute_shader fn kernel(invocation: ulong, output: Ptr<uint>)  { let mut i = uint(invocation); output.* = { i }; }",
        ),
        ("main.resin", "import { \"library.resin\" };"),
    ]);
    let module = project.compile().unwrap();
    assert!(!module.entries.contains_key("kernel"));
    assert_eq!(module.shaders.len(), 1);
    assert!(!module.shaders.values().next().unwrap().embedded);
}

#[test]
fn standard_library_imports_work_outside_the_repository() {
    let project = Project::new(&[(
        "main.resin",
        "export { main }; import { \"$/status.resin\", \"$/graphics.resin\", \"$/image.resin\" }; fn main() -> (int | Err<_>)  { runtime_status_from_code(0)?; (runtime_status_code(Incomplete {}) + 35) }",
    )]);
    assert_eq!(project.run().status.code(), Some(42));
    Project::new(&[(
        "main.resin",
        "export { main }; import { \"$/status.resin\" }; fn main()  { resin_status_string(0); }",
    )])
    .error("UnboundValue");
    Project::new(&[(
        "main.resin",
        "export { main }; import { \"$/window.resin\" }; fn main()  { gpu_new(); }",
    )])
    .error("UnboundValue");
}

#[test]
fn library_root_can_be_relocated_and_does_not_capture_relative_imports() {
    let project = Project::new(&[
        (
            "custom/math/library.resin",
            "export { answer }; fn answer() -> int  { 42 }",
        ),
        (
            "custom/renderer/value.resin",
            "export { rendered }; import { \"$/math/library.resin\" }; fn rendered() -> int  { answer() }",
        ),
        (
            "std/library.resin",
            "export { local }; fn local() -> int  { 1 }",
        ),
        (
            "main.resin",
            "export { main }; import { \"$/renderer/value.resin\", \"std/library.resin\" }; fn main() -> int  { rendered() + local() }",
        ),
    ]);
    let server = support::service::Service::configured(|config, _| {
        config.library_root = project.0.path().join("custom")
    });
    let output = server
        .command()
        .current_dir(project.0.path())
        .arg("main.resin")
        .output()
        .unwrap();
    assert_eq!(
        output.status.code(),
        Some(43),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stdout.is_empty());
}

#[test]
fn invalid_import_paths_report_the_importing_file() {
    for path in [
        "missing.resin",
        "$/missing.resin",
        "$/../Cargo.toml",
        "$/math/../../Cargo.toml",
        "$//absolute.resin",
        "$/",
        "",
    ] {
        let error = Project::new(&[("main.resin", &format!("import {{ \"{path}\" }};"))])
            .error("main.resin:1:");
        assert!(!error.is_empty());
    }
}

#[test]
fn compiler_builtins_cannot_be_redefined_in_any_module_or_scope() {
    for name in ["absurd"] {
        for source in [
            format!("fn {name} () -> ()  {{}}"),
            format!("extern {{ \"stdlib.h\": {{ fn {name} () -> int; }} }};"),
            format!("fn main () -> ()  {{ let mut {name} = 1; }}"),
            format!("fn main () -> ()  {{ let mut {name}: int; }}"),
            format!("fn f ({name}: int) -> ()  {{}}"),
            format!("extern {{ \"stdlib.h\": {{ fn f ({name}: int) -> (); }} }};"),
        ] {
            Project::new(&[("main.resin", &source)]).error("ReservedBuiltin");
            Project::new(&[
                ("library.resin", &source),
                ("main.resin", "import { \"library.resin\" };"),
            ])
            .error("ReservedBuiltin");
        }
        Project::new(&[
            (
                "library.resin",
                &format!("export {{ {name} }}; fn {name}() -> int  {{ 1 }}"),
            ),
            ("main.resin", "import { \"library.resin\" };"),
        ])
        .error("ReservedBuiltin");
    }
}

#[test]
fn builtin_spellings_are_valid_field_names() {
    let project = Project::new(&[(
        "main.resin",
        "export { main }; struct Fields { absurd: int, shader: int, } fn main () -> int  { let mut value = Fields { absurd = 20, shader = 22 }; value.absurd + value.shader }",
    )]);
    assert_eq!(project.run().status.code(), Some(42));
}

#[test]
fn entry_bindings_are_verified() {
    let mut module = support::module("export { main }; fn main () -> ()  {}");
    module
        .entries
        .insert("main".into(), FunctionId::from_index(999));
    assert!(resin_lir::verify(&module).is_err());
    assert!(resin_lir::VerifiedModule::new(module).is_err());
}

#[test]
fn files_reject_runtime_bindings_and_statements() {
    for statement in [
        "let mut x = 1;",
        "let mut x: int;",
        "x = 2;",
        "print(fmt(\"hello\", ()));",
        "();",
        "while (1 == 0) {};",
    ] {
        for declarations in ["", "type Item = int; fn helper() -> ()  {}"] {
            Project::new(&[("main.resin", &format!("{declarations} {statement}"))])
                .error("parse error");
        }
    }
}

#[test]
fn lowering_rejects_runtime_module_items_even_in_constructed_asts() {
    for statement in
        support::statements("let mut x = 1; let mut y: int; print(fmt(\"hello\", ()));")
    {
        let mut file = support::parse("");
        file.stmts.push(statement);
        assert_eq!(
            pipeline::generate(&file).unwrap_err().kind,
            GenerateErrorKind::InvalidModuleItem
        );
        let project = Project::new(&[("main.resin", "")]);
        let mut program = pipeline::load(&project.0.path().join("main.resin")).unwrap();
        program.modules[0].file = std::sync::Arc::new(file);
        assert!(
            pipeline::generate_program(&program)
                .unwrap_err()
                .to_string()
                .contains("InvalidModuleItem")
        );
    }
}

#[test]
fn declarations_do_not_create_a_module_initializer() {
    let empty = support::module("");
    assert!(empty.functions.is_empty());
    let types = support::module("export { Item }; struct Item { value: int, }");
    assert!(types.functions.is_empty());
    assert!(types.entries.is_empty());
    let module =
        support::module("export { answer, Item }; type Item = int; fn answer() -> int  { 42 }");
    assert_eq!(module.functions.len(), 1);
    assert_eq!(module.functions[0].name.as_deref(), Some("answer"));
    assert_eq!(module.entries.len(), 1);
    let project = support::project::Project::new(&module, Some("answer")).unwrap();
    let source = fs::read_to_string(project.generated.c_source().unwrap()).unwrap();
    assert!(source.contains("int main(int r_argc, char **r_argv) {\n  (void)r_argc; (void)r_argv;\n  atexit(resin_cleanup);\n  return r_fn0();\n}"));
}

#[test]
fn functions_cannot_capture_another_functions_locals() {
    Project::new(&[(
        "main.resin",
        "fn main() -> ()  { let mut value = 1; } fn read() -> int  { value }",
    )])
    .error("UnboundValue");
}

#[test]
fn shader_objects_can_reference_private_helpers() {
    let module = support::module(
        "export { main }; import { \"$/gpu.resin\" }; @compute_shader fn kernel(invocation: ulong, output: Ptr<uint>)  { let mut i = uint(invocation); output.* = { i }; } fn main() -> (() | Err<_>)  { if (0 == 1) { gpu_new()?:create_compute_pipeline(kernel)?; }; (()) }",
    );
    assert!(!module.entries.contains_key("kernel"));
    assert!(
        module
            .functions
            .iter()
            .flat_map(|f| &f.blocks)
            .flat_map(|b| &b.instrs)
            .any(|i| matches!(i, resin_lir::Instr::GpuComputePipeline { .. }))
    );
}

#[test]
fn free_operations_are_exported_independently_of_structs() {
    let source = r#"export { Counter , counter_new, add, read };
        struct Counter { value: int,
            
            
            
        }
fn counter_new(value: int) -> Counter  { Counter { value = value } }

fn add(self: Ref<Counter>, a: int, b: int)  { self.value = self.value + a + b; }

fn read(self: Ref<Counter>) -> int  { self.value }


    "#;
    let file = support::parse(source);
    assert_eq!(file.stmts.len(), 4);
    let resin_ast::StmtKind::Struct { name, methods, .. } = &file.stmts[0].val else {
        panic!("expected a struct declaration");
    };
    assert_eq!(name.val.as_ref(), "Counter");
    assert!(methods.is_empty());
    assert!(resin_ast::format_source(&file).contains("(struct"));
    let project = Project::new(&[
        ("counter.resin", source),
        (
            "main.resin",
            r#"export { main }; import { "counter.resin" };
            fn main() -> int  {
                let mut c = counter_new(30);
                c:add(5, 7);
                let mut p = &c;
                if (p.*:read() == 42 && c:read() == 42) { 0 } else { 1 }
            }
        "#,
        ),
    ]);
    let output = project.run();
    assert_eq!(
        output.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn operation_calls_validate_all_arguments() {
    for (source, message) in [
        (
            "struct A {} fn a_f(self: int) {} fn g(a: A) { a:a_f(); }",
            "TypeMismatch",
        ),
        (
            "struct A {} fn a_f() {} fn a_f() {} fn g() { a_f(); }",
            "ambiguous overload",
        ),
        (
            "struct A {  }\nfn f(self: A)  {}\n  fn g()  { f(); }",
            "expected 1 argument",
        ),
        (
            "struct A {} fn a_f() {} fn g(a: A) { a:a_f(); }",
            "expected 0 arguments",
        ),
    ] {
        let error = pipeline::source_module(source).unwrap_err().to_string();
        assert!(error.contains(message), "{error}");
    }
    let project = Project::new(&[
        ("a.resin", "export { A }; struct A {}"),
        (
            "main.resin",
            "import { \"a.resin\" }; impl A { fn f()  {} }",
        ),
    ]);
    project.error("parse error");
}

#[test]
fn method_syntax_and_field_calls_have_distinct_meanings() {
    let source = r#"export { main };
        struct Counter { read: (int) -> int,
            
            
        }
fn read(counter: Ref<Counter>, n: int) -> int  { (counter.read)(n) + 40 }

fn counter_other(self: int) -> int  { self }

        fn field(n: int) -> int  { n + 1 }

        fn main() -> int  {
            let mut c = Counter { read = field };
            if (c:read(1) == 42 && (c.read)(1) == 2 && read(c, 1) == 42 && counter_other(42) == 42) { 0 } else { 1 }
        }
    "#;
    let project = Project::new(&[("main.resin", source)]);
    let output = project.run();
    assert_eq!(
        output.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let error = pipeline::generate(&support::parse(
        "struct Record { call: (int) -> int, } fn f(r: Record) -> int  { r:call(1) }",
    ))
    .unwrap_err();
    assert!(error.to_string().contains("UnboundValue"));
}

#[test]
fn indexing_methods_require_ulong_and_do_not_replace_nominal_methods() {
    for arg in ["1_f", "1 == 1", "", "0, 1", "-1", "0_ui", "0_i", "0_l"] {
        let source = format!("fn f()  {{ let mut values = [1, 2]; values:at({arg}); }}");
        assert!(
            pipeline::generate(&support::parse(&source)).is_err(),
            "{source}"
        );
    }
    let project = Project::new(&[(
        "main.resin",
        "export { main }; struct Item { value: int,  }\nfn at(item: Item, flag: bool) -> int  { if (flag) { item.value } else { 0 } }\n  fn main() -> int  { let mut item = Item { value = 42 }; item:at(1 == 1) }",
    )]);
    assert_eq!(project.run().status.code(), Some(42));
}

#[test]
fn aliases_share_the_nominal_namespace_and_origin() {
    let project = Project::new(&[
        (
            "library.resin",
            "export { Alias, Item , read }; struct Item { value: int,  }\nfn read(value: Ref<Item>) -> int  { value.value }\n type Alias = Item; ",
        ),
        (
            "main.resin",
            "export { main }; import { \"library.resin\" }; fn main() -> int  { let mut value = Item { value = 42 }; if (value:read() == read(value)) { 0 } else { 1 } }",
        ),
    ]);
    assert_eq!(project.run().status.code(), Some(0));
    let project = Project::new(&[
        ("library.resin", "export { Item }; struct Item {}"),
        (
            "main.resin",
            "import { \"library.resin\" }; type Alias = Item; impl Alias { fn f()  {} }",
        ),
    ]);
    project.error("parse error");
    for source in [
        "struct Item {} impl Item { fn f()  {} }",
        "struct Item {} type Alias = Item; impl Alias { fn f()  {} }",
        "type Number = int; impl Number { fn f()  {} }",
    ] {
        let document = support::frontend::cst(source, None);
        assert!(!support::frontend::ast(&document).errors.is_empty());
    }
}

#[test]
fn struct_methods_resolve_later_aliases_and_recursive_siblings() {
    let project = Project::new(&[(
        "main.resin",
        r#"export { main };
        struct Owner {
            value: int,
            
            
            
            
        }
fn owner_new(value: int) -> Shared  { Shared { value = value } }

fn owner_read(self: Shared) -> int  { self.value }

fn owner_even(n: int) -> bool  { if (n == 0) { 1 == 1 } else { owner_odd(n - 1) } }

fn owner_odd(n: int) -> bool  { if (n == 0) { 1 == 0 } else { owner_even(n - 1) } }

        type Shared = Owner;
        fn main() -> int  {
            let owner = owner_new(42);
            if (owner_even(owner:owner_read())) { 0 } else { 1 }
        }
    "#,
    )]);
    assert_eq!(project.run().status.code(), Some(0));
}

#[test]
fn local_structs_are_field_only() {
    let source = "fn f() -> int  { struct Local { value: int, } Local { value = 42 }.value }";
    pipeline::generate(&support::parse(source)).unwrap();
    for source in [
        "struct Local { fn read() -> int { 42 } }",
        "fn f() { struct Local { fn read() -> int { 42 } } }",
    ] {
        assert!(
            !support::frontend::ast(&support::frontend::cst(source, None))
                .errors
                .is_empty()
        );
    }
}
