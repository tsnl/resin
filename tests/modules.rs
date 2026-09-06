use std::{
    ffi::OsString,
    fs,
    process::{Command, Output},
};

use resin::{
    ast,
    backend::{c, glsl},
    ir,
    toolchain::{self, TempDir},
};

mod support;

struct Project(TempDir);

impl Project {
    fn new(files: &[(&str, &str)]) -> Self {
        let project = Self(TempDir::new(&std::env::temp_dir()).unwrap());
        for (name, source) in files {
            let path = project.0.path().join(name);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, source).unwrap();
        }
        project
    }

    fn compile(&self) -> Result<ir::Module, ast::SourceError> {
        ir::generate_program(&ast::load(&self.0.path().join("main.resin"))?)
    }

    fn run(&self) -> Output {
        let module = self.compile().unwrap();
        let source = c::emit(&module, "main").unwrap();
        let executable = self.0.path().join("program");
        let compiler = std::env::var_os("CC").unwrap_or_else(|| OsString::from("cc"));
        toolchain::compile_c(&source, &executable, &compiler).unwrap();
        Command::new(executable).output().unwrap()
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
        "export { answer, Box, }; import { \"a.resin\", \"b.resin\", }; answer() -> int = { 42 };";
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
    assert!(ast::print::format_source(&file).contains("(export answer Box)"));
    assert!(
        ir::generate(&file)
            .unwrap_err()
            .to_string()
            .contains("UnresolvedImport")
    );
    let project = Project::new(&[("main.resin", "export { value }; value() -> int = { 1 };")]);
    let program = ast::load(&project.0.path().join("main.resin")).unwrap();
    let output = ast::print::format_program(&program);
    assert!(output.starts_with("(program"), "{output}");
    assert!(output.contains("main.resin\""), "{output}");
}

#[test]
fn private_helpers_and_types_are_resolved_in_their_own_files() {
    let project = Project::new(&[
        (
            "left.resin",
            "export { left }; Item = int; helper(value: Item) -> int = { int(value) }; left() -> int = { helper(Item(20)) };",
        ),
        (
            "right.resin",
            "export { right }; Item = int; helper(value: Item) -> int = { int(value) }; right() -> int = { helper(Item(22)) };",
        ),
        (
            "main.resin",
            "export { main }; import { \"left.resin\", \"right.resin\" }; helper () -> int = { 99 }; main () -> int = { left() + right() };",
        ),
    ]);
    assert_eq!(project.run().status.code(), Some(42));
}

#[test]
fn private_names_are_not_visible_to_consumers() {
    for (exports, declaration, use_site, error) in [
        (
            "",
            "hidden() -> int = { 1 };",
            "main() -> () = { hidden(); };",
            "UnboundValue",
        ),
        (
            "export {};",
            "hidden () -> int = { 1 };",
            "main() -> () = { hidden(); };",
            "UnboundValue",
        ),
        (
            "export {};",
            "Hidden = int;",
            "main() -> () = { x: Hidden; };",
            "UnboundType",
        ),
        (
            "export {};",
            "extern type Hidden;",
            "main() -> () = { x = Ptr<Hidden> (ulong(0)); };",
            "UnboundType",
        ),
        (
            "export {};",
            "extern \"stdlib.h\" abs (n: int) -> int;",
            "main() -> () = { abs(-1); };",
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
            "export { read }; read () -> int = { secret };",
        ),
        (
            "main.resin",
            "export { main }; import { \"library.resin\" }; main() -> () = { secret = 42; read(); };",
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
            "export { Number, make }; Number = int; make(counter: Ptr<int>) -> Number = { counter.* := counter.* + 1; Number(42) };",
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
            "export { main }; import { \"left.resin\", \"right.resin\", \"./base.resin\" }; main() -> int = { counter = 0; n = make(&counter); int(n) + counter - 1 };",
        ),
    ]);
    assert_eq!(project.run().status.code(), Some(42));
    let module = project.compile().unwrap();
    assert_eq!(module.types.len(), 1);
    assert_eq!(module.functions.len(), 2);
}

#[test]
fn dependencies_are_not_implicitly_reexported() {
    Project::new(&[
        ("base.resin", "export { secret }; secret() -> int = { 42 };"),
        (
            "library.resin",
            "export { read }; import { \"base.resin\" }; read() -> int = { secret() };",
        ),
        (
            "main.resin",
            "export { main }; import { \"library.resin\" }; main() -> () = { secret; };",
        ),
    ])
    .error("UnboundValue");
}

#[test]
fn conflicting_imports_report_both_definition_locations() {
    for definition in [
        "shared () -> int = { 1 };",
        "extern \"stdlib.h\" shared() -> int;",
        "Shared = int;",
        "extern type Shared;",
    ] {
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
fn imports_conflict_with_local_bindings_but_allow_nested_shadowing() {
    for local in [
        "shared () -> int = { 1 };",
        "extern \"x.h\" shared () -> int;",
    ] {
        Project::new(&[
            (
                "library.resin",
                "export { shared }; shared() -> int = { 42 };",
            ),
            (
                "main.resin",
                &format!("import {{ \"library.resin\" }}; {local}"),
            ),
        ])
        .error("conflicting binding");
    }
    let project = Project::new(&[
        (
            "library.resin",
            "export { shared }; shared() -> int = { 99 };",
        ),
        (
            "main.resin",
            "export { main }; import { \"library.resin\" }; main () -> int = { shared = 42; shared };",
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
            "export { value, value }; value() -> int = { 1 };",
            "DuplicateExport",
        ),
        ("export { Value, Value }; Value = int;", "DuplicateExport"),
        (
            "export { value, main }; main() -> () = { value = 1; };",
            "UnknownExport",
        ),
    ] {
        Project::new(&[("main.resin", source)]).error(error);
        assert!(
            ir::generate(&support::parse(source))
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
            "export { make, read }; Hidden = int; make () -> Hidden = { Hidden(42) }; read (n: Hidden) -> int = { int(n) };",
        ),
        (
            "main.resin",
            "export { main }; import { \"library.resin\" }; main () -> int = { read(make()) };",
        ),
    ]);
    assert_eq!(project.run().status.code(), Some(42));
}

#[test]
fn private_nominal_types_keep_distinct_identities() {
    Project::new(&[
        (
            "left.resin",
            "export { make }; Hidden = int; make () -> Hidden = { Hidden(42) };",
        ),
        (
            "right.resin",
            "export { read }; Hidden = int; read (n: Hidden) -> int = { int(n) };",
        ),
        (
            "main.resin",
            "export { main }; import { \"left.resin\", \"right.resin\" }; main() -> () = { read(make()); };",
        ),
    ])
    .error("TypeMismatch");
}

#[test]
fn importing_modules_does_not_execute_their_functions() {
    let project = Project::new(&[
        ("base.resin", "main() -> () = { print(\"A\", ()); };"),
        (
            "left.resin",
            "import { \"base.resin\" }; main() -> () = { print(\"B\", ()); };",
        ),
        (
            "right.resin",
            "import { \"base.resin\" }; main() -> () = { print(\"C\", ()); };",
        ),
        (
            "main.resin",
            "export { main }; import { \"left.resin\", \"right.resin\", \"./base.resin\" }; main() -> () = { print(\"D\", ()); };",
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
            "export { even }; even (n: int) -> int = { if (n == 0) { 42 } else { odd(n - 1) } }; odd (n: int) -> int = { if (n == 0) { 0 } else { even(n - 1) } };",
        ),
        (
            "main.resin",
            "export { main }; import { \"library.resin\" }; main () -> int = { even(10) };",
        ),
    ]);
    assert_eq!(project.run().status.code(), Some(42));
}

#[test]
fn private_main_is_not_an_entry_point() {
    for root in [
        "import { \"library.resin\" };",
        "export { main }; import { \"library.resin\" }; main() -> () = {};",
    ] {
        let project = Project::new(&[
            ("library.resin", "main () -> int = { 42 };"),
            ("main.resin", root),
        ]);
        if !root.starts_with("export") {
            assert!(
                c::emit(&project.compile().unwrap(), "main")
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
            ("library.resin", "export { main }; main() -> int = { 42 };"),
            (
                "main.resin",
                &format!("{exports} import {{ \"library.resin\" }};"),
            ),
        ]);
        if exports.is_empty() {
            assert!(
                c::emit(&project.compile().unwrap(), "main")
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
fn shader_entry_lookup_uses_the_entry_files_scope() {
    let project = Project::new(&[
        (
            "left.resin",
            "export { left }; kernel (i: uint) -> uint = { i + uint(1) }; left (i: uint) -> uint = { kernel(i) };",
        ),
        (
            "right.resin",
            "export { right }; kernel (i: uint) -> uint = { i + uint(2) }; right (i: uint) -> uint = { kernel(i) };",
        ),
        (
            "main.resin",
            "export { kernel, main }; import { \"left.resin\", \"right.resin\" }; kernel (i: uint) -> uint = { left(i) + right(i) }; main() -> () = { code = shader(kernel, \"compute\"); };",
        ),
    ]);
    let module = project.compile().unwrap();
    glsl::emit(&module, "kernel", glsl::Stage::Compute).unwrap();
    let project = Project::new(&[
        (
            "library.resin",
            "export { kernel }; kernel (i: uint) -> uint = { i };",
        ),
        ("main.resin", "import { \"library.resin\" };"),
    ]);
    assert!(glsl::emit(&project.compile().unwrap(), "kernel", glsl::Stage::Compute).is_err());
}

#[test]
fn standard_library_imports_work_outside_the_repository() {
    let project = Project::new(&[(
        "main.resin",
        "export { main }; import { \"std/status.resin\", \"std/graphics.resin\", \"std/image.resin\" }; main () -> int = { check(0); resin_status_incomplete() + 35 };",
    )]);
    assert_eq!(project.run().status.code(), Some(42));
    Project::new(&[(
        "main.resin",
        "export { main }; import { \"std/status.resin\" }; main() -> () = { exit(0); };",
    )])
    .error("UnboundValue");
    Project::new(&[(
        "main.resin",
        "export { main }; import { \"std/window.resin\" }; main() -> () = { resin_gpu_create(0); };",
    )])
    .error("UnboundValue");
}

#[test]
fn standard_library_can_be_relocated_and_does_not_capture_relative_imports() {
    let project = Project::new(&[
        (
            "custom/library.resin",
            "export { answer }; answer() -> int = { 42 };",
        ),
        ("library.resin", "export { local }; local() -> int = { 1 };"),
        (
            "main.resin",
            "export { main }; import { \"std/library.resin\", \"library.resin\" }; main() -> () = { print(\"{0}\", (answer() + local(),)); };",
        ),
    ]);
    let output = Command::new(env!("CARGO_BIN_EXE_resin"))
        .current_dir(project.0.path())
        .env("RESIN_STDLIB", project.0.path().join("custom"))
        .arg("main.resin")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.stdout, b"43");
}

#[test]
fn invalid_import_paths_report_the_importing_file() {
    for path in [
        "missing.resin",
        "std/missing.resin",
        "std/../Cargo.toml",
        "std/",
        "",
    ] {
        let error = Project::new(&[("main.resin", &format!("import {{ \"{path}\" }};"))])
            .error("main.resin:1:");
        assert!(!error.is_empty());
    }
}

#[test]
fn compiler_builtins_cannot_be_redefined_in_any_module_or_scope() {
    for name in ["print", "shader"] {
        for source in [
            format!("{name} () -> () = {{}};"),
            format!("extern \"stdlib.h\" {name} () -> int;"),
            format!("main () -> () = {{ {name} = 1; }};"),
            format!("main () -> () = {{ {name}: int; }};"),
            format!("f ({name}: int) -> () = {{}};"),
            format!("extern \"stdlib.h\" f ({name}: int) -> ();"),
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
                &format!("export {{ {name} }}; {name}() -> int = {{ 1 }};"),
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
        "export { main }; Fields = { print: int, shader: int }; main () -> int = { value = Fields { print = 20, shader = 22 }; value.print + value.shader };",
    )]);
    assert_eq!(project.run().status.code(), Some(42));
}

#[test]
fn entry_bindings_are_verified() {
    let mut module = support::module("export { main }; main () -> () = {};");
    module
        .entries
        .insert("main".into(), ir::FunctionId::from_index(999));
    assert!(ir::verify(&module).is_err());
    assert!(c::emit(&module, "main").is_err());
}

#[test]
fn files_reject_runtime_bindings_and_statements() {
    for statement in [
        "x = 1;",
        "x: int;",
        "x := 2;",
        "print(\"hello\", ());",
        "();",
        "while (1 == 0) {};",
    ] {
        for declarations in ["", "Item = int; helper() -> () = {};"] {
            Project::new(&[("main.resin", &format!("{declarations} {statement}"))])
                .error("parse error");
        }
    }
}

#[test]
fn lowering_rejects_runtime_module_items_even_in_constructed_asts() {
    for statement in support::statements("x = 1; y: int; print(\"hello\", ());") {
        let mut file = support::parse("");
        file.stmts.push(statement);
        assert_eq!(
            ir::generate(&file).unwrap_err().kind,
            ir::GenerateErrorKind::InvalidModuleItem
        );
        let project = Project::new(&[("main.resin", "")]);
        let mut program = ast::load(&project.0.path().join("main.resin")).unwrap();
        program.modules[0].file = file;
        assert!(
            ir::generate_program(&program)
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
    let types = support::module("export { Item }; Item = { value: int };");
    assert!(types.functions.is_empty());
    assert!(types.entries.is_empty());
    let module = support::module("export { answer, Item }; Item = int; answer() -> int = { 42 };");
    assert_eq!(module.functions.len(), 1);
    assert_eq!(module.functions[0].name.as_deref(), Some("answer"));
    assert_eq!(module.entries.len(), 1);
    let source = c::emit(&module, "answer").unwrap();
    assert!(source.contains("int main(void) {\n  atexit(resin_cleanup);\n  return r_fn0(0);\n}"));
}

#[test]
fn functions_cannot_capture_another_functions_locals() {
    Project::new(&[(
        "main.resin",
        "main() -> () = { value = 1; }; read() -> int = { value };",
    )])
    .error("UnboundValue");
}

#[test]
fn shader_objects_can_reference_private_helpers() {
    let module = support::module(
        "export { main }; kernel(i: uint) -> uint = { i }; main() -> () = { code = shader(kernel, \"compute\"); };",
    );
    assert!(!module.entries.contains_key("kernel"));
    assert!(
        module
            .functions
            .iter()
            .flat_map(|f| &f.blocks)
            .flat_map(|b| &b.instrs)
            .any(|i| matches!(i, ir::Instr::Shader { .. }))
    );
}
