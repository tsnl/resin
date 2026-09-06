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
        let source = c::emit(&module).unwrap();
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
    let source = "export { answer, Box, }; import { \"a.resin\", \"b.resin\", }; answer = 42;";
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
    let project = Project::new(&[("main.resin", "export { value }; value = 1;")]);
    let program = ast::load(&project.0.path().join("main.resin")).unwrap();
    let output = ast::print::format_program(&program);
    assert!(output.starts_with("(program"), "{output}");
    assert!(output.contains("main.resin\""), "{output}");
}

#[test]
fn private_helpers_globals_and_types_are_resolved_in_their_own_files() {
    let project = Project::new(&[
        (
            "left.resin",
            "export { left }; Item = int; value = Item(20); helper () -> int = { int(value) }; left () -> int = { helper() };",
        ),
        (
            "right.resin",
            "export { right }; Item = int; value = Item(22); helper () -> int = { int(value) }; right () -> int = { helper() };",
        ),
        (
            "main.resin",
            "import { \"left.resin\", \"right.resin\" }; helper () -> int = { 99 }; main () -> int = { left() + right() };",
        ),
    ]);
    assert_eq!(project.run().status.code(), Some(42));
}

#[test]
fn private_names_are_not_visible_to_consumers() {
    for (exports, declaration, use_site, error) in [
        ("", "hidden = 1;", "hidden;", "UnboundValue"),
        (
            "export {};",
            "hidden () -> int = { 1 };",
            "hidden();",
            "UnboundValue",
        ),
        ("export {};", "Hidden = int;", "x: Hidden;", "UnboundType"),
        (
            "export {};",
            "extern type Hidden;",
            "x = Ptr (Hidden) (ulong(0));",
            "UnboundType",
        ),
        (
            "export {};",
            "extern \"stdlib.h\" abs (n: int) -> int;",
            "abs(-1);",
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
            "import { \"library.resin\" }; secret = 42; read();",
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
            "export { Number, make, counter }; Number = int; counter = 0; make () -> Number = { counter := counter + 1; Number(42) };",
        ),
        (
            "left.resin",
            "export { Number, make, counter }; import { \"base.resin\" };",
        ),
        (
            "right.resin",
            "export { Number, make, counter }; import { \"base.resin\" };",
        ),
        (
            "main.resin",
            "import { \"left.resin\", \"right.resin\", \"./base.resin\" }; main () -> int = { n: Number; n := make(); int(n) + counter - 1 };",
        ),
    ]);
    assert_eq!(project.run().status.code(), Some(42));
    let module = project.compile().unwrap();
    assert_eq!(module.types.len(), 1);
    assert_eq!(module.globals.len(), 1);
}

#[test]
fn dependencies_are_not_implicitly_reexported() {
    Project::new(&[
        ("base.resin", "export { secret }; secret = 42;"),
        (
            "library.resin",
            "export { read }; import { \"base.resin\" }; read () -> int = { secret };",
        ),
        ("main.resin", "import { \"library.resin\" }; secret;"),
    ])
    .error("UnboundValue");
}

#[test]
fn conflicting_imports_report_both_definition_locations() {
    for definition in [
        "shared = 1;",
        "shared () -> int = { 1 };",
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
        "shared = 1;",
        "shared: int;",
        "shared () -> int = { 1 };",
        "extern \"x.h\" shared () -> int;",
    ] {
        Project::new(&[
            ("library.resin", "export { shared }; shared = 42;"),
            (
                "main.resin",
                &format!("import {{ \"library.resin\" }}; {local}"),
            ),
        ])
        .error("conflicting binding");
    }
    let project = Project::new(&[
        ("library.resin", "export { shared }; shared = 99;"),
        (
            "main.resin",
            "import { \"library.resin\" }; main () -> int = { shared = 42; shared };",
        ),
    ]);
    assert_eq!(project.run().status.code(), Some(42));
}

#[test]
fn invalid_exports_are_rejected_even_in_the_entry_file() {
    for (source, error) in [
        ("export { missing };", "UnknownExport"),
        ("export { Missing };", "UnknownExport"),
        ("export { value, value }; value = 1;", "DuplicateExport"),
        ("export { Value, Value }; Value = int;", "DuplicateExport"),
        ("export { value }; value: int;", "UninitializedValue"),
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
            "import { \"library.resin\" }; main () -> int = { read(make()) };",
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
            "import { \"left.resin\", \"right.resin\" }; read(make());",
        ),
    ])
    .error("TypeMismatch");
}

#[test]
fn initialization_runs_once_in_dependency_and_import_list_order() {
    let project = Project::new(&[
        ("base.resin", "print(\"A\", ());"),
        ("left.resin", "import { \"base.resin\" }; print(\"B\", ());"),
        (
            "right.resin",
            "import { \"base.resin\" }; print(\"C\", ());",
        ),
        (
            "main.resin",
            "import { \"left.resin\", \"right.resin\", \"./base.resin\" }; print(\"D\", ());",
        ),
    ]);
    let output = project.run();
    assert!(output.status.success());
    assert_eq!(output.stdout, b"ABCD");
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
            "import { \"library.resin\" }; main () -> int = { even(10) };",
        ),
    ]);
    assert_eq!(project.run().status.code(), Some(42));
}

#[test]
fn private_main_is_not_an_entry_point() {
    for root in ["", "main () -> () = {};"] {
        let project = Project::new(&[
            ("library.resin", "main () -> int = { 42 };"),
            (
                "main.resin",
                &format!("import {{ \"library.resin\" }}; {root}"),
            ),
        ]);
        assert!(project.run().status.success());
    }
}

#[test]
fn an_imported_main_can_be_an_entry_point() {
    let project = Project::new(&[
        ("library.resin", "export { main }; main () -> int = { 42 };"),
        ("main.resin", "import { \"library.resin\" };"),
    ]);
    assert_eq!(project.run().status.code(), Some(42));
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
            "import { \"left.resin\", \"right.resin\" }; kernel (i: uint) -> uint = { left(i) + right(i) }; code = shader(kernel, \"compute\");",
        ),
    ]);
    let module = project.compile().unwrap();
    glsl::emit(&module, "kernel", glsl::Stage::Compute).unwrap();
    let project = Project::new(&[
        ("library.resin", "kernel (i: uint) -> uint = { i };"),
        ("main.resin", "import { \"library.resin\" };"),
    ]);
    assert!(glsl::emit(&project.compile().unwrap(), "kernel", glsl::Stage::Compute).is_err());
}

#[test]
fn standard_library_imports_work_outside_the_repository() {
    let project = Project::new(&[(
        "main.resin",
        "import { \"std/status.resin\", \"std/graphics.resin\", \"std/image.resin\" }; main () -> int = { check(0); resin_status_incomplete + 35 };",
    )]);
    assert_eq!(project.run().status.code(), Some(42));
    Project::new(&[("main.resin", "import { \"std/status.resin\" }; exit(0);")])
        .error("UnboundValue");
    Project::new(&[(
        "main.resin",
        "import { \"std/window.resin\" }; resin_gpu_create(0);",
    )])
    .error("UnboundValue");
}

#[test]
fn standard_library_can_be_relocated_and_does_not_capture_relative_imports() {
    let project = Project::new(&[
        ("custom/library.resin", "export { answer }; answer = 42;"),
        ("library.resin", "export { local }; local = 1;"),
        (
            "main.resin",
            "import { \"std/library.resin\", \"library.resin\" }; print(\"{0}\", (answer + local,));",
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
            format!("{name} = 1;"),
            format!("{name}: int;"),
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
                &format!("export {{ {name} }}; {name} = 1;"),
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
        "Fields = { print: int, shader: int }; main () -> int = { value = Fields { print = 20, shader = 22 }; value.print + value.shader };",
    )]);
    assert_eq!(project.run().status.code(), Some(42));
}

#[test]
fn entry_bindings_are_verified() {
    let mut module = support::module("main () -> () = {};");
    module.entries.insert(
        "main".into(),
        ir::Entry::Function(ir::FunctionId::from_index(999)),
    );
    assert!(ir::verify(&module).is_err());
    assert!(c::emit(&module).is_err());
    module.entries.insert(
        "main".into(),
        ir::Entry::Global(ir::GlobalId::from_index(999)),
    );
    assert!(ir::verify(&module).is_err());
}
