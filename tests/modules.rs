#[path = "support/toolchain.rs"]
mod config;
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
        let executable = self
            .0
            .path()
            .join(format!("program{}", std::env::consts::EXE_SUFFIX));
        let compiler = std::env::var_os("CC")
            .unwrap_or_else(|| OsString::from(resin::toolchain::DEFAULT_C_COMPILER));
        toolchain::compile_c(&source, &executable, &config::c(&compiler)).unwrap();
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
    let source = "export { answer, Box, }; import { \"a.resin\", \"b.resin\", }; def answer() -> int = { 42 };";
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
    let project = Project::new(&[(
        "main.resin",
        "export { value }; def value() -> int = { 1 };",
    )]);
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
            "export { left }; type Item = int; def helper(value: Item) -> int = { int(value) }; def left() -> int = { helper(Item(20)) };",
        ),
        (
            "right.resin",
            "export { right }; type Item = int; def helper(value: Item) -> int = { int(value) }; def right() -> int = { helper(Item(22)) };",
        ),
        (
            "main.resin",
            "export { main }; import { \"left.resin\", \"right.resin\" }; def helper () -> int = { 99 }; def main () -> int = { left() + right() };",
        ),
    ]);
    assert_eq!(project.run().status.code(), Some(42));
}

#[test]
fn private_names_are_not_visible_to_consumers() {
    for (exports, declaration, use_site, error) in [
        (
            "",
            "def hidden() -> int = { 1 };",
            "def main() -> () = { hidden(); };",
            "UnboundValue",
        ),
        (
            "export {};",
            "def hidden () -> int = { 1 };",
            "def main() -> () = { hidden(); };",
            "UnboundValue",
        ),
        (
            "export {};",
            "type Hidden = int;",
            "def main() -> () = { var x: Hidden; };",
            "UnboundType",
        ),
        (
            "export {};",
            "extern type Hidden;",
            "def main() -> () = { var x = Ptr<Hidden> (ulong(0)); };",
            "UnboundType",
        ),
        (
            "export {};",
            "extern \"stdlib.h\" def abs (n: int) -> int;",
            "def main() -> () = { abs(-1); };",
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
            "export { read }; def read () -> int = { secret };",
        ),
        (
            "main.resin",
            "export { main }; import { \"library.resin\" }; def main() -> () = { var secret = 42; read(); };",
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
            "export { Number, make }; struct Number { value: int }; def make(counter: Ptr<int>) -> Number = { counter.* := counter.* + 1; Number { value = 42 } };",
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
            "export { main }; import { \"left.resin\", \"right.resin\", \"./base.resin\" }; def main() -> int = { var counter = 0; var n = make(&counter); n.value + counter - 1 };",
        ),
    ]);
    assert_eq!(project.run().status.code(), Some(42));
    let module = project.compile().unwrap();
    assert_eq!(
        module.types.iter().filter(|d| d.name().is_some()).count(),
        2
    );
    assert_eq!(module.functions.len(), 2);
}

#[test]
fn dependencies_are_not_implicitly_reexported() {
    Project::new(&[
        (
            "base.resin",
            "export { secret }; def secret() -> int = { 42 };",
        ),
        (
            "library.resin",
            "export { read }; import { \"base.resin\" }; def read() -> int = { secret() };",
        ),
        (
            "main.resin",
            "export { main }; import { \"library.resin\" }; def main() -> () = { secret; };",
        ),
    ])
    .error("UnboundValue");
}

#[test]
fn conflicting_imports_report_both_definition_locations() {
    for definition in [
        "def shared () -> int = { 1 };",
        "extern \"stdlib.h\" def shared() -> int;",
        "type Shared = int;",
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
        "def shared () -> int = { 1 };",
        "extern \"x.h\" def shared () -> int;",
    ] {
        Project::new(&[
            (
                "library.resin",
                "export { shared }; def shared() -> int = { 42 };",
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
            "export { shared }; def shared() -> int = { 99 };",
        ),
        (
            "main.resin",
            "export { main }; import { \"library.resin\" }; def main () -> int = { var shared = 42; shared };",
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
            "export { value, value }; def value() -> int = { 1 };",
            "DuplicateExport",
        ),
        (
            "export { Value, Value }; type Value = int;",
            "DuplicateExport",
        ),
        (
            "export { value, main }; def main() -> () = { var value = 1; };",
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
            "export { make, read }; struct Hidden { value: int }; def make () -> Hidden = { Hidden { value = 42 } }; def read (n: Hidden) -> int = { n.value };",
        ),
        (
            "main.resin",
            "export { main }; import { \"library.resin\" }; def main () -> int = { read(make()) };",
        ),
    ]);
    assert_eq!(project.run().status.code(), Some(42));
}

#[test]
fn private_nominal_types_keep_distinct_identities() {
    Project::new(&[
        (
            "left.resin",
            "export { make }; struct Hidden { value: int }; def make () -> Hidden = { Hidden { value = 42 } };",
        ),
        (
            "right.resin",
            "export { read }; struct Hidden { value: int }; def read (n: Hidden) -> int = { n.value };",
        ),
        (
            "main.resin",
            "export { main }; import { \"left.resin\", \"right.resin\" }; def main() -> () = { read(make()); };",
        ),
    ])
    .error("TypeMismatch");
}

#[test]
fn importing_modules_does_not_execute_their_functions() {
    let project = Project::new(&[
        (
            "base.resin",
            "def main() -> () = { print(fmt(\"A\", ())); };",
        ),
        (
            "left.resin",
            "import { \"base.resin\" }; def main() -> () = { print(fmt(\"B\", ())); };",
        ),
        (
            "right.resin",
            "import { \"base.resin\" }; def main() -> () = { print(fmt(\"C\", ())); };",
        ),
        (
            "main.resin",
            "export { main }; import { \"left.resin\", \"right.resin\", \"./base.resin\" }; def main() -> () = { print(fmt(\"D\", ())); };",
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
            "export { even }; def even (n: int) -> int = { if (n == 0) { 42 } else { odd(n - 1) } }; def odd (n: int) -> int = { if (n == 0) { 0 } else { even(n - 1) } };",
        ),
        (
            "main.resin",
            "export { main }; import { \"library.resin\" }; def main () -> int = { even(10) };",
        ),
    ]);
    assert_eq!(project.run().status.code(), Some(42));
}

#[test]
fn private_main_is_not_an_entry_point() {
    for root in [
        "import { \"library.resin\" };",
        "export { main }; import { \"library.resin\" }; def main() -> () = {};",
    ] {
        let project = Project::new(&[
            ("library.resin", "def main () -> int = { 42 };"),
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
            (
                "library.resin",
                "export { main }; def main() -> int = { 42 };",
            ),
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
            "export { left }; @compute_shader def kernel(i: uint, output: Ptr<uint>) = { output.* := { i + uint(1) }; }; def left (i: uint, output: Ptr<uint>) = { kernel(i, output); };",
        ),
        (
            "right.resin",
            "export { right }; @compute_shader def kernel(i: uint, output: Ptr<uint>) = { output.* := { i + uint(2) }; }; def right (i: uint, output: Ptr<uint>) = { kernel(i, output); };",
        ),
        (
            "main.resin",
            "export { kernel, main }; import { \"left.resin\", \"right.resin\" }; @compute_shader def kernel(i: uint, output: Ptr<uint>) = { left(i, output); right(i, output); }; def main() -> () = { var code = kernel.spirv; };",
        ),
    ]);
    let module = project.compile().unwrap();
    glsl::emit(&module, "kernel", glsl::Stage::Compute).unwrap();
    let project = Project::new(&[
        (
            "library.resin",
            "export { kernel }; @compute_shader def kernel(i: uint, output: Ptr<uint>) = { output.* := { i }; };",
        ),
        ("main.resin", "import { \"library.resin\" };"),
    ]);
    assert!(glsl::emit(&project.compile().unwrap(), "kernel", glsl::Stage::Compute).is_err());
}

#[test]
fn standard_library_imports_work_outside_the_repository() {
    let project = Project::new(&[(
        "main.resin",
        "export { main }; import { \"std/status.resin\", \"std/graphics.resin\", \"std/image.resin\" }; def main() -> Result<int, _> = { RuntimeStatus.from_code(0)?; ok(RuntimeStatus.code(Incomplete {}) + 35) };",
    )]);
    assert_eq!(project.run().status.code(), Some(42));
    Project::new(&[(
        "main.resin",
        "export { main }; import { \"std/status.resin\" }; def main() = { resin_status_string(0); };",
    )])
    .error("UnboundValue");
    Project::new(&[(
        "main.resin",
        "export { main }; import { \"std/window.resin\" }; def main() = { Gpu.new(); };",
    )])
    .error("UnboundType");
}

#[test]
fn standard_library_can_be_relocated_and_does_not_capture_relative_imports() {
    let project = Project::new(&[
        (
            "custom/library.resin",
            "export { answer }; def answer() -> int = { 42 };",
        ),
        (
            "library.resin",
            "export { local }; def local() -> int = { 1 };",
        ),
        (
            "main.resin",
            "export { main }; import { \"std/library.resin\", \"library.resin\" }; def main() -> () = { print(fmt(\"{0}\", (answer() + local(),))); };",
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
    for name in ["print"] {
        for source in [
            format!("def {name} () -> () = {{}};"),
            format!("extern \"stdlib.h\" def {name} () -> int;"),
            format!("def main () -> () = {{ var {name} = 1; }};"),
            format!("def main () -> () = {{ var {name}: int; }};"),
            format!("def f ({name}: int) -> () = {{}};"),
            format!("extern \"stdlib.h\" def f ({name}: int) -> ();"),
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
                &format!("export {{ {name} }}; def {name}() -> int = {{ 1 }};"),
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
        "export { main }; struct Fields { print: int, shader: int }; def main () -> int = { var value = Fields { print = 20, shader = 22 }; value.print + value.shader };",
    )]);
    assert_eq!(project.run().status.code(), Some(42));
}

#[test]
fn entry_bindings_are_verified() {
    let mut module = support::module("export { main }; def main () -> () = {};");
    module
        .entries
        .insert("main".into(), ir::FunctionId::from_index(999));
    assert!(ir::verify(&module).is_err());
    assert!(c::emit(&module, "main").is_err());
}

#[test]
fn files_reject_runtime_bindings_and_statements() {
    for statement in [
        "var x = 1;",
        "var x: int;",
        "x := 2;",
        "print(fmt(\"hello\", ()));",
        "();",
        "while (1 == 0) {};",
    ] {
        for declarations in ["", "type Item = int; def helper() -> () = {};"] {
            Project::new(&[("main.resin", &format!("{declarations} {statement}"))])
                .error("parse error");
        }
    }
}

#[test]
fn lowering_rejects_runtime_module_items_even_in_constructed_asts() {
    for statement in support::statements("var x = 1; var y: int; print(fmt(\"hello\", ()));") {
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
    let types = support::module("export { Item }; struct Item { value: int };");
    assert!(types.functions.is_empty());
    assert!(types.entries.is_empty());
    let module =
        support::module("export { answer, Item }; type Item = int; def answer() -> int = { 42 };");
    assert_eq!(module.functions.len(), 1);
    assert_eq!(module.functions[0].name.as_deref(), Some("answer"));
    assert_eq!(module.entries.len(), 1);
    let source = c::emit(&module, "answer").unwrap();
    assert!(source.contains("int main(int r_argc, char **r_argv) {\n  (void)r_argc; (void)r_argv;\n  atexit(resin_cleanup);\n  return r_fn0(0);\n}"));
}

#[test]
fn functions_cannot_capture_another_functions_locals() {
    Project::new(&[(
        "main.resin",
        "def main() -> () = { var value = 1; }; def read() -> int = { value };",
    )])
    .error("UnboundValue");
}

#[test]
fn shader_objects_can_reference_private_helpers() {
    let module = support::module(
        "export { main }; @compute_shader def kernel(i: uint, output: Ptr<uint>) = { output.* := { i }; }; def main() -> () = { var code = kernel.spirv; };",
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

#[test]
fn inherent_methods_keep_impl_nodes_and_follow_exported_types() {
    let source = r#"
        export { Counter };
        struct Counter { value: int };
        impl Counter {
            def new(value: int) -> Counter = { Counter { value = value } };
            def add(self: Ptr<Counter>, a: int, b: int) = { self.value := self.value + a + b; };
            def read(self: Counter) -> int = { self.value };
        }
    "#;
    let file = support::parse(source);
    assert_eq!(file.stmts.len(), 2);
    let ast::StmtKind::Impl { receiver, methods } = &file.stmts[1].val else {
        panic!("expected an impl declaration");
    };
    assert_eq!(receiver.val.as_ref(), "Counter");
    assert_eq!(methods.len(), 3);
    assert!(ast::print::format_source(&file).contains("(impl"));
    let project = Project::new(&[
        ("counter.resin", source),
        (
            "main.resin",
            r#"
            export { main }; import { "counter.resin" };
            def main() -> int = {
                var c = Counter.new(30);
                c.add(5, 7);
                var p = &c;
                if (p.read() == 42 && c.read() == 42) { 0 } else { 1 }
            };
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
fn methods_validate_declarations_and_call_receivers() {
    for (source, message) in [
        (
            "struct A {}; impl A { def f(self: int) = {}; } def g(a: A) = { a.f(); };",
            "method receiver does not match",
        ),
        (
            "struct A {}; impl A { def f() = {}; def f() = {}; }",
            "DuplicateValue",
        ),
        (
            "struct A {}; impl A { def f(self: A) = {}; } def g() = { A.f(); };",
            "TypeMismatch",
        ),
        (
            "struct A {}; impl A { def f() = {}; } def g(a: A) = { a.f(); };",
            "method receiver does not match",
        ),
    ] {
        let error = ir::generate(&support::parse(source))
            .unwrap_err()
            .to_string();
        assert!(error.contains(message), "{error}");
    }
    let project = Project::new(&[
        ("a.resin", "export { A }; struct A {};"),
        (
            "main.resin",
            "import { \"a.resin\" }; impl A { def f() = {}; }",
        ),
    ]);
    project.error("impl requires a type defined in this module");
}

#[test]
fn method_syntax_and_field_calls_have_distinct_meanings() {
    let source = r#"
        export { main };
        struct Counter { read: (int) -> int };
        def field(n: int) -> int = { n + 1 };
        impl Counter {
            def read(counter: Counter, n: int) -> int = { (counter.read)(n) + 40 };
            def other(self: int) -> int = { self };
        }
        def main() -> int = {
            var c = Counter { read = field };
            if (c.read(1) == 42 && (c.read)(1) == 2 && Counter.read(c, 1) == 42 && Counter.other(42) == 42) { 0 } else { 1 }
        };
    "#;
    let project = Project::new(&[("main.resin", source)]);
    let output = project.run();
    assert_eq!(
        output.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let error = ir::generate(&support::parse(
        "struct Record { call: (int) -> int }; def f(r: Record) -> int = { r.call(1) };",
    ))
    .unwrap_err();
    assert!(error.to_string().contains("unknown method"));
}

#[test]
fn indexing_methods_require_ulong_and_do_not_replace_nominal_methods() {
    for arg in ["1_f", "1 == 1", "", "0, 1", "-1", "0_ui", "0_i", "0_l"] {
        let source = format!("def f() = {{ var values = [1, 2]; values.at({arg}); }};");
        assert!(ir::generate(&support::parse(&source)).is_err(), "{source}");
    }
    let project = Project::new(&[(
        "main.resin",
        "export { main }; struct Item { value: int }; impl Item { def at(item: Item, flag: bool) -> int = { if (flag) { item.value } else { 0 } }; } def main() -> int = { var item = Item { value = 42 }; item.at(1 == 1) };",
    )]);
    assert_eq!(project.run().status.code(), Some(42));
}

#[test]
fn aliases_share_the_nominal_namespace_and_origin() {
    let project = Project::new(&[
        (
            "library.resin",
            "export { Alias, Item }; struct Item { value: int }; type Alias = Item; impl Alias { def read(value: Item) -> int = { value.value }; }",
        ),
        (
            "main.resin",
            "export { main }; import { \"library.resin\" }; def main() -> int = { var value = Item { value = 42 }; if (value.read() == Alias.read(value)) { 0 } else { 1 } };",
        ),
    ]);
    assert_eq!(project.run().status.code(), Some(0));
    let project = Project::new(&[
        ("library.resin", "export { Item }; struct Item {};"),
        (
            "main.resin",
            "import { \"library.resin\" }; type Alias = Item; impl Alias { def f() = {}; }",
        ),
    ]);
    project.error("defined in this module");
    let source = "struct Item {}; type Alias = Item; impl Item { def f() = {}; } impl Alias { def f() = {}; }";
    assert!(
        ir::generate(&support::parse(source))
            .unwrap_err()
            .to_string()
            .contains("duplicate method")
    );
    let source = "type Number = int; impl Number { def f() = {}; }";
    assert!(
        ir::generate(&support::parse(source))
            .unwrap_err()
            .to_string()
            .contains("nominal struct")
    );
}
