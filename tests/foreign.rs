use std::{ffi::OsString, fs, process::Command};

use resin::{
    ast,
    backend::c,
    ir,
    toolchain::{self, TempDir},
};

mod support;
use support::module;

fn run(source: &str) -> std::process::Output {
    let temp = TempDir::new(&std::env::temp_dir()).unwrap();
    let executable = temp.path().join("program");
    let source = c::emit(&module(source)).unwrap();
    let cc = std::env::var_os("CC").unwrap_or_else(|| OsString::from("cc"));
    toolchain::compile_c(&source, &executable, &cc)
        .unwrap_or_else(|error| panic!("{error}\n{source}"));
    Command::new(executable).output().unwrap()
}

fn error(source: &str) -> String {
    let temp = TempDir::new(&std::env::temp_dir()).unwrap();
    let path = temp.path().join("source.resin");
    fs::write(&path, source).unwrap();
    ir::generate_program(&ast::load(&path).unwrap())
        .unwrap_err()
        .to_string()
}

#[test]
fn foreign_functions_are_unary_values_with_c_argument_wrappers() {
    let temp = TempDir::new(&std::env::temp_dir()).unwrap();
    let header = temp.path().join("foreign.h");
    fs::write(&header, "static inline int answer(void) { return 42; }\nstatic inline void assign(int *out, int value) { *out = value; }\n").unwrap();
    let source = format!(
        r#"
        extern "{header}" answer () -> int;
        extern "{header}" assign (out: Ptr<int>, value: int) -> ();
        extern "stdlib.h" abs (n: int) -> int;
        call (f: () -> int) -> int = {{ f() }};
        main () -> int = {{
            value = 0;
            set = assign;
            set(&value, call(answer));
            abs(-value)
        }};
    "#,
        header = header.display()
    );
    let output = run(&source);
    assert_eq!(
        output.status.code(),
        Some(42),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn pointers_roundtrip_and_address_expressions_evaluate_once() {
    let output = run(r#"
        calls = 0;
        identity (p: Ptr<{ value: int }>) -> Ptr<{ value: int }> = { calls := calls + 1; p };
        main () -> int = {
            record = { value = 1 };
            pointer = &identity(&record).value;
            copy = Ptr<int> (ulong (pointer));
            copy.* := 41;
            record.value + calls
        };
    "#);
    assert_eq!(output.status.code(), Some(42));
}

#[test]
fn foreign_aggregate_values_and_implicit_pointer_casts_are_rejected() {
    for source in [
        "extern type Native; extern \"native.h\" consume (value: Native) -> ();",
        "extern \"native.h\" consume (value: { x: int }) -> ();",
        "extern \"native.h\" produce () -> { x: int };",
        "extern \"native.h\" callback (f: () -> int) -> ();",
        "extern \"bad\\nheader\" invalid () -> int;",
    ] {
        assert!(
            error(source).contains("InvalidForeignSignature"),
            "{source}"
        );
    }
    for source in [
        "extern type Native; value: Native;",
        "extern type Native; identity (n: Native) -> Native = { n };",
        "extern type Native; Wrapped = { value: Native };",
        "extern type Native; read (n: Ptr<Native>) -> () = { n.*; };",
    ] {
        assert!(error(source).contains("OpaqueValue"), "{source}");
    }
    for source in [
        "x = 0; f (p: Ptr<int>) -> () = {}; f(ulong (0));",
        "x = 0; f (p: Ptr<ubyte>) -> () = {}; f(&x);",
        "x = Ptr<int> (float32 (0.0));",
    ] {
        assert!(error(source).contains("TypeMismatch"), "{source}");
    }
    assert!(error("p = &(1 + 2);").contains("NotAPlace"));
}

#[test]
fn imports_are_relative_deduplicated_and_checked_for_cycles() {
    let temp = TempDir::new(&std::env::temp_dir()).unwrap();
    let main = temp.path().join("main.resin");
    let nested = temp.path().join("nested");
    fs::create_dir(&nested).unwrap();
    fs::write(
        temp.path().join("common.resin"),
        "export { helper }; helper () -> int = { 42 };",
    )
    .unwrap();
    fs::write(
        nested.join("library.resin"),
        "export { helper }; import { \"../common.resin\" };",
    )
    .unwrap();
    fs::write(
        &main,
        "import { \"nested/library.resin\", \"common.resin\" }; main () -> int = { helper() };",
    )
    .unwrap();
    let module = ir::generate_program(&ast::load(&main).unwrap()).unwrap();
    assert_eq!(module.functions.len(), 3);
    fs::write(
        temp.path().join("common.resin"),
        "import { \"main.resin\" };",
    )
    .unwrap();
    assert!(
        ast::load(&main)
            .unwrap_err()
            .to_string()
            .contains("cyclic source import")
    );
    fs::write(temp.path().join("common.resin"), "def invalid").unwrap();
    assert!(
        ast::load(&main)
            .unwrap_err()
            .to_string()
            .contains("common.resin")
    );
}

#[test]
fn shader_requires_a_named_function_and_a_literal_stage() {
    for source in [
        "kernel (i: uint) -> uint = { i }; code = shader(kernel);",
        "kernel (i: uint) -> uint = { i }; code = shader(kernel, \"geometry\");",
        "kernel (i: uint) -> uint = { i }; name = \"compute\"; code = shader(kernel, name);",
        "kernel (i: uint) -> uint = { i }; alias = kernel; code = shader(alias, \"compute\");",
        "extern \"stdlib.h\" abs (i: int) -> int; code = shader(abs, \"compute\");",
    ] {
        assert!(error(source).contains("InvalidShader"), "{source}");
    }
    let module = module("kernel (i: uint) -> uint = { i }; code = shader(kernel, \"compute\");");
    assert_eq!(module.globals[0].ty, ir::Ty::shader());
    assert!(
        c::emit(&module)
            .unwrap_err()
            .to_string()
            .contains("SPIR-V compilation")
    );
}

#[test]
fn shader_cannot_be_shadowed_by_a_normal_function() {
    assert!(error("shader (n: int) -> int = { n + 1 };").contains("ReservedBuiltin"));
}
