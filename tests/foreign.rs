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
    let executable = temp
        .path()
        .join(format!("program{}", std::env::consts::EXE_SUFFIX));
    let source = c::emit(&module(source), "main").unwrap();
    let cc = std::env::var_os("CC")
        .unwrap_or_else(|| OsString::from(resin::toolchain::DEFAULT_C_COMPILER));
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
        r#"export {{ main }};

        extern "{header}" def answer () -> int;
        extern "{header}" def assign (out: Ptr<int>, value: int);
        extern "stdlib.h" def abs (n: int) -> int;
        def call (f: () -> int) -> int = {{ f() }};
        def main () -> int = {{
            var value = 0;
            var set = assign;
            set(&value, call(answer));
            abs(-value)
        }};
    "#,
        header = header.to_string_lossy().replace('\\', "/")
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
    let output = run(r#"export { main };

        def identity(p: Ptr<{ value: int }>, calls: Ptr<int>) -> Ptr<{ value: int }> = {
            calls.* := calls.* + 1;
            p
        };
        def main () -> int = {
            var calls = 0;
            var record = { value = 1 };
            var pointer = &identity(&record, &calls).value;
            var copy = Ptr<int> (ulong (pointer));
            copy.* := 41;
            record.value + calls
        };
    "#);
    assert_eq!(output.status.code(), Some(42));
}

#[test]
fn foreign_aggregate_values_and_implicit_pointer_casts_are_rejected() {
    for source in [
        "extern type Native; extern \"native.h\" def consume (value: Native) -> ();",
        "extern \"native.h\" def consume (value: { x: int }) -> ();",
        "extern \"native.h\" def produce () -> { x: int };",
        "extern \"native.h\" def callback (f: () -> int) -> ();",
        "extern \"bad\\nheader\" def invalid () -> int;",
    ] {
        assert!(
            error(source).contains("InvalidForeignSignature"),
            "{source}"
        );
    }
    for source in [
        "export { main }; extern type Native; def main() -> () = { var value: Native; };",
        "extern type Native; def identity (n: Native) -> Native = { n };",
        "extern type Native; type Wrapped = { value: Native };",
        "extern type Native; def read (n: Ptr<Native>) -> () = { n.*; };",
    ] {
        assert!(error(source).contains("OpaqueValue"), "{source}");
    }
    for source in [
        "export { main }; def f (p: Ptr<int>) -> () = {}; def main() -> () = { var x = 0; f(ulong (0)); };",
        "export { main }; def f (p: Ptr<ubyte>) -> () = {}; def main() -> () = { var x = 0; f(&x); };",
        "export { main }; def main() -> () = { var x = Ptr<int> (float32 (0.0)); };",
    ] {
        assert!(error(source).contains("TypeMismatch"), "{source}");
    }
    assert!(
        error("export { main }; def main() -> () = { var p = &(1 + 2); };").contains("NotAPlace")
    );
}

#[test]
fn imports_are_relative_deduplicated_and_checked_for_cycles() {
    let temp = TempDir::new(&std::env::temp_dir()).unwrap();
    let main = temp.path().join("main.resin");
    let nested = temp.path().join("nested");
    fs::create_dir(&nested).unwrap();
    fs::write(
        temp.path().join("common.resin"),
        "export { helper }; def helper () -> int = { 42 };",
    )
    .unwrap();
    fs::write(
        nested.join("library.resin"),
        "export { helper }; import { \"../common.resin\" };",
    )
    .unwrap();
    fs::write(
        &main,
        "export { main }; import { \"nested/library.resin\", \"common.resin\" }; def main () -> int = { helper() };",
    )
    .unwrap();
    let module = ir::generate_program(&ast::load(&main).unwrap()).unwrap();
    assert_eq!(module.functions.len(), 2);
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
        "export { kernel, main }; def kernel (i: uint) -> uint = { i }; def main() -> () = { var code = shader(kernel); };",
        "export { kernel, main }; def kernel (i: uint) -> uint = { i }; def main() -> () = { var code = shader(kernel, \"geometry\"); };",
        "export { kernel, main }; def kernel (i: uint) -> uint = { i }; def main() -> () = { var name = \"compute\"; var code = shader(kernel, name); };",
        "export { kernel, main }; def kernel (i: uint) -> uint = { i }; def main() -> () = { var alias = kernel; var code = shader(alias, \"compute\"); };",
        "export { main }; extern \"stdlib.h\" def abs (i: int) -> int; def main() -> () = { var code = shader(abs, \"compute\"); };",
    ] {
        assert!(error(source).contains("InvalidShader"), "{source}");
    }
    let module = module(
        "export { kernel, main }; def kernel (i: uint) -> uint = { i }; def main() -> () = { var code = shader(kernel, \"compute\"); };",
    );
    let main = &module.functions[module.entries["main"].index()];
    assert_eq!(main.locals[1].ty, ir::Ty::shader());
    assert!(
        c::emit(&module, "main")
            .unwrap_err()
            .to_string()
            .contains("SPIR-V compilation")
    );
}

#[test]
fn shader_cannot_be_shadowed_by_a_normal_function() {
    assert!(error("def shader (n: int) -> int = { n + 1 };").contains("ReservedBuiltin"));
}
