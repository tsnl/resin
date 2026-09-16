use std::fs;
use support::pipeline;
use tempfile::TempDir;

mod support;
use support::module;

fn run(source: &str) -> std::process::Output {
    support::project::Project::new(&module(source), Some("main"))
        .unwrap()
        .run()
}

fn error(source: &str) -> String {
    let temp = TempDir::new_in(std::env::temp_dir()).unwrap();
    let path = temp.path().join("source.resin");
    fs::write(&path, source).unwrap();
    pipeline::file_module(&path).unwrap_err().to_string()
}

#[test]
fn foreign_functions_forward_separate_c_arguments() {
    let temp = TempDir::new_in(std::env::temp_dir()).unwrap();
    let header = temp.path().join("foreign.h");
    fs::write(&header, "static inline int answer(void) { return 42; }\nstatic inline void assign(int *out, int value) { *out = value; }\n").unwrap();
    let source = format!(
        r#"export {{ main }};

        extern {{
            "{header}": {{
                fn answer () -> int;
                fn assign (out: Ptr<int>, value: int);
            }},
            "stdlib.h": {{
                fn abs (n: int) -> int;
            }},
        }};

        fn call (f: () -> int) -> int  {{ f() }}
        fn main () -> int  {{
            let mut value = 0;
            let mut set = assign;
            set(&value, call(answer));
            abs(-value)
        }}"#,
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

        struct FieldsValue<T0> { value: T0, }
fn identity(p: Ptr<FieldsValue<int>>, calls: Ptr<int>) -> Ptr<FieldsValue<int>>  {
            calls.* = calls.* + 1;
            p
        }
        fn main () -> int  {
            let mut calls = 0;
            let mut record = FieldsValue<_> { value = 1 };
            let mut pointer = &identity(&record, &calls).value;
            let mut copy = Ptr<int>(ulong(pointer));
            copy.* = 41;
            record.value + calls
        }
    "#);
    assert_eq!(output.status.code(), Some(42));
}

#[test]
fn foreign_aggregate_values_and_implicit_pointer_casts_are_rejected() {
    for source in [
        "extern { \"native.h\": { fn consume (value: Native) -> (); } }; extern type Native;",
        "extern { \"native.h\": { fn consume (value: FieldsX<int>) -> (); } }; struct FieldsX<T0> { x: T0, }",
        "extern { \"native.h\": { fn produce () -> FieldsX<int>; } }; struct FieldsX<T0> { x: T0, }",
        "extern { \"native.h\": { fn callback (f: () -> int) -> (); } };",
    ] {
        assert!(
            error(source).contains("InvalidForeignSignature"),
            "{source}"
        );
    }
    assert!(
        error("extern { \"bad\\nheader\": { fn invalid () -> int; } };")
            .contains("InvalidForeignHeader")
    );
    for source in [
        "export { main }; extern type Native; fn main() -> ()  { let mut value: Native; }",
        "extern type Native; fn identity (n: Native) -> Native  { n }",
        "extern type Native; struct Wrapped { value: Native, }",
        "extern type Native; fn read (n: Ptr<Native>) -> ()  { n.*; }",
    ] {
        assert!(error(source).contains("OpaqueValue"), "{source}");
    }
    for source in [
        "export { main }; fn f (p: Ptr<int>) -> ()  {} fn main() -> ()  { let mut x = 0; f(ulong(0)); }",
        "export { main }; fn f (p: Ptr<ubyte>) -> ()  {} fn main() -> ()  { let mut x: int; x = 0; f(&x); }",
        "export { main }; fn main() -> ()  { let mut x = Ptr<int>(float32(0.0)); }",
    ] {
        assert!(error(source).contains("TypeMismatch"), "{source}");
    }
    assert!(
        error("export { main }; fn main() -> ()  { let mut p = &(1 + 2); }").contains("NotAPlace")
    );
}

#[test]
fn imports_are_relative_deduplicated_and_checked_for_cycles() {
    let temp = TempDir::new_in(std::env::temp_dir()).unwrap();
    let main = temp.path().join("main.resin");
    let nested = temp.path().join("nested");
    fs::create_dir(&nested).unwrap();
    fs::write(
        temp.path().join("common.resin"),
        "export { helper }; fn helper () -> int  { 42 }",
    )
    .unwrap();
    fs::write(
        nested.join("library.resin"),
        "export { helper }; import { \"../common.resin\" };",
    )
    .unwrap();
    fs::write(
        &main,
        "export { main }; import { \"nested/library.resin\", \"common.resin\" }; fn main () -> int  { helper() }",
    )
    .unwrap();
    let module = pipeline::file_module(&main).unwrap();
    assert_eq!(module.functions.len(), 2);
    fs::write(
        temp.path().join("common.resin"),
        "import { \"main.resin\" };",
    )
    .unwrap();
    assert!(
        pipeline::load(&main)
            .unwrap_err()
            .to_string()
            .contains("cyclic source import")
    );
    fs::write(temp.path().join("common.resin"), "fn invalid").unwrap();
    assert!(
        pipeline::load(&main)
            .unwrap_err()
            .to_string()
            .contains("common.resin")
    );
}

#[test]
fn shader_declarations_validate_signatures_and_do_not_expose_bytecode() {
    for source in [
        "@compute_shader fn kernel(index: ulong, output: Ptr<uint>)  {} fn main()  { kernel.spirv; }",
        "fn kernel(i: uint) -> uint  { i } fn main()  { let mut code = kernel.spirv; }",
        "@geometry_shader fn kernel(i: uint) -> uint  { i }",
        "@compute_shader @vertex_shader fn kernel(i: uint) -> uint  { i }",
        "@compute_shader fn kernel(i: int) -> int  { i }",
        "@compute_shader fn kernel(i: uint) -> uint  { i }",
        "@compute_shader fn kernel(invocation: ulong, output: Ptr<uint>)  { let mut i = uint(invocation); output.* = { i }; } fn main()  { let mut alias = kernel; let mut code = alias.spirv; }",
        "extern { \"stdlib.h\": { fn abs(i: int) -> int; } }; fn main()  { let mut code = abs.spirv; }",
    ] {
        assert!(!error(source).is_empty(), "{source}");
    }
}

#[test]
fn shader_is_an_ordinary_available_function_name() {
    module("fn shader(n: int) -> int  { n + 1 }");
}

#[test]
fn compute_declarations_require_ulong_indices() {
    for ty in ["uint", "int", "long"] {
        let source = format!(
            "@compute_shader fn kernel(index: {ty}, output: Ptr<ulong>)  {{ output.* = ulong(index); }}"
        );
        assert!(error(&source).contains("expected (ulong, Ptr<T>)"));
    }
    module("@compute_shader fn kernel(index: ulong, output: Ptr<ulong>)  { output.* = index; }");
}
