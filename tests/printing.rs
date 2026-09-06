use std::{ffi::OsString, process::Command};

use resin::{
    ast::AstGen,
    backend::{c, glsl},
    ir::{self, Instr, Ty},
    toolchain::{self, TempDir},
};

mod support;
use support::module;

fn run(source: &str) -> std::process::Output {
    run_c(&c::emit(&module(source), "main").unwrap())
}

fn run_c(source: &str) -> std::process::Output {
    let temp = TempDir::new(&std::env::temp_dir()).unwrap();
    let executable = temp
        .path()
        .join(format!("program{}", std::env::consts::EXE_SUFFIX));
    let cc = std::env::var_os("CC")
        .unwrap_or_else(|| OsString::from(resin::toolchain::DEFAULT_C_COMPILER));
    toolchain::compile_c(source, &executable, &cc)
        .unwrap_or_else(|error| panic!("{error}\n{source}"));
    Command::new(executable).output().unwrap()
}

fn prints(source: &str, expected: &[u8]) {
    let output = run(source);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.stdout, expected, "{source}");
    assert!(output.stderr.is_empty());
}

#[test]
fn print_is_unary_and_returns_unit() {
    let m =
        module(r#"export { main }; def main() -> () = { var n = 42; print("x = {0}\n", (n,)); };"#);
    let calls: Vec<_> = m
        .functions
        .iter()
        .flat_map(|f| &f.blocks)
        .flat_map(|b| &b.instrs)
        .filter_map(|instr| match instr {
            Instr::CallBuiltin {
                name,
                params,
                result,
            } if name.as_ref() == "print" => Some((params, result)),
            _ => None,
        })
        .collect();
    assert_eq!(calls.len(), 1);
    let (params, result) = calls[0];
    assert!(matches!(params.as_slice(), [Ty::Record { fields }] if fields.len() == 2));
    assert_eq!(result, &Ty::Unit);
    prints(
        r#"export { main }; def main() -> () = { var n = 42; print("x = {0}\n", (n,)); };"#,
        b"x = 42\n",
    );
}

#[test]
fn formats_are_length_delimited_and_do_not_add_newlines() {
    prints(
        r#"export { main }; def main() -> () = { print("", ()); print("héllo\t\"\\\r\n\0%", ()); };"#,
        "héllo\t\"\\\r\n\0%".as_bytes(),
    );
    prints(
        r#"export { main }; def main() -> () = { print("{{{1}}}: {0}, {1}", ("{not a format}%\0", "世界")); };"#,
        "{世界}: {not a format}%\0, 世界".as_bytes(),
    );
    prints(
        r#"export { main }; def main () -> () = { var fmt = "{0}!"; print(fmt, ("",)); };"#,
        b"!",
    );
}

#[test]
fn string_storage_is_terminated_without_changing_its_logical_length() {
    let m =
        module("export { main }; def main() -> () = { var text = \"hello\"; var empty = \"\"; };");
    for (local, length) in m.functions[0].locals.iter().skip(1).zip([5, 0]) {
        assert_eq!(
            local.ty,
            Ty::Array {
                element: Box::new(Ty::UInt8),
                length
            }
        );
    }
    let c = c::emit(&m, "main").unwrap();
    assert!(c.contains("items[5 + 1]"), "{c}");
    assert!(c.contains("items[0 + 1]"), "{c}");
    prints(
        r#"export { main };

        extern "string.h" def strlen(text: Ptr<ubyte>) -> ulong;
        def main() -> int = {
            var text = "héllo";
            var empty = "";
            var copy = text;
            var record = { text = copy };
            var pointer = Ptr<ubyte>(&record.text);
            if (strlen(pointer) == ulong(6) && (pointer + 6).* == ubyte(0)
                && strlen(Ptr<ubyte>(&empty)) == ulong(0)) {
                print("{0}{1}", (text, empty));
                0
            } else { 1 }
        };
    "#,
        "héllo".as_bytes(),
    );
}

#[test]
fn embedded_and_explicit_trailing_nuls_are_not_truncated() {
    prints(
        r#"export { main };

        extern "string.h" def strlen(text: Ptr<ubyte>) -> ulong;
        def main() -> int = {
            var text = "a\0b\0";
            var pointer = Ptr<ubyte>(&text);
            if (strlen(pointer) == ulong(1) && (pointer + 2).* == ubyte(98)
                && (pointer + 3).* == ubyte(0) && (pointer + 4).* == ubyte(0)) {
                print("before\0{0}after", (text,));
                0
            } else { 1 }
        };
    "#,
        b"before\0a\0b\0after",
    );
    prints(
        r#"export { main }; def main() -> () = { var bytes = [ubyte(65), ubyte(0), ubyte(66), ubyte(0)]; print("{0}", (bytes,)); };"#,
        b"A\0B\0",
    );
}

#[test]
fn numeric_widths_and_scalar_types() {
    prints(r#"export { main };

def main() -> () = {
    print("{0} {1} {2} {3} {4} {5} {6} {7}\n", (
        sbyte (-128), short (-32768), int (-2147483648), long (-9223372036854775808),
        ubyte (255), ushort (65535), uint (4294967295), ulong (18446744073709551615)
    ));
    print("{0} {1} {2} {3} {4}", (float32 (1.2), float64 (1.25), 1 == 1, 1 == 2, ()));
};"#,
    b"-128 -32768 -2147483648 -9223372036854775808 255 65535 4294967295 18446744073709551615\n1.2 1.25 true false ()");
}

#[test]
fn nominal_scalars_are_unwrapped() {
    prints(
        r#"export { main }; type Meters = int; type Distance = Meters; def main() -> () = { print("{0}", (Distance (Meters (42)),)); };"#,
        b"42",
    );
}

#[test]
fn arguments_evaluate_once_in_source_order_even_when_unused() {
    prints(
        r#"export { main }; def main() -> () = { var n = 0; print("{1} {0} {1}", ((n := n + 1), (n := n + 1), (n := n + 1))); print(" {0}", (n,)); };"#,
        b"2 1 2 3",
    );
}

#[test]
fn ordinary_and_recursive_functions_can_print() {
    prints(
        r#"
        export { main };
        def show (n: int) -> () = { print("{0}", (n,)); };
        def countdown (n: int) -> int = { print("{0}", (n,)); if (n > 0) { countdown(n - 1) } else { 0 } };
        def main () -> int = {
            var f = show;
            f(4);
            countdown(2)
        };
        "#,
        b"4210",
    );
}

#[test]
fn print_cannot_be_shadowed_by_a_local_or_parameter() {
    for source in [
        "export { main }; def main () -> () = { var print = 1; };",
        "def apply (print: (int) -> int) -> int = { print(41) };",
    ] {
        let error = ir::generate(&support::parse(source)).unwrap_err();
        assert!(
            matches!(error.kind, ir::GenerateErrorKind::ReservedBuiltin { .. }),
            "{error}"
        );
    }
}

#[test]
fn invalid_formats_fail_before_writing() {
    for format in [
        "prefix {",
        "prefix }",
        "prefix {}",
        "prefix {a}",
        "prefix {1}",
        "prefix {0:02}",
        "prefix {99999999999999999999999999}",
    ] {
        let output = run(&format!(
            "export {{ main }}; def main() -> () = {{ print({format:?}, (42,)); }};"
        ));
        assert_eq!(output.status.code(), Some(1), "{format}");
        assert!(output.stdout.is_empty(), "{format}");
        assert!(String::from_utf8_lossy(&output.stderr).contains("resin:"));
    }
    assert_eq!(
        run(r#"export { main }; def main() -> () = { print("{0}", ()); };"#)
            .status
            .code(),
        Some(1)
    );
}

#[test]
fn invalid_print_types_are_rejected() {
    for (source, diagnostic) in [
        (
            r#"export { main }; def main() -> () = { print("{0}", 1); };"#,
            "InvalidPrintArguments",
        ),
        (
            r#"export { main }; def main() -> () = { print(1, (2,)); };"#,
            "InvalidPrintArguments",
        ),
        (
            r#"export { main }; def main() -> () = { print("hello"); };"#,
            "InvalidPrintArguments",
        ),
        (
            r#"export { main }; def main() -> () = { print("{0}", (1,), (2,)); };"#,
            "InvalidPrintArguments",
        ),
        (
            r#"export { main }; def main() -> () = { print("{0}", ({ x = 1 },)); };"#,
            "UnprintableType",
        ),
        (
            r#"export { main }; def main() -> () = { print("{0}", ([1, 2],)); };"#,
            "UnprintableType",
        ),
        (
            r#"export { main }; def f () -> int = { 1 }; def main() -> () = { print("{0}", (f,)); };"#,
            "UnprintableType",
        ),
    ] {
        let mut parser = tree_sitter::Parser::new();
        parser
            .set_language(&tree_sitter_resin::LANGUAGE.into())
            .unwrap();
        let tree = parser.parse(source, None).unwrap();
        assert!(!tree.root_node().has_error(), "{source}");
        let ast = AstGen::new(source)
            .gen_source_file(tree.root_node())
            .unwrap();
        let error = ir::generate(&ast).unwrap_err().to_string();
        assert!(error.contains(diagnostic), "{source}: {error}");
    }
}

#[test]
fn shader_print_has_a_host_only_diagnostic() {
    let m =
        module(r#"export { kernel }; def kernel (i: uint) -> uint = { print("{0}", (i,)); i };"#);
    let error = glsl::emit(&m, "kernel", glsl::Stage::Compute).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("print is only supported in host programs")
    );
}

#[test]
fn generated_c_uses_the_shared_runtime_header() {
    let source = c::emit(
        &module(r#"export { main }; def main() -> () = { print("hello", ()); };"#),
        "main",
    )
    .unwrap();
    assert!(source.starts_with("#include <resin_runtime.h>\n"));
    assert!(source.contains("resin_print("));
    assert!(!source.contains("static void r_cleanup"));
}

#[test]
fn c_runtime_accepts_empty_buffers_and_pointer_values() {
    let output = run_c(
        r#"
        #include <resin_runtime.h>
        int main(void) {
            resin_print(NULL, 0, NULL, 0);
            const ResinPrintArg args[] = {
                { .kind = RESIN_PRINT_POINTER, .value = { .unsigned_value = 0x1234 } },
                { .kind = RESIN_PRINT_BYTES, .value = { .bytes = { NULL, 0 } } }
            };
            resin_print((const uint8_t *)"{0}{1}", 6, args, 2);
            return 0;
        }
    "#,
    );
    assert!(output.status.success());
    assert_eq!(output.stdout, b"0x1234");
}
