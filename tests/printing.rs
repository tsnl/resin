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
    run_c(&c::emit(&module(source)).unwrap())
}

fn run_c(source: &str) -> std::process::Output {
    let temp = TempDir::new(&std::env::temp_dir()).unwrap();
    let executable = temp.path().join("program");
    let cc = std::env::var_os("CC").unwrap_or_else(|| OsString::from("cc"));
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
    let m = module(r#"n = 42; print("x = {0}\n", (n,));"#);
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
    prints(r#"n = 42; print("x = {0}\n", (n,));"#, b"x = 42\n");
}

#[test]
fn formats_are_length_delimited_and_do_not_add_newlines() {
    prints(
        r#"print("", ()); print("héllo\t\"\\\r\n\0%", ());"#,
        "héllo\t\"\\\r\n\0%".as_bytes(),
    );
    prints(
        r#"print("{{{1}}}: {0}, {1}", ("{not a format}%\0", "世界"));"#,
        "{世界}: {not a format}%\0, 世界".as_bytes(),
    );
    prints(r#"fmt = "{0}!"; main = () => print(fmt, ("",));"#, b"!");
}

#[test]
fn numeric_widths_and_scalar_types() {
    prints(r#"print("{0} {1} {2} {3} {4} {5} {6} {7}\n", (
        sbyte (-128), short (-32768), int (-2147483648), long (-9223372036854775808),
        ubyte (255), ushort (65535), uint (4294967295), ulong (18446744073709551615)
    )); print("{0} {1} {2} {3} {4}", (float32 (1.2), float64 (1.25), 1 == 1, 1 == 2, ()));"#,
    b"-128 -32768 -2147483648 -9223372036854775808 255 65535 4294967295 18446744073709551615\n1.2 1.25 true false ()");
}

#[test]
fn nominal_scalars_are_unwrapped() {
    prints(
        r#"Meters = int; Distance = Meters; print("{0}", (Distance (Meters (42)),));"#,
        b"42",
    );
}

#[test]
fn arguments_evaluate_once_in_source_order_even_when_unused() {
    prints(
        r#"n = 0; print("{1} {0} {1}", ((n := n + 1), (n := n + 1), (n := n + 1))); print(" {0}", (n,));"#,
        b"2 1 2 3",
    );
}

#[test]
fn closures_and_recursive_functions_can_print() {
    prints(
        r#"make = (n: int) => () => print("{0}", (n,)); f = make(4); f();
        countdown = (n: int) => int { print("{0}", (n,)); if (n > 0) { countdown(n - 1) } else { 0 } };
        main = () => countdown(2);"#,
        b"4210",
    );
}

#[test]
fn print_can_be_shadowed_by_an_ordinary_function() {
    prints(
        r#"main = () => { print = (n: int) => { n + 1 }; if (print(41) == 42) { 0 } else { 1 } };"#,
        b"",
    );
    prints(
        r#"apply = (print: (int) -> int) => print(41); main = () => { apply((n: int) => { n - 41 }) };"#,
        b"",
    );
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
        let output = run(&format!("print({format:?}, (42,));"));
        assert_eq!(output.status.code(), Some(1), "{format}");
        assert!(output.stdout.is_empty(), "{format}");
        assert!(String::from_utf8_lossy(&output.stderr).contains("resin:"));
    }
    assert_eq!(run(r#"print("{0}", ());"#).status.code(), Some(1));
}

#[test]
fn invalid_print_types_are_rejected() {
    for (source, diagnostic) in [
        (r#"print("{0}", 1);"#, "InvalidPrintArguments"),
        (r#"print(1, (2,));"#, "InvalidPrintArguments"),
        (r#"print("hello");"#, "InvalidPrintArguments"),
        (r#"print("{0}", (1,), (2,));"#, "InvalidPrintArguments"),
        (r#"print("{0}", ({ x = 1 },));"#, "UnprintableType"),
        (r#"print("{0}", ([1, 2],));"#, "UnprintableType"),
        (r#"print("{0}", (() => 1,));"#, "UnprintableType"),
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
    let m = module(r#"kernel = (i: uint) => { print("{0}", (i,)); i };"#);
    let error = glsl::emit(&m, "kernel", glsl::Stage::Compute).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("print is only supported in host programs")
    );
}

#[test]
fn generated_c_uses_the_shared_runtime_header() {
    let source = c::emit(&module(r#"print("hello", ());"#)).unwrap();
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
