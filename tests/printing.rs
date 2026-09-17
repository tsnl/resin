use resin_types::prelude::*;
use std::{ffi::OsString, process::Command};
use support::pipeline;
use support::toolchain;
use tempfile::TempDir;

mod support;
use support::module;

fn run(source: &str) -> std::process::Output {
    support::project::Project::new(&module(source), Some("main"))
        .unwrap()
        .run()
}

fn run_c(source: &str) -> std::process::Output {
    let temp = TempDir::new_in(std::env::temp_dir()).unwrap();
    let executable = temp
        .path()
        .join(format!("program{}", std::env::consts::EXE_SUFFIX));
    let cc = std::env::var_os("CC")
        .unwrap_or_else(|| OsString::from(resin_toolchain::DEFAULT_C_COMPILER));
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
fn print_is_an_ordinary_source_function_returning_unit() {
    prints(
        r#"export { main }; import { "$/string.resin" };
        fn output(value: str)  { print(value) }
        fn main()  { output("x = "); print(fmt("{0}\n", (42,))); }"#,
        b"x = 42\n",
    );
}

#[test]
fn formats_are_length_delimited_and_do_not_add_newlines() {
    prints(
        r#"export { main }; import { "$/string.resin" }; fn main() -> ()  { print(""); print("héllo\t\"\\\r\n\0%"); }"#,
        "héllo\t\"\\\r\n\0%".as_bytes(),
    );
    prints(
        r#"export { main }; import { "$/string.resin" }; fn main() -> ()  { print(fmt("{{{1}}}: {0}, {1}", ("{not a format}%\0", "世界"))); }"#,
        "{世界}: {not a format}%\0, 世界".as_bytes(),
    );
    prints(
        r#"export { main }; import { "$/string.resin" }; fn main () -> ()  { let mut format_text = "{0}!"; print(fmt(format_text, ("",))); }"#,
        b"!",
    );
}

#[test]
fn string_storage_is_terminated_without_changing_its_logical_length() {
    let m = module(
        "export { main }; fn main() -> ()  { let mut text = \"hello\"; let mut empty = \"\"; }",
    );
    for local in m.functions[0].locals.iter().skip(1) {
        assert_eq!(local.ty, Ty::Str);
    }
    let project = support::project::Project::new(&m, Some("main")).unwrap();
    let c = std::fs::read_to_string(project.generated.c_source().unwrap()).unwrap();
    assert!(c.contains("static uint8_t r_literal_"), "{c}");
    prints(
        r#"export { main };

        extern {
            "string.h": {
                fn strlen(text: Ptr<ubyte>) -> ulong;
            },
        };
        import { "$/string.resin" };

        struct FieldsText<T0> { text: T0, }
fn main() -> int  {
            let mut text = "héllo";
            let mut empty = "";
            let mut copy = text;
            let mut record = FieldsText<_> { text = copy };
            let mut pointer = record.text.data;
            if (strlen(pointer) == ulong(6) && Ptr<ubyte>(ulong(pointer) + ulong(6)).* == ubyte(0)
                && strlen(empty.data) == ulong(0)) {
                print(fmt("{0}{1}", (text, empty)));
                0
            } else { 1 }
        }"#,
        "héllo".as_bytes(),
    );
}

#[test]
fn embedded_and_explicit_trailing_nuls_are_not_truncated() {
    prints(
        r#"export { main };

        extern {
            "string.h": {
                fn strlen(text: Ptr<ubyte>) -> ulong;
            },
        };
        import { "$/string.resin" };

        fn main() -> int  {
            let mut text = "a\0b\0";
            let mut pointer = text.data;
            if (strlen(pointer) == ulong(1) && Ptr<ubyte>(ulong(pointer) + ulong(2)).* == ubyte(98)
                && Ptr<ubyte>(ulong(pointer) + ulong(3)).* == ubyte(0) && Ptr<ubyte>(ulong(pointer) + ulong(4)).* == ubyte(0)) {
                print(fmt("before\0{0}after", (text,)));
                0
            } else { 1 }
        }"#,
        b"before\0a\0b\0after",
    );
    prints(
        r#"export { main }; import { "$/shared.resin", "$/string.resin", "$/span.resin" }; fn main() -> () | Err<_> { let buffer_owner = arc_ptr_alloc([ubyte(65), ubyte(0), ubyte(66), ubyte(0)])?; let buffer: Ref<_> = buffer_owner:get().*; print(fmt("{0}", (Span<ubyte> { data = Ptr<ubyte>(buffer_owner:get()), length = 4_ul }:bytes(),))); }"#,
        b"A\0B\0",
    );
}

#[test]
fn numeric_widths_and_scalar_types() {
    prints(r#"export { main }; import { "$/string.resin" };

fn main() -> ()  {
    print(fmt("{0} {1} {2} {3} {4} {5} {6} {7}\n", (
        sbyte(-128), short(-32768), int(-2147483648), long(-9223372036854775808),
        ubyte(255), ushort(65535), uint(4294967295), ulong(18446744073709551615)
    )));
    print(fmt("{0} {1} {2} {3} {4}", (float32(1.2), float64(1.25), 1 == 1, 1 == 2, ())));
}"#,
    b"-128 -32768 -2147483648 -9223372036854775808 255 65535 4294967295 18446744073709551615\n1.2 1.25 true false ()");
}

#[test]
fn aliases_preserve_scalar_printing() {
    prints(
        r#"export { main }; import { "$/string.resin" }; type Meters = int; type Distance = Meters; fn main() -> ()  { print(fmt("{0}", (Distance(Meters(42)),))); }"#,
        b"42",
    );
}

#[test]
fn arguments_evaluate_once_in_source_order_even_when_unused() {
    prints(
        r#"export { main }; import { "$/string.resin" }; fn main() -> ()  { let mut n = 0; print(fmt("{1} {0} {1}", ({ n = n + 1; n }, { n = n + 1; n }, { n = n + 1; n }))); print(fmt(" {0}", (n,))); }"#,
        b"2 1 2 3",
    );
}

#[test]
fn ordinary_and_recursive_functions_can_print() {
    prints(
        r#"export { main }; import { "$/string.resin" };
        fn show (n: int) -> ()  { print(fmt("{0}", (n,))); }
        fn countdown (n: int) -> int  { print(fmt("{0}", (n,))); if (n > 0) { countdown(n - 1) } else { 0 } }
        fn main () -> int  {
            let mut f = show;
            f(4);
            countdown(2)
        }
        "#,
        b"4210",
    );
}

#[test]
fn imported_print_can_be_shadowed_by_a_local_or_parameter() {
    let output = run(r#"export { main }; import { "$/string.resin" };
        fn increment(value: int) -> int  { value + 1 }
        fn apply(print: (int) -> int) -> int  { print(41) }
        fn main() -> int  {
            let mut print = 7;
            if (print == 7 && apply(increment) == 42) { 0 } else { 1 }
        }
    "#);
    assert!(output.status.success());
    assert!(output.stdout.is_empty());
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
            "export {{ main }}; import {{ \"$/string.resin\" }}; fn main() -> ()  {{ print(fmt({format:?}, (42,))); }}"
        ));
        assert_eq!(output.status.code(), Some(1), "{format}");
        assert!(output.stdout.is_empty(), "{format}");
        assert!(String::from_utf8_lossy(&output.stderr).contains("resin:"));
    }
    assert_eq!(
        run(r#"export { main }; import { "$/string.resin" }; fn main() -> ()  { print(fmt("{0}", ())); }"#)
            .status
            .code(),
        Some(1)
    );
}

#[test]
fn invalid_print_types_are_rejected() {
    for (source, diagnostic) in [
        (
            r#"export { main }; import { "$/string.resin" }; fn main() -> ()  { print(fmt("{0}", 1)); }"#,
            "InvalidFormatArguments",
        ),
        (
            r#"export { main }; import { "$/string.resin" }; fn main() -> ()  { print(fmt(1, (2,))); }"#,
            "no overload of `fmt` matches",
        ),
        (
            r#"export { main }; import { "$/string.resin" }; fn main() -> ()  { print(fmt("hello")); }"#,
            "no overload of `fmt` matches",
        ),
        (
            r#"export { main }; import { "$/string.resin" }; fn main() -> ()  { print(fmt("{0}", (1,), (2,))); }"#,
            "no overload of `fmt` matches",
        ),
    ] {
        let error = pipeline::source_module(source).unwrap_err().to_string();
        assert!(error.contains(diagnostic), "{source}: {error}");
    }
}

#[test]
fn shader_print_rejects_host_only_string_types() {
    let error = pipeline::shader_error(
        r#"export { kernel }; import { "$/string.resin", "$/span.resin" }; @compute_shader fn kernel(invocation: ulong, text: Ptr<Span<ubyte>>)  { print(text.*); }"#,
    );
    assert!(
        error.contains("shader cannot call foreign function resin_print"),
        "{error}"
    );
}

#[test]
fn generated_c_uses_the_shared_runtime_header() {
    let project = support::project::Project::new(
        &module(
            r#"export { main }; import { "$/string.resin" }; fn main() -> ()  { print("hello"); }"#,
        ),
        Some("main"),
    )
    .unwrap();
    let source = std::fs::read_to_string(project.generated.c_source().unwrap()).unwrap();
    assert_eq!(
        source
            .lines()
            .filter(|line| *line == "#include <resin_runtime.h>")
            .count(),
        1
    );
    assert!(source.contains("resin_print("));
    assert!(!source.contains("static void r_cleanup"));
}

#[test]
fn c_runtime_accepts_empty_buffers_and_pointer_values() {
    let output = run_c(
        r#"
        #include <resin_runtime.h>
        int main(void) {
            resin_print(NULL, 0);
            const ResinPrintArg args[] = {
                { .kind = RESIN_PRINT_POINTER, .value = { .unsigned_value = 0x1234 } },
                { .kind = RESIN_PRINT_BYTES, .value = { .bytes = { NULL, 0 } } }
            };
            ResinArc *text = resin_format((const uint8_t *)"{0}{1}", 6, args, 2);
            const uint8_t *bytes = resin_arc_data(text);
            resin_print(bytes, resin_arc_span_length(text));
            resin_arc_release(text);
            return 0;
        }
    "#,
    );
    assert!(output.status.success());
    assert_eq!(output.stdout, b"0x1234");
}

#[test]
fn formatted_strings_retain_storage_and_release_the_last_owner() {
    prints(
        r#"export { main }; import { "$/string.resin", "$/shared.resin" };
        fn make() -> String  { fmt("{0}\0{1}", ("hi", 42)) }
        fn main() -> int  {
            let mut weak = weak_span_empty::<ubyte>();
            {
                let mut original = make();
                weak = original.storage:downgrade();
                let mut alias = original;
                original = fmt("replacement", ());
                print(alias);
                print(fmt(fmt("{{0}} {0}", (7,)), (alias:bytes(),)));
            };
            match (weak:upgrade()) {
                None => { 0 },
                ArcSpan<ubyte>(live) => { 1 },
            }
        }
    "#,
        b"hi\x0042hi\x0042 7",
    );
}

#[test]
fn literal_strings_survive_returns_and_keep_explicit_nuls() {
    prints(
        r#"export { main };

        extern {
            "string.h": {
                fn strlen(p: Ptr<ubyte>) -> ulong;
            },
        };
        import { "$/string.resin" };
        fn literal() -> str  { "a\0b" }
        fn main() -> int  {
            let mut text = literal();
            let mut copy = text;
            print(copy);
            if (text.length == 3_ul && strlen(text.data) == 1_ul && text:at(2_ul) == 98_ub) { 0 } else { 1 }
        }"#,
        b"a\0b",
    );
}

#[test]
fn from_bytes_copies_unterminated_spans_verbatim_and_owns_the_result() {
    prints(
        r#"export { main }; import { "$/string.resin", "$/span.resin", "$/shared.resin" };
        type Caption = String;
        fn copied() -> String | Err<_> {
            let source = arc_ptr_alloc([65_ub, 0_ub, 66_ub])?;
            let mut result = string_from_bytes(Span<ubyte> { data = source:get():lea(0_ul), length = 3_ul });
            source:get():at(0_ul) = 90_ub;
            result
        }
        fn main() -> int | Err<_> {
            let mut weak = weak_span_empty::<ubyte>();
            {
                let mut text = copied()?;
                let alias = text:clone();
                weak = text.storage:downgrade();
                text = string_from_str("{0}} braces");
                print(alias);
                print(text);
                let mut end = Ptr<ubyte>(ulong(alias:get().data) + 3_ul);
                if (alias:get().length != 3_ul || end.* != 0_ub) { print("bad terminator"); };
                let mut empty = string_from_bytes(Span<ubyte> { data = Ptr<ubyte>(0_ul), length = 0_ul });
                if (empty:get().length != 0_ul || empty:get().data.* != 0_ub) { print("bad empty string"); };
            };
            match (weak:upgrade()) {
                None => { 0 },
                ArcSpan<ubyte>(live) => { 1 },
            }
        }
    "#,
        b"A\0B{0}} braces",
    );
}

#[test]
fn byte_spans_print_their_length_including_nuls_and_empty_views() {
    prints(
        r#"export { main }; import { "$/shared.resin", "$/string.resin", "$/span.resin" }; fn main() -> () | Err<_> {
            let buffer_owner = arc_ptr_alloc([65_ub, 0_ub, 66_ub, 67_ub])?; let buffer: Ref<_> = buffer_owner:get().*;
            let mut view = Span<ubyte> { data = buffer_owner:get():lea(0), length = 3_ul };
            let mut empty = Span<ubyte> { data = Ptr<ubyte>(0_ul), length = 0_ul };
            print(fmt("[{0}][{1}]", (view:bytes(), empty:bytes())));
        }"#,
        b"[A\0B][]",
    );
}

#[test]
fn literal_strings_have_a_distinct_type_and_require_explicit_byte_views() {
    let m = module(
        r#"import { "$/span.resin" }; fn literal() -> str  { "bytes" } fn view() -> Span<ubyte>  { bytes("bytes") }"#,
    );
    let literal = m
        .functions
        .iter()
        .find(|f| f.name.as_deref() == Some("literal"))
        .unwrap();
    let view = m
        .functions
        .iter()
        .find(|f| f.name.as_deref() == Some("view"))
        .unwrap();
    assert_eq!(literal.result, Ty::Str);
    assert!(matches!(view.result, Ty::Defined { .. }));
    assert_ne!(m.types.id(&literal.result), m.types.id(&view.result));
    for source in [
        r#"import { "$/string.resin", "$/span.resin" }; fn bad() -> Span<ubyte>  { "bytes" }"#,
        r#"import { "$/string.resin", "$/span.resin" }; fn bad(bytes: Span<ubyte>) -> str  { str(bytes) }"#,
        r#"import { "$/string.resin", "$/span.resin" }; fn bad() -> str  { str() }"#,
        r#"import { "$/string.resin", "$/span.resin" }; fn bad() -> str  { str { data = "bytes".data, length = 5_ul } }"#,
        r#"import { "$/string.resin", "$/span.resin" }; fn bad()  { string_from_str(bytes("bytes")); }"#,
        r#"import { "$/string.resin", "$/span.resin" }; fn bad()  { string_from_bytes("bytes"); }"#,
    ] {
        assert!(pipeline::source_module(source).is_err(), "{source}");
    }
}

#[test]
fn literal_byte_views_preserve_storage_while_owned_strings_copy_it() {
    prints(
        r#"export { main }; import { "$/string.resin", "$/span.resin" };
        fn view(text: str) -> Span<ubyte>  { bytes(text) }
        fn main() -> int  {
            let mut text = "hé\0";
            let mut buffer = view(text);
            let mut owned = string_from_str(text);
            let mut empty = string_from_str("");
            if (text.length == 4_ul && buffer.length == text.length &&
                ulong(buffer.data) == ulong(text.data) && text:at(1_ul) == 195_ub &&
                ulong(owned:get().data) != ulong(text.data) && owned:get().length == 4_ul &&
                Ptr<ubyte>(ulong(owned:get().data) + 4_ul).* == 0_ub &&
                empty:get().length == 0_ul && empty:get().data.* == 0_ub) {
                print(fmt(view("{0}{1}{2}"), (text, buffer:bytes(), owned:bytes())));
                0
            } else { 1 }
        }"#,
        "hé\0hé\0hé\0".as_bytes(),
    );
}

#[test]
fn raw_byte_views_and_owned_strings_preserve_non_utf8() {
    prints(
        r#"export { main }; import { "$/shared.resin", "$/string.resin", "$/span.resin" }; fn main() -> () | Err<_> {
            let data_owner = arc_ptr_alloc([255_ub, 0_ub, 254_ub])?; let data: Ref<_> = data_owner:get().*;
            let mut buffer = Span<ubyte> { data = data_owner:get():lea(0_ul), length = 3_ul };
            let mut owned = string_from_bytes(buffer);
            data:at(0_ul) = 65_ub;
            print(buffer);
            print(fmt("{0}", (owned:bytes(),)));
        }"#,
        b"A\0\xfe\xff\0\xfe",
    );
}

#[test]
fn formats_owned_temporary_results() {
    prints(
        r#"export { main }; import { "$/string.resin" };
        fn text() -> String  { fmt("{0}+i{1}", (40, 85)) }
        fn main()  { print(fmt("{0}\n", (text(),))); }"#,
        b"40+i85\n",
    );
}

#[test]
fn repr_renders_fields_arrays_tuples_and_active_union_payloads() {
    prints(r#"export { main }; import { "$/string.resin" };
        struct Complex { real: float64, imaginary: float64, }
        struct Problem { message: str, detail: String, }
        fn main()  {
            print(repr(Complex { real = -1.0, imaginary = 2.5 }));
            print("\n");
            print(repr(([1, 2, 3], true, None, "a\n\0\"\\世界")));
            print("\n");
            let mut problem: int | Problem = Problem { message = "bad", detail = string_from_str("input") };
            print(repr(problem));
            print("\n");
            print(fmt("{0} {1}", ([4, 5], string_from_str("raw"))));
        }"#,
        "Complex { real = -1, imaginary = 2.5 }\n([1, 2, 3], true, None, \"a\\n\\0\\\"\\\\世界\")\nProblem { message = \"bad\", detail = \"input\" }\n[4, 5] raw".as_bytes());
}

#[test]
fn entry_errors_display_owned_payload_contents() {
    let output = run(r#"export { main }; import { "$/string.resin" };
        struct Problem { message: String, code: int, }
        fn main() -> (() | Err<Problem>)  {
            Err(Problem { message = string_from_str("bad input"), code = 7 })
        }"#);
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(
        output.stderr,
        b"unhandled error: Problem { message = \"bad input\", code = 7 }\n"
    );
}

#[test]
fn text_representation_hooks_require_a_borrowed_receiver_and_byte_view() {
    let error = pipeline::source_module(
        r#"export { main };
        struct Bad {  }
fn repr_bytes(self: Bad) -> int  { 1 }

        fn main()  { let mut bad = Bad {}; }"#,
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("repr_bytes must take"), "{error}");
}

#[test]
fn free_generic_text_hooks_format_values_and_print_borrows_owned_strings() {
    prints(
        r#"export { main }; import { "$/string.resin" };
        struct Label<T> { text: str, value: T }
        fn repr_bytes<T>(value: Ref<Label<T>>) -> (Ptr<ubyte>, ulong) {
            (value.text.data, value.text.length)
        }
        fn main() {
            let text = string_from_str("again");
            print(text);
            print(text);
            print(fmt(" {0} {1}", (
                Label<int> { text = "integer", value = 42 },
                Label<bool> { text = "boolean", value = true },
            )));
        }"#,
        b"againagain integer boolean",
    );
}

#[test]
fn verifier_rejects_invalid_text_view_callbacks() {
    let mut module = module(
        r#"export { main }; import { "$/string.resin" };
        fn main()  { print(repr(string_from_str("x"))); }"#,
    );
    let hook = *module.text_views.values().next().unwrap();
    module.functions[hook.index()].result = Ty::Unit;
    let error = resin_lir::verify(&module).unwrap_err();
    assert!(matches!(
        error.kind,
        resin_lir::VerifyErrorKind::InvalidTextView
    ));
}

#[test]
fn repr_does_not_follow_pointers() {
    prints(
        r#"export { main }; import { "$/string.resin" };
        fn main()  { print(repr(Ptr<int>(1_ul))); }"#,
        b"0x1",
    );
}
