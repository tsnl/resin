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
        r#"export { main }; import { "$/string.resin", "$/stdio.resin" };
        fn output(value: str)  { print(value) }
        fn main()  { output("x = "); { let borrowed = fmt("{0}\n", (42,)); print(borrowed) }; }"#,
        b"x = 42\n",
    );
}

#[test]
fn printing_and_string_construction_have_separate_module_exports() {
    prints(
        r#"export { main }; import { "$/stdio.resin" };
        fn main() { print("hello\0world"); }"#,
        b"hello\0world",
    );
    for source in [
        r#"import { "$/string.resin" }; fn main() { print("unexpected"); }"#,
        r#"import { "$/stdio.resin" }; fn main() { fmt("unexpected", ()); }"#,
    ] {
        let error = pipeline::source_module(source).unwrap_err();
        assert!(error.to_string().contains("UnboundValue"), "{error}");
    }
}

#[test]
fn formats_are_length_delimited_and_do_not_add_newlines() {
    prints(
        r#"export { main }; import { "$/string.resin", "$/stdio.resin" }; fn main() -> ()  { print(""); print("héllo\t\"\\\r\n\0%"); }"#,
        "héllo\t\"\\\r\n\0%".as_bytes(),
    );
    prints(
        r#"export { main }; import { "$/string.resin", "$/stdio.resin" }; fn main() -> ()  { let text = fmt("{{{1}}}: {0}, {1}", ("{not a format}%\0", "世界")); print(text); }"#,
        "{世界}: {not a format}%\0, 世界".as_bytes(),
    );
    prints(
        r#"export { main }; import { "$/string.resin", "$/stdio.resin" }; fn main () -> ()  { let mut format_text = "{0}!"; { let borrowed = fmt(format_text, ("",)); print(borrowed) }; }"#,
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
                fn strlen(text: Ptr<u8>) -> u64;
            },
        };
        import { "$/string.resin", "$/stdio.resin" };

        struct FieldsText<T0> { text: T0, }
fn main() -> i32  {
            let mut text = "héllo";
            let mut empty = "";
            let mut copy = text;
            let mut record = FieldsText<_> { text = copy };
            let mut pointer = record.text.data;
            if (strlen(pointer) == u64(6) && Ptr<u8>(u64(pointer) + u64(6)).* == u8(0)
                && strlen(empty.data) == u64(0)) {
                { let borrowed = fmt("{0}{1}", (text, empty)); print(borrowed) };
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
                fn strlen(text: Ptr<u8>) -> u64;
            },
        };
        import { "$/string.resin", "$/stdio.resin" };

        fn main() -> i32  {
            let mut text = "a\0b\0";
            let mut pointer = text.data;
            if (strlen(pointer) == u64(1) && Ptr<u8>(u64(pointer) + u64(2)).* == u8(98)
                && Ptr<u8>(u64(pointer) + u64(3)).* == u8(0) && Ptr<u8>(u64(pointer) + u64(4)).* == u8(0)) {
                { let borrowed = fmt("before\0{0}after", (text,)); print(borrowed) };
                0
            } else { 1 }
        }"#,
        b"before\0a\0b\0after",
    );
    prints(
        r#"export { main }; import { "$/shared.resin", "$/string.resin", "$/stdio.resin", "$/span.resin" }; fn main() -> () | Err<_> { let buffer_owner = arc_ptr_alloc([u8(65), u8(0), u8(66), u8(0)])?; let buffer: Ref<_> = buffer_owner:get().*; { let borrowed_1 = fmt("{0}", ({ let borrowed = Span<u8> { data = Ptr<u8>(buffer_owner:get()), length = u64(4) }; borrowed:bytes() },)); print(borrowed_1) }; }"#,
        b"A\0B\0",
    );
}

#[test]
fn numeric_widths_and_scalar_types() {
    prints(r#"export { main }; import { "$/string.resin", "$/stdio.resin" };

fn main() -> ()  {
    { let borrowed = fmt("{0} {1} {2} {3} {4} {5} {6} {7}\n", (
        i8(-128), i16(-32768), i32(-2147483648), i64(-9223372036854775808),
        u8(255), u16(65535), u32(4294967295), u64(18446744073709551615)
    )); print(borrowed) };
    { let borrowed = fmt("{0} {1} {2} {3} {4}", (f32(1.2), f64(1.25), 1 == 1, 1 == 2, ())); print(borrowed) };
}"#,
    b"-128 -32768 -2147483648 -9223372036854775808 255 65535 4294967295 18446744073709551615\n1.2 1.25 true false ()");
}

#[test]
fn aliases_preserve_scalar_printing() {
    prints(
        r#"export { main }; import { "$/string.resin", "$/stdio.resin" }; type Meters = i32; type Distance = Meters; fn main() -> ()  { { let borrowed = fmt("{0}", (Distance(Meters(42)),)); print(borrowed) }; }"#,
        b"42",
    );
}

#[test]
fn arguments_evaluate_once_in_source_order_even_when_unused() {
    prints(
        r#"export { main }; import { "$/string.resin", "$/stdio.resin" }; fn main() -> ()  { let mut n = 0; { let borrowed = fmt("{1} {0} {1}", ({ n = n + 1; n }, { n = n + 1; n }, { n = n + 1; n })); print(borrowed) }; { let borrowed = fmt(" {0}", (n,)); print(borrowed) }; }"#,
        b"2 1 2 3",
    );
}

#[test]
fn ordinary_and_recursive_functions_can_print() {
    prints(
        r#"export { main }; import { "$/string.resin", "$/stdio.resin" };
        fn show (n: i32) -> ()  { { let borrowed = fmt("{0}", (n,)); print(borrowed) }; }
        fn countdown (n: i32) -> i32  { { let borrowed = fmt("{0}", (n,)); print(borrowed) }; if (n > 0) { countdown(n - 1) } else { 0 } }
        fn main () -> i32  {
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
    let output = run(
        r#"export { main }; import { "$/string.resin", "$/stdio.resin" };
        fn increment(value: i32) -> i32  { value + 1 }
        fn apply(print: (i32) -> i32) -> i32  { print(41) }
        fn main() -> i32  {
            let mut print = 7;
            if (print == 7 && apply(increment) == 42) { 0 } else { 1 }
        }
    "#,
    );
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
            "export {{ main }}; import {{ \"$/string.resin\", \"$/stdio.resin\" }}; fn main() -> ()  {{ let text = fmt({format:?}, (42,)); print(text); }}"
        ));
        assert_eq!(output.status.code(), Some(1), "{format}");
        assert!(output.stdout.is_empty(), "{format}");
        assert!(String::from_utf8_lossy(&output.stderr).contains("resin:"));
    }
    assert_eq!(
        run(r#"export { main }; import { "$/string.resin", "$/stdio.resin" }; fn main() -> ()  { { let borrowed = fmt("{0}", ()); print(borrowed) }; }"#)
            .status
            .code(),
        Some(1)
    );
}

#[test]
fn invalid_print_types_are_rejected() {
    for (source, diagnostic) in [
        (
            r#"export { main }; import { "$/string.resin", "$/stdio.resin" }; fn main() -> ()  { { let borrowed = fmt("{0}", 1); print(borrowed) }; }"#,
            "InvalidFormatArguments",
        ),
        (
            r#"export { main }; import { "$/string.resin", "$/stdio.resin" }; fn main() -> ()  { print(fmt(1, (2,))); }"#,
            "no overload of `fmt` matches",
        ),
        (
            r#"export { main }; import { "$/string.resin", "$/stdio.resin" }; fn main() -> ()  { print(fmt("hello")); }"#,
            "no overload of `fmt` matches",
        ),
        (
            r#"export { main }; import { "$/string.resin", "$/stdio.resin" }; fn main() -> ()  { print(fmt("{0}", (1,), (2,))); }"#,
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
        r#"export { kernel }; import { "$/string.resin", "$/stdio.resin", "$/span.resin" }; @compute_shader fn kernel(invocation: u64, text: Ptr<Span<u8>>)  { print(text.*); }"#,
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
            r#"export { main }; import { "$/string.resin", "$/stdio.resin" }; fn main() -> ()  { print("hello"); }"#,
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
        r#"export { main }; import { "$/string.resin", "$/stdio.resin", "$/shared.resin" };
        fn make() -> String  { fmt("{0}\0{1}", ("hi", 42)) }
        fn main() -> i32  {
            let mut weak = weak_span_empty::<u8>();
            {
                let mut original = make();
                weak = original.storage:downgrade();
                let mut alias = original;
                original = fmt("replacement", ());
                print(alias);
                let pattern = fmt("{{0}} {0}", (7,));
                let text = fmt(pattern, (alias:bytes(),));
                print(text);
            };
            match (weak:upgrade()) {
                None => { 0 },
                ArcSpan<u8>(live) => { 1 },
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
                fn strlen(p: Ptr<u8>) -> u64;
            },
        };
        import { "$/string.resin", "$/stdio.resin" };
        fn literal() -> str  { "a\0b" }
        fn main() -> i32  {
            let mut text = literal();
            let mut copy = text;
            print(copy);
            if (text.length == u64(3) && strlen(text.data) == u64(1) && text:at(u64(2)) == u8(98)) { 0 } else { 1 }
        }"#,
        b"a\0b",
    );
}

#[test]
fn from_bytes_copies_unterminated_spans_verbatim_and_owns_the_result() {
    prints(
        r#"export { main }; import { "$/string.resin", "$/stdio.resin", "$/span.resin", "$/shared.resin" };
        type Caption = String;
        fn copied() -> String | Err<_> {
            let source = arc_ptr_alloc([u8(65), u8(0), u8(66)])?;
            let mut result = { let borrowed = Span<u8> { data = source:get():lea(u64(0)), length = u64(3) }; string_from_bytes(borrowed) };
            source:get():at_mut(u64(0)) = u8(90);
            result
        }
        fn main() -> i32 | Err<_> {
            let mut weak = weak_span_empty::<u8>();
            {
                let mut text = copied()?;
                let alias = text:clone();
                weak = text.storage:downgrade();
                text = string_from_str("{0}} braces");
                print(alias);
                print(text);
                let mut end = Ptr<u8>(u64(alias:get().data) + u64(3));
                if (alias:get().length != u64(3) || end.* != u8(0)) { print("bad terminator"); };
                let mut empty = { let borrowed = Span<u8> { data = Ptr<u8>(u64(0)), length = u64(0) }; string_from_bytes(borrowed) };
                if (empty:get().length != u64(0) || empty:get().data.* != u8(0)) { print("bad empty string"); };
            };
            match (weak:upgrade()) {
                None => { 0 },
                ArcSpan<u8>(live) => { 1 },
            }
        }
    "#,
        b"A\0B{0}} braces",
    );
}

#[test]
fn byte_spans_print_their_length_including_nuls_and_empty_views() {
    prints(
        r#"export { main }; import { "$/shared.resin", "$/string.resin", "$/stdio.resin", "$/span.resin" }; fn main() -> () | Err<_> {
            let buffer_owner = arc_ptr_alloc([u8(65), u8(0), u8(66), u8(67)])?; let buffer: Ref<_> = buffer_owner:get().*;
            let mut view = Span<u8> { data = buffer_owner:get():lea(0), length = u64(3) };
            let mut empty = Span<u8> { data = Ptr<u8>(u64(0)), length = u64(0) };
            { let borrowed = fmt("[{0}][{1}]", (view:bytes(), empty:bytes())); print(borrowed) };
        }"#,
        b"[A\0B][]",
    );
}

#[test]
fn literal_strings_have_a_distinct_type_and_require_explicit_byte_views() {
    let m = module(
        r#"import { "$/span.resin" }; fn literal() -> str  { "bytes" } fn view() -> Span<u8>  { bytes("bytes") }"#,
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
        r#"import { "$/string.resin", "$/span.resin" }; fn bad() -> Span<u8>  { "bytes" }"#,
        r#"import { "$/string.resin", "$/span.resin" }; fn bad(bytes: Span<u8>) -> str  { str(bytes) }"#,
        r#"import { "$/string.resin", "$/span.resin" }; fn bad() -> str  { str() }"#,
        r#"import { "$/string.resin", "$/span.resin" }; fn bad() -> str  { str { data = "bytes".data, length = u64(5) } }"#,
        r#"import { "$/string.resin", "$/span.resin" }; fn bad()  { string_from_str(bytes("bytes")); }"#,
        r#"import { "$/string.resin", "$/span.resin" }; fn bad()  { string_from_bytes("bytes"); }"#,
    ] {
        assert!(pipeline::source_module(source).is_err(), "{source}");
    }
}

#[test]
fn literal_byte_views_preserve_storage_while_owned_strings_copy_it() {
    prints(
        r#"export { main }; import { "$/string.resin", "$/stdio.resin", "$/span.resin" };
        fn view(text: str) -> Span<u8>  { bytes(text) }
        fn main() -> i32  {
            let mut text = "hé\0";
            let mut buffer = view(text);
            let mut owned = string_from_str(text);
            let mut empty = string_from_str("");
            if (text.length == u64(4) && buffer.length == text.length &&
                u64(buffer.data) == u64(text.data) && text:at(u64(1)) == u8(195) &&
                u64(owned:get().data) != u64(text.data) && owned:get().length == u64(4) &&
                Ptr<u8>(u64(owned:get().data) + u64(4)).* == u8(0) &&
                empty:get().length == u64(0) && empty:get().data.* == u8(0)) {
                { let borrowed_1 = { let borrowed = view("{0}{1}{2}"); fmt(borrowed, (text, buffer:bytes(), owned:bytes())) }; print(borrowed_1) };
                0
            } else { 1 }
        }"#,
        "hé\0hé\0hé\0".as_bytes(),
    );
}

#[test]
fn raw_byte_views_and_owned_strings_preserve_non_utf8() {
    prints(
        r#"export { main }; import { "$/shared.resin", "$/string.resin", "$/stdio.resin", "$/span.resin" }; fn main() -> () | Err<_> {
            let data_owner = arc_ptr_alloc([u8(255), u8(0), u8(254)])?; let data: RefMut<_> = data_owner:get().*;
            let mut buffer = Span<u8> { data = data_owner:get():lea(u64(0)), length = u64(3) };
            let mut owned = string_from_bytes(buffer);
            data:at_mut(u64(0)) = u8(65);
            print(buffer);
            { let borrowed = fmt("{0}", (owned:bytes(),)); print(borrowed) };
        }"#,
        b"A\0\xfe\xff\0\xfe",
    );
}

#[test]
fn formats_owned_results_from_functions() {
    prints(
        r#"export { main }; import { "$/string.resin", "$/stdio.resin" };
        fn text() -> String  { fmt("{0}+i{1}", (40, 85)) }
        fn main()  { { let borrowed = fmt("{0}\n", (text(),)); print(borrowed) }; }"#,
        b"40+i85\n",
    );
}

#[test]
fn repr_renders_fields_arrays_tuples_and_active_union_payloads() {
    prints(r#"export { main }; import { "$/string.resin", "$/stdio.resin" };
        struct Complex { real: f64, imaginary: f64, }
        struct Problem { message: str, detail: String, }
        fn main()  {
            { let borrowed = repr(Complex { real = -1.0, imaginary = 2.5 }); print(borrowed) };
            print("\n");
            { let borrowed = repr(([1, 2, 3], true, None, "a\n\0\"\\世界")); print(borrowed) };
            print("\n");
            let mut problem: i32 | Problem = Problem { message = "bad", detail = string_from_str("input") };
            { let borrowed = repr(problem); print(borrowed) };
            print("\n");
            { let borrowed = fmt("{0} {1}", ([4, 5], string_from_str("raw"))); print(borrowed) };
        }"#,
        "Complex { real = -1, imaginary = 2.5 }\n([1, 2, 3], true, None, \"a\\n\\0\\\"\\\\世界\")\nProblem { message = \"bad\", detail = \"input\" }\n[4, 5] raw".as_bytes());
}

#[test]
fn entry_errors_display_owned_payload_contents() {
    let output = run(r#"export { main }; import { "$/string.resin" };
        struct Problem { message: String, code: i32, }
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
fn repr_bytes(self: Bad) -> i32  { 1 }

        fn main()  { let mut bad = Bad {}; }"#,
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("repr_bytes must take"), "{error}");
}

#[test]
fn free_generic_text_hooks_format_values_and_print_borrows_owned_strings() {
    prints(
        r#"export { main }; import { "$/string.resin", "$/stdio.resin" };
        struct Label<T> { text: str, value: T }
        fn repr_bytes<T>(value: Ref<Label<T>>) -> (Ptr<u8>, u64) {
            (value.text.data, value.text.length)
        }
        fn main() {
            let text = string_from_str("again");
            print(text);
            print(text);
            { let borrowed = fmt(" {0} {1}", (
                Label<i32> { text = "integer", value = 42 },
                Label<bool> { text = "boolean", value = true },
            )); print(borrowed) };
        }"#,
        b"againagain integer boolean",
    );
}

#[test]
fn verifier_rejects_invalid_text_view_callbacks() {
    let mut module = module(
        r#"export { main }; import { "$/string.resin", "$/stdio.resin" };
        fn main()  { { let borrowed = repr(string_from_str("x")); print(borrowed) }; }"#,
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
        r#"export { main }; import { "$/string.resin", "$/stdio.resin" };
        fn main()  { { let borrowed = repr(Ptr<i32>(u64(1))); print(borrowed) }; }"#,
        b"0x1",
    );
}
