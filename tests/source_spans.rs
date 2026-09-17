#[allow(dead_code)]
mod support;

use resin_hir::Hir;
use resin_lir::Profile;
use resin_source::{Loader, Source};
use std::path::PathBuf;

fn build_hir(source: &str) -> Hir {
    let library = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("resin");
    let mut loader = Loader::new(library);
    support::frontend::analyze(Source::new("span-test.resin", source), &mut loader, None)
}

fn compile(
    source: &str,
    entry: &str,
    profile: Profile,
) -> Result<resin_lir::VerifiedModule, Vec<resin_source::SourceError>> {
    support::pipeline::verified_lir(&build_hir(source), entry, profile)
}

fn run(source: &str) -> std::process::Output {
    let module = compile(source, "main", Profile::Host)
        .unwrap_or_else(|errors| panic!("{errors:?}"))
        .into_module();
    support::project::Project::new(&module, Some("main"))
        .unwrap()
        .run()
}

const INDEX: &str = r#"intrinsic "pointer_index" fn index<T>(data: Ptr<T>, length: ulong, position: ulong) -> Ptr<T>;
"#;

#[test]
fn generic_pointer_index_preserves_stride_mutation_and_host_bounds_diagnostics() {
    for (position, succeeds) in [(1, true), (3, false)] {
        let source = format!(
            r#"export {{ main }};
import {{ "$/shared.resin" }};

            {INDEX}
            fn main() -> int | Err<_> {{
                let values_owner = arc_ptr_alloc([3_ul, 7_ul, 11_ul])?; let values: Ref<_> = values_owner:get().*;
                index(values_owner:get():lea(0), 3, {position}).* = 42_ul;
                if (values:at(1) == 42_ul) {{ 0 }} else {{ 1 }}
            }}
        "#
        );
        let output = support::project::Project::new(&support::module(&source), Some("main"))
            .unwrap()
            .run();
        assert_eq!(
            output.status.success(),
            succeeds,
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        if !succeeds {
            assert!(String::from_utf8_lossy(&output.stderr).contains("index"));
        }
    }
}

#[test]
fn source_methods_preserve_element_stride_aliasing_and_explicit_literal_borrows() {
    let output = run(r#"export { main };
        import { "$/shared.resin", "$/span.resin" };
        fn main() -> int | Err<_> {
            let values_owner = arc_ptr_alloc([3_ul, 7_ul, 11_ul])?; let values: Ref<_> = values_owner:get().*;
            let mut view = Span<ulong> { data = values_owner:get():lea(0), length = 3_ul };
            let mut alias = view:slice(1, 2);
            alias:at(0) = 42_ul;
            let mut raw = alias:as_bytes();
            let mut literal = bytes("A\0B");
            if (values:at(1) == 42_ul && raw.length == 16_ul
                && literal.length == 3_ul && literal:at(1) == 0_ub) { 0 } else { 1 }
        }
    "#);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn empty_slices_allow_one_past_the_end_without_advancing_null() {
    let output = run(r#"export { main };
        import { "$/shared.resin", "$/span.resin" };
        fn main() -> int | Err<_> {
            let values_owner = arc_ptr_alloc([1_ui, 2_ui])?; let values: Ref<_> = values_owner:get().*;
            let mut view = Span<uint> { data = values_owner:get():lea(0), length = 2_ul };
            let mut end = view:slice(2, 0);
            let mut empty = Span<uint> { data = Ptr<uint>(0_ul), length = 0_ul }:slice(0, 0);
            if (end.length == 0_ul && ulong(end.data) == ulong(view.data) + 2_ul * size_of(uint)
                && empty.length == 0_ul && ulong(empty.data) == 0_ul) { 0 } else { 1 }
        }
    "#);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn primitive_boundaries_report_invalid_ranges_indices_and_byte_counts() {
    for (storage, cases) in [
        (
            "let values = arc_ptr_alloc([1_ul, 2_ul, 3_ul])?; let mut view = Span<ulong> { data = values:get():lea(0), length = 3_ul };",
            [
                ("view:at(3)", "index"),
                ("view:slice(2, 2)", "slice out of bounds"),
                (
                    "view:slice(0xffffffffffffffff_ul, 0)",
                    "slice out of bounds",
                ),
                (
                    "Span<ulong> { data = Ptr<ulong>(0_ul), length = 0xffffffffffffffff_ul }:as_bytes()",
                    "byte length overflow",
                ),
            ],
        ),
        (
            "let mut view = Span<uint> { data = Ptr<uint>(0_ul), length = 2_ul };",
            [
                ("view:slice(3, 0)", "span slice out of bounds"),
                ("view:slice(1, 2)", "span slice out of bounds"),
                (
                    "view:slice(0xffffffffffffffff_ul, 1)",
                    "span slice out of bounds",
                ),
                (
                    "Span<uint> { data = Ptr<uint>(0_ul), length = 0xffffffffffffffff_ul }:as_bytes()",
                    "span byte length overflow",
                ),
            ],
        ),
    ] {
        for (expression, message) in cases {
            let source = format!(
                r#"export {{ main }}; import {{ "$/span.resin", "$/string.resin", "$/stdio.resin", "$/shared.resin" }};
                fn main() -> () | Err<_> {{
                    {storage}
                    {expression};
                    print("unreachable");
                }}"#
            );
            let output = run(&source);
            assert!(!output.status.success(), "{expression}");
            assert!(output.stdout.is_empty(), "{expression}");
            assert!(
                String::from_utf8_lossy(&output.stderr).contains(message),
                "{expression}: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
    }
}

#[test]
fn byte_views_reject_nonnumeric_elements_after_specialization() {
    let compilation = compile(
        r#"export { main };
        import { "$/shared.resin", "$/span.resin" };
        struct Entry { value: uint, }
        fn main() -> () | Err<_> {
            let entry_owner = arc_ptr_alloc(Entry { value = 1_ui })?; let entry: Ref<_> = entry_owner:get().*;
            Span<Entry> { data = entry_owner:get(), length = 1_ul }:as_bytes();
        }
    "#,
        "main",
        Profile::Host,
    );
    let error = match compilation {
        Ok(_) => panic!("expected LIR instantiation to fail"),
        Err(errors) => errors,
    };
    assert!(
        error
            .iter()
            .any(|error| error.to_string().contains("numeric elements")),
        "{error:?}"
    );
}

#[test]
fn shader_span_indexing_uses_record_layout_and_device_pointer_stride() {
    let compilation = compile(
        r#"export { kernel };
        import { "$/span.resin" };
        struct Root { values: Span<uint>, }
        @compute_shader fn kernel(index: ulong, root: Ptr<Root>)  {
            root.values:at(index) = 42_ui;
        }
    "#,
        "kernel",
        Profile::Shader,
    );
    let project =
        support::project::Project::new(&compilation.unwrap().into_module(), None).unwrap();
    for shader in project.generated.shaders() {
        support::shaders::validate(shader.unoptimized_spirv());
    }
}

#[test]
fn shader_local_addresses_cannot_become_physical_pointer_index_operands() {
    let compilation = compile(
        r#"export { kernel };
        intrinsic "pointer_index" fn index<T>(data: Ptr<T>, length: ulong, position: ulong) -> Ptr<T>;
        @compute_shader fn kernel(invocation: ulong, output: Ptr<uint>)  {
            let mut values = [1_ui, 2_ui];
            output.* = index(&values:at(0), 2, 0).*;
        }
    "#,
        "kernel",
        Profile::Shader,
    );
    let errors = compilation.err().unwrap();
    assert!(
        errors.iter().any(|error| error
            .to_string()
            .contains("cannot take the address of a Ref")),
        "{errors:?}"
    );
}

#[test]
fn reference_returning_index_wrappers_preserve_nested_places() {
    let output = run(r#"export { main };
    import { "$/shared.resin", "$/span.resin" };
    struct Payload { value: int, }
    struct Entry { nested: Payload, }
    fn entry_at(items: Ref<Span<Entry>>, index: ulong) -> Ref<Entry>  { items:at(index) }
    fn main() -> int | Err<_> {
        let items_owner = arc_ptr_alloc([Entry { nested = Payload { value = 1 } }, Entry { nested = Payload { value = 2 } }])?; let items: Ref<_> = items_owner:get().*;
        let mut span = Span<Entry> { data = Ptr<Entry>(items_owner:get()), length = 2_ul };
        entry_at(span, 1_ul).nested.value = 42;
        let p: Ref<int> = entry_at(span, 1_ul).nested.value;
        p = p + 1;
        let mut copied = Payload { value = entry_at(span, 1_ul).nested.value };
        copied.value = 99;
        if (items(1).nested.value == 43 && copied.value == 99 && items(0).nested.value == 1) { 0 } else { 1 }
    }
    "#);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn opaque_native_elements_cannot_be_indexed_or_sliced() {
    for operation in ["view:at(0_ul)", "view:slice(0_ul, 1_ul)"] {
        let source = format!(
            r#"export {{ main }};
            import {{ "$/span.resin" }};
            extern type NativeHandle;
            fn main()  {{
                let mut view = Span<NativeHandle> {{ data = Ptr<NativeHandle>(0_ul), length = 1_ul }};
                {operation};
            }}
        "#
        );
        let compilation = compile(&source, "main", Profile::Host);
        let error = match compilation {
            Ok(_) => panic!("expected LIR instantiation to fail"),
            Err(errors) => errors,
        };
        assert!(
            error
                .iter()
                .any(|error| error.to_string().contains("OpaqueValue")),
            "{error:?}"
        );
    }
}
