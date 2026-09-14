#[allow(dead_code)]
mod support;

use resin_compiler::{Compilation, Compiler, Target};
use resin_source::{Loader, Source};
use std::{path::PathBuf, sync::Arc};

fn compile(source: &str, target: Target) -> Arc<Compilation> {
    let library = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("resin");
    let mut loader = Loader::new(library);
    Compiler::new().compile(
        Source::new("span-test.resin", source),
        &mut loader,
        &[target],
    )
}

fn run(body: &str) -> std::process::Output {
    let source = format!(
        r#"
        export {{ main }};
        import {{ "$/span.resin" }};
        def main() -> int = {{ {body} }};
    "#
    );
    let compilation = compile(
        &source,
        Target::Host {
            entry: "main".into(),
        },
    );
    let module = compilation
        .module()
        .unwrap_or_else(|error| panic!("{error}"));
    support::project::Project::new(module, Some("main"))
        .unwrap()
        .run()
}

#[test]
fn source_methods_preserve_element_stride_aliasing_and_explicit_literal_borrows() {
    let output = run(r#"
        var values = [3_ul, 7_ul, 11_ul];
        var view = Span<ulong> { data = values.at(0), length = 3_ul };
        var alias = view.slice(1, 2);
        alias.at(0).* := 42_ul;
        var raw = alias.as_bytes();
        var literal = bytes("A\0B");
        if (values.at(1).* == 42_ul && raw.length == 16_ul
            && literal.length == 3_ul && literal.at(1).* == 0_ub) { 0 } else { 1 }
    "#);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn empty_slices_allow_one_past_the_end_without_advancing_null() {
    let output = run(r#"
        var values = [1_ui, 2_ui];
        var view = Span<uint> { data = values.at(0), length = 2_ul };
        var end = view.slice(2, 0);
        var empty = Span<uint> { data = Ptr<uint>(0_ul), length = 0_ul }.slice(0, 0);
        if (end.length == 0_ul && ulong(end.data) == ulong(view.data) + 2_ul * size_of(uint)
            && empty.length == 0_ul && ulong(empty.data) == 0_ul) { 0 } else { 1 }
    "#);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn primitive_boundaries_report_invalid_ranges_indices_and_byte_counts() {
    for (expression, message) in [
        ("view.at(3)", "index"),
        ("view.slice(2, 2)", "slice out of bounds"),
        (
            "view.slice(0xffffffffffffffff_ul, 0)",
            "slice out of bounds",
        ),
        (
            "Span<ulong> { data = Ptr<ulong>(0_ul), length = 0xffffffffffffffff_ul }.as_bytes()",
            "byte length overflow",
        ),
    ] {
        let output = run(&format!(
            r#"
            var values = [1_ul, 2_ul, 3_ul];
            var view = Span<ulong> {{ data = values.at(0), length = 3_ul }};
            {expression}; 0
        "#
        ));
        assert!(!output.status.success(), "{expression}");
        assert!(
            String::from_utf8_lossy(&output.stderr).contains(message),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
fn byte_views_reject_nonnumeric_elements_after_specialization() {
    let compilation = compile(
        r#"
        export { main };
        import { "$/span.resin" };
        struct Entry { value: uint };
        def main() = {
            var entry = Entry { value = 1_ui };
            Span<Entry> { data = &entry, length = 1_ul }.as_bytes();
        };
    "#,
        Target::Host {
            entry: "main".into(),
        },
    );
    let error = compilation.module().unwrap_err();
    assert!(error.to_string().contains("numeric elements"), "{error}");
}

#[test]
fn embedded_shader_bytes_wrap_into_the_library_span_explicitly() {
    let compilation = compile(
        r#"
        export { main };
        import { "$/span.resin" };
        @compute_shader def kernel(index: ulong, output: Ptr<uint>) = { output.* := uint(index); };
        def main() -> int = {
            var code = Span<ubyte>(kernel.spirv);
            if (code.length > 0_ul && code.at(0).* == 3_ub) { 0 } else { 1 }
        };
    "#,
        Target::Host {
            entry: "main".into(),
        },
    );
    let output = support::project::Project::new(compilation.module().unwrap(), Some("main"))
        .unwrap()
        .run();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn shader_span_indexing_uses_record_layout_and_device_pointer_stride() {
    let compilation = compile(
        r#"
        export { kernel };
        import { "$/span.resin" };
        struct Root { values: Span<uint> };
        @compute_shader def kernel(index: ulong, root: Ptr<Root>) = {
            root.values.at(index).* := 42_ui;
        };
    "#,
        Target::Shader {
            entry: "kernel".into(),
        },
    );
    let project = support::project::Project::new(compilation.module().unwrap(), None).unwrap();
    for shader in project.generated.shaders() {
        support::shaders::validate(shader.unoptimized_spirv());
    }
}

#[test]
fn shader_local_addresses_cannot_become_physical_pointer_index_operands() {
    let compilation = compile(
        r#"
        export { kernel };
        intrinsic "pointer_index" def index<T>(data: Ptr<T>, length: ulong, position: ulong) -> Ptr<T>;
        @compute_shader def kernel(invocation: ulong, output: Ptr<uint>) = {
            var values = [1_ui, 2_ui];
            output.* := index(values.at(0), 2, 0).*;
        };
    "#,
        Target::Shader {
            entry: "kernel".into(),
        },
    );
    let module = compilation.module().unwrap();
    let error = support::project::Project::new(module, None).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("shader-local addresses cannot escape"),
        "{error}"
    );
}

#[test]
fn pointer_returning_index_wrappers_preserve_nested_places() {
    let compilation = compile(
        r#"
    export { main };
    import { "$/span.resin" };
    struct Payload { value: int };
    struct Entry { nested: Payload };
    def at(items: Span<Entry>, index: ulong) -> Ptr<Entry> = { items.at(index) };
    def main() -> int = {
        var items = [Entry { nested = Payload { value = 1 } }, Entry { nested = Payload { value = 2 } }];
        var span = Span<Entry> { data = Ptr<Entry>(&items), length = 2_ul };
        at(span, 1_ul).nested.value := 42;
        var p = &at(span, 1_ul).*.nested.value;
        p.* := p.* + 1;
        var copied = at(span, 1_ul).*.nested;
        copied.value := 99;
        if (items(1).nested.value == 43 && copied.value == 99 && items(0).nested.value == 1) { 0 } else { 1 }
    };
    "#,
        Target::Host {
            entry: "main".into(),
        },
    );
    let output = support::project::Project::new(compilation.module().unwrap(), Some("main"))
        .unwrap()
        .run();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn opaque_native_elements_cannot_be_indexed_or_sliced() {
    for operation in ["view.at(0_ul)", "view.slice(0_ul, 1_ul)"] {
        let source = format!(
            r#"
            export {{ main }};
            import {{ "$/span.resin" }};
            extern type NativeHandle;
            def main() = {{
                var view = Span<NativeHandle> {{ data = Ptr<NativeHandle>(0_ul), length = 1_ul }};
                {operation};
            }};
        "#
        );
        let compilation = compile(
            &source,
            Target::Host {
                entry: "main".into(),
            },
        );
        let error = compilation.module().unwrap_err();
        assert!(error.to_string().contains("OpaqueValue"), "{error}");
    }
}
