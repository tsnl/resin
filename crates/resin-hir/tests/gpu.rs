use resin_hir::{Module, TermKind};
use resin_types::prelude::*;

fn generate(source: &str) -> Result<Module, resin_hir::GenerateError> {
    let document = resin_cst::Document::reparse(source.into(), None);
    let ast = resin_ast::generate(&document).expect("valid GPU test syntax");
    resin_hir::generate(&ast)
}

const ALLOCATOR: &str = r#"
struct Failure {};
struct Device {};
impl Device {
    def new() -> Device = { Device {} };
    @gpu_allocator
    def malloc(self: Device, bytes: ulong, alignment: ulong, memory: int) -> Result<GpuPtr<ubyte>, Failure> = { err(Failure {}) };
}
"#;

#[test]
fn gpu_new_infers_element_from_context_and_keeps_associated_constructor() {
    let source = format!(
        "{ALLOCATOR}\ndef main() -> Result<GpuPtr<int>, Failure> = {{ var gpu = Device.new(); gpu.new(42) }};"
    );
    let module = generate(&source).unwrap();
    let function = module
        .functions
        .iter()
        .find(|function| function.name.as_ref() == "main")
        .unwrap();
    let TermKind::Block { tail, .. } = &function.body.as_ref().unwrap().kind else {
        panic!()
    };
    let TermKind::GpuNew { args, .. } = &tail.kind else {
        panic!()
    };
    assert_eq!(args.argument.ty, Ty::Int32);
}

#[test]
fn explicit_gpu_allocation_and_slicing_keep_element_and_owner_types() {
    let source = format!(
        "{ALLOCATOR}\ndef main() -> Result<GpuPtr<int>, Failure> = {{ var gpu = Device.new(); var values = GpuSpan<int>.allocate(gpu, 8)?; var part = values.slice(2, 3).read_only(); ok(part.at(1)) }};"
    );
    generate(&source).unwrap();
    generate(&format!("{ALLOCATOR}\ndef main() -> Result<GpuPtr<int>, Failure> = {{ GpuPtr<int>.new(Device.new(), 42) }};")).unwrap();
}

#[test]
fn gpu_field_addresses_retain_gpu_pointer_type() {
    let module = generate("struct Item { value: int }; def field(p: GpuPtr<Item>) -> GpuPtr<int> = { &p.value }; def explicit(p: GpuPtr<Item>) -> GpuPtr<int> = { &p.*.value };").unwrap();
    assert_eq!(module.functions.len(), 2);
    assert!(
        generate(
            "struct Item { value: int }; def field(p: GpuPtr<Item>) -> Ptr<int> = { &p.value };"
        )
        .is_err()
    );
}

#[test]
fn nested_host_indirection_preserves_gpu_field_addresses_but_host_metadata_stays_raw() {
    generate(
        r#"
        struct Item { value: int };
        def raw(p: Ptr<GpuPtr<Item>>) -> GpuPtr<int> = { &p.value };
        def shared(p: Arc<GpuPtr<Item>>) -> GpuPtr<int> = { &p.value };
        def mixed(p: Ptr<Arc<GpuPtr<Item>>>) -> GpuPtr<int> = { &p.value };
        def raw_metadata(p: Ptr<GpuPtr<Item>>) -> Ptr<GpuPtr<Item>> = { &p.* };
        def shared_metadata(p: Arc<GpuPtr<Item>>) -> Ptr<GpuPtr<Item>> = { &p.* };
    "#,
    )
    .unwrap();
    assert!(generate("struct Item { value: int }; def erase(p: Ptr<GpuPtr<Item>>) -> Ptr<int> = { &p.value };").is_err());
}

#[test]
fn gpu_method_receivers_keep_the_storage_category() {
    let types = r#"
        struct Item { value: int };
        struct Outer { item: Item };
        impl Item {
            def gpu(self: GpuPtr<Item>) -> int = { self.value };
            def host(self: Ptr<Item>) -> int = { self.value };
            def value(self: Item) -> int = { self.value };
        }
    "#;
    generate(&format!("{types} def gpu(p: GpuPtr<Outer>) -> int = {{ p.item.gpu() }}; def copied(p: GpuPtr<Item>) -> int = {{ p.value() }}; def replaced(p: GpuPtr<int>) -> int = {{ p.replace(4_i) }};")).unwrap();
    let error = generate(&format!(
        "{types} def erase(p: GpuPtr<Outer>) -> int = {{ p.item.host() }};"
    ))
    .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("cannot be borrowed as a raw Ptr"),
        "{error}"
    );
    assert!(
        generate(&format!(
            "{types} def forge(item: Item) -> int = {{ item.gpu() }};"
        ))
        .is_err()
    );
}

#[test]
fn gpu_pointer_casts_and_forged_gpu_views_are_rejected() {
    assert!(generate("def erase(p: GpuPtr<int>) -> Ptr<int> = { Ptr<int>(p) };").is_err());
    assert!(generate("def forge(p: Ptr<int>) -> GpuPtr<int> = { GpuPtr<int>(p) };").is_err());
    assert!(generate("def forge(p: GpuPtr<int>) -> GpuSpan<int> = { GpuSpan<int> { data = p, length = 12_ul } };").is_err());
}

#[test]
fn allocation_rejects_managed_elements_and_ordinary_new_is_unchanged() {
    assert!(
        generate(&format!(
            "{ALLOCATOR}\ndef main() -> _ = {{ Device.new().new(Arc<int>(1_i)) }};"
        ))
        .is_err()
    );
    generate("struct Other {}; impl Other { def new(self: Other, value: int) -> int = { value }; } def main() -> int = { Other {}.new(42) };").unwrap();
}

#[test]
fn native_gpu_allocator_accepts_nominal_arc_owner() {
    generate("extern type Native; struct Owner { handle: Ptr<Native> }; def allocate(gpu: Arc<Owner>) -> _ = { GpuPtr<ubyte>.allocate_native(gpu.handle, gpu, 8, 4, 0) };").unwrap();
}

#[test]
fn gpu_new_defaults_unconstrained_integer_elements_to_long() {
    let module = generate(&format!(
        "{ALLOCATOR}\ndef main() -> _ = {{ Device.new().new(42) }};"
    ))
    .unwrap();
    let main = module
        .functions
        .iter()
        .find(|function| function.name.as_ref() == "main")
        .unwrap();
    let Ty::Result { value, .. } = &main.signature.result.ty else {
        panic!()
    };
    assert_eq!(
        **value,
        Ty::GpuPointer {
            pointee: Box::new(Ty::Int64)
        }
    );
}

const SHADER: &str = "struct Params { scale: float32, values: Span<int> }; @compute_shader def kernel(index: ulong, root: Ptr<Params>) = {};";

#[test]
fn projection_infers_host_fields_from_shader_root() {
    let module = generate(&format!("{ALLOCATOR} {SHADER} def main(values: GpuSpan<int>) -> Result<GpuArguments, Failure> = {{ kernel.project(Device.new(), {{ scale = 2.0, values = values }}) }};")).unwrap();
    let main = module
        .functions
        .iter()
        .find(|function| function.name.as_ref() == "main")
        .unwrap();
    let TermKind::Block { tail, .. } = &main.body.as_ref().unwrap().kind else {
        panic!()
    };
    let TermKind::GpuProject { args, .. } = &tail.kind else {
        panic!()
    };
    let Ty::Record { fields } = &args.params[1] else {
        panic!()
    };
    assert_eq!(fields[0].ty, Ty::Float32);
    assert_eq!(
        fields[1].ty,
        Ty::GpuSpan {
            element: Box::new(Ty::Int32)
        }
    );
}

#[test]
fn projection_rejects_raw_pointers_and_runtime_shader_aliases() {
    let raw = format!(
        "{ALLOCATOR} {SHADER} def main(values: Span<int>) -> _ = {{ kernel.project(Device.new(), {{ scale = 2.0_f, values = values }}) }};"
    );
    assert!(generate(&raw).is_err());
    let alias = format!(
        "{ALLOCATOR} {SHADER} def main(values: GpuSpan<int>) -> _ = {{ var alias = kernel; alias.project(Device.new(), {{ scale = 2.0_f, values = values }}) }};"
    );
    assert!(generate(&alias).is_err());
    let ordinary = format!(
        "{ALLOCATOR} def ordinary(index: ulong, root: Ptr<int>) = {{}}; def main() -> _ = {{ ordinary.project(Device.new(), 1_i) }};"
    );
    assert!(generate(&ordinary).is_err());
}

#[test]
fn projection_rejects_shaders_without_root_and_pointer_graphs() {
    let no_root = format!(
        "{ALLOCATOR} struct Color {{ r: float32, g: float32, b: float32, a: float32 }}; @fragment_shader def fragment(color: Color) -> Color = {{ color }}; def main() -> _ = {{ fragment.project(Device.new(), 1_i) }};"
    );
    assert!(generate(&no_root).is_err());
    let graph = format!(
        "{ALLOCATOR} struct Node {{ next: Ptr<int> }}; @compute_shader def kernel(index: ulong, root: Ptr<Node>) = {{}}; def main(pointer: GpuPtr<int>) -> _ = {{ kernel.project(Device.new(), {{ next = pointer }}) }};"
    );
    generate(&graph).unwrap();
    let graph = format!(
        "{ALLOCATOR} struct Node {{ next: Ptr<int> }}; struct Root {{ node: Ptr<Node> }}; @compute_shader def kernel(index: ulong, root: Ptr<Root>) = {{}}; def main(pointer: GpuPtr<Node>) -> _ = {{ kernel.project(Device.new(), {{ node = pointer }}) }};"
    );
    assert!(generate(&graph).is_err());
}

#[test]
fn gpu_builtins_accept_precomputed_argument_tuples() {
    let source = format!(
        "{ALLOCATOR} {SHADER} def main() -> Result<GpuArguments, Failure> = {{ var gpu = Device.new(); var allocation = (gpu, 4_ul); var values = GpuSpan<int>.allocate(allocation)?; var initialization = (gpu, 42_i); var pointer = GpuPtr<int>.new(initialization)?; var arguments = (gpu, {{scale = 1.0_f, values = values}}); kernel.project(arguments) }};"
    );
    generate(&source).unwrap();
}

#[test]
fn generic_gpu_constructor_completion_and_hover_describe_inference() {
    use resin_source::Source;
    use std::{collections::BTreeMap, sync::Arc};

    let text = format!(
        "{ALLOCATOR} def main(gpu: Device) -> Result<(), Failure> = {{ gpu.new(42)?; GpuPtr<int>.new(gpu, 7)?; GpuSpan<int>.allocate(gpu, 3)?; ok(()) }};"
    );
    let source = Source::new("gpu.resin", text.clone());
    let document = Arc::new(resin_cst::Document::reparse(text.clone(), None));
    let file = resin_ast::generate(&document).unwrap();
    let analysis = resin_hir::analyze_program(&resin_ast::Program {
        modules: vec![resin_ast::SourceModule {
            source: source.clone(),
            file,
            imports: vec![],
        }],
    });
    assert!(
        analysis.diagnostics.is_empty(),
        "{:?}",
        analysis.diagnostics
    );
    let documents = BTreeMap::from([(source.clone(), document)]);
    for (call, name, signature) in [
        (
            "gpu.new(",
            "new",
            "new: (value: T) -> Result<GpuPtr<T>, Failure>",
        ),
        (
            "GpuPtr<int>.new(",
            "new",
            "new: (gpu: _, value: int) -> Result<GpuPtr<int>, _>",
        ),
        (
            "GpuSpan<int>.allocate(",
            "allocate",
            "allocate: (gpu: _, count: ulong) -> Result<GpuSpan<int>, _>",
        ),
    ] {
        let offset = text.find(call).unwrap() + call.rfind('.').unwrap() + 1;
        let completion = analysis
            .semantics
            .completions(&documents, &source, offset)
            .into_iter()
            .find(|item| item.name == name)
            .unwrap();
        assert_eq!(completion.detail, signature);
        let hover = analysis
            .semantics
            .hover(&documents, &source, offset)
            .unwrap();
        assert_eq!(hover.text, signature);
        assert!(
            analysis
                .semantics
                .definition(&documents, &source, offset)
                .is_none()
        );
    }
}

#[test]
fn generic_gpu_constructor_completion_recovers_after_a_dot() {
    use resin_source::Source;
    use std::{collections::BTreeMap, sync::Arc};

    for (receiver, expected) in [
        ("gpu", "new"),
        ("GpuPtr<int>", "new"),
        ("GpuSpan<int>", "allocate"),
    ] {
        let text = format!("{ALLOCATOR} def main(gpu: Device) = {{ {receiver}.");
        let source = Source::new("gpu.resin", text.clone());
        let document = Arc::new(resin_cst::Document::reparse(text.clone(), None));
        let file = resin_ast::recover(&document).file;
        let analysis = resin_hir::analyze_program(&resin_ast::Program {
            modules: vec![resin_ast::SourceModule {
                source: source.clone(),
                file,
                imports: vec![],
            }],
        });
        let documents = BTreeMap::from([(source.clone(), document)]);
        let items = analysis
            .semantics
            .completions(&documents, &source, text.len());
        assert!(
            items.iter().any(|item| item.name == expected),
            "{receiver}: {items:?}"
        );
    }
}

#[test]
fn shader_projection_completion_and_hover_follow_declaration_identity() {
    use resin_source::Source;
    use std::{collections::BTreeMap, sync::Arc};

    let text = format!(
        "{ALLOCATOR} {SHADER} def main(gpu: Device, values: GpuSpan<int>) -> Result<GpuArguments, Failure> = {{ kernel.project(gpu, {{ scale = 2.0_f, values = values }}) }};"
    );
    let source = Source::new("gpu.resin", text.clone());
    let document = Arc::new(resin_cst::Document::reparse(text.clone(), None));
    let file = resin_ast::generate(&document).unwrap();
    let analysis = resin_hir::analyze_program(&resin_ast::Program {
        modules: vec![resin_ast::SourceModule {
            source: source.clone(),
            file,
            imports: vec![],
        }],
    });
    assert!(
        analysis.diagnostics.is_empty(),
        "{:?}",
        analysis.diagnostics
    );
    let documents = BTreeMap::from([(source.clone(), document)]);
    let offset = text.find("kernel.project").unwrap() + "kernel.".len();
    let items = analysis.semantics.completions(&documents, &source, offset);
    let completion = items.iter().find(|item| item.name == "project").unwrap();
    assert!(
        completion.detail.contains("scale: float32"),
        "{completion:?}"
    );
    assert!(
        completion.detail.contains("values: GpuSpan<int>"),
        "{completion:?}"
    );
    assert!(
        completion.detail.ends_with("-> Result<GpuArguments, _>"),
        "{completion:?}"
    );
    assert_eq!(
        analysis
            .semantics
            .hover(&documents, &source, offset)
            .unwrap()
            .text,
        completion.detail
    );
}

#[test]
fn projection_completion_excludes_aliases_unrooted_shaders_and_field_receivers() {
    use resin_source::Source;
    use std::{collections::BTreeMap, sync::Arc};

    let unrooted = "struct Color { r: float32, g: float32, b: float32, a: float32 }; @fragment_shader def fragment(color: Color) -> Color = { color };";
    for (receiver, expected) in [
        ("kernel", true),
        ("alias", false),
        ("fragment", false),
        ("record.kernel", false),
    ] {
        let text = format!(
            "{ALLOCATOR} {SHADER} {unrooted} def main() = {{ var alias = kernel; var record = {{ kernel = kernel }}; {receiver}."
        );
        let source = Source::new("gpu.resin", text.clone());
        let document = Arc::new(resin_cst::Document::reparse(text.clone(), None));
        let file = resin_ast::recover(&document).file;
        let analysis = resin_hir::analyze_program(&resin_ast::Program {
            modules: vec![resin_ast::SourceModule {
                source: source.clone(),
                file,
                imports: vec![],
            }],
        });
        let documents = BTreeMap::from([(source.clone(), document)]);
        let items = analysis
            .semantics
            .completions(&documents, &source, text.len());
        assert_eq!(
            items.iter().any(|item| item.name == "project"),
            expected,
            "{receiver}: {items:?}"
        );
    }
}
