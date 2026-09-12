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

const PIPELINES: &str = r#"
struct PipelineOwner { gpu: Device };
impl PipelineOwner {
    @gpu_pipeline_context
    def context(self: Arc<PipelineOwner>) -> Device = { self.gpu };
}
impl Device {
    @gpu_compute_pipeline
    def compute(self: Device, code: Span<ubyte>) -> Result<Arc<PipelineOwner>, Failure> = {
        ok(Arc<PipelineOwner>(PipelineOwner { gpu = self }))
    };
    @gpu_graphics_pipeline
    def graphics(self: Device, vertex: Span<ubyte>, fragment: Span<ubyte>) -> Result<Arc<PipelineOwner>, Failure> = {
        ok(Arc<PipelineOwner>(PipelineOwner { gpu = self }))
    };
}
struct Commands {};
impl Commands {
    @gpu_dispatch
    def dispatch(self: Commands, pipeline: Arc<PipelineOwner>, root: GpuArguments, x: uint, y: uint, z: uint) -> Result<(), Failure> = { ok(()) };
    @gpu_draw
    def draw(self: Commands, pipeline: Arc<PipelineOwner>, root: GpuArguments | None, count: uint) -> Result<(), Failure> = { ok(()) };
}
"#;

fn pipelines(source: &str) -> Result<Module, resin_hir::GenerateError> {
    generate(&format!("{ALLOCATOR} {PIPELINES} {source}"))
}

#[test]
fn dispatch_infers_host_fields_from_the_pipeline_root() {
    let module = pipelines(&format!("{SHADER} def main(values: GpuSpan<int>) -> Result<(), Failure> = {{ var pipeline = Device.new().compute(kernel)?; Commands {{}}.dispatch(pipeline, {{ scale = 2.0, values = values }}, 1, 1, 1) }};")).unwrap();
    let main = module
        .functions
        .iter()
        .find(|function| function.name.as_ref() == "main")
        .unwrap();
    let TermKind::Block { tail, .. } = &main.body.as_ref().unwrap().kind else {
        panic!()
    };
    let TermKind::GpuPipelineDispatch {
        args, allocator, ..
    } = &tail.kind
    else {
        panic!()
    };
    assert!(allocator.is_some());
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
    assert!(module.shaders.values().all(|entry| entry.embedded));
}

#[test]
fn pipeline_types_cross_functions_and_dispatch_accepts_precomputed_arguments() {
    pipelines(&format!("{SHADER}
        def create(gpu: Device) -> Result<GpuComputePipeline<Params, Arc<PipelineOwner>>, Failure> = {{ gpu.compute(kernel) }};
        def dispatch(pipeline: GpuComputePipeline<Params, Arc<PipelineOwner>>, values: GpuSpan<int>) -> Result<(), Failure> = {{
            var args = (pipeline, {{ scale = 1.0_f, values = values }}, 1_ui, 1_ui, 1_ui);
            Commands {{}}.dispatch(args)
        }};
        def associated(gpu: Device) -> _ = {{ Device.compute(gpu, kernel) }};
    ")).unwrap();
}

#[test]
fn creation_rejects_bytecode_runtime_aliases_and_undecorated_functions() {
    for expression in [
        "gpu.compute(kernel.spirv)",
        "gpu.compute(alias)",
        "gpu.compute(ordinary)",
        "gpu.compute(choose())",
    ] {
        let source = format!(
            "{SHADER} def ordinary(index: ulong, root: Ptr<Params>) = {{}};
            def choose() -> (ulong, Ptr<Params>) -> () = {{ kernel }};
            def main(gpu: Device) -> _ = {{ var alias = kernel; {expression} }};"
        );
        assert!(pipelines(&source).is_err(), "{expression}");
    }
}

#[test]
fn dispatch_rejects_raw_pointers_and_incompatible_pipeline_roots() {
    for tail in [
        "Commands {}.dispatch(pipeline, { scale = 1.0_f, values = raw }, 1, 1, 1)",
        "Commands {}.dispatch(pipeline, { scale = 1.0_f, wrong = values }, 1, 1, 1)",
        "Commands {}.draw(pipeline, { scale = 1.0_f, values = values }, 3)",
        "Commands {}.dispatch(pipeline, None, 1, 1, 1)",
    ] {
        assert!(pipelines(&format!("{SHADER} def main(gpu: Device, values: GpuSpan<int>, raw: Span<int>) -> _ = {{ var pipeline = gpu.compute(kernel)?; {tail} }};")).is_err(), "{tail}");
    }
    assert!(pipelines(&format!("{SHADER} def create(gpu: Device) -> Result<GpuComputePipeline<int, Arc<PipelineOwner>>, Failure> = {{ gpu.compute(kernel) }};")).is_err());
}

#[test]
fn standalone_projection_and_untyped_pipeline_owners_cannot_dispatch() {
    assert!(pipelines(&format!("{SHADER} def main(gpu: Device, values: GpuSpan<int>) -> _ = {{ kernel.project(gpu, {{ scale = 2.0_f, values = values }}) }};")).is_err());
    assert!(pipelines("def main(owner: Arc<PipelineOwner>, arguments: GpuArguments) -> _ = { Commands {}.dispatch(owner, arguments, 1, 1, 1) };").is_err());
    assert!(pipelines("def forge(owner: Arc<PipelineOwner>) -> GpuComputePipeline<int, Arc<PipelineOwner>> = { GpuComputePipeline<int, Arc<PipelineOwner>>(owner) };").is_err());
}

#[test]
fn native_bridge_signatures_cannot_escape_through_method_references() {
    for tail in [
        "var make = Device.compute; make(gpu, kernel.spirv)",
        "(Device.compute)(gpu, kernel.spirv)",
        "var make = gpu.compute; make(kernel.spirv)",
        "var record = Commands.dispatch; record(Commands {}, owner, arguments, 1, 1, 1)",
        "(Commands.dispatch)(Commands {}, owner, arguments, 1, 1, 1)",
    ] {
        assert!(pipelines(&format!("{SHADER} def main(gpu: Device, owner: Arc<PipelineOwner>, arguments: GpuArguments) -> _ = {{ {tail} }};")).is_err(), "{tail}");
    }
}

#[test]
fn gpu_bridge_decorators_require_one_valid_native_signature() {
    for source in [
        "struct Device {}; impl Device { @gpu_compute_pipeline def create(self: Device, shader: int) -> int = { 0_i }; }",
        "struct Device {}; impl Device { @gpu_dispatch def dispatch(self: Device) = {}; }",
        "struct Device {}; impl Device { @gpu_pipeline_context def context(self: Device) -> Device = { self }; }",
        "struct Device {}; impl Device { @gpu_compute_pipeline @gpu_graphics_pipeline def create(self: Device) = {}; }",
        "@gpu_compute_pipeline def create() = {};",
    ] {
        assert!(generate(source).is_err(), "{source}");
    }
    assert!(pipelines("impl PipelineOwner { @gpu_pipeline_context def again(self: Arc<PipelineOwner>) -> Device = { self.gpu }; }").is_err());
}

#[test]
fn pipeline_creation_rejects_pointer_graph_roots() {
    let graph = "struct Node { next: Ptr<int> }; struct Root { node: Ptr<Node> }; @compute_shader def kernel(index: ulong, root: Ptr<Root>) = {}; def main(gpu: Device) -> _ = { gpu.compute(kernel) };";
    assert!(pipelines(graph).is_err());
}

#[test]
fn gpu_allocation_builtins_accept_precomputed_argument_tuples() {
    generate(&format!("{ALLOCATOR} def main() -> Result<GpuPtr<int>, Failure> = {{ var gpu = Device.new(); var allocation = (gpu, 4_ul); var values = GpuSpan<int>.allocate(allocation)?; var initialization = (gpu, 42_i); GpuPtr<int>.new(initialization) }};")).unwrap();
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
fn shader_completion_does_not_expose_standalone_projection() {
    use resin_source::Source;
    use std::{collections::BTreeMap, sync::Arc};
    let text = format!("{SHADER} def main() = {{ kernel.");
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
        !items.iter().any(|item| item.name == "project"),
        "{items:?}"
    );
}
