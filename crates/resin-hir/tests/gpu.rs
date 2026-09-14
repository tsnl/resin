use resin_hir::{Module, TermKind, Type};

fn generate(source: &str) -> Result<Module, resin_hir::GenerateError> {
    let document = resin_cst::Document::reparse(source.into(), None);
    let ast = resin_ast::generate(&document).expect("valid GPU test syntax");
    resin_hir::generate(&ast)
}

// These names deliberately differ from the library. Contracts are explicit
// declarations; no compiler type or method is selected from a public spelling.
const PIPELINES: &str = r#"
struct Failure {};
struct DeviceReference<T> { allocation: GpuView };
struct HostRange<T> { data: Ptr<T>, length: ulong };
struct DeviceRange<T> { data: DeviceReference<T>, length: ulong };
intrinsic "gpu_pointer_projection" def pointer_projection<T>(value: DeviceReference<T>) -> Ptr<T>;
intrinsic "gpu_span_projection" def span_projection<T>(value: DeviceRange<T>) -> HostRange<T>;
struct ComputeProgram<Root, Owner> { contract: GpuPipelineContract };
struct GraphicsProgram<Root, Owner> { contract: GpuPipelineContract };
intrinsic "gpu_compute_pipeline_type" def compute_type<Root, Owner>(contract: GpuPipelineContract) -> ComputeProgram<Root, Owner>;
intrinsic "gpu_graphics_pipeline_type" def graphics_type<Root, Owner>(contract: GpuPipelineContract) -> GraphicsProgram<Root, Owner>;
struct Device {
    @gpu_allocator
    def malloc(self: Device, bytes: ulong, alignment: ulong, memory: int) -> Result<GpuView, Failure> = { err(Failure {}) };
    @gpu_compute_pipeline
    def compute(self: Device, code: { data: Ptr<ubyte>, length: ulong }) -> Result<PipelineOwner, Failure> = { err(Failure {}) };
    @gpu_graphics_pipeline
    def graphics(self: Device, vertex: { data: Ptr<ubyte>, length: ulong }, fragment: { data: Ptr<ubyte>, length: ulong }) -> Result<PipelineOwner, Failure> = { err(Failure {}) };
};
struct PipelineOwner { owner: StrongOwner,
    @gpu_pipeline_context
    def context(self: PipelineOwner) -> Device = { Device {} };
};
struct Commands {
    @gpu_dispatch
    def dispatch(self: Commands, pipeline: PipelineOwner, arguments: GpuArguments, x: uint, y: uint, z: uint) -> Result<(), Failure> = { ok(()) };
    @gpu_draw
    def draw(self: Commands, pipeline: PipelineOwner, arguments: GpuArguments | None, count: uint) -> Result<(), Failure> = { ok(()) };
};
struct Params { scale: float32, values: HostRange<int> };
@compute_shader def kernel(index: ulong, root: Ptr<Params>) = {};
"#;

fn pipelines(source: &str) -> Result<Module, resin_hir::GenerateError> {
    generate(&format!("{PIPELINES}\n{source}"))
}

#[test]
fn former_gpu_type_names_are_ordinary_generic_declarations() {
    generate(r#"
        struct GpuPtr<T> { value: T, def new(value: T) -> GpuPtr<T> = { GpuPtr<T> { value = value } }; };
        struct GpuSpan<T> { value: T };
        struct GpuComputePipeline<T> { value: T };
        struct GpuGraphicsPipeline<T> { value: T };
        def main() -> int = { GpuPtr<int>.new(42).value };
    "#).unwrap();
}

#[test]
fn allocator_registration_does_not_synthesize_constructor_methods() {
    let error = pipelines("def main(device: Device) = { device.new(42); };").unwrap_err();
    assert!(
        error.to_string().contains("unknown method `new`"),
        "{error}"
    );
}

#[test]
fn opaque_gpu_views_cannot_be_dereferenced_or_cast_to_raw_addresses() {
    for body in ["value.*", "Ptr<int>(value)", "ulong(value)", "value.data"] {
        let error = generate(&format!("def invalid(value: GpuView) = {{ {body}; }};")).unwrap_err();
        assert!(!error.to_string().contains("Unbound"), "{error}");
    }
}

#[test]
fn dispatch_infers_source_host_fields_from_the_pipeline_root() {
    let module = pipelines(
        r#"
        def main(values: DeviceRange<int>) -> Result<(), Failure> = {
            var pipeline = Device {}.compute(kernel)?;
            Commands {}.dispatch(pipeline, { scale = 2.0, values = values }, 1, 1, 1)
        };
    "#,
    )
    .unwrap();
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
    let Type::Record { fields } = &args.params[2] else {
        panic!()
    };
    assert_eq!(fields[0].ty, Type::Float32);
    let Type::Defined { arguments, .. } = &fields[1].ty else {
        panic!()
    };
    assert_eq!(arguments, &[Type::Int32]);
    assert!(module.shaders.values().all(|entry| entry.embedded));
}

#[test]
fn pipeline_types_cross_functions_and_accept_precomputed_arguments() {
    pipelines(r#"
        def create(gpu: Device) -> Result<ComputeProgram<Params, PipelineOwner>, Failure> = { gpu.compute(kernel) };
        def dispatch(pipeline: ComputeProgram<Params, PipelineOwner>, values: DeviceRange<int>) -> Result<(), Failure> = {
            var arguments = (pipeline, { scale = 1.0_f, values = values }, 1_ui, 1_ui, 1_ui);
            Commands {}.dispatch(arguments.0, arguments.1, arguments.2, arguments.3, arguments.4)
        };
        def associated(gpu: Device) -> _ = { Device.compute(gpu, kernel) };
    "#).unwrap();
}

#[test]
fn creation_requires_direct_decorated_shader_declarations() {
    for expression in [
        "gpu.compute(kernel.spirv)",
        "gpu.compute(alias)",
        "gpu.compute(ordinary)",
        "gpu.compute(choose())",
    ] {
        let error = pipelines(&format!(
            r#"
            def ordinary(index: ulong, root: Ptr<Params>) = {{}};
            def choose() -> (ulong, Ptr<Params>) -> () = {{ kernel }};
            def main(gpu: Device) -> _ = {{ var alias = kernel; {expression} }};
        "#
        ))
        .unwrap_err();
        assert!(
            !error.to_string().contains("Unbound"),
            "{expression}: {error}"
        );
    }
}

#[test]
fn dispatch_rejects_raw_views_wrong_fields_and_incompatible_stages() {
    for tail in [
        "Commands {}.dispatch(pipeline, { scale = 1.0_f, values = raw }, 1, 1, 1)",
        "Commands {}.dispatch(pipeline, { scale = 1.0_f, wrong = values }, 1, 1, 1)",
        "Commands {}.draw(pipeline, { scale = 1.0_f, values = values }, 3)",
        "Commands {}.dispatch(pipeline, None, 1, 1, 1)",
    ] {
        let error = pipelines(&format!("def main(gpu: Device, values: DeviceRange<int>, raw: HostRange<int>) -> _ = {{ var pipeline = gpu.compute(kernel)?; {tail} }};")).unwrap_err();
        assert!(!error.to_string().contains("Unbound"), "{tail}: {error}");
    }
    assert!(pipelines("def create(gpu: Device) -> Result<ComputeProgram<int, PipelineOwner>, Failure> = { gpu.compute(kernel) };").is_err());
}

#[test]
fn native_bridges_cannot_escape_through_method_references() {
    for declaration in [
        "def escape(gpu: Device) = { var factory = gpu.compute; };",
        "def escape() = { var factory = Device.compute; };",
        "def escape(commands: Commands) = { var dispatch = commands.dispatch; };",
    ] {
        let error = pipelines(declaration).unwrap_err();
        assert!(!error.to_string().contains("Unbound"), "{error}");
    }
}
