use resin_hir::{Module, TermKind, Type};

mod common;
use common::hir_module;

fn generate(source: &str) -> Result<Module, resin_source::SourceError> {
    hir_module(source)
}

#[test]
fn compute_workgroup_size_is_an_ordinary_source_name() {
    let error = generate("def size() -> ulong = { compute_workgroup_size };").unwrap_err();
    assert!(error.to_string().contains("UnboundValue"), "{error}");
    generate("def compute_workgroup_size() -> ulong = { 1_ul }; def size() -> ulong = { var compute_workgroup_size = 1_ul; compute_workgroup_size := 2_ul; compute_workgroup_size };").unwrap();
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
    def malloc(self: Device, bytes: ulong, alignment: ulong, memory: int) -> (GpuView | Err<Failure>) = { Err(Failure {}) };
    @gpu_compute_pipeline
    def compute(self: Device, code: { data: Ptr<ubyte>, length: ulong }) -> (PipelineOwner | Err<Failure>) = { Err(Failure {}) };
    @gpu_graphics_pipeline
    def graphics(self: Device, vertex: { data: Ptr<ubyte>, length: ulong }, fragment: { data: Ptr<ubyte>, length: ulong }) -> (PipelineOwner | Err<Failure>) = { Err(Failure {}) };
};
struct PipelineOwner { owner: StrongOwner,
    @gpu_pipeline_context
    def context(self: PipelineOwner) -> Device = { Device {} };
};
struct Commands {
    @gpu_dispatch
    def dispatch(self: Commands, pipeline: PipelineOwner, arguments: GpuArguments, x: uint, y: uint, z: uint) -> (() | Err<Failure>) = { (()) };
    @gpu_draw
    def draw(self: Commands, pipeline: PipelineOwner, arguments: GpuArguments | None, count: uint) -> (() | Err<Failure>) = { (()) };
};
struct Params { scale: float32, values: HostRange<int> };
@compute_shader def kernel(index: ulong, root: Ptr<Params>) = {};
"#;

fn pipelines(source: &str) -> Result<Module, resin_source::SourceError> {
    generate(&format!("{PIPELINES}\n{source}"))
}

#[test]
fn contract_declarations_may_follow_their_users_and_dependencies() {
    let (contracts, declarations): (Vec<_>, Vec<_>) = PIPELINES
        .lines()
        .partition(|line| line.starts_with("intrinsic "));
    let contracts = contracts.into_iter().rev().collect::<Vec<_>>().join("\n");
    let declarations = declarations.join("\n");
    generate(&format!(
        r#"
        {declarations}
        def main(values: DeviceRange<int>) -> (() | Err<Failure>) = {{
            var pipeline = Device {{}}.compute(kernel)?;
            Commands {{}}.dispatch(pipeline, {{ scale = 2.0, values = values }}, 1, 1, 1)
        }};
        {contracts}
    "#
    ))
    .unwrap();
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
    for (body, expected) in [
        ("value.*", "dereference requires a pointer"),
        ("Ptr<int>(value)", "TypeMismatch"),
        ("ulong(value)", "TypeMismatch"),
        ("value.data", "field access requires a record"),
    ] {
        let error = generate(&format!("def invalid(value: GpuView) = {{ {body}; }};")).unwrap_err();
        assert!(error.to_string().contains(expected), "{body}: {error}");
    }
}

#[test]
fn dispatch_infers_source_host_fields_from_the_pipeline_root() {
    let module = pipelines(
        r#"
        def main(values: DeviceRange<int>) -> (() | Err<Failure>) = {
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
        def create(gpu: Device) -> (ComputeProgram<Params, PipelineOwner> | Err<Failure>) = { gpu.compute(kernel) };
        def dispatch(pipeline: ComputeProgram<Params, PipelineOwner>, values: DeviceRange<int>) -> (() | Err<Failure>) = {
            var arguments = (pipeline, { scale = 1.0_f, values = values }, 1_ui, 1_ui, 1_ui);
            Commands {}.dispatch(arguments.0, arguments.1, arguments.2, arguments.3, arguments.4)
        };
        def associated(gpu: Device) -> _ = { Device.compute(gpu, kernel) };
    "#).unwrap();
}

#[test]
fn creation_requires_direct_decorated_shader_declarations() {
    for (expression, expected) in [
        (
            "gpu.compute({ data = Ptr<ubyte>(0_ul), length = 0_ul })",
            "requires decorated shader declarations",
        ),
        ("gpu.compute(alias)", "requires direct shader declarations"),
        (
            "gpu.compute(ordinary)",
            "requires a decorated shader declaration",
        ),
        (
            "gpu.compute(choose())",
            "requires direct shader declarations",
        ),
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
            error.to_string().contains(expected),
            "{expression}: {error}"
        );
    }
}

#[test]
fn dispatch_rejects_raw_views_wrong_fields_and_incompatible_stages() {
    for (tail, expected) in [
        (
            "Commands {}.dispatch(pipeline, { scale = 1.0_f, values = raw }, 1, 1, 1)",
            "incompatible inferred types",
        ),
        (
            "Commands {}.dispatch(pipeline, { scale = 1.0_f, wrong = values }, 1, 1, 1)",
            "missing field `values`",
        ),
        (
            "Commands {}.draw(pipeline, { scale = 1.0_f, values = values }, 3)",
            "draw requires a graphics pipeline",
        ),
        (
            "Commands {}.dispatch(pipeline, None, 1, 1, 1)",
            "incompatible inferred types",
        ),
    ] {
        let error = pipelines(&format!("def main(gpu: Device, values: DeviceRange<int>, raw: HostRange<int>) -> _ = {{ var pipeline = gpu.compute(kernel)?; {tail} }};")).unwrap_err();
        assert!(error.to_string().contains(expected), "{tail}: {error}");
    }
    let error = pipelines("def create(gpu: Device) -> (ComputeProgram<int, PipelineOwner> | Err<Failure>) = { gpu.compute(kernel) };").unwrap_err();
    assert!(error.to_string().contains("destination union"), "{error}");
}

#[test]
fn native_bridges_cannot_escape_through_method_references() {
    for (declaration, expected) in [
        (
            "def escape(gpu: Device) = { var factory = gpu.compute; };",
            "unknown field `compute`",
        ),
        (
            "def escape() = { var factory = Device.compute; };",
            "only source methods can be referenced as function values",
        ),
        (
            "def escape(commands: Commands) = { var dispatch = commands.dispatch; };",
            "unknown field `dispatch`",
        ),
    ] {
        let error = pipelines(declaration).unwrap_err();
        assert!(error.to_string().contains(expected), "{error}");
    }
}
