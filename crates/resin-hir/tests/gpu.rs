use resin_hir::{Module, TermKind, Type};

mod common;
use common::hir_module;

fn generate(source: &str) -> Result<Module, resin_source::SourceError> {
    hir_module(source)
}

#[test]
fn compute_workgroup_size_is_an_ordinary_source_name() {
    let error = generate("fn size() -> ulong  { compute_workgroup_size }").unwrap_err();
    assert!(error.to_string().contains("UnboundValue"), "{error}");
    generate("fn compute_workgroup_size() -> ulong  { 1_ul } fn size() -> ulong  { let mut compute_workgroup_size = 1_ul; compute_workgroup_size = 2_ul; compute_workgroup_size }").unwrap();
}

// These names deliberately differ from the library. Contracts are explicit
// declarations; no compiler type or method is selected from a public spelling.
const PIPELINES: &str = r#"struct Failure {}
struct DeviceReference<T> { allocation: GpuView; }
struct HostRange<T> { data: Ptr<T>; length: ulong; }
struct DeviceRange<T> { data: DeviceReference<T>; length: ulong; }
intrinsic "gpu_pointer_projection" fn pointer_projection<T>(value: DeviceReference<T>) -> Ptr<T>;
intrinsic "gpu_span_projection" fn span_projection<T>(value: DeviceRange<T>) -> HostRange<T>;
struct ComputeProgram<Root, Owner> { contract: GpuPipelineContract; }
struct GraphicsProgram<Root, Owner> { contract: GpuPipelineContract; }
intrinsic "gpu_compute_pipeline_type" fn compute_type<Root, Owner>(contract: GpuPipelineContract) -> ComputeProgram<Root, Owner>;
intrinsic "gpu_graphics_pipeline_type" fn graphics_type<Root, Owner>(contract: GpuPipelineContract) -> GraphicsProgram<Root, Owner>;
struct Device {
    
    
    
}
@gpu_allocator
    fn malloc(self: Device, bytes: ulong, alignment: ulong, memory: int) -> (GpuView | Err<Failure>)  { Err(Failure {}) }

@gpu_compute_pipeline
    fn compute(self: Device, code: (Ptr<ubyte>, ulong)) -> (PipelineOwner | Err<Failure>)  { Err(Failure {}) }

@gpu_graphics_pipeline
    fn graphics(self: Device, vertex: (Ptr<ubyte>, ulong), fragment: (Ptr<ubyte>, ulong)) -> (PipelineOwner | Err<Failure>)  { Err(Failure {}) }

struct PipelineOwner { owner: StrongOwner;
    
}
@gpu_pipeline_context
    fn context(self: PipelineOwner) -> Device  { Device {} }

struct Commands {
    
    
}
@gpu_dispatch
    fn dispatch(self: Commands, pipeline: PipelineOwner, arguments: GpuArguments, x: uint, y: uint, z: uint) -> (() | Err<Failure>)  { (()) }

@gpu_draw
    fn draw(self: Commands, pipeline: PipelineOwner, arguments: GpuArguments | None, count: uint) -> (() | Err<Failure>)  { (()) }

struct HostParams<T> { scale: float32; values: T; }
struct WrongParams<T> { scale: float32; wrong: T; }
struct Params { scale: float32; values: HostRange<int>; }
@compute_shader fn kernel(index: ulong, root: Ptr<Params>)  {}
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
        r#"{declarations}
        struct FieldsScaleValues<T0, T1> {{ scale: T0; values: T1; }}
fn main(values: DeviceRange<int>) -> (() | Err<Failure>) = {{
            let mut pipeline = Device {{}}:compute(kernel)?;
            Commands {{}}.dispatch(pipeline, FieldsScaleValues<_, _> {{ scale = 2.0, values  values }}, 1, 1, 1)
        }};
        {contracts}
    "#
    ))
    .unwrap();
}

#[test]
fn former_gpu_type_names_are_ordinary_generic_declarations() {
    generate(
        r#"struct GpuPtr<T> { value: T;  }
fn gpu_ptr_new<T>(value: T) -> GpuPtr<T>  { GpuPtr<T> { value = value } }

        struct GpuSpan<T> { value: T; }
        struct GpuComputePipeline<T> { value: T; }
        struct GpuGraphicsPipeline<T> { value: T; }
        fn main() -> int  { gpu_ptr_new::<int>(42).value }
    "#,
    )
    .unwrap();
}

#[test]
fn allocator_registration_does_not_synthesize_constructor_methods() {
    let error = pipelines("fn main(device: Device)  { device:new(42); }").unwrap_err();
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
        let error = generate(&format!("fn invalid(value: GpuView)  {{ {body}; }}")).unwrap_err();
        assert!(error.to_string().contains(expected), "{body}: {error}");
    }
}

#[test]
fn dispatch_infers_source_host_fields_from_the_pipeline_root() {
    let module = pipelines(
        r#"struct FieldsScaleValues<T0, T1> { scale: T0; values: T1; }
fn main(values: DeviceRange<int>) -> (() | Err<Failure>)  {
            let mut pipeline = Device {}:compute(kernel)?;
            Commands {}:dispatch(pipeline, FieldsScaleValues<_, _> { scale = 2.0, values = values }, 1, 1, 1)
        }
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
    let Type::Defined {
        arguments: fields, ..
    } = &args.params[2]
    else {
        panic!()
    };
    assert_eq!(fields[0], Type::Float32);
    let Type::Defined { arguments, .. } = &fields[1] else {
        panic!()
    };
    assert_eq!(arguments, &[Type::Int32]);
    assert!(module.shaders.values().all(|entry| entry.embedded));
}

#[test]
fn pipeline_types_cross_functions_and_accept_precomputed_arguments() {
    pipelines(r#"struct FieldsScaleValues<T0, T1> { scale: T0; values: T1; }
fn create(gpu: Device) -> (ComputeProgram<Params, PipelineOwner> | Err<Failure>)  { gpu:compute(kernel) }
        fn dispatch(pipeline: ComputeProgram<Params, PipelineOwner>, values: DeviceRange<int>) -> (() | Err<Failure>)  {
            let mut arguments = (pipeline, FieldsScaleValues<_, _> { scale = 1.0_f, values = values }, 1_ui, 1_ui, 1_ui);
            Commands {}:dispatch(arguments.0, arguments.1, arguments.2, arguments.3, arguments.4)
        }
        fn associated(gpu: Device) -> _  { compute(gpu, kernel) }
    "#).unwrap();
}

#[test]
fn creation_requires_direct_decorated_shader_declarations() {
    for (expression, expected) in [
        (
            "gpu.compute((Ptr<ubyte>(0_ul), 0_ul))",
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
            r#"fn ordinary(index: ulong, root: Ptr<Params>)  {{}}
            fn choose() -> (ulong, Ptr<Params>) -> ()  {{ kernel }}
            fn main(gpu: Device) -> _  {{ let mut alias = kernel; {expression} }}
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
            "Commands {}.dispatch(pipeline, HostParams<_> { scale = 1.0_f, values = raw }, 1, 1, 1)",
            "incompatible inferred types",
        ),
        (
            "Commands {}.dispatch(pipeline, WrongParams<_> { scale = 1.0_f, wrong = values }, 1, 1, 1)",
            "must match the shader root",
        ),
        (
            "Commands {}.draw(pipeline, HostParams<_> { scale = 1.0_f, values = values }, 3)",
            "draw requires a graphics pipeline",
        ),
        (
            "Commands {}.dispatch(pipeline, None, 1, 1, 1)",
            "incompatible inferred types",
        ),
    ] {
        let error = pipelines(&format!("fn main(gpu: Device, values: DeviceRange<int>, raw: HostRange<int>) -> _  {{ let mut pipeline = gpu:compute(kernel)?; {tail} }}")).unwrap_err();
        assert!(error.to_string().contains(expected), "{tail}: {error}");
    }
    let error = pipelines("fn create(gpu: Device) -> (ComputeProgram<int, PipelineOwner> | Err<Failure>)  { gpu:compute(kernel) }").unwrap_err();
    assert!(error.to_string().contains("destination union"), "{error}");
}

#[test]
fn native_bridges_cannot_escape_through_method_references() {
    for (declaration, expected) in [
        (
            "fn escape(gpu: Device)  { let mut factory = gpu.compute; }",
            "unknown field `compute`",
        ),
        (
            "fn escape()  { let mut factory = compute; }",
            "only source methods can be referenced as function values",
        ),
        (
            "fn escape(commands: Commands)  { let mut dispatch = commands.dispatch; }",
            "unknown field `dispatch`",
        ),
    ] {
        let error = pipelines(declaration).unwrap_err();
        assert!(error.to_string().contains(expected), "{error}");
    }
}
