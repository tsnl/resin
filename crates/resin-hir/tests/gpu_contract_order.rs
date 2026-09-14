use resin_hir::Module;

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
        def main(values: DeviceRange<int>) -> Result<(), Failure> = {{
            var pipeline = Device {{}}.compute(kernel)?;
            Commands {{}}.dispatch(pipeline, {{ scale = 2.0, values = values }}, 1, 1, 1)
        }};
        {contracts}
    "#
    ))
    .unwrap();
}
