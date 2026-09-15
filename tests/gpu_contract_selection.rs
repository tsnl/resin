use resin_frontend::Frontend;
use resin_source::{Loader, Source};

const PIPELINES: &str = r#"
struct Failure {};
struct ComputeProgram<Root, Owner> { token: GpuPipelineContract };
struct GraphicsProgram<Root, Owner> { token: GpuPipelineContract };
intrinsic "gpu_compute_pipeline_type" def compute_type<R, O>(token: GpuPipelineContract) -> ComputeProgram<R, O>;
intrinsic "gpu_graphics_pipeline_type" def graphics_type<R, O>(token: GpuPipelineContract) -> GraphicsProgram<R, O>;
struct Device {
    @gpu_allocator
    def allocate(self: Device, bytes: ulong, alignment: ulong, memory: int) -> Result<GpuView, Failure> = { err(Failure {}) };
    @gpu_compute_pipeline
    def compute(self: Device, code: {data: Ptr<ubyte>, length: ulong}) -> Result<PipelineOwner, Failure> = { err(Failure {}) };
    @gpu_graphics_pipeline
    def graphics(self: Device, vertex: {data: Ptr<ubyte>, length: ulong}, fragment: {data: Ptr<ubyte>, length: ulong}) -> Result<PipelineOwner, Failure> = { err(Failure {}) };
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
struct Params { value: Ptr<int> };
"#;

const USE_PROJECTION: &str = r#"
def record(commands: Commands, pipeline: ComputeProgram<Params, PipelineOwner>, value: Alpha<int>) -> Result<(), Failure> = {
    commands.dispatch(pipeline, {value = value}, 1, 1, 1)
};
"#;

fn pointer(name: &str) -> String {
    format!(
        r#"struct {name}<T> {{ view: GpuView }};
        intrinsic "gpu_pointer_projection" def project_{}<T>(value: {name}<T>) -> Ptr<T>;"#,
        name.to_lowercase()
    )
}

#[test]
fn ambiguous_projection_diagnostics_are_independent_of_declaration_and_import_order() {
    let mut messages = Vec::new();
    for imported in [false, true] {
        for names in [["Alpha", "Beta"], ["Beta", "Alpha"]] {
            let declarations = if imported {
                format!(
                    r#"import {{ "{}.resin", "{}.resin" }};"#,
                    names[0], names[1]
                )
            } else {
                names.map(pointer).join("\n")
            };
            let source = Source::new(
                "main.resin",
                format!("{declarations}\n{PIPELINES}\n{USE_PROJECTION}"),
            );
            let mut loader = Loader::new(Default::default());
            for name in names {
                let file = format!("{name}.resin");
                let target = Source::new(
                    file.as_str(),
                    format!("export {{ {name} }}; {}", pointer(name)),
                );
                loader.set_import(&source, &file, target).unwrap();
            }
            let analysis = Frontend::new().analyze(source, &mut loader);
            let errors = analysis.diagnostics();
            assert_eq!(errors.len(), 1, "{errors:?}");
            // Importing moves the call's source span; the diagnostic itself is stable.
            messages.push(errors[0].message.split_once(": ").unwrap().1.to_owned());
        }
    }
    for message in &messages {
        assert_eq!(message, &messages[0]);
        assert!(
            message.contains("ambiguous GPU projections: `Alpha`, `Beta`"),
            "{message}"
        );
    }
}

#[test]
fn unfinished_bridge_completions_use_registered_source_pipeline_names() {
    for receiver in ["device", "Device", "commands", "Commands"] {
        let source = Source::new(
            "main.resin",
            format!(
                "{PIPELINES}\ndef completion(device: Device, commands: Commands) = {{ {receiver}.; }};"
            ),
        );
        let analysis =
            Frontend::new().analyze(source.clone(), &mut Loader::new(Default::default()));
        let offset = source.text().rfind(".;").unwrap() + 1;
        let completions = analysis.completions(&source, offset);
        let methods = if receiver.eq_ignore_ascii_case("device") {
            [
                ("compute", "ComputeProgram"),
                ("graphics", "GraphicsProgram"),
            ]
        } else {
            [("dispatch", "ComputeProgram"), ("draw", "GraphicsProgram")]
        };
        for (method, pipeline) in methods {
            let completion = completions.iter().find(|item| item.name == method).unwrap();
            assert!(
                completion.detail.contains(&format!("{pipeline}<T,")),
                "{completion:?}"
            );
            assert!(
                !completion.detail.contains("GpuComputePipeline"),
                "{completion:?}"
            );
            assert!(
                !completion.detail.contains("GpuGraphicsPipeline"),
                "{completion:?}"
            );
        }
    }
}
