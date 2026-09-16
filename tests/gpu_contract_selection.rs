#[allow(dead_code)]
mod support;

use resin_source::{Loader, Source};

const PIPELINES: &str = r#"struct Failure {}
struct ComputeProgram<Root, Owner> { token: GpuPipelineContract, }
struct GraphicsProgram<Root, Owner> { token: GpuPipelineContract, }
intrinsic "gpu_compute_pipeline_type" fn compute_type<R, O>(token: GpuPipelineContract) -> ComputeProgram<R, O>;
intrinsic "gpu_graphics_pipeline_type" fn graphics_type<R, O>(token: GpuPipelineContract) -> GraphicsProgram<R, O>;
struct Device {
    
    
    
}
@gpu_allocator
    fn allocate(self: Device, bytes: ulong, alignment: ulong, memory: int) -> (GpuView | Err<Failure>)  { Err(Failure {}) }

@gpu_compute_pipeline
    fn compute(self: Device, code: (Ptr<ubyte>, ulong)) -> (PipelineOwner | Err<Failure>)  { Err(Failure {}) }

@gpu_graphics_pipeline
    fn graphics(self: Device, vertex: (Ptr<ubyte>, ulong), fragment: (Ptr<ubyte>, ulong)) -> (PipelineOwner | Err<Failure>)  { Err(Failure {}) }

struct PipelineOwner { owner: StrongOwner,
    
}
@gpu_pipeline_context
    fn context(self: PipelineOwner) -> Device  { Device {} }

struct Commands {
    
    
}
@gpu_dispatch
    fn dispatch(self: Commands, pipeline: PipelineOwner, arguments: GpuArguments, x: uint, y: uint, z: uint) -> (() | Err<Failure>)  { (()) }

@gpu_draw
    fn draw(self: Commands, pipeline: PipelineOwner, arguments: GpuArguments | None, count: uint) -> (() | Err<Failure>)  { (()) }

struct Params { value: Ptr<int>, }
"#;

const USE_PROJECTION: &str = r#"struct FieldsValue<T0> { value: T0, }
fn record(commands: Commands, pipeline: ComputeProgram<Params, PipelineOwner>, value: Alpha<int>) -> (() | Err<Failure>)  {
    commands:dispatch(pipeline, FieldsValue<_> {value = value}, 1, 1, 1)
}
"#;

fn pointer(name: &str) -> String {
    format!(
        r#"struct {name}<T> {{ view: GpuView; }}
        intrinsic "gpu_pointer_projection" fn project_{}<T>(value: {name}<T>) -> Ptr<T>;"#,
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
            let analysis = support::frontend::analyze(source, &mut loader, None);
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
                "{PIPELINES}\nfn completion(device: Device, commands: Commands)  {{ {receiver}.; }}"
            ),
        );
        let analysis =
            support::frontend::analyze(source.clone(), &mut Loader::new(Default::default()), None);
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
