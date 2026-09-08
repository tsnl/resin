//! Backend clients supply verified LIR and receive source files plus a build graph.
use resin_lir::{BasicBlock, BlockId, Function, Instr, Local, Module, Terminator, VerifiedModule};
use resin_source::prelude::*;
use resin_types::prelude::*;
use std::fs;
use tempfile::TempDir;

fn constant_function(parameter: Ty, result: Ty, value: Value) -> Function {
    Function {
        name: None,
        foreign: None,
        result,
        locals: vec![Local {
            name: None,
            ty: parameter,
        }],
        entry: BlockId::from_index(0),
        blocks: vec![BasicBlock {
            name: None,
            instrs: vec![Instr::Push { value }],
            terminator: Terminator::Return,
        }],
    }
}

fn module() -> Module {
    let pointer = Ty::Pointer {
        pointee: Box::new(Ty::UInt64),
    };
    Module {
        functions: vec![
            constant_function(Ty::Unit, Ty::Int32, Value::Int32 { value: 42 }),
            constant_function(Ty::parameter(&[Ty::UInt64, pointer]), Ty::Unit, Value::Unit),
        ],
        entries: [("main".into(), FunctionId::from_index(0))].into(),
        shaders: [(
            FunctionId::from_index(1),
            ShaderEntry {
                stage: "compute".into(),
                embedded: false,
            },
        )]
        .into(),
        ..Default::default()
    }
}

fn embedded_module() -> Module {
    let mut module = module();
    let shader = FunctionId::from_index(1);
    module.shaders.get_mut(&shader).unwrap().embedded = true;
    module.functions[0].blocks[0].instrs.splice(
        0..0,
        [
            Instr::Shader {
                function: shader,
                stage: "compute".into(),
            },
            Instr::Discard,
        ],
    );
    module
}

#[test]
fn generated_project_outlives_its_verified_input_and_retains_opaque_names() {
    let directory = TempDir::new_in(std::env::temp_dir()).unwrap();
    let project = {
        let mut module = embedded_module();
        module.origins.functions.insert(
            FunctionId::from_index(0),
            SourceLocation {
                source: Source::new("editor://buffer/λ", "main"),
                span: Span { start: 0, end: 4 },
            },
        );
        let checked = VerifiedModule::new(module).unwrap();
        resin_codegen::generate(checked.view(), Some("main"), directory.path()).unwrap()
    };
    assert_eq!(project.directory(), directory.path());
    assert_eq!(project.name(), "editor://buffer/λ");
    assert_eq!(project.entry(), Some("main"));
    let source = fs::read_to_string(project.c_source().unwrap()).unwrap();
    assert!(source.contains("int main("));
    let shader = &project.shaders()[0];
    assert_eq!(shader.function(), FunctionId::from_index(1));
    assert_eq!(shader.stage(), Stage::Compute);
    assert!(source.contains(&format!(
        "#include \"{}\"",
        shader.header().file_name().unwrap().to_str().unwrap()
    )));
    assert!(source.contains(&format!("{}_length", shader.symbol())));
    assert!(
        fs::read_to_string(shader.source())
            .unwrap()
            .starts_with("#version 460\n")
    );
    assert!(project.build_file().is_file());
    for output in [shader.spirv(), shader.header(), project.program().unwrap()] {
        assert!(!output.exists(), "generation must not invoke native tools");
    }
}

#[test]
fn shader_only_generation_batches_declared_functions_without_a_host_entry() {
    let directory = TempDir::new_in(std::env::temp_dir()).unwrap();
    let mut module = module();
    module.entries.clear();
    module.functions.push(module.functions[1].clone());
    module.shaders.insert(
        FunctionId::from_index(2),
        module.shaders[&FunctionId::from_index(1)].clone(),
    );
    let checked = VerifiedModule::new(module).unwrap();
    let project = resin_codegen::generate(checked.view(), None, directory.path()).unwrap();
    assert!(project.c_source().is_none());
    assert!(project.program().is_none());
    assert!(project.entry().is_none());
    assert_eq!(project.name(), "shaders");
    assert_eq!(project.shaders().len(), 2);
    let first = &project.shaders()[0];
    let second = &project.shaders()[1];
    assert_ne!(first.source(), second.source());
    assert_ne!(first.symbol(), second.symbol());
    for shader in project.shaders() {
        assert!(
            fs::read_to_string(shader.source())
                .unwrap()
                .contains("void main()")
        );
    }
}

#[test]
fn host_generation_does_not_lower_unrequested_shader_bodies() {
    let directory = TempDir::new_in(std::env::temp_dir()).unwrap();
    let mut module = module();
    module.functions[1].blocks[0].instrs.splice(
        0..0,
        [
            Instr::Push {
                value: Value::Str {
                    value: "host only".as_bytes().to_vec().into(),
                },
            },
            Instr::Discard,
        ],
    );
    let checked = VerifiedModule::new(module).unwrap();
    let project = resin_codegen::generate(checked.view(), Some("main"), directory.path()).unwrap();
    assert!(project.shaders().is_empty());
    assert!(project.c_source().unwrap().is_file());
    let shader_directory = directory.path().join("shaders");
    assert!(resin_codegen::generate(checked.view(), None, &shader_directory).is_err());
    assert!(!shader_directory.exists());
}

#[test]
fn lowering_failure_leaves_existing_outputs_untouched() {
    let directory = TempDir::new_in(std::env::temp_dir()).unwrap();
    let checked = VerifiedModule::new(embedded_module()).unwrap();
    let project = resin_codegen::generate(checked.view(), Some("main"), directory.path()).unwrap();
    let before_c = fs::read(project.c_source().unwrap()).unwrap();
    let before_glsl = fs::read(project.shaders()[0].source()).unwrap();
    let error =
        resin_codegen::generate(checked.view(), Some("missing"), directory.path()).unwrap_err();
    assert!(error.to_string().contains("not exported"));
    let mut bad = embedded_module();
    bad.functions[1].blocks[0].instrs.splice(
        0..0,
        [
            Instr::Push {
                value: Value::Str {
                    value: "host only".as_bytes().to_vec().into(),
                },
            },
            Instr::Discard,
        ],
    );
    let checked = VerifiedModule::new(bad).unwrap();
    let error =
        resin_codegen::generate(checked.view(), Some("main"), directory.path()).unwrap_err();
    assert!(error.to_string().contains("shader string literals"));
    assert_eq!(fs::read(project.c_source().unwrap()).unwrap(), before_c);
    assert_eq!(
        fs::read(project.shaders()[0].source()).unwrap(),
        before_glsl
    );
}

#[test]
fn build_graph_orders_shader_compilation_embedding_and_c_compilation() {
    let directory = TempDir::new_in(std::env::temp_dir()).unwrap();
    let checked = VerifiedModule::new(embedded_module()).unwrap();
    let project = resin_codegen::generate(checked.view(), Some("main"), directory.path()).unwrap();
    let graph = fs::read_to_string(project.build_file()).unwrap();
    assert!(graph.contains("include toolchain.ninja"));
    assert!(graph.contains("shader_1.spv: compile_shader shader_1.glsl | toolchain.state"));
    assert!(graph.contains("shader_1.h: embed_shader shader_1.spv | toolchain.state"));
    assert!(graph.contains("compile_program main.c | toolchain.state $runtime_library shader_1.h"));
    assert!(graph.contains("$resin --embed $in --symbol $symbol --output $out"));
}
