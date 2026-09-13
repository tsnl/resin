//! Target requests are compilation inputs; declaration analysis has no target artifact.
use resin_compiler::{Compiler, CompilerConfig, Target};
use resin_source::{Loader, Source};
use std::{num::NonZeroUsize, sync::Arc};

fn host(entry: &str) -> Target {
    Target::Host {
        entry: entry.into(),
    }
}
fn shader(entry: &str) -> Target {
    Target::Shader {
        entry: entry.into(),
    }
}
fn loader() -> Loader {
    Loader::new(resin_source::library_root())
}

#[test]
fn declarations_do_not_instantiate_unused_concrete_operations() {
    let source = Source::new(
        "entry",
        "export { main, invalid }; def main() -> int = { 42 }; def invalid() -> bool = { (1 == 1) + (1 == 1) };",
    );
    let mut compiler = Compiler::new();
    let mut loader = loader();
    let declarations = compiler.analyze(source.clone(), &mut loader);
    assert!(declarations.diagnostics().is_empty());
    assert!(declarations.hir().is_ok());
    assert!(
        declarations
            .module()
            .unwrap_err()
            .to_string()
            .contains("did not request")
    );
    let compiled = compiler.compile(source.clone(), &mut loader, &[host("main")]);
    assert_eq!(compiled.module().unwrap().functions.len(), 1);
    assert!(!Arc::ptr_eq(&declarations, &compiled));
    let invalid = compiler.compile(source, &mut loader, &[host("invalid")]);
    assert!(invalid.hir().is_ok());
    assert!(invalid.module().is_err());
    assert_eq!(invalid.diagnostics().len(), 1);
    assert!(compiled.module().is_ok());
}

#[test]
fn target_sets_are_canonicalized_for_scheduling_and_cache_matching() {
    let source = Source::new(
        "entry",
        "export { first, second }; def first() -> int = { 1 }; def second() -> int = { 2 };",
    );
    let mut compiler = Compiler::new();
    let mut loader = loader();
    let both = compiler.compile(
        source.clone(),
        &mut loader,
        &[host("second"), host("first")],
    );
    let repeated = compiler.compile(
        source.clone(),
        &mut loader,
        &[host("first"), host("second"), host("first")],
    );
    assert!(Arc::ptr_eq(&both, &repeated));
    let first = compiler.compile(source, &mut loader, &[host("first")]);
    assert!(!Arc::ptr_eq(&both, &first));
    assert_eq!(first.module().unwrap().entries.len(), 1);
    assert_eq!(both.module().unwrap().entries.len(), 2);
}

#[test]
fn the_same_declaration_can_be_requested_on_host_and_shader() {
    let source = Source::new(
        "entry",
        "export { kernel }; @compute_shader def kernel(i: ulong, out: Ptr<uint>) = { out.* := uint(i); };",
    );
    let mut compiler = Compiler::new();
    let mut loader = loader();
    let host_only = compiler.compile(source.clone(), &mut loader, &[host("kernel")]);
    assert!(host_only.module().unwrap().shaders.is_empty());
    let shader_only = compiler.compile(source.clone(), &mut loader, &[shader("kernel")]);
    assert!(!Arc::ptr_eq(&host_only, &shader_only));
    assert!(shader_only.module().unwrap().entries.is_empty());
    assert_eq!(shader_only.module().unwrap().shaders.len(), 1);
    let both = compiler.compile(
        source.clone(),
        &mut loader,
        &[shader("kernel"), host("kernel")],
    );
    assert_eq!(both.module().unwrap().functions.len(), 2);
    let limited = Compiler::with_config(CompilerConfig {
        max_monomorphs_per_function: NonZeroUsize::new(1).unwrap(),
    })
    .compile(source, &mut loader, &[shader("kernel"), host("kernel")]);
    assert!(limited.module().is_err());
    assert!(
        limited.diagnostics()[0]
            .message
            .contains("limit of 1 monomorphs")
    );
}

#[test]
fn invalid_requests_fail_without_discarding_source_analysis() {
    let source = Source::new(
        "entry",
        "export { main }; def main() = {}; def private() = {};",
    );
    let mut compiler = Compiler::new();
    let mut loader = loader();
    for targets in [
        vec![],
        vec![host("missing")],
        vec![host("private")],
        vec![shader("main")],
    ] {
        let compilation = compiler.compile(source.clone(), &mut loader, &targets);
        assert!(compilation.hir().is_ok());
        assert!(compilation.module().is_err());
        assert_eq!(compilation.diagnostics().len(), 1);
    }
}

#[test]
fn unused_source_initialization_errors_still_prevent_compilation() {
    let source = Source::new(
        "entry",
        "export { main }; def main() = {}; def unused() -> int = { var x: int; x };",
    );
    let compilation = Compiler::new().compile(source, &mut loader(), &[host("main")]);
    assert!(compilation.hir().is_err());
    assert!(
        compilation.diagnostics()[0]
            .message
            .contains("UninitializedValue")
    );
}

#[test]
fn demanded_nominals_retain_field_conversions_and_real_drop_identities() {
    let source = Source::new(
        "entry",
        "export { main }; struct Unused {}; struct Owner { n: int, def drop(self: Ptr<Owner>) = {}; }; def main() -> int = { var owner = Owner { n = 42 }; owner.n };",
    );
    let compilation = Compiler::new().compile(source, &mut loader(), &[host("main")]);
    let module = compilation.module().unwrap();
    assert_eq!(
        module.types.iter().filter(|ty| ty.name().is_some()).count(),
        1
    );
    assert_eq!(module.types[0].name().unwrap().as_ref(), "Owner");
    let drop = module.types[0].drop_hook().unwrap();
    assert!(
        module.functions[drop.index()]
            .name
            .as_ref()
            .unwrap()
            .contains("drop")
    );
    assert_eq!(module.functions.len(), 2);
}
