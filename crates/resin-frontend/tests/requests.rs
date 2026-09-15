//! LIR instantiation is a generate-time request; analysis has no target artifact.
use resin_frontend::{Frontend, FrontendConfig, FrontendOutput, Target};
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

fn module(output: &FrontendOutput, targets: &[Target]) -> resin_lir::Module {
    output
        .build_lir(targets)
        .unwrap_or_else(|errors| panic!("{errors:?}"))
        .into_module()
}

fn failed(output: &FrontendOutput, targets: &[Target]) -> Vec<resin_source::SourceError> {
    match output.build_lir(targets) {
        Ok(_) => panic!("expected LIR instantiation to fail"),
        Err(errors) => errors,
    }
}

#[test]
fn declarations_do_not_instantiate_unused_concrete_operations() {
    let source = Source::new(
        "entry",
        "export { main, invalid }; def main() -> int = { 42 }; def invalid() -> bool = { (1 == 1) + (1 == 1) };",
    );
    let mut frontend = Frontend::new();
    let mut loader = loader();
    let declarations = frontend.build_hir(source.clone(), &mut loader);
    assert!(declarations.diagnostics().is_empty());
    assert!(declarations.hir().is_ok());
    let compiled = frontend.build_hir(source.clone(), &mut loader);
    assert!(Arc::ptr_eq(&declarations, &compiled));
    assert_eq!(module(&compiled, &[host("main")]).functions.len(), 1);
    let errors = failed(&compiled, &[host("invalid")]);
    assert_eq!(errors.len(), 1);
}

#[test]
fn target_sets_are_canonicalized_for_scheduling() {
    let source = Source::new(
        "entry",
        "export { first, second }; def first() -> int = { 1 }; def second() -> int = { 2 };",
    );
    let mut frontend = Frontend::new();
    let mut loader = loader();
    let output = frontend.build_hir(source, &mut loader);
    let both = module(&output, &[host("second"), host("first")]);
    let repeated = module(&output, &[host("first"), host("second"), host("first")]);
    assert_eq!(both.entries.len(), 2);
    assert_eq!(repeated.entries.len(), 2);
    assert_eq!(module(&output, &[host("first")]).entries.len(), 1);
}

#[test]
fn the_same_declaration_can_be_requested_on_host_and_shader() {
    let source = Source::new(
        "entry",
        "export { kernel }; @compute_shader def kernel(i: ulong, out: Ptr<uint>) = { out.* := uint(i); };",
    );
    let mut frontend = Frontend::new();
    let mut loader = loader();
    let output = frontend.build_hir(source.clone(), &mut loader);
    let host_only = module(&output, &[host("kernel")]);
    assert!(host_only.shaders.is_empty());
    let shader_only = module(&output, &[shader("kernel")]);
    assert!(shader_only.entries.is_empty());
    assert_eq!(shader_only.shaders.len(), 1);
    let both = module(&output, &[shader("kernel"), host("kernel")]);
    assert_eq!(both.functions.len(), 2);
    let limited = Frontend::with_config(FrontendConfig {
        max_monomorphs_per_function: NonZeroUsize::new(1).unwrap(),
    })
    .build_hir(source, &mut loader);
    assert!(
        failed(&limited, &[shader("kernel"), host("kernel")])[0]
            .to_string()
            .contains("limit of 1 monomorphs")
    );
}

#[test]
fn invalid_requests_fail_without_discarding_source_analysis() {
    let source = Source::new(
        "entry",
        "export { main }; def main() = {}; def private() = {};",
    );
    let mut frontend = Frontend::new();
    let mut loader = loader();
    let output = frontend.build_hir(source, &mut loader);
    assert!(output.hir().is_ok());
    for targets in [
        vec![],
        vec![host("missing")],
        vec![host("private")],
        vec![shader("main")],
    ] {
        let errors = failed(&output, &targets);
        assert_eq!(errors.len(), 1);
    }
}

#[test]
fn unused_source_initialization_errors_still_prevent_compilation() {
    let source = Source::new(
        "entry",
        "export { main }; def main() = {}; def unused() -> int = { var x: int; x };",
    );
    let output = Frontend::new().build_hir(source, &mut loader());
    assert!(output.hir().is_err());
    assert!(
        output.diagnostics()[0]
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
    let output = Frontend::new().build_hir(source, &mut loader());
    let module = module(&output, &[host("main")]);
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

#[test]
fn unsupported_shader_operations_fail_during_compilation_with_application_notes() {
    let source = Source::new(
        "entry",
        "export { main, kernel }; def main() -> int = { 42 }; @compute_shader def kernel(i: ulong, out: Ptr<uint>) = { out.* := uint(i) / 2_ui; };",
    );
    let mut frontend = Frontend::new();
    let mut loader = loader();
    let output = frontend.build_hir(source.clone(), &mut loader);
    assert!(output.build_lir(&[host("main")]).is_ok());
    assert!(output.build_lir(&[host("kernel")]).is_ok());
    let errors = failed(&output, &[shader("kernel")]);
    assert!(output.hir().is_ok());
    let error = &errors[0];
    assert!(error.diagnostic.contains("unsupported shader builtin"));
    let span = error.span.expect("operation span");
    assert_eq!(&source.text()[span.start..span.end], "uint(i) / 2_ui");
    assert!(
        error
            .related
            .iter()
            .any(|note| note.message.contains("kernel") && note.message.contains("Shader"))
    );
}

#[test]
fn shader_recursion_is_rejected_before_publishing_lir() {
    let source = Source::new(
        "entry",
        "export { kernel }; def helper(i: ulong, out: Ptr<uint>) = { kernel(i, out); }; @compute_shader def kernel(i: ulong, out: Ptr<uint>) = { helper(i, out); };",
    );
    let mut frontend = Frontend::new();
    let mut loader = loader();
    let output = frontend.build_hir(source, &mut loader);
    assert!(output.build_lir(&[host("kernel")]).is_ok());
    assert!(output.hir().is_ok());
    let errors = failed(&output, &[shader("kernel")]);
    assert!(
        errors[0]
            .to_string()
            .contains("recursive shader call graph")
    );
    assert!(
        errors[0]
            .related
            .iter()
            .any(|note| note.message.contains("helper"))
    );
}

#[test]
fn shader_calls_cannot_enter_foreign_functions_or_store_function_values() {
    for (helper, body, message) in [
        (
            "extern \"stdlib.h\" def abs(i: int) -> int;",
            "out.* := uint(abs(int(i)));",
            "foreign",
        ),
        (
            "def helper(i: uint) -> uint = { i };",
            "var f = helper; out.* := f(uint(i));",
            "does not support type",
        ),
    ] {
        let source = Source::new(
            "entry",
            format!(
                "export {{ kernel }}; {helper} @compute_shader def kernel(i: ulong, out: Ptr<uint>) = {{ {body} }};"
            ),
        );
        let output = Frontend::new().build_hir(source, &mut loader());
        assert!(output.hir().is_ok());
        let errors = failed(&output, &[shader("kernel")]);
        assert!(
            errors
                .iter()
                .any(|error| error.diagnostic.contains(message)),
            "{errors:?}"
        );
    }
}
