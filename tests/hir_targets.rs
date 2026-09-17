//! LIR construction is a generate-time request; analysis has no target artifact.
#[allow(dead_code)]
mod support;
use resin_hir::Hir;
use resin_lir::{LoweringOptions, Profile};
use resin_source::{Loader, Source, SourceError, SourceNote};
use std::num::NonZeroUsize;

fn host(entry: &str) -> (&str, Profile) {
    (entry, Profile::Host)
}
fn shader(entry: &str) -> (&str, Profile) {
    (entry, Profile::Shader)
}
fn loader() -> Loader {
    Loader::new(resin_source::library_root())
}

fn lower(
    output: &Hir,
    requests: &[(&str, Profile)],
    options: &LoweringOptions,
) -> Result<resin_lir::VerifiedModule, Vec<SourceError>> {
    let hir = output.hir().map_err(|error| vec![error])?;
    let entries = requests
        .iter()
        .map(|(name, profile)| {
            resin_lir::Entry::exported(hir, *name, *profile)
                .map_err(|error| lir_error(output, error))
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| vec![error])?;
    let lir = support::frontend::lower(hir, &entries, options).map_err(|errors| {
        errors
            .into_iter()
            .map(|error| lir_error(output, error))
            .collect::<Vec<_>>()
    })?;
    resin_lir::VerifiedModule::new(lir).map_err(|error| {
        vec![SourceError::new(
            output.source().clone(),
            None,
            format!("invalid LIR: {error}"),
        )]
    })
}

fn module(output: &Hir, requests: &[(&str, Profile)]) -> resin_lir::Module {
    lower(output, requests, &LoweringOptions::default())
        .unwrap_or_else(|errors| panic!("{errors:?}"))
        .into_module()
}

fn failed(output: &Hir, requests: &[(&str, Profile)]) -> Vec<SourceError> {
    match lower(output, requests, &LoweringOptions::default()) {
        Ok(_) => panic!("expected LIR construction to fail"),
        Err(errors) => errors,
    }
}

fn lir_error(output: &Hir, error: resin_lir::Error) -> SourceError {
    let mut diagnostic = SourceError::new(
        error
            .source
            .clone()
            .unwrap_or_else(|| output.source().clone()),
        Some(error.span),
        error.to_string(),
    );
    diagnostic
        .related
        .extend(error.applications.into_iter().filter_map(|application| {
            application.location.map(|location| SourceNote {
                location,
                message: format!(
                    "while instantiating {} with {:?} for {:?}",
                    application.function, application.arguments, application.profile
                ),
            })
        }));
    diagnostic
}

#[test]
fn declarations_do_not_instantiate_unused_concrete_operations() {
    let source = Source::new(
        "entry",
        "export { main, invalid }; fn main() -> i32  { 42 } fn invalid() -> bool  { (1 == 1) + (1 == 1) }",
    );
    let mut loader = loader();
    let declarations = support::frontend::analyze(source.clone(), &mut loader, None);
    assert!(declarations.diagnostics().is_empty());
    assert!(declarations.hir().is_ok());
    let compiled = support::frontend::analyze(source.clone(), &mut loader, Some(&declarations));
    assert!(declarations.same(&compiled));
    assert_eq!(module(&compiled, &[host("main")]).functions.len(), 1);
    let errors = failed(&compiled, &[host("invalid")]);
    assert_eq!(errors.len(), 1);
}

#[test]
fn target_sets_are_canonicalized_for_scheduling() {
    let source = Source::new(
        "entry",
        "export { first, second }; fn first() -> i32  { 1 } fn second() -> i32  { 2 }",
    );
    let mut loader = loader();
    let output = support::frontend::analyze(source, &mut loader, None);
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
        "export { kernel }; @compute_shader fn kernel(i: u64, out: Ptr<u32>)  { out.* = u32(i); }",
    );
    let mut loader = loader();
    let output = support::frontend::analyze(source.clone(), &mut loader, None);
    let host_only = module(&output, &[host("kernel")]);
    assert!(host_only.shaders.is_empty());
    let shader_only = module(&output, &[shader("kernel")]);
    assert!(shader_only.entries.is_empty());
    assert_eq!(shader_only.shaders.len(), 1);
    let both = module(&output, &[shader("kernel"), host("kernel")]);
    assert_eq!(both.functions.len(), 2);
    let limited = support::frontend::analyze(source, &mut loader, None);
    let options = LoweringOptions {
        max_monomorphs_per_function: NonZeroUsize::new(1).unwrap(),
    };
    let errors = match lower(&limited, &[shader("kernel"), host("kernel")], &options) {
        Ok(_) => panic!("expected monomorph limit"),
        Err(errors) => errors,
    };
    assert!(errors[0].to_string().contains("limit of 1 monomorphs"));
}

#[test]
fn invalid_requests_fail_without_discarding_source_analysis() {
    let source = Source::new("entry", "export { main }; fn main()  {} fn private()  {}");
    let mut loader = loader();
    let output = support::frontend::analyze(source, &mut loader, None);
    assert!(output.hir().is_ok());
    for targets in [
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
        "export { main }; fn main()  {} fn unused() -> i32  { let mut x: i32; x }",
    );
    let output = support::frontend::analyze(source, &mut loader(), None);
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
        "export { main }; struct Unused {} struct Owner { n: i32,  }\nfn drop(self: RefMut<Owner>)  {}\n fn main() -> i32  { let mut owner = Owner { n = 42 }; owner.n }",
    );
    let output = support::frontend::analyze(source, &mut loader(), None);
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
        "export { main, kernel }; fn main() -> i32  { 42 } @compute_shader fn kernel(i: u64, out: Ptr<u32>)  { out.* = u32(f64(i) / f64(2.0)); }",
    );
    let mut loader = loader();
    let output = support::frontend::analyze(source.clone(), &mut loader, None);
    assert!(lower(&output, &[host("main")], &LoweringOptions::default()).is_ok());
    assert!(lower(&output, &[host("kernel")], &LoweringOptions::default()).is_ok());
    let errors = failed(&output, &[shader("kernel")]);
    assert!(output.hir().is_ok());
    let error = &errors[0];
    assert!(error.diagnostic.contains("does not support type"));
    let span = error.span.expect("operation span");
    assert_eq!(&source.text()[span.start..span.end], "f64(i) / f64(2.0)");
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
        "export { kernel }; fn helper(i: u64, out: Ptr<u32>)  { kernel(i, out); } @compute_shader fn kernel(i: u64, out: Ptr<u32>)  { helper(i, out); }",
    );
    let mut loader = loader();
    let output = support::frontend::analyze(source, &mut loader, None);
    assert!(lower(&output, &[host("kernel")], &LoweringOptions::default()).is_ok());
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
            "extern { \"stdlib.h\": { fn abs(i: i32) -> i32; } };",
            "out.* = u32(abs(i32(i)));",
            "foreign",
        ),
        (
            "fn helper(i: u32) -> u32  { i }",
            "let mut f = helper; out.* = f(u32(i));",
            "does not support type",
        ),
    ] {
        let source = Source::new(
            "entry",
            format!(
                "export {{ kernel }}; {helper} @compute_shader fn kernel(i: u64, out: Ptr<u32>)  {{ {body} }}"
            ),
        );
        let output = support::frontend::analyze(source, &mut loader(), None);
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

#[test]
fn shader_pointer_casts_fail_before_codegen_including_generic_helpers() {
    for (helper, body, cast) in [
        ("", "u64(out);", "u64(out)"),
        ("", "Ptr<u32>(i);", "Ptr<u32>(i)"),
        ("", "Ptr<u8>(out);", "Ptr<u8>(out)"),
        (
            "fn address<T>(p: Ptr<T>) -> u64  { u64(p) }",
            "address(out);",
            "u64(p)",
        ),
    ] {
        let source = Source::new(
            "pointer.resin",
            format!(
                "export {{ kernel }}; {helper} @compute_shader fn kernel(i: u64, out: Ptr<u32>)  {{ {body} }}"
            ),
        );
        let output = support::frontend::analyze(source.clone(), &mut loader(), None);
        assert!(lower(&output, &[host("kernel")], &LoweringOptions::default()).is_ok());
        let errors = failed(&output, &[shader("kernel")]);
        let error = &errors[0];
        assert!(
            error
                .diagnostic
                .contains("shader pointer casts are unsupported"),
            "{error}"
        );
        let span = error.span.expect("cast has a source span");
        assert_eq!(&source.text()[span.start..span.end], cast);
    }
}
