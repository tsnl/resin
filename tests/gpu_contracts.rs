mod support;

use resin_source::{Loader, Source, library_root};
use resin_types::{GpuProjectionOperation, Ty};

const CONTRACTS: &str = r#"
    struct DeviceReference<T> { allocation: GpuView };
    struct HostRange<T> { pointer: Ptr<T>, count: ulong };
    struct DeviceRange<T> { pointer: DeviceReference<T>, count: ulong };
    intrinsic "gpu_pointer_projection" def project_pointer<T>(value: DeviceReference<T>) -> Ptr<T>;
    intrinsic "gpu_span_projection" def project_range<T>(value: DeviceRange<T>) -> HostRange<T>;
"#;

#[test]
fn source_gpu_contracts_retain_nominal_identity_and_produce_concrete_plans() {
    let source = format!(
        r#"
        {CONTRACTS}
        def materialize(value: DeviceRange<uint>, output: Ptr<HostRange<uint>>) = {{}};
    "#
    );
    let hir = support::hir(&source);
    let definitions = hir
        .types
        .iter()
        .filter(|definition| definition.gpu_projection.is_some())
        .collect::<Vec<_>>();
    assert_eq!(definitions.len(), 2);
    for definition in definitions {
        let projection = definition.gpu_projection.as_ref().unwrap();
        assert!(
            hir.functions[projection.declaration.index()]
                .name
                .starts_with("project_")
        );
    }
    let module = support::module(&source);
    let function = module
        .functions
        .iter()
        .find(|f| f.name.as_deref() == Some("materialize"))
        .unwrap();
    let source = &function.locals[0].ty;
    let Ty::Pointer { pointee: target } = &function.locals[1].ty else {
        panic!("target parameter");
    };
    let plan = resin_types::gpu_projection_plan(&module.types, source, target).unwrap();
    assert!(matches!(
        plan.operation,
        GpuProjectionOperation::Sequence {
            element: Ty::UInt32
        }
    ));
    assert!(
        resin_types::gpu_projection_plan(
            &module.types,
            source,
            &Ty::Pointer {
                pointee: Box::new(Ty::Float32)
            }
        )
        .is_err()
    );
}

#[test]
fn projection_registration_rejects_invalid_representations_and_retagged_elements() {
    for source in [
        r#"struct View<T> { raw: Ptr<T> }; intrinsic "gpu_pointer_projection" def register<T>(value: View<T>) -> Ptr<T>;"#,
        r#"struct View<T> { allocation: GpuView }; intrinsic "gpu_pointer_projection" def register<T>(value: View<T>) -> Ptr<ulong>;"#,
        r#"struct View<T> { allocation: GpuView }; intrinsic "gpu_pointer_projection" def register<T>(value: View<T>) -> Ptr<T>; intrinsic "gpu_pointer_projection" def duplicate<T>(value: View<T>) -> Ptr<T>;"#,
    ] {
        let error = support::frontend::check_hir(&support::program(source))
            .into_module()
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("invalid signature or wrapper storage"),
            "{error}"
        );
    }
}

#[test]
fn source_cannot_call_a_projection_contract_to_escape_a_device_address() {
    let source = format!(
        r#"
        {CONTRACTS}
        def escape(value: DeviceReference<uint>) -> Ptr<uint> = {{ project_pointer(value) }};
    "#
    );
    let hir = support::hir(&source);
    let error = support::frontend::lower(&hir, &[], &resin_lir::LoweringOptions::default())
        .unwrap_err()
        .remove(0);
    assert!(
        error.to_string().contains("cannot be called directly"),
        "{error}"
    );
}

#[test]
fn pipeline_contracts_are_explicit_and_preserve_source_parameter_identity() {
    let source = r#"
        struct Compute<R, O> { token: GpuPipelineContract };
        intrinsic "gpu_compute_pipeline_type" def register<R, O>(token: GpuPipelineContract) -> Compute<R, O>;
        def materialize(value: Compute<uint, StrongOwner>) = {};
    "#;
    let hir = support::hir(source);
    let declaration = hir
        .types
        .iter()
        .find(|source| source.gpu_pipeline.is_some())
        .unwrap();
    assert_eq!(declaration.name.as_ref(), "Compute");
    let module = support::module(source);
    let function = module
        .functions
        .iter()
        .find(|function| function.name.as_deref() == Some("materialize"))
        .unwrap();
    let contract =
        resin_types::gpu_pipeline_contract(&module.types, &function.locals[0].ty).unwrap();
    assert_eq!(contract.root, Ty::UInt32);
    assert_eq!(contract.owner, Ty::StrongOwner);
}

#[test]
fn pipeline_type_contracts_reject_invalid_storage_and_direct_calls() {
    for source in [
        r#"struct Compute<R, O> { token: StrongOwner }; intrinsic "gpu_compute_pipeline_type" def register<R, O>(token: GpuPipelineContract) -> Compute<R, O>;"#,
        r#"struct Compute<R, O> { token: GpuPipelineContract }; intrinsic "gpu_compute_pipeline_type" def register<R, O>(token: GpuPipelineContract) -> Compute<O, R>;"#,
        r#"struct Compute<R, O> { token: GpuPipelineContract, extra: uint }; intrinsic "gpu_compute_pipeline_type" def register<R, O>(token: GpuPipelineContract) -> Compute<R, O>;"#,
    ] {
        assert!(
            support::frontend::check_hir(&support::program(source))
                .into_module()
                .is_err()
        );
    }
    let source = r#"
        struct Compute<R, O> { token: GpuPipelineContract };
        intrinsic "gpu_compute_pipeline_type" def register<R, O>(token: GpuPipelineContract) -> Compute<R, O>;
        def forge(token: GpuPipelineContract) -> Compute<uint, StrongOwner> = { register(token) };
    "#;
    let hir = support::hir(source);
    assert!(
        support::frontend::lower(&hir, &[], &resin_lir::LoweringOptions::default())
            .unwrap_err()
            .remove(0)
            .to_string()
            .contains("cannot be called directly")
    );
}

#[test]
fn source_drop_hooks_reject_gpu_elements_before_and_after_importing() {
    for (declaration, owner) in [
        (
            "struct Managed { value: int, def drop(self: Ptr<Managed>) = {}; };",
            "Managed",
        ),
        (
            "struct Managed<T> { value: T, def drop(self: Ptr<Managed<T>>) = {}; };",
            "Managed<int>",
        ),
    ] {
        let use_site = format!(
            r#"
            def invalid(gpu: Gpu) -> Result<(), _> = {{
                gpu.alloc::<{owner}>(0_ul)?;
                ok(())
            }};
            "#
        );
        let local = format!("import {{ \"$/gpu.resin\" }}; {declaration} {use_site}");
        let error = support::pipeline::source_module(&local).unwrap_err();
        assert!(
            error.to_string().contains("plain shared storage"),
            "{error}"
        );

        let directory = tempfile::tempdir().unwrap();
        std::fs::write(
            directory.path().join("owner.resin"),
            format!("export {{ Managed }}; {declaration}"),
        )
        .unwrap();
        let entry = directory.path().join("main.resin");
        std::fs::write(
            &entry,
            format!("import {{ \"$/gpu.resin\", \"owner.resin\" }}; {use_site}"),
        )
        .unwrap();
        let error = support::pipeline::file_module(&entry).unwrap_err();
        assert!(
            error.to_string().contains("plain shared storage"),
            "{error}"
        );
    }
}

#[test]
fn source_gpu_library_resolves_generic_allocation_and_explicit_access() {
    let source = Source::new(
        "gpu-library.resin",
        r#"
        export { main };
        import { "$/gpu.resin", "$/span.resin" };
        struct Pair { left: int, right: int };
        def main() -> Result<(), _> = {
            var gpu = Gpu.new()?;
            var scalar = gpu.create(Pair { left = 1_i, right = 2_i })?;
            var value = scalar.load();
            value.right := 3_i;
            scalar.store(value);
            scalar.replace(value);
            var values = gpu.alloc::<int>(4_ul)?;
            values.at(1_ul).store(7_i);
            var tail = values.slice(1_ul, 2_ul);
            var alias = tail.data.slice(1_ul, 1_ul);
            var output = [0_i, 0_i];
            tail.read_only().copy_to(Span<int> { data = &output.at(0_ul), length = 2_ul });
            var readback = gpu.alloc_in::<ubyte>(64_ul, memory_readback)?;
            ok(())
        };
    "#,
    );
    let mut loader = Loader::new(library_root());
    let analysis = support::frontend::analyze(source, &mut loader, None);
    analysis.hir().unwrap();
}
