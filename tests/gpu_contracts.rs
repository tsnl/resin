mod support;

use resin_source::{Loader, Source, library_root};
use resin_types::{GpuProjectionOperation, Ty};

const CONTRACTS: &str = r#"struct DeviceReference<T> { allocation: GpuView, }
    struct HostRange<T> { pointer: PtrMut<T>, count: u64, }
    struct DeviceRange<T> { pointer: DeviceReference<T>, count: u64, }
    intrinsic "gpu_pointer_projection" fn project_pointer<T>(value: DeviceReference<T>) -> PtrMut<T>;
    intrinsic "gpu_span_projection" fn project_range<T>(value: DeviceRange<T>) -> HostRange<T>;
"#;

#[test]
fn source_gpu_contracts_retain_nominal_identity_and_produce_concrete_plans() {
    let source = format!(
        r#"{CONTRACTS}
        fn materialize(value: DeviceRange<u32>, output: PtrMut<HostRange<u32>>)  {{}}
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
    let Ty::Pointer {
        pointee: target, ..
    } = &function.locals[1].ty
    else {
        panic!("target parameter");
    };
    let plan = resin_types::gpu_projection_plan(&module.types, source, target).unwrap();
    assert!(matches!(
        plan.operation,
        GpuProjectionOperation::Sequence {
            writable: true,
            element: Ty::UInt32
        }
    ));
    assert!(
        resin_types::gpu_projection_plan(
            &module.types,
            source,
            &Ty::Pointer {
                mutable: true,
                pointee: Box::new(Ty::Float32)
            }
        )
        .is_err()
    );
}

#[test]
fn projection_registration_rejects_invalid_representations_and_retagged_elements() {
    for source in [
        r#"struct View<T> { raw: PtrMut<T>, } intrinsic "gpu_pointer_projection" fn register<T>(value: View<T>) -> PtrMut<T>;"#,
        r#"struct View<T> { allocation: GpuView, } intrinsic "gpu_pointer_projection" fn register<T>(value: View<T>) -> PtrMut<u64>;"#,
        r#"struct View<T> { allocation: GpuView, } intrinsic "gpu_pointer_projection" fn register<T>(value: View<T>) -> PtrMut<T>; intrinsic "gpu_pointer_projection" fn duplicate<T>(value: View<T>) -> PtrMut<T>;"#,
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
        r#"{CONTRACTS}
        fn escape(value: DeviceReference<u32>) -> PtrMut<u32>  {{ project_pointer(value) }}
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
    let source = r#"struct Compute<R, O> { token: GpuPipelineContract, }
        intrinsic "gpu_compute_pipeline_type" fn register<R, O>(token: GpuPipelineContract) -> Compute<R, O>;
        fn materialize(value: Compute<u32, StrongOwner>)  {}
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
        r#"struct Compute<R, O> { token: StrongOwner, } intrinsic "gpu_compute_pipeline_type" fn register<R, O>(token: GpuPipelineContract) -> Compute<R, O>;"#,
        r#"struct Compute<R, O> { token: GpuPipelineContract, } intrinsic "gpu_compute_pipeline_type" fn register<R, O>(token: GpuPipelineContract) -> Compute<O, R>;"#,
        r#"struct Compute<R, O> { token: GpuPipelineContract, extra: u32, } intrinsic "gpu_compute_pipeline_type" fn register<R, O>(token: GpuPipelineContract) -> Compute<R, O>;"#,
    ] {
        assert!(
            support::frontend::check_hir(&support::program(source))
                .into_module()
                .is_err()
        );
    }
    let source = r#"struct Compute<R, O> { token: GpuPipelineContract, }
        intrinsic "gpu_compute_pipeline_type" fn register<R, O>(token: GpuPipelineContract) -> Compute<R, O>;
        fn forge(token: GpuPipelineContract) -> Compute<u32, StrongOwner>  { register(token) }
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
            "struct Managed { value: i32,  }\nfn drop(self: RefMut<Managed>)  {}\n",
            "Managed",
        ),
        (
            "struct Managed<T> { value: T,  }\nfn drop<T>(self: RefMut<Managed<T>>)  {}\n",
            "Managed<i32>",
        ),
    ] {
        let use_site = format!(
            r#"fn invalid(gpu: Gpu) -> (() | Err<_>)  {{
                gpu:alloc::<{owner}>(u64(0))?;
                (())
            }}
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
        r#"export { main };
        import { "$/shared.resin", "$/gpu.resin", "$/span.resin" };
        struct Pair { left: i32, right: i32, }
        fn main() -> (() | Err<_>)  {
            let mut gpu = gpu_new()?;
            let mut scalar = gpu:create(Pair { left = i32(1), right = i32(2) })?;
            let mut value = scalar:load();
            value.right = i32(3);
            scalar:store(value);
            scalar:replace(Pair { left = i32(4), right = i32(5) });
            let mut values = gpu:alloc::<i32>(u64(4))?;
            { let borrowed = values:at(u64(1)); borrowed:store(i32(7)) };
            let mut tail = values:slice(u64(1), u64(2));
            let mut alias = tail:slice(u64(1), u64(1));
            let output_owner = arc_ptr_alloc([i32(0), i32(0)])?; let output: Ref<_> = output_owner:get().*;
            { let borrowed = tail:read_only(); borrowed:copy_to(SpanMut<i32> { data = output_owner:get():lea(u64(0)), length = u64(2) }) };
            let mut readback = gpu:alloc_in::<u8>(u64(64), memory_readback)?;
            (())
        }
    "#,
    );
    let mut loader = Loader::new(library_root());
    let analysis = support::frontend::analyze(source, &mut loader, None);
    analysis.hir().unwrap();
}
