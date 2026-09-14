mod support;

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
    let hir = resin_hir::generate(&support::parse(&source)).unwrap();
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
        let error = resin_hir::generate(&support::parse(source)).unwrap_err();
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
    let hir = resin_hir::generate(&support::parse(&source)).unwrap();
    let error = resin_lir::generate(&hir).unwrap_err();
    assert!(
        error.to_string().contains("cannot be called directly"),
        "{error}"
    );
}
