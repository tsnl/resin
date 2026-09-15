use resin_hir::{TermKind, Type};

mod common;
use common::hir_module;

fn generate(source: &str) -> Result<resin_hir::Module, resin_source::SourceError> {
    hir_module(source)
}

#[test]
fn owner_operations_keep_generic_payloads_in_ordinary_signatures() {
    let module = generate(
        r#"
        intrinsic "owner_allocate" def allocate<T>(count: ulong, initial: T) -> StrongOwner | None;
        intrinsic "owner_data" def data<T>(owner: Ptr<StrongOwner>) -> Ptr<T>;
        struct Shared<T> { owner: StrongOwner,
            def get(self: Ptr<Shared<T>>) -> Ptr<T> = { data::<T>(&self.owner) };
        };
        def borrow<T>(value: Ptr<Shared<T>>) -> Ptr<T> = { value.get() };
    "#,
    )
    .unwrap();
    assert!(matches!(
        module
            .functions
            .iter()
            .find(|f| f.name.as_ref() == "allocate")
            .unwrap()
            .body
            .as_ref()
            .unwrap()
            .kind,
        TermKind::Intrinsic { .. }
    ));
    let Type::Pointer { pointee } = &module
        .functions
        .iter()
        .find(|f| f.name.as_ref() == "data")
        .unwrap()
        .signature
        .result
        .ty
    else {
        panic!("typed payload pointer")
    };
    assert!(matches!(**pointee, Type::Parameter { .. }));
    assert_eq!(module.types.last().unwrap().type_params.len(), 1);
}

#[test]
fn owner_handles_cannot_be_constructed_or_dereferenced() {
    for source in [
        "def bad() -> StrongOwner = { StrongOwner(0_ul) };",
        "def bad(value: StrongOwner) = { value.*; };",
        "def bad(value: WeakOwner) = { value.*; };",
        "def bad(value: WeakOwner) -> StrongOwner = { StrongOwner(value) };",
    ] {
        assert!(generate(source).is_err(), "{source}");
    }
}

#[test]
fn owner_primitives_reject_forged_contracts() {
    for source in [
        r#"intrinsic "owner_allocate" def bad<T>(count: int, initial: T) -> StrongOwner | None;"#,
        r#"intrinsic "owner_data" def bad<T>(owner: StrongOwner) -> Ptr<T>;"#,
        r#"intrinsic "owner_upgrade" def bad(owner: Ptr<StrongOwner>) -> StrongOwner | None;"#,
        r#"intrinsic "owner_downgrade" def bad(owner: Ptr<StrongOwner>) -> StrongOwner;"#,
    ] {
        assert!(generate(source).is_err(), "{source}");
    }
}
