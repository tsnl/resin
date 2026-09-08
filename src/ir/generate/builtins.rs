//! Builtin source types lower to ordinary nominal records and shared owners.
use crate::ir::{RecordField, Ty, TyperContext};

pub(super) fn typer() -> TyperContext {
    let mut typer = TyperContext::new();
    let definition = typer
        .create_type(
            "String",
            Ty::Record {
                fields: vec![RecordField {
                    name: "bytes".into(),
                    ty: Ty::formatted_bytes(),
                }],
            },
        )
        .expect("builtin String layout");
    typer.string_type = Some(Ty::Defined { definition });
    super::builtin_methods::register(&mut typer);
    typer
}
