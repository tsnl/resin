//! Builtin source types lower to ordinary nominal records and shared owners.
use crate::lower::context::Context;
use resin_common::types::{RecordField, Ty};

pub(super) fn typer() -> Context {
    let mut typer = Context::new();
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
    typer.set_string_type(Ty::Defined { definition });
    super::builtin_methods::register(&mut typer);
    typer
}

pub(super) fn shader_properties() -> Ty {
    Ty::Record {
        fields: vec![RecordField {
            name: "spirv".into(),
            ty: Ty::shader(),
        }],
    }
}
