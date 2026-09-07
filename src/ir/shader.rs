//! Declaration-level shader candidates. Artifact selection never follows function values.
use super::{Function, Ty, TyperContext};
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShaderEntry {
    pub stage: Arc<str>,
    pub embedded: bool,
}

pub fn validate(typer: &TyperContext, function: &Function, stage: &str) -> Result<(), String> {
    fn shape(typer: &TyperContext, ty: &Ty) -> Result<Ty, String> {
        let mut ty = ty.clone();
        while matches!(ty, Ty::Defined { .. }) {
            ty = typer.body(&ty).map_err(|e| e.to_string())?;
        }
        Ok(ty)
    }
    fn vector(typer: &TyperContext, ty: &Ty, names: &[&str]) -> bool {
        matches!(shape(typer, ty), Ok(Ty::Record { fields }) if fields.len() == names.len()
            && fields.iter().zip(names).all(|(field, name)| field.name.as_ref() == *name && field.ty == Ty::Float32))
    }
    if function.foreign.is_some() {
        return Err("foreign functions cannot be shader entries".into());
    }
    let param = function.locals.first().ok_or("invalid shader parameter")?;
    let param = shape(typer, &param.ty)?;
    let (input, root) = match &param {
        Ty::Record { fields }
            if fields.len() == 2
                && fields[0].name.as_ref() == "_0"
                && fields[1].name.as_ref() == "_1"
                && matches!(fields[1].ty, Ty::Pointer { .. }) =>
        {
            (&fields[0].ty, true)
        }
        _ => (&param, false),
    };
    let input = shape(typer, input)?;
    let result = shape(typer, &function.result)?;
    let valid = match stage {
        "compute" => input == Ty::UInt32 && result == if root { Ty::Unit } else { Ty::UInt32 },
        "vertex" => {
            input == Ty::Int32
                && matches!(&result, Ty::Record { fields }
            if fields.len() == 2 && fields[0].name.as_ref() == "position" && fields[1].name.as_ref() == "color"
                && vector(typer, &fields[0].ty, &["x", "y", "z", "w"])
                && vector(typer, &fields[1].ty, &["r", "g", "b", "a"]))
        }
        "fragment" => {
            vector(typer, &input, &["r", "g", "b", "a"])
                && vector(typer, &result, &["r", "g", "b", "a"])
        }
        _ => false,
    };
    if valid {
        Ok(())
    } else {
        Err(format!(
            "invalid @{stage}_shader signature: {}",
            match stage {
                "compute" => "expected (uint, Ptr<T>) -> () or (uint) -> uint",
                "vertex" => "expected int or (int, Ptr<T>) returning a position/color record",
                "fragment" =>
                    "expected Color or (Color, Ptr<T>) returning Color with float32 r/g/b/a fields",
                _ => "unknown shader stage",
            }
        ))
    }
}
