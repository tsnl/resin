//! Declaration-level shader candidates. Artifact selection never follows function values.
use super::{Ty, TyperContext};
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShaderEntry {
    pub stage: Arc<str>,
    pub embedded: bool,
}

/// A checked stage interface, shared by declaration checking and GLSL emission.
#[derive(Debug, Clone)]
pub enum Interface {
    Compute {
        index: Ty,
    },
    Vertex {
        index: Ty,
        root: bool,
        position: Ty,
        color: Ty,
    },
    Fragment {
        color: Ty,
        root: bool,
    },
}

pub fn validate(
    typer: &TyperContext,
    parameter: &Ty,
    result: &Ty,
    foreign: bool,
    stage: &str,
) -> Result<Interface, String> {
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
    if foreign {
        return Err("foreign functions cannot be shader entries".into());
    }
    let param = shape(typer, parameter)?;
    let (input, root) = match &param {
        Ty::Record { fields }
            if fields.len() == 2
                && fields[0].name.as_ref() == "_0"
                && fields[1].name.as_ref() == "_1"
                && matches!(fields[1].ty, Ty::Pointer { .. }) =>
        {
            (&fields[0].ty, true)
        }
        _ => (parameter, false),
    };
    let input_shape = shape(typer, input)?;
    let result = shape(typer, result)?;
    let interface = match stage {
        "compute" if root && input_shape == Ty::UInt64 && result == Ty::Unit => {
            Some(Interface::Compute {
                index: input.clone(),
            })
        }
        "vertex" if input_shape == Ty::Int32 => match &result {
            Ty::Record { fields }
                if fields.len() == 2
                    && fields[0].name.as_ref() == "position"
                    && fields[1].name.as_ref() == "color"
                    && vector(typer, &fields[0].ty, &["x", "y", "z", "w"])
                    && vector(typer, &fields[1].ty, &["r", "g", "b", "a"]) =>
            {
                Some(Interface::Vertex {
                    index: input.clone(),
                    root,
                    position: fields[0].ty.clone(),
                    color: fields[1].ty.clone(),
                })
            }
            _ => None,
        },
        "fragment"
            if vector(typer, input, &["r", "g", "b", "a"])
                && vector(typer, &result, &["r", "g", "b", "a"]) =>
        {
            Some(Interface::Fragment {
                color: input.clone(),
                root,
            })
        }
        _ => None,
    };
    if let Some(interface) = interface {
        Ok(interface)
    } else {
        Err(format!(
            "invalid @{stage}_shader signature: {}",
            match stage {
                "compute" => "expected (ulong, Ptr<T>) -> ()",
                "vertex" => "expected int or (int, Ptr<T>) returning a position/color record",
                "fragment" =>
                    "expected Color or (Color, Ptr<T>) returning Color with float32 r/g/b/a fields",
                _ => "unknown shader stage",
            }
        ))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    Compute,
    Vertex,
    Fragment,
}

impl std::str::FromStr for Stage {
    type Err = String;

    fn from_str(name: &str) -> Result<Self, Self::Err> {
        match name {
            "compute" => Ok(Self::Compute),
            "vertex" => Ok(Self::Vertex),
            "fragment" => Ok(Self::Fragment),
            _ => Err("stage must be compute, vertex, or fragment".into()),
        }
    }
}

impl Stage {
    pub fn name(self) -> &'static str {
        match self {
            Self::Compute => "compute",
            Self::Vertex => "vertex",
            Self::Fragment => "fragment",
        }
    }
    pub fn entry(self) -> &'static str {
        match self {
            Self::Compute => "kernel",
            Self::Vertex => "vertex",
            Self::Fragment => "fragment",
        }
    }
}
