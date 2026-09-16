//! Collect the C declarations that completed LIR requires from captured headers.
use crate::{
    headers::{NativeHeaders, NativeInclude},
    http::failure,
};
use resin_protocol::{ErrorCode, Failure};
use resin_toolchain::{ForeignFunction, ForeignInputs, ForeignScalar};
use resin_types::prelude::*;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

pub(crate) struct Prepared {
    pub inputs: Arc<ForeignInputs>,
    pub bindings: BTreeMap<FunctionId, Arc<str>>,
}

pub(crate) fn prepare(
    module: &resin_lir::Module,
    headers: &NativeHeaders,
) -> Result<Prepared, Failure> {
    let mut includes = BTreeSet::new();
    for header in &module.foreign_headers {
        includes.insert(include(headers, header)?);
    }
    let mut functions = BTreeMap::new();
    let mut bindings = BTreeMap::new();
    for (index, function) in module.functions.iter().enumerate() {
        let Some(foreign) = &function.foreign else {
            continue;
        };
        includes.insert(include(headers, &foreign.header)?);
        let name = function
            .name
            .as_ref()
            .ok_or_else(|| invalid("foreign function has no C name"))?
            .to_string();
        let declaration = ForeignFunction {
            name,
            params: foreign
                .params
                .iter()
                .map(scalar)
                .collect::<Result<_, _>>()?,
            result: scalar(&function.result)?,
        };
        bindings.insert(
            FunctionId::from_index(index),
            Arc::from(declaration.name.as_str()),
        );
        if functions
            .insert(declaration.name.clone(), declaration.clone())
            .is_some_and(|prior| prior != declaration)
        {
            return Err(invalid(format!(
                "conflicting Resin signatures for C symbol `{}`",
                declaration.name
            )));
        }
    }
    Ok(Prepared {
        inputs: Arc::new(ForeignInputs {
            files: headers.files.clone(),
            includes: includes.into_iter().collect(),
            include_directories: headers
                .include_directories
                .iter()
                .map(ToString::to_string)
                .collect(),
            functions: functions.into_values().collect(),
        }),
        bindings,
    })
}

fn include(headers: &NativeHeaders, header: &resin_lir::ForeignHeader) -> Result<String, Failure> {
    match headers.bindings.get(header) {
        Some(NativeInclude::Staged { path }) => Ok(path.to_string()),
        Some(NativeInclude::System { spelling }) => Ok(spelling.to_string()),
        None => Err(invalid(format!(
            "missing captured native header `{}`",
            header.spelling
        ))),
    }
}

fn scalar(ty: &Ty) -> Result<ForeignScalar, Failure> {
    Ok(match ty {
        Ty::Unit => ForeignScalar::Void,
        Ty::Bool => ForeignScalar::Bool,
        Ty::Int8 => ForeignScalar::Integer {
            bits: 8,
            signed: true,
        },
        Ty::UInt8 => ForeignScalar::Integer {
            bits: 8,
            signed: false,
        },
        Ty::Int16 => ForeignScalar::Integer {
            bits: 16,
            signed: true,
        },
        Ty::UInt16 => ForeignScalar::Integer {
            bits: 16,
            signed: false,
        },
        Ty::Int32 => ForeignScalar::Integer {
            bits: 32,
            signed: true,
        },
        Ty::UInt32 => ForeignScalar::Integer {
            bits: 32,
            signed: false,
        },
        Ty::Int64 => ForeignScalar::Integer {
            bits: 64,
            signed: true,
        },
        Ty::UInt64 => ForeignScalar::Integer {
            bits: 64,
            signed: false,
        },
        Ty::Float32 => ForeignScalar::Float { bits: 32 },
        Ty::Float64 => ForeignScalar::Float { bits: 64 },
        Ty::Pointer { .. } => ForeignScalar::Pointer,
        _ => {
            return Err(invalid(format!(
                "foreign declaration requires a scalar C type, found {ty:?}"
            )));
        }
    })
}

fn invalid(message: impl Into<String>) -> Failure {
    failure(ErrorCode::CompilationFailed, message)
}
