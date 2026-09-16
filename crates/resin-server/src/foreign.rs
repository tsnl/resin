//! Translate completed LIR foreign declarations into independently reusable C adapters.
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
        let header = include(headers, &foreign.header)?;
        includes.insert(header.clone());
        let name = function
            .name
            .as_ref()
            .ok_or_else(|| invalid("foreign function has no C name"))?
            .to_string();
        let mut adapter = ForeignFunction {
            symbol: String::new(),
            name,
            params: foreign
                .params
                .iter()
                .map(scalar)
                .collect::<Result<_, _>>()?,
            result: scalar(&function.result)?,
        };
        let mut hash = blake3::Hasher::new();
        hash.update(b"resin-foreign-adapter-v1\0");
        for text in [header, format!("{adapter:?}")] {
            hash.update(&(text.len() as u64).to_le_bytes());
            hash.update(text.as_bytes());
        }
        adapter.symbol = format!("resin_foreign_{}", hash.finalize().to_hex());
        bindings.insert(
            FunctionId::from_index(index),
            Arc::from(adapter.symbol.as_str()),
        );
        functions.insert(adapter.symbol.clone(), adapter);
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
