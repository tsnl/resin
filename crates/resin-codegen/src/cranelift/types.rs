use super::{failure, scalar_type, unsupported};
use crate::{Error, NativeInputs};
use cranelift_codegen::ir;
use cranelift_module::{DataDescription, DataId, FuncId, Linkage, Module};
use cranelift_object::ObjectModule;
use resin_types::prelude::*;
use std::collections::BTreeMap;

pub(super) struct Types<'a> {
    pub module: &'a resin_lir::Module,
    pub table: &'a TypeTable,
    layouts: BTreeMap<Ty, resin_types::layout::Layout>,
    literals: BTreeMap<Vec<u8>, DataId>,
    pub lifecycle: BTreeMap<Ty, Lifecycle>,
    shaders: BTreeMap<FunctionId, (DataId, usize)>,
    pub runtime: BTreeMap<std::sync::Arc<str>, std::sync::Arc<str>>,
}

pub(super) struct Lifecycle {
    pub retain: FuncId,
    pub drop: FuncId,
}

impl<'a> Types<'a> {
    pub fn new(
        checked: resin_lir::Verified<'a>,
        module: &mut ObjectModule,
        inputs: &NativeInputs,
    ) -> Result<Self, Error> {
        let table = &checked.analysis().types;
        let mut layouts = BTreeMap::new();
        for ty in table.types() {
            if matches!(ty, Ty::Foreign { .. }) {
                continue;
            }
            let layout = resin_types::host_layout(table, &ty).map_err(failure)?;
            if layout.size > u32::MAX as usize {
                return Err(unsupported("host value exceeds maximum stack slot size"));
            }
            layouts.insert(ty.clone(), layout);
        }
        let mut lifecycle = BTreeMap::new();
        let mut signature = module.make_signature();
        signature.params.push(ir::AbiParam::new(ir::types::I64));
        for ty in table.types().filter(|ty| ty.needs_drop(table)) {
            let id = table.id(&ty).unwrap().index();
            let retain = module
                .declare_function(&format!("resin_retain_{id}"), Linkage::Local, &signature)
                .map_err(failure)?;
            let drop = module
                .declare_function(&format!("resin_drop_{id}"), Linkage::Local, &signature)
                .map_err(failure)?;
            lifecycle.insert(ty, Lifecycle { retain, drop });
        }
        let mut types = Self {
            module: checked.module(),
            table,
            layouts,
            literals: BTreeMap::new(),
            lifecycle,
            shaders: BTreeMap::new(),
            runtime: inputs.runtime.clone(),
        };
        for function in &checked.module().functions {
            for block in &function.blocks {
                for instruction in &block.instrs {
                    if let resin_lir::Instr::Push { value } = instruction {
                        types.collect(module, value)?;
                    }
                }
            }
        }
        for (&function, bytes) in &inputs.shaders {
            if !checked.module().shaders.contains_key(&function) || bytes.len() % 4 != 0 {
                return Err(unsupported(
                    "shader input must name a declared shader and contain complete SPIR-V words",
                ));
            }
            let data = module
                .declare_anonymous_data(false, false)
                .map_err(failure)?;
            let mut description = DataDescription::new();
            description.define(bytes.to_vec().into_boxed_slice());
            description.set_align(4);
            module.define_data(data, &description).map_err(failure)?;
            types.shaders.insert(function, (data, bytes.len()));
        }
        for (&function, shader) in &checked.module().shaders {
            if shader.embedded && !types.shaders.contains_key(&function) {
                return Err(unsupported(format!(
                    "missing shader binary for function {}",
                    function.index()
                )));
            }
        }
        Ok(types)
    }

    pub fn layout(&self, ty: &Ty) -> &resin_types::layout::Layout {
        &self.layouts[ty]
    }

    pub fn shape<'b>(&'b self, mut ty: &'b Ty) -> &'b Ty {
        while let Ty::Defined { definition } = ty {
            ty = self.table[definition.index()].body().unwrap();
        }
        ty
    }

    pub fn scalar(&self, ty: &Ty) -> Option<ir::Type> {
        scalar_type(self.shape(ty)).ok()
    }

    pub fn value_type(&self, ty: &Ty) -> ir::Type {
        self.scalar(ty).unwrap_or(ir::types::I64)
    }

    pub fn signature(&self, module: &ObjectModule, params: &[Ty], result: &Ty) -> ir::Signature {
        let mut signature = module.make_signature();
        if let Some(result) = self.scalar(result) {
            signature.returns.push(ir::AbiParam::new(result));
        } else {
            signature.params.push(ir::AbiParam::new(ir::types::I64));
        }
        for ty in params {
            signature
                .params
                .push(ir::AbiParam::new(self.value_type(ty)));
        }
        signature
    }

    pub fn validate_value(&self, ty: &Ty) -> Result<(), Error> {
        if matches!(ty, Ty::Foreign { .. }) {
            return Err(unsupported("opaque foreign values have no representation"));
        }
        Ok(())
    }

    pub fn shader(&self, function: FunctionId) -> (DataId, usize) {
        self.shaders[&function]
    }

    pub fn literal(&self, bytes: &[u8]) -> DataId {
        self.literals[bytes]
    }

    fn collect(&mut self, module: &mut ObjectModule, value: &Value) -> Result<(), Error> {
        match value {
            Value::Str { value } if !self.literals.contains_key(value.as_ref()) => {
                let data = module
                    .declare_anonymous_data(false, false)
                    .map_err(failure)?;
                let mut description = DataDescription::new();
                description.define(
                    value
                        .iter()
                        .copied()
                        .chain([0])
                        .collect::<Vec<_>>()
                        .into_boxed_slice(),
                );
                description.set_align(1);
                module.define_data(data, &description).map_err(failure)?;
                self.literals.insert(value.to_vec(), data);
            }
            Value::Array { value } => {
                for value in &value.elements {
                    self.collect(module, value)?;
                }
            }
            Value::Record { value } => {
                for field in &value.fields {
                    self.collect(module, &field.value)?;
                }
            }
            _ => {}
        }
        Ok(())
    }
}
