use std::{collections::HashMap, ops::Deref, sync::Arc};

use super::{Ty, TypeDef, TypeId};
use crate::ir::{Instr, Module, Value, verify::FunctionTypes};

/// The canonical definitions of every type in one compiled program. IDs are
/// indices into `definitions`; the map only accelerates structural lookup.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TypeTable {
    definitions: Vec<TypeDef>,
    ids: HashMap<Ty, TypeId>,
}

impl Deref for TypeTable {
    type Target = [TypeDef];

    fn deref(&self) -> &Self::Target {
        &self.definitions
    }
}

impl From<Vec<TypeDef>> for TypeTable {
    fn from(definitions: Vec<TypeDef>) -> Self {
        let ids = definitions
            .iter()
            .enumerate()
            .map(|(index, definition)| {
                let id = TypeId::from_index(index);
                (definition.ty(id), id)
            })
            .collect();
        Self { definitions, ids }
    }
}

impl TypeTable {
    pub fn id(&self, ty: &Ty) -> Option<TypeId> {
        self.ids.get(ty).copied()
    }

    pub fn types(&self) -> impl Iterator<Item = Ty> + '_ {
        self.iter()
            .enumerate()
            .map(|(index, definition)| definition.ty(TypeId::from_index(index)))
    }

    pub(crate) fn reserve(&mut self, name: Arc<str>) -> TypeId {
        let id = TypeId::from_index(self.len());
        self.definitions.push(TypeDef::Nominal { name, body: None });
        self.ids.insert(Ty::Defined { definition: id }, id);
        id
    }

    pub(crate) fn define(&mut self, id: TypeId, body: Ty) {
        let TypeDef::Nominal { body: slot, .. } = &mut self.definitions[id.index()] else {
            unreachable!("only nominal definitions have pending bodies")
        };
        *slot = Some(body);
    }

    pub(crate) fn pop_nominal(&mut self) {
        let id = TypeId::from_index(self.len() - 1);
        assert!(matches!(
            self.definitions.pop(),
            Some(TypeDef::Nominal { .. })
        ));
        self.ids.remove(&Ty::Defined { definition: id });
    }

    /// Intern a resolved type without changing nominal identity. Nominal slots
    /// are reserved before their bodies, so recursive pointers terminate here.
    pub fn intern(&mut self, ty: &Ty) -> TypeId {
        if let Some(id) = self.id(ty) {
            return id;
        }
        assert!(!matches!(ty, Ty::Defined { .. }), "unreserved nominal type");
        let id = TypeId::from_index(self.len());
        self.definitions.push(TypeDef::Structural(ty.clone()));
        self.ids.insert(ty.clone(), id);
        self.components(ty);
        id
    }

    fn components(&mut self, ty: &Ty) {
        match ty {
            Ty::Union { variants } => {
                for member in variants {
                    self.intern(member);
                }
                self.intern(&Ty::UInt32);
            }
            Ty::Result { value, error } => {
                self.intern(value);
                self.intern(error);
                self.intern(&Ty::UInt32);
            }
            Ty::Pointer { pointee } => {
                self.intern(pointee);
            }
            Ty::Span { element } | Ty::Array { element, .. } => {
                self.intern(element);
            }
            Ty::Record { fields } => {
                for field in fields {
                    self.intern(&field.ty);
                }
            }
            Ty::Function { param, result } => {
                self.intern(param);
                self.intern(result);
            }
            _ => {}
        }
    }

    fn value(&mut self, value: &Value) {
        match value {
            Value::Type { ty } => {
                self.intern(ty);
            }
            Value::Array { value } => {
                for value in &value.elements {
                    self.value(value);
                }
            }
            Value::Record { value } => {
                for field in &value.fields {
                    self.value(&field.value);
                }
            }
            _ => {}
        }
    }

    /// Complete the table after stack verification has resolved every operand.
    /// The verified module shares this table across all shader and host emitters.
    pub(crate) fn collect(module: &Module, analysis: &[FunctionTypes]) -> Self {
        let mut table = module.types.clone();
        for definition in module.types.iter() {
            let body = definition.body().expect("verified type definition");
            table.intern(body);
            table.components(body);
        }
        for (function, flow) in module.functions.iter().zip(analysis) {
            table.intern(&function.ty().expect("verified parameter"));
            for local in &function.locals {
                table.intern(&local.ty);
            }
            for ty in flow
                .inputs
                .iter()
                .flatten()
                .chain(flow.results.iter().flatten().flatten())
            {
                table.intern(ty);
            }
            for block in &function.blocks {
                for instr in &block.instrs {
                    if let Instr::Push { value } = instr {
                        table.value(value);
                    }
                }
            }
        }
        table
    }
}
