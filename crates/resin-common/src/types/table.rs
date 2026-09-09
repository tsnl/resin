use std::{collections::HashMap, ops::Deref, sync::Arc};

use super::{Ty, TypeDef, TypeId};

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
        self.definitions.push(TypeDef::Nominal {
            name,
            body: None,
            drop: None,
        });
        self.ids.insert(Ty::Defined { definition: id }, id);
        id
    }

    pub(crate) fn define(&mut self, id: TypeId, body: Ty) {
        let TypeDef::Nominal { body: slot, .. } = &mut self.definitions[id.index()] else {
            unreachable!("only nominal definitions have pending bodies")
        };
        *slot = Some(body);
    }

    pub(crate) fn set_drop(&mut self, id: TypeId, function: crate::types::FunctionId) {
        let TypeDef::Nominal { drop, .. } = &mut self.definitions[id.index()] else {
            unreachable!("only nominal definitions have destruction hooks")
        };
        *drop = Some(function);
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
            if !matches!(ty, Ty::Defined { .. }) {
                self.components(ty);
            }
            return id;
        }
        assert!(!matches!(ty, Ty::Defined { .. }), "unreserved nominal type");
        let id = TypeId::from_index(self.len());
        self.definitions.push(TypeDef::Structural(ty.clone()));
        self.ids.insert(ty.clone(), id);
        self.components(ty);
        id
    }

    pub(crate) fn components(&mut self, ty: &Ty) {
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
            Ty::Pointer { pointee } | Ty::Arc { pointee } | Ty::Weak { pointee } => {
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
}
