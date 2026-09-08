//! Payload type identities shared by every union in a compiled module.
//!
//! Nominal records already have a `TypeId`; primitives and structural types do
//! not. This table extends that ID space for both host and shader emission, so
//! a member's tag never depends on its position in a particular union. Numeric
//! IDs are compilation-local, not an ABI across separately compiled programs.
use crate::ir::{Case, Module, Ty, verify::FunctionTypes};
use std::collections::{BTreeSet, HashMap};

pub(crate) struct VariantTags(HashMap<Ty, u32>);

impl VariantTags {
    pub fn new(module: &Module, analysis: &[FunctionTypes]) -> Self {
        fn visit(ty: &Ty, seen: &mut BTreeSet<Ty>) {
            if !seen.insert(ty.clone()) {
                return;
            }
            match ty {
                Ty::Union { variants } => {
                    for member in variants {
                        visit(member, seen);
                    }
                }
                Ty::Result { value, error } => {
                    visit(value, seen);
                    visit(error, seen);
                }
                Ty::Pointer { pointee } => visit(pointee, seen),
                Ty::Span { element } | Ty::Array { element, .. } => visit(element, seen),
                Ty::Record { fields } => {
                    for field in fields {
                        visit(&field.ty, seen);
                    }
                }
                Ty::Function { param, result } => {
                    visit(param, seen);
                    visit(result, seen);
                }
                _ => {}
            }
        }
        let mut seen = BTreeSet::new();
        for definition in &module.types {
            if let Some(body) = definition.body() {
                visit(body, &mut seen);
            }
        }
        for function in &module.functions {
            visit(&function.result, &mut seen);
            for local in &function.locals {
                visit(&local.ty, &mut seen);
            }
        }
        for flow in analysis {
            for ty in flow
                .inputs
                .iter()
                .flatten()
                .chain(flow.results.iter().flatten().flatten())
            {
                visit(ty, &mut seen);
            }
        }
        let mut next = u32::try_from(module.types.len()).expect("too many types");
        Self(
            seen.into_iter()
                // Nominal IDs are already assigned. Unions only contribute
                // their members: normalization and verification forbid a
                // union from being another union's payload.
                .filter(|ty| !matches!(ty, Ty::Defined { .. } | Ty::Union { .. }))
                .map(|ty| {
                    next = next.checked_add(1).expect("too many union member types");
                    (ty, next)
                })
                .collect(),
        )
    }

    pub fn tag(&self, case: &Case) -> u32 {
        match case {
            Case::Ok => 0,
            Case::Err => 1,
            Case::Type(Ty::Defined { definition }) => definition.tag(),
            Case::Type(ty) => self.0[ty],
        }
    }
}
