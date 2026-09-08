//! Compiler-provided method definitions. Lookup, argument checking, and editor
//! analysis use ordinary declarations. Generated bodies consume the unpacked
//! parameters on the IR stack, so indexing can preserve shader-local addresses
//! without passing them through a runtime call or tuple value.

use std::sync::Arc;

use crate::ir::{
    Instr, Ty, TyperContext,
    typecheck::{FunctionBody, FunctionDecl},
};

pub(super) fn register(typer: &mut TyperContext) {
    typer.register_method_definitions(definitions);
}

fn pointer(ty: Ty) -> Ty {
    Ty::Pointer {
        pointee: Box::new(ty),
    }
}

fn definitions(receiver: &Ty) -> Vec<(Arc<str>, FunctionDecl)> {
    match receiver {
        Ty::Pointer { pointee } => vec![method(
            "replace",
            vec![receiver.clone(), *pointee.clone()],
            *pointee.clone(),
            vec![Instr::Replace],
        )],
        Ty::Array { element, .. } => vec![method(
            "at",
            vec![pointer(receiver.clone()), Ty::UInt64],
            pointer(*element.clone()),
            vec![Instr::AccessDynamic],
        )],
        Ty::Span { element } => vec![method(
            "at",
            vec![receiver.clone(), Ty::UInt64],
            pointer(*element.clone()),
            vec![Instr::AccessDynamic],
        )],
        Ty::Arc { pointee } => vec![
            method(
                "get",
                vec![pointer(receiver.clone())],
                pointer(*pointee.clone()),
                vec![Instr::Load, Instr::ArcData],
            ),
            method(
                "downgrade",
                vec![receiver.clone()],
                Ty::Weak {
                    pointee: pointee.clone(),
                },
                vec![Instr::Downgrade],
            ),
        ],
        Ty::Weak { pointee } => vec![method(
            "upgrade",
            vec![receiver.clone()],
            Ty::union_of([
                Ty::Arc {
                    pointee: pointee.clone(),
                },
                Ty::None,
            ]),
            vec![Instr::Upgrade],
        )],
        _ => vec![],
    }
}

fn method(
    name: &str,
    params: Vec<Ty>,
    result: Ty,
    instructions: Vec<Instr>,
) -> (Arc<str>, FunctionDecl) {
    (
        name.into(),
        FunctionDecl {
            body: FunctionBody::Generated(instructions),
            params,
            result,
        },
    )
}
