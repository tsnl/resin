//! Compiler-provided method definitions. Lookup, argument checking, and editor
//! analysis use ordinary declarations. Generated bodies consume the unpacked
//! parameters on the IR stack, so indexing can preserve shader-local addresses
//! without passing them through a runtime call or tuple value.
use crate::lower::context::Context;
use crate::lower::namespaces::{FunctionBody, FunctionDecl};

use std::sync::Arc;

use crate::types::{Intrinsic, Ty};

pub(super) fn register(typer: &mut Context) {
    typer.register_method_definitions(definitions);
}

fn pointer(ty: Ty) -> Ty {
    Ty::Pointer {
        pointee: Box::new(ty),
    }
}

fn definitions(receiver: &Ty, typer: &Context) -> Vec<(Arc<str>, FunctionDecl)> {
    if typer.string_type() == Some(receiver) {
        return vec![method(
            "from_str",
            vec![Ty::byte_span()],
            receiver.clone(),
            Intrinsic::StringFromStr,
        )];
    }
    match receiver {
        Ty::Pointer { pointee } => vec![method(
            "replace",
            vec![receiver.clone(), *pointee.clone()],
            *pointee.clone(),
            Intrinsic::Replace,
        )],
        Ty::Array { element, .. } => vec![method(
            "at",
            vec![pointer(receiver.clone()), Ty::UInt64],
            pointer(*element.clone()),
            Intrinsic::Index,
        )],
        Ty::Span { element } => vec![method(
            "at",
            vec![receiver.clone(), Ty::UInt64],
            pointer(*element.clone()),
            Intrinsic::Index,
        )],
        Ty::Arc { pointee } => vec![
            method(
                "get",
                vec![pointer(receiver.clone())],
                pointer(*pointee.clone()),
                Intrinsic::ArcGet,
            ),
            method(
                "downgrade",
                vec![receiver.clone()],
                Ty::Weak {
                    pointee: pointee.clone(),
                },
                Intrinsic::Downgrade,
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
            Intrinsic::Upgrade,
        )],
        _ => vec![],
    }
}

fn method(
    name: &str,
    params: Vec<Ty>,
    result: Ty,
    intrinsic: Intrinsic,
) -> (Arc<str>, FunctionDecl) {
    (
        name.into(),
        FunctionDecl {
            body: FunctionBody::Intrinsic(intrinsic),
            params,
            result,
        },
    )
}
