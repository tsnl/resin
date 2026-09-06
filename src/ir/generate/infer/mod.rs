//! Infer explicit type holes before emitting any function instructions.

mod check;
mod constraints;
mod scan;
mod solver;
mod types;

use std::{
    collections::{BTreeMap, HashMap},
    sync::Arc,
};

use super::{GenerateError, GenerateErrorKind, scope::Scopes};
use crate::{
    ast::{self, SourceFile, Span, StmtKind},
    ir::{Ty, TypeId, TyperContext},
};
use check::Checker;
use scan::Scan;
use types::Type;

type Result<T> = std::result::Result<T, GenerateError>;

pub(super) fn error(span: Span, message: impl Into<Arc<str>>) -> GenerateError {
    GenerateError {
        span,
        kind: GenerateErrorKind::Inference {
            message: message.into(),
        },
    }
}

#[derive(Default)]
pub(super) struct Inferred {
    // Node identity, not source spans: synthesized nodes can share a span.
    // Keys are never dereferenced and live only while lowering the borrowed AST.
    pub holes: HashMap<*const ast::Type, Ty>,
    pub expressions: HashMap<*const ast::Term, Ty>,
    pub definitions: HashMap<*const ast::Ident, TypeId>,
}

pub(super) fn file(
    file: &SourceFile,
    typer: &mut TyperContext,
    scopes: &Scopes,
) -> Result<Inferred> {
    let functions: Vec<_> = file
        .stmts
        .iter()
        .filter_map(|stmt| {
            if let StmtKind::Function {
                name,
                params,
                result,
                body,
            } = &stmt.val
            {
                let mut scan = Scan::default();
                scan.ty(result);
                for (name, ann) in params {
                    scan.bind(&name.val);
                    scan.ty(ann);
                }
                scan.term(body);
                Some((name, params, result, body, scan))
            } else {
                None
            }
        })
        .collect();
    if !functions.iter().any(|(_, _, _, _, scan)| scan.needed) {
        return Ok(Inferred::default());
    }
    let mut checker = Checker::new(typer, scopes.untraced());
    let mut results = Vec::new();
    let mut names = BTreeMap::new();
    for (i, (name, params, result, _, _)) in functions.iter().enumerate() {
        let params = params
            .iter()
            .map(|(_, ann)| checker.annotation(ann, false))
            .collect::<Result<Vec<_>>>()?;
        let result = checker.annotation(result, true)?;
        checker.functions.insert(
            name.val.clone(),
            Type::function(Type::parameter(params), result.clone()),
        );
        names.insert(name.val.to_string(), i);
        results.push(result);
    }
    for stmt in &file.stmts {
        if let StmtKind::ForeignFunction {
            name,
            params,
            result,
            ..
        } = &stmt.val
        {
            let params = params
                .iter()
                .map(|(_, ann)| checker.annotation(ann, false))
                .collect::<Result<Vec<_>>>()?;
            let result = checker.annotation(result, false)?;
            checker.functions.insert(
                name.val.clone(),
                Type::function(Type::parameter(params), result),
            );
        }
    }
    let edges: Vec<Vec<usize>> = functions
        .iter()
        .map(|(_, _, _, _, scan)| {
            scan.references
                .iter()
                .filter_map(|name| names.get(name).copied())
                .collect()
        })
        .collect();
    for group in scan::groups(&edges) {
        let first_expression = checker.expressions.len();
        for &i in &group {
            let (_, params, _, body, scan) = &functions[i];
            if !scan.needed {
                continue;
            }
            checker.push();
            checker.result = results[i].clone();
            for (name, ann) in *params {
                let ty = checker.annotation(ann, false)?;
                checker.bind(name, ty)?;
            }
            checker.term(body, Some(results[i].clone()))?;
            checker.pop();
        }
        let mut roots: Vec<_> = group.iter().map(|i| results[*i].clone()).collect();
        roots.extend(
            checker.expressions[first_expression..]
                .iter()
                .map(|(_, _, ty)| ty.clone()),
        );
        checker.solve(&roots)?;
        for &i in &group {
            checker.solver.require(&results[i], functions[i].2.span)?;
        }
        for (_, span, ty) in &checker.expressions[first_expression..] {
            checker.solver.require(ty, *span)?;
        }
    }
    let holes = checker
        .holes
        .iter()
        .map(|(node, span, ty)| Ok((*node, checker.solver.require(ty, *span)?)))
        .collect::<Result<_>>()?;
    let expressions = checker
        .expressions
        .iter()
        .map(|(node, span, ty)| Ok((*node, checker.solver.require(ty, *span)?)))
        .collect::<Result<_>>()?;
    Ok(Inferred {
        holes,
        expressions,
        definitions: checker.definitions,
    })
}
