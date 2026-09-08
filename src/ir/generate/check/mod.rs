//! Check every function and resolve its types before emitting instructions.

mod constraints;
mod expressions;
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
    ir::{Ty, TypeId, TyperContext, typer::SourceModuleId},
};
use expressions::Checker;
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
pub(super) struct Checked {
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
    source_module: SourceModuleId,
) -> Result<Checked> {
    let functions: Vec<_> = file
        .declarations()
        .filter_map(|stmt| {
            if let StmtKind::Function {
                name,
                params,
                result,
                body,
                ..
            } = &stmt.val
            {
                let mut scan = Scan::default();
                for (name, _) in params {
                    scan.bind(&name.val);
                }
                scan.term(body);
                Some((name, params, result, body, scan))
            } else {
                None
            }
        })
        .collect();
    let mut checker = Checker::new(typer, scopes.untraced(), source_module);
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
    for stmt in file.declarations() {
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
            let (_, params, _, body, _) = &functions[i];
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
            if let Err(error) = checker.solver.require(ty, *span) {
                // Empty arrays need an element annotation even when discarded.
                if matches!(
                    checker.solver.head(ty),
                    Type::Node(types::Head::Array(0), _)
                ) {
                    return Err(GenerateError::typing(
                        *span,
                        crate::ir::TypeError {
                            kind: crate::ir::TypeErrorKind::EmptyArrayNeedsElementType,
                        },
                    ));
                }
                return Err(error);
            }
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
    Ok(Checked {
        holes,
        expressions,
        definitions: checker.definitions,
    })
}
