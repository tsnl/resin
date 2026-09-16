//! Constant declarations use ordinary expression typing, then checked scalar evaluation.
//! Each initializer is completed before its type can be influenced by a later use.
use super::*;
use crate::lower::elaborate;
use resin_ast::{ConstSpec, TermKind as AstTerm};

struct Initializer<'a> {
    name: &'a Ident,
    ann: Option<&'a resin_ast::Type>,
    expression: &'a resin_ast::Term,
    iota: usize,
}

fn initializers(specs: &[ConstSpec]) -> Result<Vec<Initializer<'_>>> {
    let mut result = Vec::new();
    for (iota, spec) in specs.iter().enumerate() {
        if spec.init.is_empty() {
            return Err(GenerateError::inference(
                spec.span,
                "each const specification requires an explicit initializer",
            ));
        }
        if spec.names.len() != spec.init.len() {
            return Err(GenerateError::inference(
                spec.span,
                "const names and initializers must have equal counts",
            ));
        }
        for (name, expression) in spec.names.iter().zip(&spec.init) {
            result.push(Initializer {
                name,
                ann: spec.ann.as_ref(),
                expression,
                iota,
            });
        }
    }
    Ok(result)
}

// This whitelist also supplies the module dependency graph. Runtime expressions
// are rejected even in an unselected branch of a short-circuit expression.
fn references<'a>(term: &'a resin_ast::Term, names: &mut Vec<&'a Ident>) -> Result<()> {
    match &term.val {
        AstTerm::Bool { .. }
        | AstTerm::Num { .. }
        | AstTerm::String { .. }
        | AstTerm::SizeOf { .. } => Ok(()),
        AstTerm::Var { name } => {
            names.push(name);
            Ok(())
        }
        AstTerm::Builtin { args, .. } => {
            for arg in args {
                references(arg, names)?;
            }
            Ok(())
        }
        AstTerm::Call { func, args } => {
            if let AstTerm::Var { name } = &func.val
                && matches!(name.val.as_ref(), "size_of" | "align_of")
                && matches!(
                    args.as_slice(),
                    [resin_ast::Term {
                        val: AstTerm::Type { .. },
                        ..
                    }]
                )
            {
                return Ok(());
            }
            if matches!(func.val, AstTerm::Type { .. }) && args.len() == 1 {
                return references(&args[0], names);
            }
            Err(GenerateError::inference(
                term.span,
                "const initializer requires constant operands, scalar conversions, or type layout queries",
            ))
        }
        _ => Err(GenerateError::inference(
            term.span,
            "expression is not a compile-time scalar constant",
        )),
    }
}

impl Checker<'_> {
    pub(super) fn module_constants(&mut self, file: &SourceFile) {
        let mut pending = Vec::new();
        for stmt in &file.stmts {
            if let StmtKind::Const { specs } = &stmt.val {
                match initializers(specs) {
                    Ok(items) => pending.extend(items),
                    Err(error) => self.errors.push(error),
                }
            }
        }
        let names: BTreeMap<_, _> = pending
            .iter()
            .enumerate()
            .filter(|(_, item)| item.name.val.as_ref() != "_")
            .map(|(index, item)| (item.name.val.as_ref(), index))
            .collect();
        let edges: Vec<Vec<usize>> = pending
            .iter()
            .map(|item| {
                let mut refs = Vec::new();
                if let Err(error) = references(item.expression, &mut refs) {
                    self.errors.push(error);
                }
                refs.into_iter()
                    .filter_map(|name| names.get(name.val.as_ref()).copied())
                    .collect()
            })
            .collect();
        for group in groups(&edges) {
            if group.len() > 1 || group.iter().any(|index| edges[*index].contains(index)) {
                self.errors.push(GenerateError::inference(
                    pending[group[0]].name.span,
                    "cyclic constant definitions",
                ));
                continue;
            }
            for index in group {
                let item = &pending[index];
                let value = self.constant_initializer(item);
                self.install_constant(item.name, value);
            }
        }
    }

    pub(super) fn local_constants(&mut self, specs: &[ConstSpec]) -> Result<()> {
        let items = initializers(specs)?;
        // Every name in a local specification becomes visible after all its
        // initializers, preserving access to an outer same-name constant.
        for row in 0..specs.len() {
            let values: Vec<_> = items
                .iter()
                .filter(|item| item.iota == row)
                .map(|item| (item.name, self.constant_initializer(item)))
                .collect();
            for (name, value) in values {
                self.install_constant(name, value);
            }
        }
        Ok(())
    }

    fn install_constant(&mut self, name: &Ident, value: Result<crate::Term>) {
        let binding = if name.val.as_ref() == "_" {
            None
        } else {
            self.bind(name, Type::Invalid, DefinitionKind::Constant)
                .map_err(|error| self.errors.push(error))
                .ok()
        };
        match value {
            Ok(value) => {
                if let Some(binding) = binding {
                    self.scopes.set_constant(binding, value);
                }
            }
            Err(error) => self.errors.push(error),
        }
    }

    fn constant_initializer(&mut self, item: &Initializer<'_>) -> Result<crate::Term> {
        let mut names = Vec::new();
        references(item.expression, &mut names)?;
        for name in names {
            if name.val.as_ref() != "iota" && self.scopes.constant(&name.val).is_none() {
                return Err(GenerateError::inference(
                    name.span,
                    format!("`{}` is not a constant", name.val),
                ));
            }
        }
        // Local constants must not solve the enclosing function's equations or
        // let a later use choose a different type for this declaration.
        let constraints = std::mem::take(&mut self.typing.constraints);
        let expressions = std::mem::take(&mut self.expressions);
        self.iota = Some(item.iota);
        let result = self.complete_constant(item);
        self.iota = None;
        self.typing.constraints = constraints;
        self.expressions = expressions;
        result
    }

    fn complete_constant(&mut self, item: &Initializer<'_>) -> Result<crate::Term> {
        let expected = item.ann.map(|ann| self.ann(ann, false).ty);
        let (_, term) = self.term(item.expression, expected);
        let roots = self
            .expressions
            .iter()
            .map(|(_, ty)| ty.clone())
            .chain([term.ty.clone()])
            .collect::<Vec<_>>();
        self.errors.extend(self.typing.solve(&roots));
        let completed = elaborate::function(
            &term,
            &[],
            &self.typing.solver,
            &self.typing.methods,
            self.typing.typer,
            &Default::default(),
            &Default::default(),
        )?;
        let value = evaluate(&completed.body, self.typing.typer)?;
        scalar_term(value, completed.body.ty, item.expression.span)
    }
}

fn concrete(ty: &crate::Type, span: Span) -> Result<Ty> {
    Solver::default().require(&Type::from_hir(ty), span)
}

fn scalar_term(value: Value, ty: crate::Type, span: Span) -> Result<crate::Term> {
    let kind = match value {
        Value::Str { value } => crate::TermKind::Constant {
            value: crate::Constant::Str { value },
        },
        Value::Bool { value } => crate::TermKind::Constant {
            value: crate::Constant::Bool { value },
        },
        value => crate::TermKind::Numeric {
            text: match value {
                Value::Int8 { value } => value.to_string(),
                Value::Int16 { value } => value.to_string(),
                Value::Int32 { value } => value.to_string(),
                Value::Int64 { value } => value.to_string(),
                Value::UInt8 { value } => value.to_string(),
                Value::UInt16 { value } => value.to_string(),
                Value::UInt32 { value } => value.to_string(),
                Value::UInt64 { value } => value.to_string(),
                Value::Float32 { value } => value.to_string(),
                Value::Float64 { value } => value.to_string(),
                _ => {
                    return Err(GenerateError::inference(
                        span,
                        "constants require a numeric, bool, or str type",
                    ));
                }
            }
            .into(),
        },
    };
    Ok(crate::Term { span, ty, kind })
}

fn evaluate(term: &crate::Term, context: &Context) -> Result<Value> {
    use crate::TermKind;
    let error = |message| GenerateError::inference(term.span, message);
    let ty = concrete(&term.ty, term.span)?;
    if !ty.is_numeric() && !matches!(ty, Ty::Bool | Ty::Str) {
        return Err(error(
            "constants require a numeric, bool, or str type".into(),
        ));
    }
    match &term.kind {
        TermKind::Constant {
            value: crate::Constant::Bool { value },
        } => Ok(Value::Bool { value: *value }),
        TermKind::Constant {
            value: crate::Constant::Str { value },
        } => Ok(Value::Str {
            value: value.clone(),
        }),
        TermKind::Numeric { text } => resin_types::literal::parse(text, &ty).map_err(error),
        TermKind::Builtin { name, args, .. } => {
            let operands = args
                .iter()
                .map(|arg| evaluate(arg, context))
                .collect::<Result<Vec<_>>>()?;
            resin_types::constant_operation(name, &operands).map_err(error)
        }
        TermKind::Convert { arg } | TermKind::Use { arg } if arg.ty == term.ty => {
            evaluate(arg, context)
        }
        TermKind::Convert { arg } => {
            resin_types::convert_constant(&evaluate(arg, context)?, &ty).map_err(error)
        }
        TermKind::If { cond, then, els } => {
            let Value::Bool { value } = evaluate(cond, context)? else {
                unreachable!("checked boolean");
            };
            let then = evaluate(then, context)?;
            let els = evaluate(els, context)?;
            Ok(if value { then } else { els })
        }
        TermKind::Layout { of, size } => {
            let ty = layout_type(of, context, term.span, 0)?;
            let layout = resin_types::layout::layout(&[], &ty).map_err(|e| error(e.to_string()))?;
            Ok(Value::UInt64 {
                value: if *size { layout.size } else { layout.align } as u64,
            })
        }
        TermKind::SizeOf { of } => {
            let ty = layout_type(of, context, term.span, 0)?;
            let layout = resin_types::layout::value(&[], &ty).map_err(|e| error(e.to_string()))?;
            Ok(Value::UInt64 {
                value: layout.size as u64,
            })
        }
        _ => Err(error(
            "expression is not a compile-time scalar constant".into(),
        )),
    }
}

// Expand only the data needed for layout. Pointers terminate recursive nominal
// types; resin-types owns all byte sizes, alignment, and overflow checks.
fn layout_type(ty: &crate::Type, context: &Context, span: Span, depth: usize) -> Result<Ty> {
    if depth > 128 {
        return Err(GenerateError::inference(
            span,
            "constant layout type is recursive or too deep",
        ));
    }
    let child = |ty| layout_type(ty, context, span, depth + 1);
    Ok(match ty {
        crate::Type::Defined {
            definition,
            arguments,
        } => {
            let scheme = &context.nominal_schemes[definition];
            let applied = Type::Apply {
                body: Box::new(Type::from_hir(&scheme.body)),
                arguments: scheme
                    .type_params
                    .iter()
                    .zip(arguments)
                    .map(|(parameter, arg)| (parameter.id, Type::from_hir(arg)))
                    .collect(),
            };
            child(&Solver::default().require_complete(&applied, span)?)?
        }
        crate::Type::Pointer { .. } => Ty::Pointer {
            pointee: Box::new(Ty::Unit),
        },
        crate::Type::Function { .. } => Ty::Function {
            params: vec![],
            result: Box::new(Ty::Unit),
        },
        crate::Type::Record { fields } => Ty::Record {
            fields: fields
                .iter()
                .map(|field| {
                    Ok(resin_types::RecordField {
                        name: field.name.clone(),
                        ty: child(&field.ty)?,
                    })
                })
                .collect::<Result<_>>()?,
        },
        crate::Type::Array { element, length } => Ty::Array {
            element: Box::new(child(element)?),
            length: *length,
        },
        crate::Type::Union { variants } => Ty::Union {
            variants: variants.iter().map(child).collect::<Result<Vec<_>>>()?,
        },
        crate::Type::Result { value, error } => Ty::Result {
            value: Box::new(child(value)?),
            error: Box::new(child(error)?),
        },
        _ => concrete(ty, span)?,
    })
}
