use hashbrown::HashMap;
use num::ToPrimitive;

use crate::ast::{self, expr, stmt};
use crate::typer::{DeclBinding, TopLevel};
use crate::ir::{self, ConstVal, ElemOp, FuncId, Node, NodeId, Ref};
use crate::types::{Scope, SlotShape, TyCtx};
use crate::{Symbol, fb};

// ---------------------------------------------------------------------------
// Value — a lowered value that may span multiple IR slots
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
enum Value {
    /// Single tensor — one Ref.
    Slot(Ref),
    /// Exploded record — ordered named slots.
    Record(Vec<(Symbol, Value)>),
}

impl Value {
    /// Flatten to a list of Refs (leaf slots).
    fn flatten(&self) -> Vec<Ref> {
        match self {
            Value::Slot(r) => vec![r.clone()],
            Value::Record(fields) => {
                fields.iter().flat_map(|(_, v)| v.flatten()).collect()
            }
        }
    }

    /// Number of leaf slots.
    fn slot_count(&self) -> u32 {
        match self {
            Value::Slot(_) => 1,
            Value::Record(fields) => fields.iter().map(|(_, v)| v.slot_count()).sum(),
        }
    }

    /// Reconstruct a Value from a SlotShape and a flat iterator of Refs.
    fn from_shape_and_refs(shape: &SlotShape, refs: &mut impl Iterator<Item = Ref>) -> Self {
        match shape {
            SlotShape::Tensor => Value::Slot(refs.next().expect("not enough refs")),
            SlotShape::Record(fields) => Value::Record(
                fields
                    .iter()
                    .map(|(name, sub)| (name.clone(), Value::from_shape_and_refs(sub, refs)))
                    .collect(),
            ),
        }
    }

    /// Project a field from a Record value.
    fn project(&self, field: &Symbol) -> Option<&Value> {
        match self {
            Value::Record(fields) => {
                fields.iter().find(|(n, _)| n == field).map(|(_, v)| v)
            }
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// FuncBuilder — per-function lowering state
// ---------------------------------------------------------------------------

struct FuncBuilder<'a> {
    id: FuncId,
    name: Symbol,
    param_names: Vec<Symbol>,
    nodes: Vec<Node>,
    next_param_idx: u32,
    top: &'a TopLevel,
}

impl<'a> FuncBuilder<'a> {
    fn new(id: FuncId, name: Symbol, top: &'a TopLevel) -> Self {
        FuncBuilder {
            id,
            name,
            param_names: vec![],
            nodes: vec![],
            next_param_idx: 0,
            top,
        }
    }

    /// Emit a node and return its NodeId.
    fn emit(&mut self, node: Node) -> NodeId {
        let id = NodeId(self.nodes.len() as u32);
        self.nodes.push(node);
        id
    }

    /// Emit a Param node.
    fn emit_param(&mut self, name: Symbol) -> Ref {
        let idx = self.next_param_idx;
        self.next_param_idx += 1;
        self.param_names.push(name);
        let nid = self.emit(Node::Param { idx });
        Ref::simple(nid)
    }

    /// Build params from a SlotShape, returning the Value.
    fn build_params_from_shape(
        &mut self,
        name: &Symbol,
        shape: &SlotShape,
    ) -> Value {
        match shape {
            SlotShape::Tensor => {
                let r = self.emit_param(name.clone());
                Value::Slot(r)
            }
            SlotShape::Record(fields) => {
                let rec_fields: Vec<(Symbol, Value)> = fields
                    .iter()
                    .map(|(fname, sub_shape)| {
                        // Param names include field info for diagnostics
                        let param_name =
                            Symbol::from(format!("{}.{}", name.text(), fname.text()));
                        let val = self.build_params_from_shape(&param_name, sub_shape);
                        (fname.clone(), val)
                    })
                    .collect();
                Value::Record(rec_fields)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Operator symbol → ElemOp mapping
// ---------------------------------------------------------------------------

fn op_from_symbol(sym: &str) -> Option<ElemOp> {
    Some(match sym {
        "+(_,_)" => ElemOp::Add,
        "-(_,_)" => ElemOp::Sub,
        "*(_,_)" => ElemOp::Mul,
        "/(_,_)" => ElemOp::Div,
        "//(_,_)" => ElemOp::IntDiv,
        "%(_,_)" => ElemOp::Rem,
        "==(_,_)" => ElemOp::Eq,
        "!=(_,_)" => ElemOp::Ne,
        "<(_,_)" => ElemOp::Lt,
        ">(_,_)" => ElemOp::Gt,
        "<=(_,_)" => ElemOp::Le,
        ">=(_,_)" => ElemOp::Ge,
        "||(_,_)" => ElemOp::Or,
        "&&(_,_)" => ElemOp::And,
        _ => return None,
    })
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

pub fn lower(file: &ast::File, top: &TopLevel) -> fb::Result<ir::Program> {
    let mut functions = vec![];
    let mut errors = vec![];

    for stmt in &file.stmts {
        if let ast::Stmt::Def(inner) = stmt {
            let stmt::Def {
                name, args, body, ..
            } = inner.as_ref();

            // Skip type definitions (uppercase names)
            if name.is_upper_id() {
                continue;
            }

            let id = match top.func_id(name) {
                Some(id) => id,
                None => continue,
            };

            match lower_function(id, name, args, body, top) {
                Ok(func) => functions.push(func),
                Err(e) => errors.extend(e.messages),
            }
        }
    }

    if !errors.is_empty() {
        return Err(fb::Outbox { messages: errors });
    }

    // Collect builtin function IDs
    let mut builtins = hashbrown::HashMap::new();
    for (_, binding) in &top.bindings {
        if let DeclBinding::Builtin { id, builtin, .. } = binding {
            let kind = match builtin {
                crate::typer::Builtin::Matmul => ir::BuiltinKind::Matmul,
                crate::typer::Builtin::CrossEntropy => ir::BuiltinKind::CrossEntropy,
                crate::typer::Builtin::Grad => continue,
            };
            builtins.insert(*id, kind);
        }
    }

    Ok(ir::Program { functions, builtins })
}

// ---------------------------------------------------------------------------
// Lower a single function
// ---------------------------------------------------------------------------

fn lower_function(
    id: FuncId,
    name: &Symbol,
    args: &[(Symbol, crate::Span)],
    body: &ast::Expr,
    top: &TopLevel,
) -> fb::Result<ir::Function> {
    let mut builder = FuncBuilder::new(id, name.clone(), top);

    // Determine slot shapes from the (now-inferred) type scheme
    let param_shapes = param_slot_shapes(name, args.len(), top);

    // Build param bindings
    let mut bindings: HashMap<Symbol, Value> = HashMap::new();
    for (i, (arg_name, _)) in args.iter().enumerate() {
        let shape = param_shapes
            .get(i)
            .cloned()
            .unwrap_or(SlotShape::Tensor);
        let val = builder.build_params_from_shape(arg_name, &shape);
        bindings.insert(arg_name.clone(), val);
    }

    let root_scope = Scope::root(bindings);
    let result = lower_expr(&mut builder, body, &root_scope)?;

    let outputs = result.flatten();

    Ok(ir::Function {
        id: builder.id,
        name: builder.name,
        params: builder.param_names,
        nodes: builder.nodes,
        outputs,
    })
}

/// Derive param slot shapes from the function's inferred type scheme.
fn param_slot_shapes(name: &Symbol, arity: usize, top: &TopLevel) -> Vec<SlotShape> {
    if let Some(scheme) = top.func_scheme(name) {
        let ctx = TyCtx::new();
        match &scheme.ty {
            crate::types::Ty::Fn { params, .. } => {
                params.iter().map(|t| SlotShape::from_ty(t, &ctx)).collect()
            }
            _ => vec![SlotShape::Tensor; arity],
        }
    } else {
        vec![SlotShape::Tensor; arity]
    }
}

// ---------------------------------------------------------------------------
// Expression lowering
// ---------------------------------------------------------------------------

fn lower_expr<'a>(
    builder: &mut FuncBuilder,
    expr: &ast::Expr,
    scope: &Scope<'a, Value>,
) -> fb::Result<Value> {
    match expr {
        ast::Expr::Literal(inner) => lower_literal(builder, &inner.val),

        ast::Expr::Name(inner) => {
            let name = &inner.name;
            // Check local scope first
            if let Some(val) = scope.lookup(name) {
                return Ok(val.clone());
            }
            // Function names cannot be used as values (first-class functions
            // are not supported). They can only appear as callees in Apply
            // or as arguments to `grad`.
            if builder.top.func_id(name).is_some() {
                return Err(fb::Outbox {
                    messages: vec![fb::Message {
                        title: format!(
                            "function '{name}' cannot be used as a value; \
                             call it directly or use `grad {name} args...`"
                        ),
                        span: Some(inner.span.clone()),
                        notes: vec![],
                    }],
                });
            }
            Err(fb::Outbox {
                messages: vec![fb::Message {
                    title: format!("undefined name: {name}"),
                    span: Some(inner.span.clone()),
                    notes: vec![],
                }],
            })
        }

        ast::Expr::Apply(inner) => {
            let expr::Apply { callee, args, .. } = inner.as_ref();
            lower_apply(builder, callee, args, scope)
        }

        ast::Expr::If(inner) => {
            let expr::If {
                cond_branch_vec,
                else_branch,
                ..
            } = inner.as_ref();
            lower_if(builder, cond_branch_vec, else_branch, scope)
        }

        ast::Expr::Chain(inner) => {
            let expr::Chain { stmt_vec, .. } = inner.as_ref();
            lower_chain(builder, stmt_vec, scope)
        }

        ast::Expr::Dot(inner) => {
            let expr::Dot { base, field, .. } = inner.as_ref();
            let base_val = lower_expr(builder, base, scope)?;
            match base_val.project(field) {
                Some(val) => Ok(val.clone()),
                None => {
                    // The base isn't a record — it's a single slot that should
                    // have been exploded. This happens when a function lacks a
                    // type signature. Emit an error with a helpful message.
                    Err(fb::Outbox {
                        messages: vec![fb::Message {
                            title: format!(
                                "no field '{field}' on value (add a type signature to enable struct explosion)"
                            ),
                            span: Some(inner.span.clone()),
                            notes: vec![],
                        }],
                    })
                }
            }
        }

        ast::Expr::Tuple(inner) => {
            // Lower each element; wrap as a record with numeric field names
            let mut fields = vec![];
            for (i, elem) in inner.elements.iter().enumerate() {
                let val = lower_expr(builder, elem, scope)?;
                fields.push((Symbol::from(format!("{i}")), val));
            }
            Ok(Value::Record(fields))
        }

        ast::Expr::Ctor(_) => {
            // Type constructors shouldn't appear in expression position during IR gen
            Ok(Value::Record(vec![]))
        }

        ast::Expr::As(inner) => {
            // View cast — pass through for now (actual view computation deferred)
            lower_expr(builder, &inner.expr, scope)
        }

        ast::Expr::Grad(inner) => {
            lower_grad(builder, &inner.func, &inner.args, &inner.span, scope)
        }

        ast::Expr::Match(inner) => {
            // Match is not yet supported in IR gen
            Err(fb::Outbox {
                messages: vec![fb::Message {
                    title: "match expressions not yet supported in IR generation".into(),
                    span: Some(inner.span.clone()),
                    notes: vec![],
                }],
            })
        }
    }
}

// ---------------------------------------------------------------------------
// Literals
// ---------------------------------------------------------------------------

fn lower_literal(
    builder: &mut FuncBuilder,
    lit: &crate::vocab::Literal,
) -> fb::Result<Value> {
    let val = match lit {
        crate::vocab::Literal::Number(n) => {
            if n.is_integer() {
                ConstVal::Int(n.value.to_i64().unwrap_or(0))
            } else {
                ConstVal::Float(n.value.to_f64().unwrap_or(0.0))
            }
        }
        crate::vocab::Literal::Bool(b) => ConstVal::Bool(b.value),
        crate::vocab::Literal::String(_) => {
            // Strings don't exist in tensor IR — emit as int 0 placeholder
            ConstVal::Int(0)
        }
    };
    let nid = builder.emit(Node::Const { val });
    Ok(Value::Slot(Ref::simple(nid)))
}

// ---------------------------------------------------------------------------
// Apply (function calls and operators)
// ---------------------------------------------------------------------------

fn lower_apply<'a>(
    builder: &mut FuncBuilder,
    callee: &ast::Expr,
    args: &[ast::Expr],
    scope: &Scope<'a, Value>,
) -> fb::Result<Value> {
    // Check if callee is a name
    if let ast::Expr::Name(name_inner) = callee {
        let name = &name_inner.name;
        let text = name.text();

        // Check for unary negation
        if text.as_str() == "~(_)" && args.len() == 1 {
            let operand = lower_expr(builder, &args[0], scope)?;
            return lower_unary_neg(builder, operand);
        }

        // Check for binary operator symbols
        if let Some(op) = op_from_symbol(text.as_str()) {
            return lower_binop(builder, op, args, scope);
        }

        // Check for known functions/builtins
        if let Some(binding) = builder.top.lookup(name) {
            match binding {
                DeclBinding::Func { id, .. } => {
                    let func_id = *id;
                    // Special case: max with 2 args → Elem(Max)
                    if text.as_str() == "max" && args.len() == 2 {
                        return lower_binop(builder, ElemOp::Max, args, scope);
                    }
                    return lower_call(builder, func_id, name, args, scope);
                }
                DeclBinding::Builtin { id, .. } => {
                    let func_id = *id;
                    return lower_call(builder, func_id, name, args, scope);
                }
                DeclBinding::TypeDef { .. } => {
                    // Type constructor used as function — not supported
                }
            }
        }

        // Check if it's a local variable being "applied" (shouldn't happen in well-typed code)
        if scope.lookup(name).is_some() {
            // Treat as function call to the variable — error for now
            return Err(fb::Outbox {
                messages: vec![fb::Message {
                    title: format!("cannot call local variable '{name}' as a function"),
                    span: Some(name_inner.span.clone()),
                    notes: vec![],
                }],
            });
        }

        return Err(fb::Outbox {
            messages: vec![fb::Message {
                title: format!("undefined function: {name}"),
                span: Some(name_inner.span.clone()),
                notes: vec![],
            }],
        });
    }

    Err(fb::Outbox {
        messages: vec![fb::Message {
            title: "complex callee expressions not yet supported".into(),
            span: Some(callee.span().clone()),
            notes: vec![],
        }],
    })
}

/// Lower `grad f x y z` — a modified call that differentiates `f`.
fn lower_grad<'a>(
    builder: &mut FuncBuilder,
    func_expr: &ast::Expr,
    args: &[ast::Expr],
    span: &crate::Span,
    scope: &Scope<'a, Value>,
) -> fb::Result<Value> {
    let func_name = match func_expr {
        ast::Expr::Name(n) => &n.name,
        _ => {
            return Err(fb::Outbox {
                messages: vec![fb::Message {
                    title: "grad argument must be a function name".into(),
                    span: Some(span.clone()),
                    notes: vec![],
                }],
            });
        }
    };
    let target_func_id = match builder.top.lookup(func_name) {
        Some(DeclBinding::Func { id, .. } | DeclBinding::Builtin { id, .. }) => *id,
        _ => {
            return Err(fb::Outbox {
                messages: vec![fb::Message {
                    title: format!("'{}' is not a function", func_name),
                    span: Some(span.clone()),
                    notes: vec![],
                }],
            });
        }
    };
    let mut flat_args = vec![];
    let mut first_arg_val: Option<Value> = None;
    for (i, arg) in args.iter().enumerate() {
        let val = lower_expr(builder, arg, scope)?;
        if i == 0 {
            first_arg_val = Some(val.clone());
        }
        flat_args.extend(val.flatten());
    }
    let nid = builder.emit(Node::Grad {
        func: target_func_id,
        args: flat_args,
    });
    // grad returns the same shape as the first argument
    let ret_shape = match &first_arg_val {
        Some(v) => value_to_slot_shape(v),
        None => SlotShape::Tensor,
    };
    let total = ret_shape.total_slots();
    let refs: Vec<Ref> = (0..total).map(|i| Ref::output(nid, i)).collect();
    let mut ref_iter = refs.into_iter();
    Ok(Value::from_shape_and_refs(&ret_shape, &mut ref_iter))
}

/// Lower a binary operator application.
fn lower_binop<'a>(
    builder: &mut FuncBuilder,
    op: ElemOp,
    args: &[ast::Expr],
    scope: &Scope<'a, Value>,
) -> fb::Result<Value> {
    if args.len() != 2 {
        return Err(fb::Outbox {
            messages: vec![fb::Message {
                title: format!(
                    "binary operator {:?} requires exactly 2 arguments, got {}",
                    op,
                    args.len()
                ),
                span: args.first().map(|a| a.span().clone()),
                notes: vec![],
            }],
        });
    }
    let lhs = lower_expr(builder, &args[0], scope)?;
    let rhs = lower_expr(builder, &args[1], scope)?;

    // If both sides are records, apply element-wise to corresponding slots
    match (&lhs, &rhs) {
        (Value::Record(lf), Value::Record(rf)) if lf.len() == rf.len() => {
            let fields: fb::Result<Vec<(Symbol, Value)>> = lf
                .iter()
                .zip(rf.iter())
                .map(|((ln, lv), (_, rv))| {
                    let result = apply_binop_to_values(builder, op, lv, rv)?;
                    Ok((ln.clone(), result))
                })
                .collect();
            Ok(Value::Record(fields?))
        }
        // Scalar op on single slots
        (Value::Slot(l), Value::Slot(r)) => {
            let nid = builder.emit(Node::Elem {
                op,
                args: vec![l.clone(), r.clone()],
            });
            Ok(Value::Slot(Ref::simple(nid)))
        }
        // Broadcast scalar across record fields
        (Value::Slot(_), Value::Record(rf)) => {
            let fields: fb::Result<Vec<(Symbol, Value)>> = rf
                .iter()
                .map(|(fname, rv)| {
                    let result = apply_binop_to_values(builder, op, &lhs, rv)?;
                    Ok((fname.clone(), result))
                })
                .collect();
            Ok(Value::Record(fields?))
        }
        (Value::Record(lf), Value::Slot(_)) => {
            let fields: fb::Result<Vec<(Symbol, Value)>> = lf
                .iter()
                .map(|(fname, lv)| {
                    let result = apply_binop_to_values(builder, op, lv, &rhs)?;
                    Ok((fname.clone(), result))
                })
                .collect();
            Ok(Value::Record(fields?))
        }
        _ => {
            // Mismatched record shapes — type error
            Err(fb::Outbox {
                messages: vec![fb::Message {
                    title: "cannot apply binary op to values with different structures".into(),
                    span: None,
                    notes: vec![],
                }],
            })
        }
    }
}

/// Recursively apply a binary op to two values.
fn apply_binop_to_values(
    builder: &mut FuncBuilder,
    op: ElemOp,
    lhs: &Value,
    rhs: &Value,
) -> fb::Result<Value> {
    match (lhs, rhs) {
        (Value::Slot(l), Value::Slot(r)) => {
            let nid = builder.emit(Node::Elem {
                op,
                args: vec![l.clone(), r.clone()],
            });
            Ok(Value::Slot(Ref::simple(nid)))
        }
        (Value::Record(lf), Value::Record(rf)) if lf.len() == rf.len() => {
            let fields: fb::Result<Vec<(Symbol, Value)>> = lf
                .iter()
                .zip(rf.iter())
                .map(|((ln, lv), (_, rv))| {
                    let result = apply_binop_to_values(builder, op, lv, rv)?;
                    Ok((ln.clone(), result))
                })
                .collect();
            Ok(Value::Record(fields?))
        }
        (Value::Slot(_), Value::Record(rf)) => {
            let fields: fb::Result<Vec<(Symbol, Value)>> = rf
                .iter()
                .map(|(fname, rv)| {
                    let result = apply_binop_to_values(builder, op, lhs, rv)?;
                    Ok((fname.clone(), result))
                })
                .collect();
            Ok(Value::Record(fields?))
        }
        (Value::Record(lf), Value::Slot(_)) => {
            let fields: fb::Result<Vec<(Symbol, Value)>> = lf
                .iter()
                .map(|(fname, lv)| {
                    let result = apply_binop_to_values(builder, op, lv, rhs)?;
                    Ok((fname.clone(), result))
                })
                .collect();
            Ok(Value::Record(fields?))
        }
        _ => Err(fb::Outbox {
            messages: vec![fb::Message {
                title: "binary op structure mismatch".into(),
                span: None,
                notes: vec![],
            }],
        }),
    }
}

/// Lower unary negation, recursing into records.
fn lower_unary_neg(builder: &mut FuncBuilder, val: Value) -> fb::Result<Value> {
    match val {
        Value::Slot(r) => {
            let nid = builder.emit(Node::Elem {
                op: ElemOp::Neg,
                args: vec![r],
            });
            Ok(Value::Slot(Ref::simple(nid)))
        }
        Value::Record(fields) => {
            let new_fields: fb::Result<Vec<(Symbol, Value)>> = fields
                .into_iter()
                .map(|(name, v)| {
                    let negated = lower_unary_neg(builder, v)?;
                    Ok((name, negated))
                })
                .collect();
            Ok(Value::Record(new_fields?))
        }
    }
}

/// Lower a function call.
fn lower_call<'a>(
    builder: &mut FuncBuilder,
    func_id: FuncId,
    func_name: &Symbol,
    args: &[ast::Expr],
    scope: &Scope<'a, Value>,
) -> fb::Result<Value> {
    // Lower and flatten all arguments
    let mut flat_args = vec![];
    for arg in args {
        let val = lower_expr(builder, arg, scope)?;
        flat_args.extend(val.flatten());
    }

    let nid = builder.emit(Node::Call {
        func: func_id,
        args: flat_args,
    });

    // Reconstruct value from return slot shape
    let ret_shape = builder.top.return_slots(func_name);
    let total = ret_shape.total_slots();
    let refs: Vec<Ref> = (0..total).map(|i| Ref::output(nid, i)).collect();
    let mut ref_iter = refs.into_iter();
    Ok(Value::from_shape_and_refs(&ret_shape, &mut ref_iter))
}

// ---------------------------------------------------------------------------
// If → Cond (inside-out)
// ---------------------------------------------------------------------------

fn lower_if<'a>(
    builder: &mut FuncBuilder,
    cond_branches: &[(ast::Expr, ast::Expr)],
    else_branch: &ast::Expr,
    scope: &Scope<'a, Value>,
) -> fb::Result<Value> {
    // Start with else branch
    let mut current_else = lower_expr(builder, else_branch, scope)?;

    // Process branches from last to first (inside-out)
    for (cond_expr, then_expr) in cond_branches.iter().rev() {
        let pred_val = lower_expr(builder, cond_expr, scope)?;
        let pred_ref = match &pred_val {
            Value::Slot(r) => r.clone(),
            _ => {
                return Err(fb::Outbox {
                    messages: vec![fb::Message {
                        title: "condition must be a scalar, not a record".into(),
                        span: Some(cond_expr.span().clone()),
                        notes: vec![],
                    }],
                })
            }
        };

        let then_val = lower_expr(builder, then_expr, scope)?;
        let then_refs = then_val.flatten();
        let else_refs = current_else.flatten();

        if then_refs.len() != else_refs.len() {
            return Err(fb::Outbox {
                messages: vec![fb::Message {
                    title: format!(
                        "if/else branch slot count mismatch: then has {}, else has {}",
                        then_refs.len(),
                        else_refs.len()
                    ),
                    span: Some(cond_expr.span().clone()),
                    notes: vec![],
                }],
            });
        }

        let cond_nid = builder.emit(Node::Cond {
            pred: pred_ref,
            then_refs,
            else_refs,
        });

        // Reconstruct value from Cond outputs
        let slot_count = then_val.slot_count();
        let cond_refs: Vec<Ref> = (0..slot_count)
            .map(|i| Ref::output(cond_nid, i))
            .collect();
        let mut ref_iter = cond_refs.into_iter();

        // Reconstruct using then_val's shape (both branches have same shape)
        let shape = value_to_slot_shape(&then_val);
        current_else = Value::from_shape_and_refs(&shape, &mut ref_iter);
    }

    Ok(current_else)
}

/// Derive a SlotShape from a Value's structure.
fn value_to_slot_shape(val: &Value) -> SlotShape {
    match val {
        Value::Slot(_) => SlotShape::Tensor,
        Value::Record(fields) => SlotShape::Record(
            fields
                .iter()
                .map(|(n, v)| (n.clone(), value_to_slot_shape(v)))
                .collect(),
        ),
    }
}

// ---------------------------------------------------------------------------
// Chain (let bindings)
// ---------------------------------------------------------------------------

fn lower_chain<'a>(
    builder: &mut FuncBuilder,
    stmts: &[ast::Stmt],
    scope: &Scope<'a, Value>,
) -> fb::Result<Value> {
    debug_assert!(!stmts.is_empty(), "parser should prevent empty chains");

    // Process statements one by one, creating child scopes
    lower_chain_inner(builder, stmts, 0, scope)
}

fn lower_chain_inner<'a>(
    builder: &mut FuncBuilder,
    stmts: &[ast::Stmt],
    idx: usize,
    scope: &Scope<'a, Value>,
) -> fb::Result<Value> {
    if idx >= stmts.len() {
        return Err(fb::Outbox {
            messages: vec![fb::Message {
                title: "chain must end with a value expression, not a let binding".into(),
                span: Some(stmts.last().unwrap().span().clone()),
                notes: vec![],
            }],
        });
    }

    let stmt = &stmts[idx];
    let is_last = idx == stmts.len() - 1;

    match stmt {
        ast::Stmt::Let(inner) => {
            let stmt::Let { pattern, init, .. } = inner.as_ref();
            let init_val = lower_expr(builder, init, scope)?;

            // Bind pattern
            let mut new_bindings: HashMap<Symbol, Value> = HashMap::new();
            bind_pattern(pattern, init_val, &mut new_bindings);

            let child = Scope::child(scope, new_bindings);
            lower_chain_inner(builder, stmts, idx + 1, &child)
        }
        ast::Stmt::Discard(inner) => {
            let stmt::Discard { val, .. } = inner.as_ref();
            let result = lower_expr(builder, val, scope)?;
            if is_last {
                Ok(result)
            } else {
                // Discard and continue
                lower_chain_inner(builder, stmts, idx + 1, scope)
            }
        }
        ast::Stmt::Def(_) | ast::Stmt::TypeSig(_) => {
            // Skip nested defs/sigs in chain
            lower_chain_inner(builder, stmts, idx + 1, scope)
        }
    }
}

fn bind_pattern(pattern: &ast::Pattern, val: Value, bindings: &mut HashMap<Symbol, Value>) {
    match pattern {
        ast::Pattern::Name(inner) => {
            bindings.insert(inner.name.clone(), val);
        }
        ast::Pattern::Hole(_) => {
            // Discard
        }
        ast::Pattern::Literal(_) | ast::Pattern::Constructor(_) => {
            // Pattern matching on literals/constructors not supported in IR gen
        }
    }
}
