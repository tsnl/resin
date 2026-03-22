use hashbrown::{HashMap, HashSet};

use crate::ast::{self, expr, stmt};
use crate::ir::FuncId;
use crate::types::{Scheme, Scope, SlotShape, Ty, TyCtx, TyVar, TypeError};
use crate::vocab::ScalarType;
use crate::{Symbol, fb};

// ---------------------------------------------------------------------------
// Top-level declarations (typer output)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Builtin {
    Matmul,
    CrossEntropy,
    Grad,
}

#[derive(Debug, Clone)]
pub enum DeclBinding {
    TypeDef {
        params: Vec<Symbol>,
        param_vars: Vec<TyVar>,
        body_ty: Ty,
        fields: Vec<(Symbol, SlotShape)>,
    },
    Func { id: FuncId, scheme: Scheme },
    Builtin { id: FuncId, builtin: Builtin, scheme: Scheme },
}

pub struct TopLevel {
    pub bindings: HashMap<Symbol, DeclBinding>,
    next_func_id: u32,
}

impl TopLevel {
    fn alloc_func_id(&mut self) -> FuncId {
        let id = FuncId(self.next_func_id);
        self.next_func_id += 1;
        id
    }

    pub fn lookup(&self, name: &Symbol) -> Option<&DeclBinding> {
        self.bindings.get(name)
    }

    pub fn func_id(&self, name: &Symbol) -> Option<FuncId> {
        match self.bindings.get(name) {
            Some(DeclBinding::Func { id, .. } | DeclBinding::Builtin { id, .. }) => Some(*id),
            _ => None,
        }
    }

    pub fn func_scheme(&self, name: &Symbol) -> Option<&Scheme> {
        match self.bindings.get(name) {
            Some(DeclBinding::Func { scheme, .. } | DeclBinding::Builtin { scheme, .. }) => {
                Some(scheme)
            }
            _ => None,
        }
    }

    pub fn return_slots(&self, name: &Symbol) -> SlotShape {
        match self.bindings.get(name) {
            Some(DeclBinding::Func { scheme, .. } | DeclBinding::Builtin { scheme, .. }) => {
                let ctx = TyCtx::new();
                match &scheme.ty {
                    Ty::Fn { ret, .. } => SlotShape::from_ty(ret, &ctx),
                    other => SlotShape::from_ty(other, &ctx),
                }
            }
            Some(DeclBinding::TypeDef { fields, .. }) => SlotShape::Record(fields.clone()),
            None => SlotShape::Tensor,
        }
    }
}

// TypeEnv replaced by Scope<'a, Scheme> from types.rs

// ---------------------------------------------------------------------------
// Function definition (collected in pass 1)
// ---------------------------------------------------------------------------

struct FuncDef<'a> {
    name: Symbol,
    args: &'a [(Symbol, crate::Span)],
    body: &'a ast::Expr,
    sig: Option<Scheme>,
    id: FuncId,
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

pub fn check(file: &ast::File) -> fb::Result<TopLevel> {
    let mut top = TopLevel {
        bindings: HashMap::new(),
        next_func_id: 0,
    };
    register_builtins(&mut top);

    // --- Pass 1: collect type defs, sigs, function stubs ---
    let mut pending_sigs: HashMap<Symbol, &ast::Expr> = HashMap::new();
    let mut func_defs: Vec<FuncDef> = vec![];

    for stmt in &file.stmts {
        match stmt {
            ast::Stmt::TypeSig(inner) => {
                let stmt::TypeSig { name, sig, .. } = inner.as_ref();
                pending_sigs.insert(name.clone(), sig);
            }
            ast::Stmt::Def(inner) => {
                let stmt::Def { name, args, body, .. } = inner.as_ref();
                if name.is_upper_id() {
                    collect_type_def(&mut top, name, args, body);
                } else {
                    let sig = pending_sigs
                        .remove(name)
                        .map(|sig_expr| parse_type_sig(sig_expr, &top));
                    let id = top.alloc_func_id();
                    let stub = match &sig {
                        Some(s) => s.clone(),
                        None => {
                            let mut ctx = TyCtx::new();
                            let ps: Vec<Ty> = args.iter().map(|_| ctx.fresh_var()).collect();
                            let r = ctx.fresh_var();
                            let bound: Vec<TyVar> = (0..ctx.next_var).map(TyVar).collect();
                            Scheme { bound, ty: Ty::Fn { params: ps, ret: Box::new(r) } }
                        }
                    };
                    top.bindings.insert(name.clone(), DeclBinding::Func { id, scheme: stub });
                    func_defs.push(FuncDef { name: name.clone(), args, body, sig, id });
                }
            }
            _ => {}
        }
    }

    // --- Pass 2: topological sort ---
    let (sorted, in_cycle) = topo_sort(&func_defs);

    // --- Pass 2b: reject recursive functions without explicit type signatures ---
    let mut errors = vec![];
    for &idx in &in_cycle {
        let func = &func_defs[idx];
        if func.sig.is_none() {
            errors.push(fb::Message {
                title: format!(
                    "recursive function '{}' requires an explicit type signature (`::`); \
                     type inference cannot determine the type of recursive or \
                     mutually recursive functions without one",
                    func.name
                ),
                span: func.args.first().map(|(_, s)| s.clone()),
                notes: vec![],
            });
        }
    }
    if !errors.is_empty() {
        return Err(fb::Outbox { messages: errors });
    }

    // --- Pass 3: Algorithm W on each function ---
    for idx in sorted {
        let func = &func_defs[idx];
        match infer_function(&top, func) {
            Ok(scheme) => {
                if let Some(DeclBinding::Func { scheme: s, .. }) =
                    top.bindings.get_mut(&func.name)
                {
                    *s = scheme;
                }
            }
            Err(e) => errors.push(fb::Message {
                title: format!("in function '{}': {}", func.name, e.message),
                span: None,
                notes: vec![],
            }),
        }
    }

    if !errors.is_empty() {
        return Err(fb::Outbox { messages: errors });
    }
    Ok(top)
}

// ---------------------------------------------------------------------------
// Topological sort
// ---------------------------------------------------------------------------

/// Returns `(sorted_order, cycle_indices)`.
/// `cycle_indices` contains the indices of functions involved in dependency
/// cycles (self-recursion or mutual recursion). These are appended to the
/// sorted order so that inference still runs, but callers can check whether
/// explicit type signatures are present.
fn topo_sort(funcs: &[FuncDef]) -> (Vec<usize>, Vec<usize>) {
    let name_to_idx: HashMap<&Symbol, usize> =
        funcs.iter().enumerate().map(|(i, f)| (&f.name, i)).collect();

    let mut deps: Vec<HashSet<usize>> = vec![HashSet::new(); funcs.len()];
    for (i, func) in funcs.iter().enumerate() {
        collect_call_deps(func.body, &name_to_idx, &mut deps[i]);
    }

    // Kahn's — process callees before callers.
    // deps[i] = set of functions that i calls (i depends on them).
    // We want callees first, so compute in-degree from reversed edges:
    // in_degree[i] = number of functions i calls (i depends on).
    let mut in_degree = vec![0usize; funcs.len()];
    for i in 0..funcs.len() {
        in_degree[i] = deps[i].len();
    }
    // Build reverse edges: rev_deps[j] = set of functions that call j
    let mut rev_deps: Vec<Vec<usize>> = vec![vec![]; funcs.len()];
    for (i, d) in deps.iter().enumerate() {
        for &j in d {
            rev_deps[j].push(i);
        }
    }

    let mut queue: Vec<usize> = (0..funcs.len()).filter(|i| in_degree[*i] == 0).collect();
    let mut sorted = vec![];
    let mut in_sorted: HashSet<usize> = HashSet::new();
    while let Some(i) = queue.pop() {
        sorted.push(i);
        in_sorted.insert(i);
        for &caller in &rev_deps[i] {
            in_degree[caller] -= 1;
            if in_degree[caller] == 0 {
                queue.push(caller);
            }
        }
    }
    // Functions still with in_degree > 0 are in cycles.
    let cycle_indices: Vec<usize> = (0..funcs.len())
        .filter(|i| !in_sorted.contains(i))
        .collect();
    sorted.extend(&cycle_indices);
    (sorted, cycle_indices)
}

fn collect_call_deps(expr: &ast::Expr, names: &HashMap<&Symbol, usize>, out: &mut HashSet<usize>) {
    match expr {
        ast::Expr::Apply(inner) => {
            if let ast::Expr::Name(n) = &inner.callee {
                if let Some(&idx) = names.get(&n.name) {
                    out.insert(idx);
                }
            }
            collect_call_deps(&inner.callee, names, out);
            for a in &inner.args {
                collect_call_deps(a, names, out);
            }
        }
        ast::Expr::If(inner) => {
            for (c, b) in &inner.cond_branch_vec {
                collect_call_deps(c, names, out);
                collect_call_deps(b, names, out);
            }
            collect_call_deps(&inner.else_branch, names, out);
        }
        ast::Expr::Chain(inner) => {
            for s in &inner.stmt_vec {
                match s {
                    ast::Stmt::Let(l) => collect_call_deps(&l.init, names, out),
                    ast::Stmt::Discard(d) => collect_call_deps(&d.val, names, out),
                    _ => {}
                }
            }
        }
        ast::Expr::Dot(inner) => collect_call_deps(&inner.base, names, out),
        ast::Expr::Grad(inner) => collect_call_deps(&inner.func, names, out),
        ast::Expr::As(inner) => collect_call_deps(&inner.expr, names, out),
        ast::Expr::Name(inner) => {
            if let Some(&idx) = names.get(&inner.name) {
                out.insert(idx);
            }
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Algorithm W
// ---------------------------------------------------------------------------

fn infer_function(top: &TopLevel, func: &FuncDef) -> Result<Scheme, TypeError> {
    let mut ctx = TyCtx::new();

    // Build root scope with top-level bindings
    let mut root_bindings: HashMap<Symbol, Scheme> = HashMap::new();
    for (name, binding) in &top.bindings {
        match binding {
            DeclBinding::Func { scheme, .. } | DeclBinding::Builtin { scheme, .. } => {
                root_bindings.insert(name.clone(), scheme.clone());
            }
            _ => {}
        }
    }

    // Add parameter bindings
    let mut param_tys = vec![];
    for (arg_name, _) in func.args {
        let tv = ctx.fresh_var();
        root_bindings.insert(arg_name.clone(), Scheme::mono(tv.clone()));
        param_tys.push(tv);
    }

    // If explicit sig provided, unify param types BEFORE body inference
    // so that record field accesses can resolve during inference.
    let sig_inst = func.sig.as_ref().map(|s| ctx.instantiate(s));
    if let Some(Ty::Fn { params: ref sig_params, .. }) = sig_inst {
        for (pt, sp) in param_tys.iter().zip(sig_params.iter()) {
            ctx.unify(pt, sp)?;
        }
    }

    let scope = Scope::root(root_bindings);

    // Infer body type
    let body_ty = infer_expr(&mut ctx, &scope, func.body, top)?;

    // Build inferred function type
    let fn_ty = Ty::Fn { params: param_tys, ret: Box::new(body_ty) };

    // Full reconciliation with sig (checks return type too)
    if let Some(sig_ty) = &sig_inst {
        ctx.unify(&fn_ty, sig_ty)?;
    }

    let resolved = ctx.resolve(&fn_ty);
    let env_fv = scope.free_vars();
    Ok(ctx.generalize(&resolved, &env_fv))
}

fn infer_expr<'a>(
    ctx: &mut TyCtx,
    scope: &Scope<'a, Scheme>,
    expr: &ast::Expr,
    top: &TopLevel,
) -> Result<Ty, TypeError> {
    match expr {
        ast::Expr::Literal(inner) => match &inner.val {
            crate::vocab::Literal::Number(n) => {
                if n.is_integer() {
                    Ok(ctx.fresh_var()) // polymorphic int literal
                } else {
                    Ok(Ty::scalar(ScalarType::F32))
                }
            }
            crate::vocab::Literal::Bool(_) => Ok(Ty::scalar(ScalarType::Bool)),
            crate::vocab::Literal::String(_) => Ok(ctx.fresh_var()),
        },

        ast::Expr::Name(inner) => {
            if let Some(scheme) = scope.lookup(&inner.name) {
                Ok(ctx.instantiate(scheme))
            } else if let Some(scheme) = top.func_scheme(&inner.name) {
                Ok(ctx.instantiate(scheme))
            } else {
                Ok(ctx.fresh_var())
            }
        }

        ast::Expr::Apply(inner) => {
            let expr::Apply { callee, args, .. } = inner.as_ref();

            if let ast::Expr::Name(n) = callee {
                let text = n.name.text();

                // Unary negation
                if text.as_str() == "~(_)" && args.len() == 1 {
                    return infer_expr(ctx, scope, &args[0], top);
                }

                // Binary operators
                if is_binop(text.as_str()) && args.len() == 2 {
                    let lhs = infer_expr(ctx, scope, &args[0], top)?;
                    let rhs = infer_expr(ctx, scope, &args[1], top)?;
                    let result_ty = unify_broadcast(ctx, &lhs, &rhs)?;
                    return match text.as_str() {
                        "==(_,_)" | "!=(_,_)" | "<(_,_)" | ">(_,_)" | "<=(_,_)"
                        | ">=(_,_)" | "||(_,_)" | "&&(_,_)" => Ok(Ty::scalar(ScalarType::Bool)),
                        _ => Ok(result_ty),
                    };
                }

            }

            // General application
            let callee_ty = infer_expr(ctx, scope, callee, top)?;
            let mut arg_tys = vec![];
            for arg in args {
                arg_tys.push(infer_expr(ctx, scope, arg, top)?);
            }
            let ret = ctx.fresh_var();
            let fn_ty = Ty::Fn { params: arg_tys, ret: Box::new(ret.clone()) };
            ctx.unify(&callee_ty, &fn_ty)?;
            Ok(ctx.resolve(&ret))
        }

        ast::Expr::Dot(inner) => {
            let base_ty = infer_expr(ctx, scope, &inner.base, top)?;
            let resolved = ctx.resolve(&base_ty);
            match &resolved {
                Ty::Record { fields } => {
                    for (fname, fty) in fields {
                        if fname == &inner.field {
                            return Ok(fty.clone());
                        }
                    }
                    Err(TypeError::new(format!("no field '{}' on record", inner.field)))
                }
                _ => Err(TypeError::new(format!(
                    "field access '.{}' requires a known record type — add a type signature",
                    inner.field
                ))),
            }
        }

        ast::Expr::If(inner) => {
            let result_ty = ctx.fresh_var();
            for (cond, body) in &inner.cond_branch_vec {
                let cond_ty = infer_expr(ctx, scope, cond, top)?;
                ctx.unify(&cond_ty, &Ty::scalar(ScalarType::Bool))?;
                let body_ty = infer_expr(ctx, scope, body, top)?;
                ctx.unify(&result_ty, &body_ty)?;
            }
            let else_ty = infer_expr(ctx, scope, &inner.else_branch, top)?;
            ctx.unify(&result_ty, &else_ty)?;
            Ok(ctx.resolve(&result_ty))
        }

        ast::Expr::Chain(inner) => infer_chain(ctx, scope, &inner.stmt_vec, top),

        ast::Expr::Tuple(inner) => {
            let mut fields = vec![];
            for (i, elem) in inner.elements.iter().enumerate() {
                let ty = infer_expr(ctx, scope, elem, top)?;
                fields.push((Symbol::from(format!("{i}")), ty));
            }
            Ok(Ty::Record { fields })
        }

        ast::Expr::Grad(inner) => {
            let f_ty = infer_expr(ctx, scope, &inner.func, top)?;
            let resolved = ctx.resolve(&f_ty);
            match resolved {
                Ty::Fn { params, .. } if !params.is_empty() => {
                    let grad_ret = params[0].clone();
                    Ok(Ty::Fn { params, ret: Box::new(grad_ret) })
                }
                _ => Ok(ctx.fresh_var()),
            }
        }

        ast::Expr::As(inner) => {
            let source_ty = infer_expr(ctx, scope, &inner.expr, top)?;
            // Parse target as a type expression with fresh vars for holes
            let mut type_vars: HashMap<Symbol, TyVar> = HashMap::new();
            let target_ty = as_value_type(parse_type_expr(&inner.target, top, ctx, &mut type_vars));
            // Unify element types for compatibility
            let rs = ctx.resolve(&source_ty);
            let rt = ctx.resolve(&target_ty);
            if let (Ty::Tensor { elem: se, .. }, Ty::Tensor { elem: te, .. }) = (&rs, &rt) {
                ctx.unify(se, te)?;
            }
            Ok(ctx.resolve(&target_ty))
        }

        ast::Expr::Ctor(_) | ast::Expr::Match(_) => Ok(ctx.fresh_var()),
    }
}

fn infer_chain<'a>(
    ctx: &mut TyCtx,
    scope: &Scope<'a, Scheme>,
    stmts: &[ast::Stmt],
    top: &TopLevel,
) -> Result<Ty, TypeError> {
    infer_chain_inner(ctx, scope, stmts, 0, top)
}

fn infer_chain_inner<'a>(
    ctx: &mut TyCtx,
    scope: &Scope<'a, Scheme>,
    stmts: &[ast::Stmt],
    idx: usize,
    top: &TopLevel,
) -> Result<Ty, TypeError> {
    if idx >= stmts.len() {
        return Ok(ctx.fresh_var());
    }
    let is_last = idx == stmts.len() - 1;
    match &stmts[idx] {
        ast::Stmt::Let(inner) => {
            let init_ty = infer_expr(ctx, scope, &inner.init, top)?;
            let mut new_bindings = HashMap::new();
            bind_pattern_ty(&inner.pattern, &init_ty, &mut new_bindings);
            let child = Scope::child(scope, new_bindings);
            infer_chain_inner(ctx, &child, stmts, idx + 1, top)
        }
        ast::Stmt::Discard(inner) => {
            let ty = infer_expr(ctx, scope, &inner.val, top)?;
            if is_last {
                Ok(ty)
            } else {
                infer_chain_inner(ctx, scope, stmts, idx + 1, top)
            }
        }
        _ => infer_chain_inner(ctx, scope, stmts, idx + 1, top),
    }
}

/// Unify two types with record broadcasting support.
/// - Record op Record: field-by-field unification
/// - Scalar/Tensor op Record: broadcast into each field
/// - Otherwise: standard unification
fn unify_broadcast(ctx: &mut TyCtx, a: &Ty, b: &Ty) -> Result<Ty, TypeError> {
    let ra = ctx.resolve(a);
    let rb = ctx.resolve(b);
    match (&ra, &rb) {
        (Ty::Record { fields: fa }, Ty::Record { fields: fb }) => {
            if fa.len() != fb.len() {
                return Err(TypeError::new("record field count mismatch in binary op"));
            }
            let mut result_fields = vec![];
            for ((na, ta), (_, tb)) in fa.iter().zip(fb.iter()) {
                let ft = unify_broadcast(ctx, ta, tb)?;
                result_fields.push((na.clone(), ft));
            }
            Ok(Ty::Record { fields: result_fields })
        }
        (_, Ty::Record { fields }) => {
            let mut result_fields = vec![];
            for (n, ft) in fields {
                let rt = unify_broadcast(ctx, &ra, ft)?;
                result_fields.push((n.clone(), rt));
            }
            Ok(Ty::Record { fields: result_fields })
        }
        (Ty::Record { fields }, _) => {
            let mut result_fields = vec![];
            for (n, ft) in fields {
                let rt = unify_broadcast(ctx, ft, &rb)?;
                result_fields.push((n.clone(), rt));
            }
            Ok(Ty::Record { fields: result_fields })
        }
        // Tensor broadcasting: scalar (0-dim) broadcasts to any rank
        (Ty::Tensor { elem: ea, dims: da }, Ty::Tensor { elem: eb, dims: db }) => {
            ctx.unify(ea, eb)?;
            if da.is_empty() {
                Ok(ctx.resolve(&rb))
            } else if db.is_empty() {
                Ok(ctx.resolve(&ra))
            } else {
                ctx.unify(&ra, &rb)?;
                Ok(ctx.resolve(&ra))
            }
        }
        _ => {
            ctx.unify(&ra, &rb)?;
            Ok(ctx.resolve(&ra))
        }
    }
}

fn bind_pattern_ty(pat: &ast::Pattern, ty: &Ty, bindings: &mut HashMap<Symbol, Scheme>) {
    if let ast::Pattern::Name(inner) = pat {
        bindings.insert(inner.name.clone(), Scheme::mono(ty.clone()));
    }
}


fn is_binop(s: &str) -> bool {
    matches!(s, "+(_,_)" | "-(_,_)" | "*(_,_)" | "/(_,_)" | "//(_,_)" | "%(_,_)"
        | "==(_,_)" | "!=(_,_)" | "<(_,_)" | ">(_,_)" | "<=(_,_)" | ">=(_,_)"
        | "||(_,_)" | "&&(_,_)")
}

// ---------------------------------------------------------------------------
// Type definition collection
// ---------------------------------------------------------------------------

fn collect_type_def(
    top: &mut TopLevel,
    name: &Symbol,
    args: &[(Symbol, crate::Span)],
    body: &ast::Expr,
) {
    let mut ctx = TyCtx::new();
    let mut vars: HashMap<Symbol, TyVar> = HashMap::new();
    let mut param_vars = vec![];
    let params: Vec<Symbol> = args.iter().map(|(n, _)| n.clone()).collect();

    // Create TyVars for each type parameter
    for param_name in &params {
        let Ty::Var(v) = ctx.fresh_var() else { unreachable!() };
        vars.insert(param_name.clone(), v);
        param_vars.push(v);
    }

    // Parse the body into a Ty using these param vars
    let body_ty = parse_type_expr(body, top, &mut ctx, &mut vars);

    // Compute SlotShape for IR gen (structural shape, params become Tensor leaves)
    let fields = match &body_ty {
        Ty::Record { fields } => fields
            .iter()
            .map(|(n, ty)| (n.clone(), SlotShape::from_ty(ty, &ctx)))
            .collect(),
        _ => vec![],
    };

    top.bindings.insert(
        name.clone(),
        DeclBinding::TypeDef { params, param_vars, body_ty, fields },
    );
}


// ---------------------------------------------------------------------------
// Type signature parsing
// ---------------------------------------------------------------------------

fn parse_type_sig(sig_expr: &ast::Expr, top: &TopLevel) -> Scheme {
    let mut ctx = TyCtx::new();
    let mut vars: HashMap<Symbol, TyVar> = HashMap::new();
    let ty = parse_type_expr(sig_expr, top, &mut ctx, &mut vars);
    Scheme { bound: vars.values().copied().collect(), ty }
}

fn parse_type_expr(
    expr: &ast::Expr,
    top: &TopLevel,
    ctx: &mut TyCtx,
    vars: &mut HashMap<Symbol, TyVar>,
) -> Ty {
    match expr {
        ast::Expr::Name(inner) => {
            let text = inner.name.text();
            if let Some(ty) = scalar_from_name(text.as_str()) { return ty; }
            if let Some(DeclBinding::TypeDef { param_vars, body_ty, .. }) = top.bindings.get(&inner.name) {
                let (param_vars, body_ty) = (param_vars.clone(), body_ty.clone());
                return instantiate_typedef(&param_vars, &body_ty, &[], ctx);
            }
            if let Some(v) = vars.get(&inner.name) {
                Ty::Var(*v)
            } else {
                let Ty::Var(v) = ctx.fresh_var() else { unreachable!() };
                vars.insert(inner.name.clone(), v);
                Ty::Var(v)
            }
        }

        ast::Expr::Apply(inner) => {
            let expr::Apply { callee, args, .. } = inner.as_ref();
            if let ast::Expr::Name(n) = callee {
                if n.name.text() == "->" && args.len() == 2 {
                    return parse_arrow_chain(expr, top, ctx, vars);
                }
                if let Some(DeclBinding::TypeDef { param_vars, body_ty, .. }) = top.bindings.get(&n.name) {
                    let (param_vars, body_ty) = (param_vars.clone(), body_ty.clone());
                    let arg_tys: Vec<Ty> = args.iter()
                        .map(|a| parse_type_expr(a, top, ctx, vars))
                        .collect();
                    return instantiate_typedef(&param_vars, &body_ty, &arg_tys, ctx);
                }
            }
            // Array type or other
            Ty::Tensor {
                elem: Box::new(parse_type_expr(args.last().unwrap_or(callee), top, ctx, vars)),
                dims: if args.len() > 1 {
                    args[..args.len() - 1].iter().map(|a| parse_type_expr(a, top, ctx, vars)).collect()
                } else {
                    vec![ctx.fresh_var()]
                },
            }
        }

        ast::Expr::Ctor(inner) => parse_ast_type(&inner.ty, top, ctx, vars),
        _ => ctx.fresh_var(),
    }
}

fn parse_arrow_chain(
    expr: &ast::Expr,
    top: &TopLevel,
    ctx: &mut TyCtx,
    vars: &mut HashMap<Symbol, TyVar>,
) -> Ty {
    let mut params = vec![];
    let mut current = expr;
    loop {
        match current {
            ast::Expr::Apply(app) => {
                if let ast::Expr::Name(n) = &app.callee {
                    if n.name.text().as_str() == "->" && app.args.len() == 2 {
                        params.push(as_value_type(parse_type_expr(&app.args[0], top, ctx, vars)));
                        current = &app.args[1];
                        continue;
                    }
                }
                break;
            }
            ast::Expr::Ctor(inner) => {
                if let ast::Type::Apply { name, args } = &inner.ty {
                    if name.text().as_str() == "->" && args.len() == 2 {
                        params.push(as_value_type(parse_type_expr(&args[0], top, ctx, vars)));
                        current = &args[1];
                        continue;
                    }
                }
                break;
            }
            _ => break,
        }
    }
    let ret = as_value_type(parse_type_expr(current, top, ctx, vars));
    Ty::Fn { params, ret: Box::new(ret) }
}

fn parse_ast_type(
    ty: &ast::Type,
    top: &TopLevel,
    ctx: &mut TyCtx,
    vars: &mut HashMap<Symbol, TyVar>,
) -> Ty {
    match ty {
        ast::Type::Name { name } => {
            if let Some(t) = scalar_from_name(name.text().as_str()) { return t; }
            if let Some(DeclBinding::TypeDef { param_vars, body_ty, .. }) = top.bindings.get(name) {
                let (param_vars, body_ty) = (param_vars.clone(), body_ty.clone());
                return instantiate_typedef(&param_vars, &body_ty, &[], ctx);
            }
            if let Some(v) = vars.get(name) {
                Ty::Var(*v)
            } else {
                let Ty::Var(v) = ctx.fresh_var() else { unreachable!() };
                vars.insert(name.clone(), v);
                Ty::Var(v)
            }
        }
        ast::Type::Apply { name, args } => {
            let text = name.text();
            if text == "->" && args.len() == 2 {
                let mut params = vec![];
                params.push(as_value_type(parse_type_expr(&args[0], top, ctx, vars)));
                let mut current = &args[1];
                loop {
                    if let ast::Expr::Ctor(ci) = current {
                        if let ast::Type::Apply { name: n2, args: a2 } = &ci.ty {
                            if n2.text().as_str() == "->" && a2.len() == 2 {
                                params.push(as_value_type(parse_type_expr(&a2[0], top, ctx, vars)));
                                current = &a2[1];
                                continue;
                            }
                        }
                    }
                    break;
                }
                let ret = as_value_type(parse_type_expr(current, top, ctx, vars));
                return Ty::Fn { params, ret: Box::new(ret) };
            }
            if text == "[]" {
                if args.len() >= 2 {
                    let dim = parse_type_expr(&args[0], top, ctx, vars);
                    let elem = parse_type_expr(&args[1], top, ctx, vars);
                    return match elem {
                        Ty::Tensor { elem: ie, mut dims } => {
                            dims.insert(0, dim);
                            Ty::Tensor { elem: ie, dims }
                        }
                        other => Ty::Tensor { elem: Box::new(other), dims: vec![dim] },
                    };
                }
                return Ty::Tensor {
                    elem: Box::new(args.first().map(|a| parse_type_expr(a, top, ctx, vars)).unwrap_or_else(|| ctx.fresh_var())),
                    dims: vec![ctx.fresh_var()],
                };
            }
            if let Some(DeclBinding::TypeDef { param_vars, body_ty, .. }) = top.bindings.get(name) {
                let (param_vars, body_ty) = (param_vars.clone(), body_ty.clone());
                let arg_tys: Vec<Ty> = args.iter()
                    .map(|a| parse_type_expr(a, top, ctx, vars))
                    .collect();
                instantiate_typedef(&param_vars, &body_ty, &arg_tys, ctx)
            } else {
                Ty::App {
                    name: name.clone(),
                    args: args.iter().map(|a| parse_type_expr(a, top, ctx, vars)).collect(),
                }
            }
        }
        ast::Type::Record { fields } => Ty::Record {
            fields: fields.iter().map(|(n, e)| (n.clone(), as_value_type(parse_type_expr(e, top, ctx, vars)))).collect(),
        },
        ast::Type::Enum { .. } => ctx.fresh_var(),
    }
}

/// Wrap an element-level type as a value type.
/// Uid type variables and bare scalars become 0-dim tensors.
/// Already-wrapped types (Tensor, Record, Fn) pass through.
fn as_value_type(ty: Ty) -> Ty {
    match &ty {
        Ty::Var(_) | Ty::Scalar(_) => Ty::Tensor { elem: Box::new(ty), dims: vec![] },
        _ => ty,
    }
}

/// Instantiate a TypeDef's body_ty by substituting type args for param_vars.
/// If fewer args than params, remaining params get fresh vars.
fn instantiate_typedef(
    param_vars: &[TyVar],
    body_ty: &Ty,
    arg_tys: &[Ty],
    ctx: &mut TyCtx,
) -> Ty {
    use crate::types::Substitution;
    let mut map = HashMap::new();
    for (i, &pv) in param_vars.iter().enumerate() {
        if i < arg_tys.len() {
            map.insert(pv, arg_tys[i].clone());
        } else {
            map.insert(pv, ctx.fresh_var());
        }
    }
    let subst = Substitution::from_map(map);
    subst.apply(body_ty)
}

fn scalar_from_name(s: &str) -> Option<Ty> {
    Some(match s {
        "f32" => Ty::scalar(ScalarType::F32),
        "f64" => Ty::scalar(ScalarType::F64),
        "i32" => Ty::scalar(ScalarType::I32),
        "i64" => Ty::scalar(ScalarType::I64),
        "u32" => Ty::scalar(ScalarType::U32),
        "u64" => Ty::scalar(ScalarType::U64),
        "bool" => Ty::scalar(ScalarType::Bool),
        _ => return None,
    })
}

// ---------------------------------------------------------------------------
// Built-in function registration
// ---------------------------------------------------------------------------

fn register_builtins(top: &mut TopLevel) {
    let mut ctx = TyCtx::new();

    macro_rules! fresh {
        () => {{ let Ty::Var(v) = ctx.fresh_var() else { unreachable!() }; v }};
    }

    // max :: a -> a -> a
    let v = fresh!();
    let id = top.alloc_func_id();
    top.bindings.insert(Symbol::from("max"), DeclBinding::Func {
        id, scheme: Scheme { bound: vec![v], ty: Ty::Fn {
            params: vec![Ty::Var(v), Ty::Var(v)], ret: Box::new(Ty::Var(v)),
        }},
    });

    // matmul :: [m][k]T -> [k][n]T -> [m][n]T
    let (t, m, k, n) = (fresh!(), fresh!(), fresh!(), fresh!());
    let id = top.alloc_func_id();
    top.bindings.insert(Symbol::from("matmul"), DeclBinding::Builtin {
        id, builtin: Builtin::Matmul,
        scheme: Scheme { bound: vec![t, m, k, n], ty: Ty::Fn {
            params: vec![
                Ty::Tensor { elem: Box::new(Ty::Var(t)), dims: vec![Ty::Var(m), Ty::Var(k)] },
                Ty::Tensor { elem: Box::new(Ty::Var(t)), dims: vec![Ty::Var(k), Ty::Var(n)] },
            ],
            ret: Box::new(Ty::Tensor { elem: Box::new(Ty::Var(t)), dims: vec![Ty::Var(m), Ty::Var(n)] }),
        }},
    });

    // cross_entropy :: [n]T -> [n]T -> T
    let (t, nv) = (fresh!(), fresh!());
    let id = top.alloc_func_id();
    top.bindings.insert(Symbol::from("cross_entropy"), DeclBinding::Builtin {
        id, builtin: Builtin::CrossEntropy,
        scheme: Scheme { bound: vec![t, nv], ty: Ty::Fn {
            params: vec![
                Ty::Tensor { elem: Box::new(Ty::Var(t)), dims: vec![Ty::Var(nv)] },
                Ty::Tensor { elem: Box::new(Ty::Var(t)), dims: vec![Ty::Var(nv)] },
            ],
            ret: Box::new(Ty::Tensor { elem: Box::new(Ty::Var(t)), dims: vec![] }),
        }},
    });

    // grad is now a keyword/special form, not a builtin function
}
