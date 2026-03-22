use hashbrown::{HashMap, HashSet};

use crate::ast::{self, expr, stmt};
use crate::ir::FuncId;
use crate::types::{ElemTy, Scheme, SlotShape, Ty, TyCtx, TyVar, TypeError};
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
    TypeDef { fields: Vec<(Symbol, SlotShape)> },
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
            Some(DeclBinding::TypeDef { fields }) => SlotShape::Record(fields.clone()),
            None => SlotShape::Tensor,
        }
    }
}

// ---------------------------------------------------------------------------
// Type environment for Algorithm W
// ---------------------------------------------------------------------------

struct TypeEnv {
    bindings: HashMap<Symbol, Scheme>,
}

impl TypeEnv {
    fn new() -> Self {
        TypeEnv { bindings: HashMap::new() }
    }

    fn insert(&mut self, name: Symbol, scheme: Scheme) {
        self.bindings.insert(name, scheme);
    }

    fn lookup(&self, name: &Symbol) -> Option<&Scheme> {
        self.bindings.get(name)
    }

    fn free_vars(&self) -> HashSet<TyVar> {
        self.bindings.values().flat_map(|s| s.free_vars()).collect()
    }
}

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
                    collect_type_def(&mut top, name, body);
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
    let sorted = topo_sort(&func_defs);

    // --- Pass 3: Algorithm W on each function ---
    let mut errors = vec![];
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

fn topo_sort(funcs: &[FuncDef]) -> Vec<usize> {
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
    while let Some(i) = queue.pop() {
        sorted.push(i);
        // i is a callee; its callers can now have their in-degree reduced
        for &caller in &rev_deps[i] {
            in_degree[caller] -= 1;
            if in_degree[caller] == 0 {
                queue.push(caller);
            }
        }
    }
    // Cycles: append remaining
    for i in 0..funcs.len() {
        if !sorted.contains(&i) {
            sorted.push(i);
        }
    }
    sorted
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
            if let Some(e) = &inner.else_branch {
                collect_call_deps(e, names, out);
            }
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
    // If explicit type sig, use it directly
    if let Some(sig) = &func.sig {
        return Ok(sig.clone());
    }

    let mut ctx = TyCtx::new();
    let mut env = TypeEnv::new();

    // Add top-level bindings to env
    for (name, binding) in &top.bindings {
        match binding {
            DeclBinding::Func { scheme, .. } | DeclBinding::Builtin { scheme, .. } => {
                env.insert(name.clone(), scheme.clone());
            }
            _ => {}
        }
    }

    // Assign fresh type vars to parameters
    let mut param_tys = vec![];
    for (arg_name, _) in func.args {
        let tv = ctx.fresh_var();
        env.insert(arg_name.clone(), Scheme::mono(tv.clone()));
        param_tys.push(tv);
    }

    // Infer body type
    let body_ty = infer_expr(&mut ctx, &mut env, func.body, top)?;

    // Build function type, resolve, generalize
    let fn_ty = Ty::Fn { params: param_tys, ret: Box::new(body_ty) };
    let resolved = ctx.resolve(&fn_ty);
    let env_fv = env.free_vars();
    Ok(ctx.generalize(&resolved, &env_fv))
}

fn infer_expr(
    ctx: &mut TyCtx,
    env: &mut TypeEnv,
    expr: &ast::Expr,
    top: &TopLevel,
) -> Result<Ty, TypeError> {
    match expr {
        ast::Expr::Literal(inner) => match &inner.val {
            crate::vocab::Literal::Number(n) => {
                if n.is_integer() {
                    Ok(ctx.fresh_var()) // polymorphic int literal
                } else {
                    Ok(Ty::scalar(ElemTy::F32))
                }
            }
            crate::vocab::Literal::Bool(_) => Ok(Ty::scalar(ElemTy::Bool)),
            crate::vocab::Literal::String(_) => Ok(ctx.fresh_var()),
        },

        ast::Expr::Name(inner) => {
            if let Some(scheme) = env.lookup(&inner.name) {
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

                // Binary operators
                if is_binop(text.as_str()) && args.len() == 2 {
                    let lhs = infer_expr(ctx, env, &args[0], top)?;
                    let rhs = infer_expr(ctx, env, &args[1], top)?;
                    ctx.unify(&lhs, &rhs)?;
                    return match text.as_str() {
                        "==(_,_)" | "!=(_,_)" | "<(_,_)" | ">(_,_)" | "<=(_,_)"
                        | ">=(_,_)" | "||(_,_)" | "&&(_,_)" => Ok(Ty::scalar(ElemTy::Bool)),
                        _ => Ok(ctx.resolve(&lhs)),
                    };
                }

                // grad — special form: (grad f) returns a function with
                // the same params as f but returning the type of f's first param
                if text.as_str() == "grad" && args.len() == 1 {
                    let f_ty = infer_expr(ctx, env, &args[0], top)?;
                    let resolved = ctx.resolve(&f_ty);
                    return match resolved {
                        Ty::Fn { params, .. } if !params.is_empty() => {
                            let grad_ret = params[0].clone();
                            Ok(Ty::Fn { params, ret: Box::new(grad_ret) })
                        }
                        _ => {
                            // Can't determine function type — return fresh
                            Ok(ctx.fresh_var())
                        }
                    };
                }
            }

            // General application
            let callee_ty = infer_expr(ctx, env, callee, top)?;
            let mut arg_tys = vec![];
            for arg in args {
                arg_tys.push(infer_expr(ctx, env, arg, top)?);
            }
            let ret = ctx.fresh_var();
            let fn_ty = Ty::Fn { params: arg_tys, ret: Box::new(ret.clone()) };
            ctx.unify(&callee_ty, &fn_ty)?;
            Ok(ctx.resolve(&ret))
        }

        ast::Expr::Dot(inner) => {
            let base_ty = infer_expr(ctx, env, &inner.base, top)?;
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
                let cond_ty = infer_expr(ctx, env, cond, top)?;
                ctx.unify(&cond_ty, &Ty::scalar(ElemTy::Bool))?;
                let body_ty = infer_expr(ctx, env, body, top)?;
                ctx.unify(&result_ty, &body_ty)?;
            }
            if let Some(else_expr) = &inner.else_branch {
                let else_ty = infer_expr(ctx, env, else_expr, top)?;
                ctx.unify(&result_ty, &else_ty)?;
            }
            Ok(ctx.resolve(&result_ty))
        }

        ast::Expr::Chain(inner) => infer_chain(ctx, env, &inner.stmt_vec, top),

        ast::Expr::Tuple(inner) => {
            let mut fields = vec![];
            for (i, elem) in inner.elements.iter().enumerate() {
                let ty = infer_expr(ctx, env, elem, top)?;
                fields.push((Symbol::from(format!("{i}")), ty));
            }
            Ok(Ty::Record { fields })
        }

        ast::Expr::Ctor(_) | ast::Expr::Match(_) => Ok(ctx.fresh_var()),
    }
}

fn infer_chain(
    ctx: &mut TyCtx,
    env: &mut TypeEnv,
    stmts: &[ast::Stmt],
    top: &TopLevel,
) -> Result<Ty, TypeError> {
    let mut last_ty = ctx.fresh_var();
    for (i, stmt) in stmts.iter().enumerate() {
        let is_last = i == stmts.len() - 1;
        match stmt {
            ast::Stmt::Let(inner) => {
                let init_ty = infer_expr(ctx, env, &inner.init, top)?;
                bind_pattern_ty(env, &inner.pattern, &init_ty);
            }
            ast::Stmt::Discard(inner) => {
                let ty = infer_expr(ctx, env, &inner.val, top)?;
                if is_last {
                    last_ty = ty;
                }
            }
            _ => {}
        }
    }
    Ok(last_ty)
}

fn bind_pattern_ty(env: &mut TypeEnv, pat: &ast::Pattern, ty: &Ty) {
    if let ast::Pattern::Name(inner) = pat {
        env.insert(inner.name.clone(), Scheme::mono(ty.clone()));
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

fn collect_type_def(top: &mut TopLevel, name: &Symbol, body: &ast::Expr) {
    if let ast::Expr::Ctor(inner) = body {
        if let ast::Type::Record { fields } = &inner.ty {
            let slot_fields: Vec<(Symbol, SlotShape)> = fields
                .iter()
                .map(|(fname, fty_expr)| (fname.clone(), slot_shape_from_type_expr(fty_expr, top)))
                .collect();
            top.bindings.insert(name.clone(), DeclBinding::TypeDef { fields: slot_fields });
        }
    }
}

fn slot_shape_from_type_expr(expr: &ast::Expr, top: &TopLevel) -> SlotShape {
    match expr {
        ast::Expr::Name(inner) => {
            if let Some(DeclBinding::TypeDef { fields }) = top.bindings.get(&inner.name) {
                SlotShape::Record(fields.clone())
            } else {
                SlotShape::Tensor
            }
        }
        ast::Expr::Apply(inner) => {
            if let ast::Expr::Name(n) = &inner.callee {
                if let Some(DeclBinding::TypeDef { fields }) = top.bindings.get(&n.name) {
                    return SlotShape::Record(fields.clone());
                }
            }
            SlotShape::Tensor
        }
        ast::Expr::Ctor(inner) => slot_shape_from_ast_type(&inner.ty, top),
        _ => SlotShape::Tensor,
    }
}

fn slot_shape_from_ast_type(ty: &ast::Type, top: &TopLevel) -> SlotShape {
    match ty {
        ast::Type::Name { name } | ast::Type::Apply { name, .. } => {
            if let Some(DeclBinding::TypeDef { fields }) = top.bindings.get(name) {
                SlotShape::Record(fields.clone())
            } else {
                SlotShape::Tensor
            }
        }
        ast::Type::Record { fields } => SlotShape::Record(
            fields.iter().map(|(n, e)| (n.clone(), slot_shape_from_type_expr(e, top))).collect(),
        ),
        ast::Type::Enum { .. } => SlotShape::Tensor,
    }
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
            if let Some(DeclBinding::TypeDef { fields }) = top.bindings.get(&inner.name) {
                return ty_from_typedef_fields(fields, ctx);
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
                if let Some(DeclBinding::TypeDef { fields }) = top.bindings.get(&n.name) {
                    return Ty::Record {
                        fields: fields.iter().map(|(n, _)| (n.clone(), ctx.fresh_var())).collect(),
                    };
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
                        params.push(parse_type_expr(&app.args[0], top, ctx, vars));
                        current = &app.args[1];
                        continue;
                    }
                }
                break;
            }
            ast::Expr::Ctor(inner) => {
                if let ast::Type::Apply { name, args } = &inner.ty {
                    if name.text().as_str() == "->" && args.len() == 2 {
                        params.push(parse_type_expr(&args[0], top, ctx, vars));
                        current = &args[1];
                        continue;
                    }
                }
                break;
            }
            _ => break,
        }
    }
    let ret = parse_type_expr(current, top, ctx, vars);
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
            if let Some(DeclBinding::TypeDef { fields }) = top.bindings.get(name) {
                return ty_from_typedef_fields(fields, ctx);
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
                params.push(parse_type_expr(&args[0], top, ctx, vars));
                let mut current = &args[1];
                loop {
                    if let ast::Expr::Ctor(ci) = current {
                        if let ast::Type::Apply { name: n2, args: a2 } = &ci.ty {
                            if n2.text().as_str() == "->" && a2.len() == 2 {
                                params.push(parse_type_expr(&a2[0], top, ctx, vars));
                                current = &a2[1];
                                continue;
                            }
                        }
                    }
                    break;
                }
                let ret = parse_type_expr(current, top, ctx, vars);
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
            if let Some(DeclBinding::TypeDef { fields }) = top.bindings.get(name) {
                ty_from_typedef_fields(fields, ctx)
            } else {
                Ty::App {
                    name: name.clone(),
                    args: args.iter().map(|a| parse_type_expr(a, top, ctx, vars)).collect(),
                }
            }
        }
        ast::Type::Record { fields } => Ty::Record {
            fields: fields.iter().map(|(n, e)| (n.clone(), parse_type_expr(e, top, ctx, vars))).collect(),
        },
        ast::Type::Enum { .. } => ctx.fresh_var(),
    }
}

/// Build a Ty::Record from a TypeDef's slot shape fields, using fresh vars for leaf tensors.
fn ty_from_typedef_fields(fields: &[(Symbol, SlotShape)], ctx: &mut TyCtx) -> Ty {
    Ty::Record {
        fields: fields
            .iter()
            .map(|(n, shape)| (n.clone(), ty_from_slot_shape(shape, ctx)))
            .collect(),
    }
}

fn ty_from_slot_shape(shape: &SlotShape, ctx: &mut TyCtx) -> Ty {
    match shape {
        SlotShape::Tensor => ctx.fresh_var(),
        SlotShape::Record(fields) => Ty::Record {
            fields: fields
                .iter()
                .map(|(n, sub)| (n.clone(), ty_from_slot_shape(sub, ctx)))
                .collect(),
        },
    }
}

fn scalar_from_name(s: &str) -> Option<Ty> {
    Some(match s {
        "f32" => Ty::scalar(ElemTy::F32),
        "f64" => Ty::scalar(ElemTy::F64),
        "i32" => Ty::scalar(ElemTy::I32),
        "i64" => Ty::scalar(ElemTy::I64),
        "u32" => Ty::scalar(ElemTy::U32),
        "u64" => Ty::scalar(ElemTy::U64),
        "bool" => Ty::scalar(ElemTy::Bool),
        _ => return None,
    })
}

// ---------------------------------------------------------------------------
// Built-in function registration
// ---------------------------------------------------------------------------

fn register_builtins(top: &mut TopLevel) {
    // max :: a -> a -> a
    let v = TyVar(1000);
    let id = top.alloc_func_id();
    top.bindings.insert(Symbol::from("max"), DeclBinding::Func {
        id, scheme: Scheme { bound: vec![v], ty: Ty::Fn {
            params: vec![Ty::Var(v), Ty::Var(v)], ret: Box::new(Ty::Var(v)),
        }},
    });

    // matmul :: [m][k]T -> [k][n]T -> [m][n]T
    let (t, m, k, n) = (TyVar(1001), TyVar(1002), TyVar(1003), TyVar(1004));
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
    let (t, nv) = (TyVar(1005), TyVar(1006));
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

    // grad :: (a -> b) -> (a -> a)  — unary operator on functions
    let (a, b) = (TyVar(1007), TyVar(1008));
    let id = top.alloc_func_id();
    top.bindings.insert(Symbol::from("grad"), DeclBinding::Builtin {
        id, builtin: Builtin::Grad,
        scheme: Scheme { bound: vec![a, b], ty: Ty::Fn {
            params: vec![
                Ty::Fn { params: vec![Ty::Var(a)], ret: Box::new(Ty::Var(b)) },
            ],
            ret: Box::new(Ty::Fn { params: vec![Ty::Var(a)], ret: Box::new(Ty::Var(a)) }),
        }},
    });
}
