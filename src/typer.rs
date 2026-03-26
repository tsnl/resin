//! # Resin Type Checker
//!
//! ## Reading order
//!
//! 1. `src/types.rs` — Data structures: `Type`, `Value`, `Scheme`,
//!    `Substitution`, `TyCtx`, `Scope`.
//! 2. `src/interp.rs` — Comptime interpreter: `Env`, `FnRegistry`,
//!    `interpret`, builtins (`concat`, `head`, etc.).
//! 3. This file (`src/typer.rs`):
//!    - `builtin_scope` (bottom) — builtins available before user code
//!    - `typecheck_file` (top) — entry point, dependency ordering, SCC loop
//!    - `typecheck_def` — type-checking a single definition
//!    - `elaborate_signature` / `elaborate_type` — AST type exprs → `Type`
//!    - `infer` — core HM inference over value expressions
//!    - `build_dep_graph` / `tarjan_scc` — dependency analysis
//!
//! ## Design
//!
//! Three principles:
//!
//! 1. **`Type` only contains normal forms.** No unevaluated function calls.
//!    To construct a `Type`, any comptime calls must be evaluated first.
//!
//! 2. **A full interpreter participates in type-checking.** Functions are
//!    type-checked, then become callable at compile time for subsequent
//!    type-checking.
//!
//! 3. **Type-checking and compilation are interleaved.** For each function
//!    (in dependency order): type-check → register in scope → register in
//!    `FnRegistry`.
//!
//! ## Architecture
//!
//! ```text
//! For each function, in dependency order:
//!
//!   1. TYPE-CHECK  ─── HM inference on the body.
//!   │                  Binds type variables.
//!   │
//!   2. EVALUATE    ─── For any comptime calls in the return type:
//!   │                  substitute bound values, run the interpreter.
//!   │                  Panic → type error.
//!   │
//!   3. REGISTER    ─── Add to scope and FnRegistry.
//!   │
//!   └── This function is now available for comptime use by later functions.
//! ```
//!
//! ## Immutability
//!
//! All data structures are immutable:
//! - `Scope` / `Env`: persistent parent-chain (extend returns new child, O(1))
//! - `Substitution`: immutable cons list (bind prepends, O(1))
//! - `TyCtx`: threaded by value through inference (fresh_var, unify return new ctx)
//! - `FnRegistry`: register returns a new registry
//! - Pipeline: fold over SCCs, threading `(scope, fn_registry)`
//!
//! ## Elaboration and Interpretation
//!
//! Two evaluation functions with a one-way dependency:
//!
//! - `elaborate_type`: AST type expressions → `Type`. Handles `Ten` (→ Tensor),
//!   uppercase names (→ TyCon), type variables, literals, and comptime function
//!   calls (delegates to `interpret` when args are concrete).
//!
//! - `interpret` (in interp.rs): AST value expressions → `Value`. Handles
//!   arithmetic, if/else, let, array construction, function calls. Pure comptime.
//!
//! `elaborate_type` calls `interpret` when it encounters a comptime call.
//! `interpret` never calls `elaborate_type`. One-way dependency.
//!
//! ## Unification
//!
//! Standard Hindley-Milner, no extensions:
//!
//! ```text
//! Var(v) ~ t              → bind v = t (with occurs check)
//! Val(a) ~ Val(b)         → ok if a == b
//! Scalar(a) ~ Scalar(b)   → ok if a == b
//! Tensor(e1,s1) ~ Tensor(e2,s2) → unify(e1,e2), unify(s1,s2)
//! Record(fs1) ~ Record(fs2)     → same fields, pairwise unify
//! Fn(p1,r1) ~ Fn(p2,r2)         → pairwise params, then ret
//! TyCon(f,as) ~ TyCon(f,bs)     → same name, pairwise args
//! otherwise                      → type error
//! ```
//!
//! ## Dependency Ordering
//!
//! Definitions are processed using Tarjan's SCC algorithm:
//! 1. Build dependency graph (scan bodies + sigs for name references)
//! 2. Find SCCs in reverse topological order (callees before callers)
//! 3. Non-recursive SCCs: type-check, then register
//! 4. Recursive SCCs: add placeholder schemes, type-check all, then register
//!
//! ## TyCon Expansion
//!
//! Type constructors (e.g., `Linear T o i = { w: Ten T [o,i], b: Ten T [o] }`)
//! are opaque during unification. Dot access (`model.w`) triggers expansion:
//! look up the TyConDef via `Scope::lookup_tycon`, substitute the concrete
//! args for params, elaborate the body to get a Record type, then look up
//! the field.

use std::sync::Arc;

use hashbrown::{HashMap, HashSet};

use crate::Symbol;
use crate::ast::{self, Expr, Pattern, Stmt, TypeSpec};
use crate::interp::{self, Env, FnRegistry, CompiledFn, PanicError};
use crate::types::*;
use crate::vocab::ScalarType;

// ---------------------------------------------------------------------------
// Top-level: typecheck a file
// ---------------------------------------------------------------------------

/// Type-check all definitions in a file, processing them in dependency order.
///
/// Uses Tarjan's SCC algorithm to topologically sort definitions so that
/// callees are type-checked before callers. For each definition: type-check it,
/// register it in scope (as a `Binding`), and register it in the function
/// registry (for comptime evaluation by later definitions).
///
/// Returns the extended scope and function registry.
pub fn typecheck_file(
    file: &ast::File,
    scope: &Arc<Scope>,
    fn_reg: &FnRegistry,
) -> Result<(Arc<Scope>, FnRegistry), TypeError> {
    let (defs, edges) = build_dep_graph(file);
    let sccs = tarjan_scc(&defs, &edges);

    let mut scope = Arc::clone(scope);
    let mut fn_reg = fn_reg.clone_env();
    let mut next_var: u32 = 2000; // Start high to avoid collision with builtin TyVars

    for scc in &sccs {
        if scc.members.len() == 1 && !scc.is_recursive {
            let name = &scc.members[0];
            let info = &defs[name];

            let (scheme, compiled, nv) =
                typecheck_def(&info.def, info.sig.as_ref(), &scope, &fn_reg, next_var)?;
            next_var = nv;

            if is_tycon_name(name) {
                scope = Scope::extend(
                    &scope,
                    name.clone(),
                    Binding::TyCon(
                        TyConDef {
                            params: info.def.args.iter().map(|(n, _)| n.clone()).collect(),
                            body: info.def.body.clone(),
                        },
                        scheme,
                    ),
                );
            } else {
                scope = Scope::extend(&scope, name.clone(), Binding::Value(scheme));
            }
            if let Some(compiled) = compiled {
                fn_reg = fn_reg.register(name.clone(), compiled);
            }
        } else {
            // Self-recursive or mutually recursive
            let mut ctx = TyCtx { next_var, subst: Substitution::empty() };
            for name in &scc.members {
                let info = &defs[name];
                let arity = info.def.args.len();
                let mut param_vars = Vec::new();
                for _ in 0..arity {
                    let (v, new_ctx) = ctx.fresh_var();
                    ctx = new_ctx;
                    param_vars.push(v);
                }
                let (ret_var, new_ctx) = ctx.fresh_var();
                ctx = new_ctx;
                let placeholder = Scheme::mono_fn(param_vars, ret_var);
                if is_tycon_name(name) {
                    scope = Scope::extend(
                        &scope,
                        name.clone(),
                        Binding::TyCon(
                            TyConDef {
                                params: info.def.args.iter().map(|(n, _)| n.clone()).collect(),
                                body: info.def.body.clone(),
                            },
                            placeholder,
                        ),
                    );
                } else {
                    scope = Scope::extend(&scope, name.clone(), Binding::Value(placeholder));
                }
            }
            next_var = ctx.next_var;

            for name in &scc.members {
                let info = &defs[name];
                let (scheme, compiled, nv) =
                    typecheck_def(&info.def, info.sig.as_ref(), &scope, &fn_reg, next_var)?;
                next_var = nv;
                if is_tycon_name(name) {
                    scope = Scope::extend(
                        &scope,
                        name.clone(),
                        Binding::TyCon(
                            TyConDef {
                                params: info.def.args.iter().map(|(n, _)| n.clone()).collect(),
                                body: info.def.body.clone(),
                            },
                            scheme,
                        ),
                    );
                } else {
                    scope = Scope::extend(&scope, name.clone(), Binding::Value(scheme));
                }
                if let Some(compiled) = compiled {
                    fn_reg = fn_reg.register(name.clone(), compiled);
                }
            }
        }
    }

    Ok((scope, fn_reg))
}

fn is_tycon_name(name: &Symbol) -> bool {
    name.text()
        .chars()
        .next()
        .map(|c| c.is_ascii_uppercase())
        .unwrap_or(false)
}

// ---------------------------------------------------------------------------
// Dependency graph
// ---------------------------------------------------------------------------

struct DefInfo {
    def: ast::stmt::Def,
    sig: Option<Expr>,
}

struct SCC {
    members: Vec<Symbol>,
    is_recursive: bool,
}

fn build_dep_graph(
    file: &ast::File,
) -> (HashMap<Symbol, DefInfo>, HashMap<Symbol, HashSet<Symbol>>) {
    // Collect all definitions
    let mut defs: HashMap<Symbol, DefInfo> = HashMap::new();
    let mut sigs: HashMap<Symbol, Expr> = HashMap::new();

    for stmt in &file.stmts {
        match stmt {
            Stmt::TypeSig(ts) => {
                sigs.insert(ts.name.clone(), ts.sig.clone());
            }
            Stmt::Def(def) => {
                let sig = sigs.remove(&def.name);
                defs.insert(
                    def.name.clone(),
                    DefInfo {
                        def: (**def).clone(),
                        sig,
                    },
                );
            }
            _ => {}
        }
    }

    // Build edges
    let def_names: HashSet<Symbol> = defs.keys().cloned().collect();
    let mut edges: HashMap<Symbol, HashSet<Symbol>> = HashMap::new();

    for (name, info) in &defs {
        let mut refs = HashSet::new();
        collect_refs_expr(&info.def.body, &mut refs);
        if let Some(sig) = &info.sig {
            collect_refs_expr(sig, &mut refs);
        }
        // Only keep refs that are other top-level defs (including self-refs for recursion detection)
        refs.retain(|r| def_names.contains(r));
        edges.insert(name.clone(), refs);
    }

    (defs, edges)
}

fn collect_refs_expr(expr: &Expr, refs: &mut HashSet<Symbol>) {
    match expr {
        Expr::Name(n) => {
            refs.insert(n.name.clone());
        }
        Expr::Apply(app) => {
            collect_refs_expr(&app.callee, refs);
            for arg in &app.args {
                collect_refs_expr(arg, refs);
            }
        }
        Expr::Dot(dot) => {
            collect_refs_expr(&dot.base, refs);
        }
        Expr::If(if_) => {
            for (c, b) in &if_.cond_branch_vec {
                collect_refs_expr(c, refs);
                collect_refs_expr(b, refs);
            }
            collect_refs_expr(&if_.else_branch, refs);
        }
        Expr::Chain(chain) => {
            for stmt in &chain.stmt_vec {
                match stmt {
                    Stmt::Let(l) => collect_refs_expr(&l.init, refs),
                    Stmt::Discard(d) => collect_refs_expr(&d.val, refs),
                    _ => {}
                }
            }
        }
        Expr::Grad(g) => {
            collect_refs_expr(&g.func, refs);
            for arg in &g.args {
                collect_refs_expr(arg, refs);
            }
        }
        Expr::ArrayLit(a) => {
            for e in &a.elements {
                collect_refs_expr(e, refs);
            }
        }
        Expr::Ctor(c) => {
            collect_refs_typespec(&c.ty, refs);
        }
        Expr::As(a) => {
            collect_refs_expr(&a.expr, refs);
            collect_refs_expr(&a.target, refs);
        }
        Expr::Match(m) => {
            collect_refs_expr(&m.scrutinee, refs);
            for (_, body) in &m.arms {
                collect_refs_expr(body, refs);
            }
        }
        Expr::Tuple(t) => {
            for e in &t.elements {
                collect_refs_expr(e, refs);
            }
        }
        Expr::Literal(_) => {}
    }
}

fn collect_refs_typespec(ts: &TypeSpec, refs: &mut HashSet<Symbol>) {
    match ts {
        TypeSpec::Name(n) => {
            refs.insert(n.name.clone());
        }
        TypeSpec::Apply(a) => {
            refs.insert(a.name.clone());
            for arg in &a.args {
                collect_refs_expr(arg, refs);
            }
        }
        TypeSpec::Record(r) => {
            for (_, expr) in &r.fields {
                collect_refs_expr(expr, refs);
            }
        }
        TypeSpec::Enum(e) => {
            for (_, payload) in &e.variants {
                if let Some(expr) = payload {
                    collect_refs_expr(expr, refs);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tarjan's SCC algorithm
// ---------------------------------------------------------------------------

fn tarjan_scc(
    defs: &HashMap<Symbol, DefInfo>,
    edges: &HashMap<Symbol, HashSet<Symbol>>,
) -> Vec<SCC> {
    struct State {
        index_counter: usize,
        stack: Vec<Symbol>,
        on_stack: HashSet<Symbol>,
        index: HashMap<Symbol, usize>,
        lowlink: HashMap<Symbol, usize>,
        result: Vec<SCC>,
    }

    fn strongconnect(
        v: &Symbol,
        edges: &HashMap<Symbol, HashSet<Symbol>>,
        all_defs: &HashSet<Symbol>,
        state: &mut State,
    ) {
        let idx = state.index_counter;
        state.index.insert(v.clone(), idx);
        state.lowlink.insert(v.clone(), idx);
        state.index_counter += 1;
        state.stack.push(v.clone());
        state.on_stack.insert(v.clone());

        if let Some(deps) = edges.get(v) {
            for w in deps {
                if !all_defs.contains(w) {
                    continue;
                }
                if !state.index.contains_key(w) {
                    strongconnect(w, edges, all_defs, state);
                    let w_low = state.lowlink[w];
                    let v_low = state.lowlink[v];
                    if w_low < v_low {
                        state.lowlink.insert(v.clone(), w_low);
                    }
                } else if state.on_stack.contains(w) {
                    let w_idx = state.index[w];
                    let v_low = state.lowlink[v];
                    if w_idx < v_low {
                        state.lowlink.insert(v.clone(), w_idx);
                    }
                }
            }
        }

        if state.lowlink[v] == state.index[v] {
            let mut members = Vec::new();
            loop {
                let w = state.stack.pop().unwrap();
                state.on_stack.remove(&w);
                members.push(w.clone());
                if w == *v {
                    break;
                }
            }
            // Check if this SCC is recursive
            let is_recursive = if members.len() > 1 {
                true
            } else {
                // Self-recursive if the single member has an edge to itself
                let name = &members[0];
                edges
                    .get(name)
                    .map(|deps| deps.contains(name))
                    .unwrap_or(false)
            };
            state.result.push(SCC {
                members,
                is_recursive,
            });
        }
    }

    let all_defs: HashSet<Symbol> = defs.keys().cloned().collect();
    let mut state = State {
        index_counter: 0,
        stack: Vec::new(),
        on_stack: HashSet::new(),
        index: HashMap::new(),
        lowlink: HashMap::new(),
        result: Vec::new(),
    };

    for name in defs.keys() {
        if !state.index.contains_key(name) {
            strongconnect(name, edges, &all_defs, &mut state);
        }
    }

    // Tarjan's produces SCCs in reverse topological order (callees first)
    state.result
}

fn find_type_sig<'a>(stmts: &'a [Stmt], name: &Symbol) -> Option<&'a Expr> {
    for stmt in stmts {
        if let Stmt::TypeSig(ts) = stmt {
            if ts.name == *name {
                return Some(&ts.sig);
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Typecheck a single definition
// ---------------------------------------------------------------------------

/// Type-check a single top-level definition.
///
/// If a type signature is provided, elaborates it into a `Scheme`, then
/// infers the body type and unifies with the declared return type.
/// Otherwise, infers everything from the body.
///
/// Returns the function's `Scheme`, an optional `CompiledFn` (for comptime
/// registration), and the updated `next_var` counter.
fn typecheck_def(
    def: &ast::stmt::Def,
    sig: Option<&Expr>,
    scope: &Arc<Scope>,
    fn_reg: &FnRegistry,
    next_var: u32,
) -> Result<(Scheme, Option<CompiledFn>, u32), TypeError> {
    let mut ctx = TyCtx { next_var, subst: Substitution::empty() };

    if let Some(sig_expr) = sig {
        // Has explicit type signature — elaborate it
        let (scheme, ctx_after) = elaborate_signature(&mut ctx, sig_expr, &def.args, fn_reg)?;
        ctx = ctx_after;

        // Create scope with args bound to their scheme-declared types
        let mut arg_scope = Arc::clone(scope);
        let params = scheme.params.clone();
        for (i, (arg_name, _)) in def.args.iter().enumerate() {
            let arg_scheme = Scheme::mono_fn(vec![], params[i].clone());
            arg_scope = Scope::extend(&arg_scope, arg_name.clone(), Binding::Value(arg_scheme));
        }

        // Infer the body type
        let (body_ty, ctx) = infer(ctx, &arg_scope, &def.body, fn_reg)?;

        // Evaluate the return type from the signature
        let ret_ty = evaluate_return_type(&scheme, &ctx, fn_reg)?;

        // Unify body type with declared return type
        let _ctx = ctx.unify(&body_ty, &ret_ty)?;

        // Build the compiled function for fn_reg use
        let compiled = CompiledFn {
            params: def.args.iter().map(|(name, _)| name.clone()).collect(),
            body: Arc::new(def.body.clone()),
        };

        Ok((scheme, Some(compiled), _ctx.next_var))
    } else {
        // No type signature — infer everything
        let mut arg_types = vec![];
        let mut arg_scope = Arc::clone(scope);
        for (arg_name, _) in &def.args {
            let (arg_ty, new_ctx) = ctx.fresh_var();
            ctx = new_ctx;
            arg_types.push(arg_ty.clone());
            let arg_scheme = Scheme::mono_fn(vec![], arg_ty);
            arg_scope = Scope::extend(&arg_scope, arg_name.clone(), Binding::Value(arg_scheme));
        }

        let (body_ty, ctx) = infer(ctx, &arg_scope, &def.body, fn_reg)?;

        let resolved_params: Vec<Type> = arg_types.iter().map(|t| ctx.resolve(t)).collect();
        let resolved_ret = ctx.resolve(&body_ty);

        let scheme = Scheme::mono_fn(resolved_params, resolved_ret);

        let compiled = CompiledFn {
            params: def.args.iter().map(|(name, _)| name.clone()).collect(),
            body: Arc::new(def.body.clone()),
        };

        Ok((scheme, Some(compiled), ctx.next_var))
    }
}

// ---------------------------------------------------------------------------
// Elaborate: AST type expression → Type
// ---------------------------------------------------------------------------

/// Elaborate a type signature AST expression into a `Scheme`.
///
/// Parses `A -> B -> C` into `Scheme { params: [A, B], ret: C }`, creating
/// fresh TyVars for unbound type variable names (e.g., `T`, `s`).
fn elaborate_signature(
    ctx: &mut TyCtx,
    sig_expr: &Expr,
    args: &[(Symbol, crate::Span)],
    fn_reg: &FnRegistry,
) -> Result<(Scheme, TyCtx), TypeError> {
    // A type signature like `A -> B -> C` is parsed as nested Ctor(Apply("->", [A, B->C])).
    // We need to flatten it into params + return type.
    let mut type_var_scope: HashMap<Symbol, TyVar> = HashMap::new();
    let (fn_ty, new_ctx) = elaborate_type(ctx.clone(), sig_expr, &mut type_var_scope, fn_reg)?;

    // Flatten the function type into params and return
    match &fn_ty {
        Type::Fn { params, ret } => {
            if params.len() != args.len() {
                return Err(TypeError::new(format!(
                    "signature has {} params but definition has {} args",
                    params.len(),
                    args.len()
                )));
            }
            let bound: Vec<(Symbol, TyVar)> = type_var_scope.into_iter().collect();
            let scheme = Scheme {
                bound,
                params: params.clone(),
                ret: ReturnType::Concrete(*ret.clone()),
            };
            Ok((scheme, new_ctx))
        }
        // If it's not a function type (e.g. just `T`), treat as zero-arg function
        other => {
            if !args.is_empty() {
                return Err(TypeError::new(
                    "signature is not a function type but definition has args",
                ));
            }
            let bound: Vec<(Symbol, TyVar)> = type_var_scope.into_iter().collect();
            let scheme = Scheme {
                bound,
                params: vec![],
                ret: ReturnType::Concrete(other.clone()),
            };
            Ok((scheme, new_ctx))
        }
    }
}

/// Convert an AST expression in type position into a `Type`.
///
/// Handles scalar keywords (`f32`, `i64`), type variables (unknown names
/// become fresh TyVars), `Ten` (→ `Tensor`), uppercase names (→ `TyCon`),
/// array literals (→ `Val`), and comptime function calls (evaluated via
/// the interpreter when all args are concrete).
fn elaborate_type(
    ctx: TyCtx,
    expr: &Expr,
    type_vars: &mut HashMap<Symbol, TyVar>,
    fn_reg: &FnRegistry,
) -> Result<(Type, TyCtx), TypeError> {
    match expr {
        Expr::Name(n) => {
            let name = &n.name;
            let text = name.text();
            // Check for scalar type keywords
            if let Some(st) = scalar_from_name(text) {
                return Ok((Type::Scalar(st), ctx));
            }
            // Check for existing type variable
            if let Some(&var) = type_vars.get(name) {
                return Ok((Type::Var(var), ctx));
            }
            // Otherwise: create a fresh type variable
            let (ty, ctx) = ctx.fresh_var();
            if let Type::Var(v) = &ty {
                type_vars.insert(name.clone(), *v);
            }
            Ok((ty, ctx))
        }

        Expr::Literal(lit) => {
            let val = interp::interpret(
                expr,
                &Env::empty(),
                fn_reg,
            ).map_err(|e| TypeError::new(e.message))?;
            Ok((Type::Val(val), ctx))
        }

        Expr::ArrayLit(arr) => {
            // Array literals in type position: elaborate each element.
            // If all are concrete values, produce a Val. Otherwise, keep as Var.
            let mut ctx = ctx;
            let mut elaborated = vec![];
            let mut all_concrete = true;
            for elem in &arr.elements {
                let (elem_ty, new_ctx) = elaborate_type(ctx, elem, type_vars, fn_reg)?;
                ctx = new_ctx;
                if matches!(elem_ty, Type::Var(_)) {
                    all_concrete = false;
                }
                elaborated.push(elem_ty);
            }
            if all_concrete {
                let vals: Result<Vec<Value>, TypeError> =
                    elaborated.iter().map(type_to_value).collect();
                let vals = vals?;
                // Build an int array from scalar int values
                let ints: Result<Vec<i64>, TypeError> = vals
                    .iter()
                    .map(|v| {
                        v.as_int()
                            .ok_or_else(|| TypeError::new("array element is not an integer"))
                    })
                    .collect();
                let ints = ints?;
                Ok((Type::Val(Value::int_vec(ints)), ctx))
            } else {
                // Contains type variables — return a fresh var that will be
                // bound later through unification
                let (ty, ctx) = ctx.fresh_var();
                Ok((ty, ctx))
            }
        }

        Expr::Ctor(ctor) => {
            match &ctor.ty {
                ast::TypeSpec::Name(n) => {
                    let name = &n.name;
                    let text = name.text();
                    if let Some(st) = scalar_from_name(text) {
                        return Ok((Type::Scalar(st), ctx));
                    }
                    // Unknown type name — treat as type variable
                    if let Some(&var) = type_vars.get(name) {
                        return Ok((Type::Var(var), ctx));
                    }
                    let (ty, ctx) = ctx.fresh_var();
                    if let Type::Var(v) = &ty {
                        type_vars.insert(name.clone(), *v);
                    }
                    Ok((ty, ctx))
                }

                ast::TypeSpec::Apply(a) => {
                    let name = &a.name;
                    let args = &a.args;
                    let text = name.text();

                    // Arrow type: A -> B
                    if text == "->" && args.len() == 2 {
                        let (param, ctx) = elaborate_type(ctx, &args[0], type_vars, fn_reg)?;
                        let (ret, ctx) = elaborate_type(ctx, &args[1], type_vars, fn_reg)?;
                        // Flatten nested arrows into multi-param function
                        match ret {
                            Type::Fn {
                                mut params,
                                ret: inner_ret,
                            } => {
                                params.insert(0, param);
                                Ok((Type::Fn { params, ret: inner_ret }, ctx))
                            }
                            _ => Ok((
                                Type::Fn {
                                    params: vec![param],
                                    ret: Box::new(ret),
                                },
                                ctx,
                            )),
                        }
                    }
                    // Ten constructor: Ten elem shape
                    else if text == "Ten" && args.len() == 2 {
                        let (elem, ctx) = elaborate_type(ctx, &args[0], type_vars, fn_reg)?;
                        let (shape, ctx) = elaborate_type(ctx, &args[1], type_vars, fn_reg)?;
                        Ok((
                            Type::Tensor {
                                elem: Box::new(elem),
                                shape: Box::new(shape),
                            },
                            ctx,
                        ))
                    }
                    // Uppercase name: type constructor
                    else if name.text().starts_with(|c: char| c.is_uppercase()) {
                        let mut elaborated_args = vec![];
                        let mut ctx = ctx;
                        for arg in args {
                            let (arg_ty, new_ctx) =
                                elaborate_type(ctx, arg, type_vars, fn_reg)?;
                            ctx = new_ctx;
                            elaborated_args.push(arg_ty);
                        }
                        Ok((
                            Type::TyCon {
                                name: name.clone(),
                                args: elaborated_args,
                            },
                            ctx,
                        ))
                    }
                    // Lowercase name: fn_reg function call in type position
                    else {
                        // Try to evaluate it via the interpreter
                        // First elaborate args to see if they're concrete
                        let mut elaborated_args = vec![];
                        let mut ctx = ctx;
                        let mut all_concrete = true;
                        for arg in args {
                            let (arg_ty, new_ctx) =
                                elaborate_type(ctx, arg, type_vars, fn_reg)?;
                            ctx = new_ctx;
                            if matches!(arg_ty, Type::Var(_)) {
                                all_concrete = false;
                            }
                            elaborated_args.push(arg_ty);
                        }
                        if all_concrete {
                            // Convert Type args to Value args and call
                            let val_args: Result<Vec<Value>, TypeError> = elaborated_args
                                .iter()
                                .map(type_to_value)
                                .collect();
                            let val_args = val_args?;
                            let result = fn_reg
                                .call(name, val_args)
                                .map_err(|e| TypeError::new(e.message))?;
                            Ok((Type::Val(result), ctx))
                        } else {
                            // Args not yet concrete — this would be a deferred return type.
                            // For now, return a type variable that will be resolved later.
                            let (ty, ctx) = ctx.fresh_var();
                            Ok((ty, ctx))
                        }
                    }
                }

                ast::TypeSpec::Record(r) => {
                    let mut type_fields = vec![];
                    let mut ctx = ctx;
                    for (name, field_expr) in &r.fields {
                        let (field_ty, new_ctx) =
                            elaborate_type(ctx, field_expr, type_vars, fn_reg)?;
                        ctx = new_ctx;
                        type_fields.push((name.clone(), field_ty));
                    }
                    Ok((Type::Record { fields: type_fields }, ctx))
                }

                ast::TypeSpec::Enum(_) => {
                    Err(TypeError::new("enum types not yet supported"))
                }
            }
        }

        Expr::Apply(app) => {
            // Function application in type position — this is a fn_reg call
            // e.g. `matmul_shape s1 s2`
            let func_name = match &app.callee {
                Expr::Name(n) => n.name.clone(),
                _ => return Err(TypeError::new("expected function name in type position")),
            };

            // Ten constructor
            if func_name.text() == "Ten" && app.args.len() == 2 {
                let (elem, ctx) = elaborate_type(ctx, &app.args[0], type_vars, fn_reg)?;
                let (shape, ctx) = elaborate_type(ctx, &app.args[1], type_vars, fn_reg)?;
                return Ok((
                    Type::Tensor {
                        elem: Box::new(elem),
                        shape: Box::new(shape),
                    },
                    ctx,
                ));
            }

            // Uppercase: type constructor
            if func_name.text().starts_with(|c: char| c.is_uppercase()) {
                let mut elaborated_args = vec![];
                let mut ctx = ctx;
                for arg in &app.args {
                    let (arg_ty, new_ctx) = elaborate_type(ctx, arg, type_vars, fn_reg)?;
                    ctx = new_ctx;
                    elaborated_args.push(arg_ty);
                }
                return Ok((
                    Type::TyCon {
                        name: func_name,
                        args: elaborated_args,
                    },
                    ctx,
                ));
            }

            // Lowercase: fn_reg function call
            let mut elaborated_args = vec![];
            let mut ctx = ctx;
            let mut all_concrete = true;
            for arg in &app.args {
                let (arg_ty, new_ctx) = elaborate_type(ctx, arg, type_vars, fn_reg)?;
                ctx = new_ctx;
                if matches!(arg_ty, Type::Var(_)) {
                    all_concrete = false;
                }
                elaborated_args.push(arg_ty);
            }
            if all_concrete {
                let val_args: Result<Vec<Value>, TypeError> =
                    elaborated_args.iter().map(type_to_value).collect();
                let val_args = val_args?;
                let result = fn_reg
                    .call(&func_name, val_args)
                    .map_err(|e| TypeError::new(e.message))?;
                Ok((Type::Val(result), ctx))
            } else {
                let (ty, ctx) = ctx.fresh_var();
                Ok((ty, ctx))
            }
        }

        _ => Err(TypeError::new(format!(
            "unsupported expression in type position: {:?}",
            std::mem::discriminant(expr)
        ))),
    }
}

fn type_to_value(ty: &Type) -> Result<Value, TypeError> {
    match ty {
        Type::Val(v) => Ok(v.clone()),
        Type::Scalar(st) => {
            // Convert scalar type tag to an integer code for fn_reg use
            Err(TypeError::new(format!(
                "cannot use scalar type {st:?} as a fn_reg value"
            )))
        }
        _ => Err(TypeError::new(format!(
            "cannot convert type to fn_reg value: {ty:?}"
        ))),
    }
}

fn evaluate_return_type(
    scheme: &Scheme,
    ctx: &TyCtx,
    _fn_reg: &FnRegistry,
) -> Result<Type, TypeError> {
    match &scheme.ret {
        ReturnType::Concrete(ty) => Ok(ctx.resolve(ty)),
        ReturnType::Deferred(_ast) => {
            // TODO: build env from resolved bound vars, interpret the AST
            Err(TypeError::new(
                "deferred return type evaluation not yet implemented",
            ))
        }
    }
}

fn scalar_from_name(name: &str) -> Option<ScalarType> {
    match name {
        "f32" => Some(ScalarType::F32),
        "f64" => Some(ScalarType::F64),
        "i32" => Some(ScalarType::I32),
        "i64" => Some(ScalarType::I64),
        "u32" => Some(ScalarType::U32),
        "u64" => Some(ScalarType::U64),
        "bool" => Some(ScalarType::Bool),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Infer: AST value expression → Type
// ---------------------------------------------------------------------------

/// Core Hindley-Milner inference on value expressions.
///
/// Walks the AST, assigning types to each subexpression. Handles names
/// (scope lookup + scheme instantiation), application (unify function type
/// with args), if/else, let chains, dot access (with TyCon expansion),
/// array literals, grad, and type constructor bodies (Ctor).
///
/// Takes `TyCtx` by value, returns a new `TyCtx` with updated substitution.
fn infer(
    ctx: TyCtx,
    scope: &Arc<Scope>,
    expr: &Expr,
    fn_reg: &FnRegistry,
) -> Result<(Type, TyCtx), TypeError> {
    match expr {
        Expr::Literal(lit) => infer_literal(ctx, &lit.val),

        Expr::Name(n) => {
            match scope.lookup_value(&n.name) {
                Some(scheme) => {
                    let mut ctx = ctx;
                    let (params, ret, _bound) = ctx.instantiate(scheme);
                    let ret_ty = match ret {
                        Some(ty) => ty,
                        None => {
                            // Deferred return type — use a fresh var for now
                            let (v, new_ctx) = ctx.fresh_var();
                            ctx = new_ctx;
                            v
                        }
                    };
                    if params.is_empty() {
                        Ok((ret_ty, ctx))
                    } else {
                        Ok((
                            Type::Fn {
                                params,
                                ret: Box::new(ret_ty),
                            },
                            ctx,
                        ))
                    }
                }
                None => Err(TypeError::new(format!("undefined: {}", n.name))),
            }
        }

        Expr::Apply(app) => {
            // Infer the function type
            let (f_ty, ctx) = infer(ctx, scope, &app.callee, fn_reg)?;

            // Infer argument types
            let mut ctx = ctx;
            let mut arg_types = vec![];
            for arg in &app.args {
                let (arg_ty, new_ctx) = infer(ctx, scope, arg, fn_reg)?;
                ctx = new_ctx;
                arg_types.push(arg_ty);
            }

            // Create expected function type
            let (ret_var, ctx) = ctx.fresh_var();
            let expected = Type::Fn {
                params: arg_types,
                ret: Box::new(ret_var.clone()),
            };
            let ctx = ctx.unify(&f_ty, &expected)?;

            Ok((ctx.resolve(&ret_var), ctx))
        }

        Expr::If(if_expr) => {
            let bool_ty = Type::Scalar(ScalarType::Bool);

            let mut ctx = ctx;
            let mut result_ty = None;

            for (cond, body) in &if_expr.cond_branch_vec {
                let (cond_ty, new_ctx) = infer(ctx, scope, cond, fn_reg)?;
                ctx = new_ctx.unify(&cond_ty, &bool_ty)?;
                let (body_ty, new_ctx) = infer(ctx, scope, body, fn_reg)?;
                ctx = new_ctx;
                match result_ty {
                    None => result_ty = Some(body_ty),
                    Some(ref prev) => {
                        ctx = ctx.unify(prev, &body_ty)?;
                    }
                }
            }

            let (else_ty, ctx) = infer(ctx, scope, &if_expr.else_branch, fn_reg)?;
            match result_ty {
                Some(ref prev) => {
                    let ctx = ctx.unify(prev, &else_ty)?;
                    Ok((ctx.resolve(&else_ty), ctx))
                }
                None => Ok((else_ty, ctx)),
            }
        }

        Expr::Chain(chain) => {
            let mut ctx = ctx;
            let mut current_scope = Arc::clone(scope);
            let mut last_ty = None;

            for stmt in &chain.stmt_vec {
                match stmt {
                    Stmt::Let(let_stmt) => {
                        let (init_ty, new_ctx) =
                            infer(ctx, &current_scope, &let_stmt.init, fn_reg)?;
                        ctx = new_ctx;
                        let name = match &let_stmt.pattern {
                            Pattern::Name(n) => n.name.clone(),
                            _ => {
                                return Err(TypeError::new("unsupported pattern in let"))
                            }
                        };
                        let scheme = Scheme::mono_fn(vec![], init_ty);
                        current_scope = Scope::extend(&current_scope, name, Binding::Value(scheme));
                    }
                    Stmt::Discard(d) => {
                        let (ty, new_ctx) = infer(ctx, &current_scope, &d.val, fn_reg)?;
                        ctx = new_ctx;
                        last_ty = Some(ty);
                    }
                    _ => return Err(TypeError::new("unsupported statement")),
                }
            }

            last_ty
                .map(|ty| (ty, ctx))
                .ok_or_else(|| TypeError::new("empty chain"))
        }

        Expr::Dot(dot) => {
            let (base_ty, ctx) = infer(ctx, scope, &dot.base, fn_reg)?;
            let resolved = ctx.resolve(&base_ty);
            match &resolved {
                Type::Record { fields } => {
                    for (name, ty) in fields {
                        if *name == dot.field {
                            return Ok((ty.clone(), ctx));
                        }
                    }
                    Err(TypeError::new(format!(
                        "no field '{}' in record",
                        dot.field
                    )))
                }
                Type::TyCon { name, args } => {
                    // Expand the TyCon to see its record structure
                    let tycon_def = scope.lookup_tycon(name).ok_or_else(|| {
                        TypeError::new(format!("unknown type constructor: {name}"))
                    })?;

                    // Build a type scope: bind TyCon params to the concrete args
                    let mut type_vars = HashMap::new();
                    let mut elab_ctx = ctx.clone();
                    for (param, arg) in tycon_def.params.iter().zip(args.iter()) {
                        // Create a type var for this param and immediately bind it to the arg
                        let var = TyVar(elab_ctx.next_var);
                        elab_ctx.next_var += 1;
                        elab_ctx.subst = elab_ctx.subst.bind(var, arg.clone());
                        type_vars.insert(param.clone(), var);
                    }

                    // Elaborate the TyCon body with those bindings
                    let (record_ty, elab_ctx_after) =
                        elaborate_type(elab_ctx, &tycon_def.body, &mut type_vars, fn_reg)?;

                    // Look up the field in the expanded record, resolving through elab_ctx's substitution
                    match &record_ty {
                        Type::Record { fields } => {
                            for (fname, fty) in fields {
                                if *fname == dot.field {
                                    return Ok((elab_ctx_after.resolve(fty), ctx));
                                }
                            }
                            Err(TypeError::new(format!(
                                "no field '{}' in type constructor '{}'",
                                dot.field, name
                            )))
                        }
                        _ => Err(TypeError::new(format!(
                            "type constructor '{}' does not expand to a record",
                            name
                        ))),
                    }
                }
                _ => Err(TypeError::new(format!(
                    "dot access on non-record type: {:?}",
                    resolved
                ))),
            }
        }

        Expr::ArrayLit(arr) => {
            // In value position, infer element types and unify
            let mut ctx = ctx;
            let (elem_var, new_ctx) = ctx.fresh_var();
            ctx = new_ctx;
            for elem in &arr.elements {
                let (elem_ty, new_ctx) = infer(ctx, scope, elem, fn_reg)?;
                ctx = new_ctx.unify(&elem_ty, &elem_var)?;
            }
            // The type of an array literal in value position — for now just return a fresh var
            // since arrays in value position are fn_reg values
            let (ty, ctx) = ctx.fresh_var();
            Ok((ty, ctx))
        }

        Expr::Grad(grad) => {
            // grad f args...
            // The type of grad f is: if f :: A -> B -> ... -> R, then grad f :: A -> B -> ... -> A
            let (f_ty, ctx) = infer(ctx, scope, &grad.func, fn_reg)?;
            let mut ctx = ctx;
            let mut arg_types = vec![];
            for arg in &grad.args {
                let (arg_ty, new_ctx) = infer(ctx, scope, arg, fn_reg)?;
                ctx = new_ctx;
                arg_types.push(arg_ty);
            }

            let resolved_f = ctx.resolve(&f_ty);
            match &resolved_f {
                Type::Fn { params, .. } => {
                    if params.is_empty() {
                        return Err(TypeError::new("grad: function has no parameters"));
                    }
                    // Unify args with params
                    if arg_types.len() != params.len() {
                        return Err(TypeError::new(format!(
                            "grad: expected {} args, got {}",
                            params.len(),
                            arg_types.len()
                        )));
                    }
                    for (arg_ty, param_ty) in arg_types.iter().zip(params.iter()) {
                        ctx = ctx.unify(arg_ty, param_ty)?;
                    }
                    // Return type is the type of the first parameter
                    Ok((ctx.resolve(&params[0]), ctx))
                }
                _ => Err(TypeError::new("grad: expected a function")),
            }
        }

        Expr::Ctor(ctor) => {
            // Type constructor in value position (e.g., a type definition body)
            let mut type_vars = HashMap::new();
            elaborate_type(ctx, &Expr::Ctor(ctor.clone()), &mut type_vars, fn_reg)
        }

        _ => Err(TypeError::new(format!(
            "unsupported expression: {:?}",
            std::mem::discriminant(expr)
        ))),
    }
}

fn infer_literal(ctx: TyCtx, lit: &crate::vocab::Literal) -> Result<(Type, TyCtx), TypeError> {
    match lit {
        crate::vocab::Literal::Number(n) => {
            // Numeric literals get a fresh type variable — the concrete numeric type
            // is determined by context (how it's used).
            let (ty, ctx) = ctx.fresh_var();
            Ok((ty, ctx))
        }
        crate::vocab::Literal::Bool(_) => Ok((Type::Scalar(ScalarType::Bool), ctx)),
        crate::vocab::Literal::String(_) => {
            let (ty, ctx) = ctx.fresh_var();
            Ok((ty, ctx))
        }
    }
}

// ---------------------------------------------------------------------------
// FnRegistry clone helper
// ---------------------------------------------------------------------------

impl FnRegistry {
    pub fn clone_env(&self) -> Self {
        // This is a shallow clone — builtins and compiled fns are Arc'd
        FnRegistry {
            builtins: self.builtins.clone(),
            compiled: self.compiled.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// Builtin scope
// ---------------------------------------------------------------------------

/// Create the initial scope with all builtin operators and functions.
///
/// Includes arithmetic (`+`, `-`, `*`, `/`, `%`, `@`), comparison (`==`, `<`, etc.),
/// logical (`||`, `&&`), array operations (`concat`, `head`, `tail`, `init`,
/// `last`, `len`), `max`, `min`, `cross_entropy`, `panic`, `matmul_builtin`,
/// and unary negation (`~`).
pub fn builtin_scope() -> Arc<Scope> {
    let mut bindings: HashMap<Symbol, Binding> = HashMap::new();

    // Register arithmetic operators
    // Arithmetic operators: A -> B -> C (independent types for broadcasting support)
    let arith_ops = ["+(_,_)", "-(_,_)", "*(_,_)", "/(_,_)", "%(_,_)"];
    for op in &arith_ops {
        bindings.insert(
            Symbol::from(*op),
            Binding::Value(Scheme {
                bound: vec![
                    (Symbol::from("A"), TyVar(1000)),
                    (Symbol::from("B"), TyVar(1015)),
                    (Symbol::from("C"), TyVar(1016)),
                ],
                params: vec![Type::Var(TyVar(1000)), Type::Var(TyVar(1015))],
                ret: ReturnType::Concrete(Type::Var(TyVar(1016))),
            }),
        );
    }

    // @ operator: T1 -> T2 -> T3 (matmul — types are independent, validated by shape computation)
    bindings.insert(
        Symbol::from("@(_,_)"),
        Binding::Value(Scheme {
            bound: vec![
                (Symbol::from("A"), TyVar(1012)),
                (Symbol::from("B"), TyVar(1013)),
                (Symbol::from("C"), TyVar(1014)),
            ],
            params: vec![Type::Var(TyVar(1012)), Type::Var(TyVar(1013))],
            ret: ReturnType::Concrete(Type::Var(TyVar(1014))),
        }),
    );

    // Comparison operators return bool
    let cmp_ops = ["==(_,_)", "!=(_,_)", "<(_,_)", ">(_,_)", "<=(_,_)", ">=(_,_)"];
    for op in &cmp_ops {
        bindings.insert(
            Symbol::from(*op),
            Binding::Value(Scheme {
                bound: vec![
                    (Symbol::from("T"), TyVar(1001)),
                ],
                params: vec![Type::Var(TyVar(1001)), Type::Var(TyVar(1001))],
                ret: ReturnType::Concrete(Type::Scalar(ScalarType::Bool)),
            }),
        );
    }

    // Logical operators
    let log_ops = ["||(_,_)", "&&(_,_)"];
    for op in &log_ops {
        bindings.insert(
            Symbol::from(*op),
            Binding::Value(Scheme {
                bound: vec![],
                params: vec![
                    Type::Scalar(ScalarType::Bool),
                    Type::Scalar(ScalarType::Bool),
                ],
                ret: ReturnType::Concrete(Type::Scalar(ScalarType::Bool)),
            }),
        );
    }

    // max, min
    for name in &["max", "min"] {
        bindings.insert(
            Symbol::from(*name),
            Binding::Value(Scheme {
                bound: vec![(Symbol::from("T"), TyVar(1002))],
                params: vec![Type::Var(TyVar(1002)), Type::Var(TyVar(1002))],
                ret: ReturnType::Concrete(Type::Var(TyVar(1002))),
            }),
        );
    }

    // cross_entropy :: Ten T [n] -> Ten T [n] -> T
    bindings.insert(
        Symbol::from("cross_entropy"),
        Binding::Value(Scheme {
            bound: vec![
                (Symbol::from("T"), TyVar(1003)),
                (Symbol::from("n"), TyVar(1004)),
            ],
            params: vec![
                Type::Tensor {
                    elem: Box::new(Type::Var(TyVar(1003))),
                    shape: Box::new(Type::Var(TyVar(1004))),
                },
                Type::Tensor {
                    elem: Box::new(Type::Var(TyVar(1003))),
                    shape: Box::new(Type::Var(TyVar(1004))),
                },
            ],
            ret: ReturnType::Concrete(Type::Var(TyVar(1003))),
        }),
    );

    // Array operations: concat, head, tail, init, last, len
    // concat :: [T] -> [T] -> [T]
    bindings.insert(
        Symbol::from("concat"),
        Binding::Value(Scheme {
            bound: vec![(Symbol::from("T"), TyVar(1005))],
            params: vec![Type::Var(TyVar(1005)), Type::Var(TyVar(1005))],
            ret: ReturnType::Concrete(Type::Var(TyVar(1005))),
        }),
    );

    // head :: [T] -> T, last :: [T] -> T
    for name in &["head", "last"] {
        bindings.insert(
            Symbol::from(*name),
            Binding::Value(Scheme {
                bound: vec![(Symbol::from("T"), TyVar(1006))],
                params: vec![Type::Var(TyVar(1006))],
                ret: ReturnType::Concrete(Type::Var(TyVar(1006))),
            }),
        );
    }

    // tail :: [T] -> [T], init :: [T] -> [T]
    for name in &["tail", "init"] {
        bindings.insert(
            Symbol::from(*name),
            Binding::Value(Scheme {
                bound: vec![(Symbol::from("T"), TyVar(1007))],
                params: vec![Type::Var(TyVar(1007))],
                ret: ReturnType::Concrete(Type::Var(TyVar(1007))),
            }),
        );
    }

    // len :: [T] -> T
    bindings.insert(
        Symbol::from("len"),
        Binding::Value(Scheme {
            bound: vec![(Symbol::from("T"), TyVar(1008))],
            params: vec![Type::Var(TyVar(1008))],
            ret: ReturnType::Concrete(Type::Var(TyVar(1008))),
        }),
    );

    // panic :: T -> T
    bindings.insert(
        Symbol::from("panic"),
        Binding::Value(Scheme {
            bound: vec![(Symbol::from("T"), TyVar(1009))],
            params: vec![Type::Var(TyVar(1009))],
            ret: ReturnType::Concrete(Type::Var(TyVar(1009))),
        }),
    );

    // matmul_builtin :: T -> T -> T (placeholder intrinsic)
    bindings.insert(
        Symbol::from("matmul_builtin"),
        Binding::Value(Scheme {
            bound: vec![(Symbol::from("T"), TyVar(1011))],
            params: vec![Type::Var(TyVar(1011)), Type::Var(TyVar(1011))],
            ret: ReturnType::Concrete(Type::Var(TyVar(1011))),
        }),
    );

    // Unary negation: ~(_) :: T -> T
    bindings.insert(
        Symbol::from("~(_)"),
        Binding::Value(Scheme {
            bound: vec![(Symbol::from("T"), TyVar(1010))],
            params: vec![Type::Var(TyVar(1010))],
            ret: ReturnType::Concrete(Type::Var(TyVar(1010))),
        }),
    );

    Scope::root(bindings)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Config, Lexer, Source, parser};

    fn parse_and_typecheck(src: &str) -> Result<(Arc<Scope>, FnRegistry), TypeError> {
        let lexer = Lexer::new(Config::default());
        let source = Source::new("test", src, lexer.config());
        let tokens = lexer.lex(source).unwrap();
        let file = parser::parse(parser::TokenStream::new(tokens)).unwrap();

        let scope = super::builtin_scope();
        let fn_reg = interp::builtin_fn_registry();
        typecheck_file(&file, &scope, &fn_reg)
    }

    #[test]
    fn test_infer_simple_def() {
        let result = parse_and_typecheck("relu x =\n  max 0 x\n");
        assert!(result.is_ok(), "Failed: {:?}", result.err());
    }

    #[test]
    fn test_infer_with_signature() {
        let result = parse_and_typecheck(
            "double :: Ten f32 [10] -> Ten f32 [10]\ndouble x =\n  x + x\n",
        );
        assert!(result.is_ok(), "Failed: {:?}", result.err());
    }

    #[test]
    fn test_infer_if_else() {
        let result = parse_and_typecheck("f x =\n  if x then\n    1\n  else\n    0\n");
        assert!(result.is_ok(), "Failed: {:?}", result.err());
    }

    #[test]
    fn test_infer_let_chain() {
        let result = parse_and_typecheck("f x =\n  y = x\n  y\n");
        assert!(result.is_ok(), "Failed: {:?}", result.err());
    }

    #[test]
    fn test_infer_binop() {
        let result = parse_and_typecheck("f x =\n  x + 1\n");
        assert!(result.is_ok(), "Failed: {:?}", result.err());
    }

    #[test]
    fn test_out_of_order_defs() {
        // f calls g, but g is defined after f
        let result = parse_and_typecheck(
            "f x =\n  g x\n\ng y =\n  y + 1\n",
        );
        assert!(result.is_ok(), "Failed: {:?}", result.err());
    }

    #[test]
    fn test_self_recursive() {
        let result = parse_and_typecheck(
            "countdown n =\n  if n <= 0 then\n    0\n  else\n    countdown (n - 1)\n",
        );
        assert!(result.is_ok(), "Failed: {:?}", result.err());
    }

    #[test]
    fn test_mutual_recursion() {
        let result = parse_and_typecheck(
            "is_even n =\n  if n == 0 then\n    true\n  else\n    is_odd (n - 1)\n\nis_odd n =\n  if n == 0 then\n    false\n  else\n    is_even (n - 1)\n",
        );
        assert!(result.is_ok(), "Failed: {:?}", result.err());
    }

    #[test]
    fn test_multiple_defs_in_order() {
        // helper is defined first, main calls helper — should work regardless of order
        let result = parse_and_typecheck(
            "helper x =\n  x + 1\n\nmain x =\n  helper (helper x)\n",
        );
        assert!(result.is_ok(), "Failed: {:?}", result.err());
    }
}
