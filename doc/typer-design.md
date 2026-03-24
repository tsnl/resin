# Resin Type Checker Design

## Overview

Resin's type checker combines Hindley-Milner inference with compile-time evaluation of pure functions in type positions. Three design principles guide it:

1. **`Type` only contains normal forms.** No unevaluated function calls live inside `Type`. To construct a `Type`, you must first evaluate. This cleanly separates "what a type is" from "how it's computed."

2. **A full interpreter participates in type-checking.** Functions are type-checked, then become available for compile-time evaluation during type-checking of subsequent functions.

3. **Type-checking and compilation are one interleaved process.** For each function (in dependency order): type-check it, compile it, register it. The compiled form serves double duty — comptime evaluation during type-checking, and later final code generation.

## Immutability

All data structures are immutable. No mutation anywhere— not in the interpreter, not in the type checker, not in the compilation pipeline.

### Persistent parent-chain (Scope and Env)

Both the type checker and interpreter use the same parent-chain structure. A node holds a small set of bindings plus an `Arc` reference to a parent. "Mutation" is shadowing — creating a new child:

- **`Scope`** maps names to `Scheme`s. Used by the type checker (`infer`, `elaborate`).
- **`Env`** maps names to `Value`s. Used by the interpreter (`interpret`).

Same structure, different contents:

```rust
struct Scope {
    bindings: HashMap<Symbol, Scheme>,
    parent: Option<Arc<Scope>>,
}

struct Env {
    bindings: HashMap<Symbol, Value>,
    parent: Option<Arc<Env>>,
}
```

Both support `extend` (returns a new child, O(1)) and `lookup` (walks the chain, O(depth)).

### Substitution: cons list

The unification substitution is an immutable singly linked list. Each `bind` prepends a node — O(1), no cloning. Lookup walks the chain — O(n), first match wins. No `compose` needed; the chain IS the composition (newer bindings shadow older ones):

```rust
#[derive(Clone)]
enum Substitution {
    Empty,
    Bind {
        var: TyVar,
        ty: Type,
        parent: Arc<Substitution>,
    },
}

impl Substitution {
    fn empty() -> Self { Substitution::Empty }

    fn bind(&self, var: TyVar, ty: Type) -> Self {
        Substitution::Bind { var, ty, parent: Arc::new(self.clone()) }
    }

    fn lookup(&self, v: TyVar) -> Option<&Type> {
        match self {
            Substitution::Empty => None,
            Substitution::Bind { var, ty, parent } =>
                if *var == v { Some(ty) } else { parent.lookup(v) },
        }
    }

    fn apply(&self, ty: &Type) -> Type {
        match ty {
            Type::Var(v) => match self.lookup(*v) {
                Some(t) => self.apply(t),   // chase chains
                None => ty.clone(),
            },
            // recurse into compound types...
            _ => { /* structural recursion */ }
        }
    }
}
```

### Type context: threaded state

The type context (fresh variable counter + substitution) is threaded through inference as a value, not mutated:

```rust
struct TyCtx {
    next_var: u32,
    subst: Substitution,
}

impl TyCtx {
    fn fresh_var(self) -> (Type, TyCtx);     // returns new var AND new ctx
    fn unify(self, a: &Type, b: &Type) -> Result<TyCtx, TypeError>;  // returns new ctx
    fn resolve(&self, ty: &Type) -> Type;
}
```

Every inference function takes `TyCtx` by value and returns a new `TyCtx`:

```rust
fn infer(ctx: TyCtx, scope: &Arc<Scope>, expr: &ast::Expr)
    -> Result<(Type, TyCtx), TypeError>;
```

### Compilation pipeline: fold over definitions

The top-level pipeline is a fold — each definition produces a new `(scope, comptime_env)` pair that feeds into the next:

```rust
fn typecheck_file(file: &ast::File) -> Result<TypedFile, TypeError> {
    let order = toposort(&file);

    let init = (builtin_scope(), builtin_comptime_env());

    order.iter().try_fold(init, |(scope, comptime), scc| {
        let (scheme, compiled) = typecheck_def(scc, &scope, &comptime)?;
        let scope = scope.extend(scc.name, scheme);
        let comptime = comptime.register(scc.name, compiled);
        Ok((scope, comptime))
    })?
}
```

No `mut` anywhere. Each step produces new immutable values.

## Architecture

```
For each function, in dependency order:

  1. TYPE-CHECK  ─── HM inference on the body.
  │                  Binds type variables.
  │
  2. EVALUATE    ─── For any comptime calls in the return type:
  │                  substitute bound values, run the interpreter.
  │                  Panic → type error.
  │
  3. COMPILE     ─── Register in comptime environment.
  │
  └── This function is now available for comptime use by later functions.
```

## Types: Only Normal Forms

```rust
enum Type {
    Var(TyVar),                                 // Unification metavariable
    Val(Value),                                 // Concrete comptime value (a tensor)
    Scalar(ScalarType),                         // f32, i64, bool — type tags (from vocab.rs)
    Tensor { elem: Box<Type>, shape: Box<Type> },
    Record { fields: Vec<(Symbol, Type)> },
    Fn { params: Vec<Type>, ret: Box<Type> },
    TyCon { name: Symbol, args: Vec<Type> },    // Opaque type constructor
}
```

### `ScalarType` is a type tag, not a value

`ScalarType` (defined in `vocab.rs`) classifies element types: `F32`, `I64`, `Bool`, etc. It's not something you compute with — it's part of the type structure. It stays as its own variant, not inside `Value`.

### Values: tensors and records

Comptime values mirror the two structural forms of `Type` — tensors and records:

```rust
enum Value {
    Tensor(TensorVal),
    Record(Vec<(Symbol, Value)>),
}

struct TensorVal {
    data: TensorData,
    shape: Vec<usize>,       // [] = scalar, [n] = vector, [m,n] = matrix, ...
}

enum TensorData {
    Int(Vec<i64>),
    Float(Vec<f64>),
    Bool(Vec<bool>),
}
```

Tensor values follow NumPy: scalars are 0-dim, shapes are 1-dim integer tensors.

| What | Value representation |
|------|---------------------|
| Dimension `10` | `Tensor({ Int([10]), shape: [] })` — scalar |
| Shape `[10, 5]` | `Tensor({ Int([10, 5]), shape: [2] })` — 1-dim |
| Boolean `true` | `Tensor({ Bool([true]), shape: [] })` — scalar |
| A model | `Record([("w", Tensor(...)), ("b", Tensor(...))])` |

## How Type Signatures Work

A function's type scheme stores parameter types as `Type` (may contain `Var`s) and the return type as a `ReturnType`:

```rust
enum ReturnType {
    Concrete(Type),              // Already fully elaborated
    Deferred(Arc<ast::Expr>),    // Needs evaluation after vars are bound
}

struct Scheme {
    bound: Vec<(Symbol, TyVar)>,
    params: Vec<Type>,
    ret: ReturnType,
}
```

Most signatures use `Concrete` — the return type elaborates directly to a `Type`. `Deferred` is for return types containing comptime function calls with unresolved arguments (e.g., `matmul_shape s1 s2`).

### At a call site

When we see `matmul a b`:

```
1. Instantiate the scheme:
   - Fresh vars: T→?0, s1→?1, s2→?2
   - Params: [Tensor(?0, ?1), Tensor(?0, ?2)]

2. Unify params with argument types (returns new TyCtx):
   - ?0 = Scalar(F32), ?1 = Val([10,5]), ?2 = Val([5])

3. Build a type scope from resolved vars:
   { T → Scalar(F32), s1 → Val([10,5]), s2 → Val([5]) }

4. Re-elaborate ret_ast with that scope:
   elaborate(ret_ast, type_scope, comptime)
   → Tensor { elem: Scalar(F32), shape: Val([10]) }
   (matmul_shape is called via the interpreter during elaboration)

5. Result: Tensor(Scalar(F32), Val([10]))
```

Return type evaluation IS just `elaborate` called again — once vars are resolved, any comptime calls have concrete args and get evaluated.

### Simple case: no comptime calls

For `relu :: Ten T s -> Ten T s`, re-elaboration of `Ten T s` is trivial: look up `T` and `s` in the scope, build the Type. Same code path.

## Elaboration and Interpretation

There are two evaluation functions with a clean separation:

- **`elaborate`** converts AST type expressions into `Type`. It handles type constructors (`Ten`, `Linear`), type variables, literals. When it encounters a comptime function call with concrete args, it calls the interpreter to evaluate it and wraps the result as `Val(...)`.

- **`interpret`** evaluates AST value expressions into `Value`. It handles function bodies: arithmetic, if/else, let bindings, array construction, function calls. Pure comptime computation.

`elaborate` calls `interpret` (downward) when it encounters a comptime call. `interpret` never calls `elaborate`. One-way dependency.

## The Interpreter

### Purely functional tree-walking

The interpreter is a pure function: `(AST, Env, ComptimeEnv) → Result<Value, PanicError>`. No mutation. Envs are extended by creating child envs:

```rust
fn interpret(
    expr: &ast::Expr,
    env: &Arc<Env>,
    comptime: &ComptimeEnv,
) -> Result<Value, PanicError> {
    match expr {
        Expr::Literal(lit) => Ok(literal_to_value(lit)),

        Expr::Name(n) => env.lookup(&n.name)
            .cloned()
            .ok_or_else(|| panic_err("undefined variable")),

        Expr::Apply(app) => {
            let arg_vals: Vec<Value> = app.args.iter()
                .map(|a| interpret(a, env, comptime))
                .collect::<Result<_, _>>()?;
            let func_name = match &*app.callee {
                Expr::Name(n) => &n.name,
                _ => return Err(panic_err("higher-order not yet supported")),
            };
            comptime.call(func_name, arg_vals)
        }

        Expr::If(if_expr) => {
            let cond = interpret(&if_expr.cond_branch_vec[0].0, env, comptime)?;
            if cond.as_bool()? {
                interpret(&if_expr.cond_branch_vec[0].1, env, comptime)
            } else {
                interpret(&if_expr.else_branch, env, comptime)
            }
        }

        Expr::Chain(chain) => {
            let mut current_env = Arc::clone(env);
            let mut last_val = None;
            for stmt in &chain.stmt_vec {
                match stmt {
                    Stmt::Let(let_) => {
                        let val = interpret(&let_.init, &current_env, comptime)?;
                        let name = match &let_.pattern {
                            Pattern::Name(n) => n.name.clone(),
                            _ => return Err(panic_err("unsupported pattern")),
                        };
                        current_env = Env::extend(&current_env, name, val);
                    }
                    Stmt::Discard(d) => {
                        last_val = Some(interpret(&d.val, &current_env, comptime)?);
                    }
                    _ => return Err(panic_err("unsupported statement")),
                }
            }
            last_val.ok_or_else(|| panic_err("empty chain"))
        }

        Expr::ArrayLit(arr) => {
            let vals: Vec<Value> = arr.elements.iter()
                .map(|e| interpret(e, env, comptime))
                .collect::<Result<_, _>>()?;
            Ok(Value::from_vec(vals))
        }

        _ => Err(panic_err("unsupported expression in comptime")),
    }
}
```

`current_env` in the `Chain` arm is a local rebinding, not mutation. Each `Env::extend` returns a new `Arc<Env>`.

### ComptimeEnv

```rust
struct ComptimeEnv {
    builtins: HashMap<Symbol, BuiltinFn>,
    compiled: HashMap<Symbol, CompiledFn>,
}

struct CompiledFn {
    params: Vec<Symbol>,
    body: Arc<ast::Expr>,
}

impl ComptimeEnv {
    fn call(&self, func: &Symbol, args: Vec<Value>) -> Result<Value, PanicError> {
        if let Some(builtin) = self.builtins.get(func) {
            return builtin(args);
        }
        if let Some(compiled) = self.compiled.get(func) {
            let arg_env = compiled.params.iter().zip(args)
                .fold(Env::empty(), |env, (name, val)| {
                    Env::extend(&env, name.clone(), val)
                });
            return interpret(&compiled.body, &arg_env, &Arc::new(self.clone()));
        }
        Err(panic_err(format!("unknown function: {func}")))
    }

    fn register(&self, name: Symbol, func: CompiledFn) -> Self {
        let mut compiled = self.compiled.clone();
        compiled.insert(name, func);
        ComptimeEnv { builtins: self.builtins.clone(), compiled }
    }
}
```

`register` returns a **new** `ComptimeEnv`. The old one is unchanged.

## TyCon Expansion (Dot Access)

Type constructors like `Linear T o i = { w: Ten T [o, i], b: Ten T [o] }` are opaque during unification — `TyCon("Linear", [f32, 10, 5])` unifies structurally. But dot access (`model.w`) requires **expanding** the TyCon to see its record structure.

### TyCon registry

```rust
struct TyConDef {
    params: Vec<Symbol>,     // [T, o, i]
    body: ast::Expr,         // the { w: Ten T [o, i], b: Ten T [o] } AST
}
```

The type checker maintains a registry alongside the `Scope` for functions. TyCon definitions are registered during the dependency-ordered fold, just like function schemes.

### Expansion during dot access

```
infer(ctx, scope, Dot(model, "w")):
    (model_ty, ctx) = infer(ctx, scope, model)

    match resolve(model_ty):
        Record { fields } → lookup "w" in fields
        TyCon { name, args } →
            def = tycon_registry.lookup(name)
            // Build scope: { T → Scalar(F32), o → Val(10), i → Val(5) }
            type_scope = zip(def.params, args)
            // Elaborate the body with those bindings
            record_ty = elaborate(def.body, type_scope, comptime)
            // Look up the field
            lookup "w" in record_ty.fields
```

This reuses the same `elaborate` function used for return types. No new machinery.

## Broadcasting

Binary ops like `+(_,_)` have type `Ten T s -> Ten T s -> Ten T s`, which requires both operands to have the same shape. But scalars (shape `[]`) broadcast to any shape.

Broadcasting is a **post-unification fixup** on binary operator applications:

```
infer_binop(ctx, scope, op, lhs, rhs):
    (lhs_ty, ctx) = infer(ctx, scope, lhs)
    (rhs_ty, ctx) = infer(ctx, scope, rhs)

    match (resolve(lhs_ty), resolve(rhs_ty)):
        (Tensor { elem: e1, shape: s1 }, Tensor { elem: e2, shape: s2 }) →
            ctx = ctx.unify(e1, e2)
            match (resolve(s1), resolve(s2)):
                (Val(shape), Val([])) → result_shape = shape  // rhs scalar broadcasts
                (Val([]), Val(shape)) → result_shape = shape  // lhs scalar broadcasts
                _ → ctx = ctx.unify(s1, s2); result_shape = resolve(s1)
            return (Tensor { elem: resolve(e1), shape: Val(result_shape) }, ctx)

        // Record broadcasting (scalar op record, record op record)
        (Scalar(..) | Tensor { shape: Val([]), .. }, Record { fields }) →
            // Apply op element-wise to each field
        (Record { fields: f1 }, Record { fields: f2 }) →
            // Apply op pairwise to matching fields
```

Record broadcasting (`lr * grads`, `model - scaled`) applies the operator recursively to each field of the record, bottoming out at tensor leaves.

## Grad

`grad f args...` differentiates `f` with respect to its first argument:

```
If f :: A -> B -> ... -> R
then grad f :: A -> B -> ... -> A
```

For type-checking, `grad` is a special form:

```
infer(ctx, scope, Grad(f, args)):
    (f_ty, ctx) = infer(ctx, scope, f)
    match resolve(f_ty):
        Fn { params, ret } →
            for (arg, param) in zip(args, params):
                (arg_ty, ctx) = infer(ctx, scope, arg)
                ctx = ctx.unify(arg_ty, param)
            return (resolve(params[0]), ctx)
```

The actual gradient computation is an IR-level concern. The type checker only needs to know the return type is always the type of the first argument.

## Name Resolution in Type Positions

In a type expression like `Ten T [o, i]`, elaboration distinguishes type variables from defined names by scope lookup:

**A bare name in type position is resolved by scope lookup. If not found, it becomes a fresh type variable.**

```
elaborate(ctx, scope, Name(n)):
    if n is a scalar keyword (f32, i64, bool, ...) → Scalar(...)
    if n is found in scope as a Type → return that Type
    else → create fresh TyVar, bind n to it in scope, return Var(...)
```

Function names like `matmul_shape` only appear as the callee of an `Apply` node and are looked up in the comptime environment.

## Interplay: Type-Checking and Interpretation

Interleaved **across** functions, sequential **within** each function:

```
╔═══════════════════════════════════════════════════╗
║  Process function "matmul_shape"                  ║
║                                                   ║
║  1. Type-check body:                              ║
║     concat, init, tail are in scope (builtins).   ║
║     Infer: matmul_shape :: Value → Value → Value  ║
║                                                   ║
║  2. Register in comptime env (returns new env):   ║
║     comptime' = comptime.register("matmul_shape") ║
║                                                   ║
╠═══════════════════════════════════════════════════╣
║  Process function "matmul"                        ║
║                                                   ║
║  1. Type-check body:                              ║
║     Elaborate param types → Type (with Vars)      ║
║     Unify with argument types → bind s1, s2       ║
║                                                   ║
║  2. Evaluate return type:                         ║
║     Interpret "Ten T (matmul_shape s1 s2)"        ║
║     using comptime' (which has matmul_shape)      ║
║     → returns concrete shape                      ║
║                                                   ║
║  3. Register in comptime env (returns new env):   ║
║     comptime'' = comptime'.register("matmul")     ║
║                                                   ║
╚═══════════════════════════════════════════════════╝
```

### Dependency ordering

Definitions are processed in dependency order using Tarjan's SCC algorithm:

1. **Build dependency graph**: Walk each definition's body and type signature, collecting all referenced names that match other top-level definitions.

2. **Find SCCs**: Tarjan's algorithm produces strongly connected components in reverse topological order (callees before callers).

3. **Process each SCC**:
   - **Non-recursive** (1 member, no self-call): type-check, then register in scope and comptime env.
   - **Self-recursive** (1 member, calls itself): add a placeholder scheme with fresh type variables to scope, type-check the body, then replace with the inferred scheme.
   - **Mutually recursive** (>1 member): add placeholder schemes for all members, type-check all together. Comptime calls within the same SCC are an error (function not yet compiled).

```rust
fn typecheck_file(file: &ast::File, scope, comptime) -> Result<...> {
    let (defs, edges) = build_dep_graph(file);
    let sccs = tarjan_scc(&defs, &edges);

    // Fold over SCCs, threading (scope, comptime_env)
    for scc in &sccs {
        if scc.members.len() == 1 && !scc.is_recursive {
            // Non-recursive: typecheck_def, register
            ...
        } else {
            // Recursive: add placeholders, typecheck, register
            ...
        }
    }
}
```

### The compilation pipeline

```
Source code
    │
    ├─ Lex → Tokens
    ├─ Parse → AST
    │
    ├─ Dependency analysis → topologically sorted SCCs
    │
    │  Fold over SCCs, threading (scope, comptime_env):
    │  ├─ Type-check (HM + interpret return types)
    │  ├─ Produce new scope with this function's Scheme
    │  └─ Produce new comptime_env with this function registered
    │
    ├─ All functions type-checked
    │
    └─ Code generation (CUDA / C / WGSL) from compiled forms
```

## Elaboration: AST type expressions → Type

```
elaborate(ctx, scope, Name("f32"))       → (Scalar(F32), ctx)
elaborate(ctx, scope, Name("T"))         → (Var(lookup_or_fresh("T")), ctx')
elaborate(ctx, scope, Literal(10))       → (Val(scalar_int(10)), ctx)
elaborate(ctx, scope, ArrayLit([a, b]))  → (Val(int_vec([...])), ctx)
elaborate(ctx, scope, Apply(Name("Ten"), [elem, shape]))
    → (Tensor { elem: ..., shape: ... }, ctx')
elaborate(ctx, scope, Apply(Name("Linear"), [T, o, i]))       // Uppercase → TyCon
    → (TyCon { name: "Linear", args: [...] }, ctx')
elaborate(ctx, scope, Apply(Name("matmul_shape"), [s1, s2]))  // Lowercase → comptime call
    → if args fully concrete: interpret, return (Val(result), ctx)
      else: CANNOT elaborate yet — deferred (stored as ret_ast in Scheme)
```

## Unification

Standard HM. No extensions.

```
Var(v) ~ t                          →  bind v = t (with occurs check)
Val(a) ~ Val(b)                     →  ok if a == b (tensor equality)
Scalar(a) ~ Scalar(b)              →  ok if a == b
Tensor(e1,s1) ~ Tensor(e2,s2)      →  unify(e1,e2), unify(s1,s2)
Record(fs1) ~ Record(fs2)          →  same fields, pairwise unify
Fn(p1,r1) ~ Fn(p2,r2)              →  pairwise params, then ret
TyCon(f,as) ~ TyCon(f,bs)          →  same name, pairwise args
otherwise                            →  type error
```
