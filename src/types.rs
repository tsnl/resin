use std::fmt;
use std::sync::Arc;

use hashbrown::{HashMap, HashSet};

use crate::Symbol;
use crate::vocab::ScalarType;

// ---------------------------------------------------------------------------
// Type variables
// ---------------------------------------------------------------------------

/// A unification metavariable, identified by a unique integer.
/// Fresh TyVars are allocated by `TyCtx::fresh_var`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TyVar(pub u32);

// ---------------------------------------------------------------------------
// Types (normal forms only — no unevaluated terms)
// ---------------------------------------------------------------------------

/// The core type representation. Only contains normal forms — no unevaluated
/// function calls or pending computations. To construct a `Type`, any comptime
/// calls must be evaluated first (see `elaborate_type` in typer.rs).
#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    /// Unification metavariable, resolved by `TyCtx::resolve`.
    Var(TyVar),
    /// A concrete comptime value embedded in a type (e.g., a shape like `[10, 5]`).
    Val(Value),
    /// Scalar element type tag: `f32`, `i64`, `bool`, etc.
    Scalar(ScalarType),
    /// Tensor type: `Ten f32 [10, 5]` → `Tensor { elem: Scalar(F32), shape: Val([10,5]) }`.
    Tensor { elem: Box<Type>, shape: Box<Type> },
    /// Record type: `{ w: Ten f32 [10], b: Ten f32 [5] }`.
    Record { fields: Vec<(Symbol, Type)> },
    /// Function type: `(params) -> ret`.
    Fn { params: Vec<Type>, ret: Box<Type> },
    /// Opaque type constructor application: `Linear f32 10 5`.
    /// Unified structurally (same name + pairwise args). Expanded via
    /// `Scope::lookup_tycon` during dot access.
    TyCon { name: Symbol, args: Vec<Type> },
}

impl Type {
    pub fn scalar(st: ScalarType) -> Self {
        Type::Tensor {
            elem: Box::new(Type::Scalar(st)),
            shape: Box::new(Type::Val(Value::int_vec(vec![]))),
        }
    }

    pub fn free_vars(&self) -> HashSet<TyVar> {
        let mut vars = HashSet::new();
        self.collect_free_vars(&mut vars);
        vars
    }

    fn collect_free_vars(&self, out: &mut HashSet<TyVar>) {
        match self {
            Type::Var(v) => {
                out.insert(*v);
            }
            Type::Tensor { elem, shape } => {
                elem.collect_free_vars(out);
                shape.collect_free_vars(out);
            }
            Type::Record { fields } => {
                for (_, ty) in fields {
                    ty.collect_free_vars(out);
                }
            }
            Type::Fn { params, ret } => {
                for p in params {
                    p.collect_free_vars(out);
                }
                ret.collect_free_vars(out);
            }
            Type::TyCon { args, .. } => {
                for a in args {
                    a.collect_free_vars(out);
                }
            }
            Type::Val(_) | Type::Scalar(_) => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Values (comptime tensor/record data)
// ---------------------------------------------------------------------------

/// A compile-time value. All comptime data is either a tensor, a record of
/// values, or a string (for panic messages). Tensor values follow NumPy
/// conventions: scalars are 0-dim (`shape: []`), shapes are 1-dim int tensors.
#[derive(Debug, Clone)]
pub enum Value {
    Tensor(TensorVal),
    Record(Vec<(Symbol, Value)>),
    String(String),
}

/// A dense tensor value with typed element data and a shape.
#[derive(Debug, Clone)]
pub struct TensorVal {
    pub data: TensorData,
    /// Tensor dimensions. `[]` = scalar, `[n]` = vector, `[m,n]` = matrix.
    pub shape: Vec<usize>,
}

/// Element storage for a tensor value.
#[derive(Debug, Clone)]
pub enum TensorData {
    Int(Vec<i64>),
    Float(Vec<f64>),
    Bool(Vec<bool>),
}

impl Value {
    pub fn scalar_int(n: i64) -> Self {
        Value::Tensor(TensorVal {
            data: TensorData::Int(vec![n]),
            shape: vec![],
        })
    }

    pub fn int_vec(ns: Vec<i64>) -> Self {
        let len = ns.len();
        Value::Tensor(TensorVal {
            data: TensorData::Int(ns),
            shape: vec![len],
        })
    }

    pub fn scalar_float(n: f64) -> Self {
        Value::Tensor(TensorVal {
            data: TensorData::Float(vec![n]),
            shape: vec![],
        })
    }

    pub fn scalar_bool(b: bool) -> Self {
        Value::Tensor(TensorVal {
            data: TensorData::Bool(vec![b]),
            shape: vec![],
        })
    }

    pub fn as_int(&self) -> Option<i64> {
        match self {
            Value::Tensor(t) => match (&t.data, t.shape.as_slice()) {
                (TensorData::Int(v), []) if v.len() == 1 => Some(v[0]),
                _ => None,
            },
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Tensor(t) => match (&t.data, t.shape.as_slice()) {
                (TensorData::Bool(v), []) if v.len() == 1 => Some(v[0]),
                _ => None,
            },
            _ => None,
        }
    }

    pub fn as_int_slice(&self) -> Option<&[i64]> {
        match self {
            Value::Tensor(t) => match &t.data {
                TensorData::Int(v) => Some(v.as_slice()),
                _ => None,
            },
            _ => None,
        }
    }

    pub fn tensor_shape(&self) -> Option<&[usize]> {
        match self {
            Value::Tensor(t) => Some(&t.shape),
            _ => None,
        }
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Value::Tensor(a), Value::Tensor(b)) => a == b,
            (Value::Record(a), Value::Record(b)) => {
                a.len() == b.len()
                    && a.iter()
                        .zip(b.iter())
                        .all(|((na, va), (nb, vb))| na == nb && va == vb)
            }
            (Value::String(a), Value::String(b)) => a == b,
            _ => false,
        }
    }
}

impl PartialEq for TensorVal {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape && self.data == other.data
    }
}

impl PartialEq for TensorData {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (TensorData::Int(a), TensorData::Int(b)) => a == b,
            (TensorData::Float(a), TensorData::Float(b)) => {
                a.len() == b.len()
                    && a.iter()
                        .zip(b.iter())
                        .all(|(x, y)| x.to_bits() == y.to_bits())
            }
            (TensorData::Bool(a), TensorData::Bool(b)) => a == b,
            _ => false,
        }
    }
}

// ---------------------------------------------------------------------------
// Type schemes
// ---------------------------------------------------------------------------

/// How a function's return type is represented in its scheme.
///
/// `Concrete`: the return type is fully elaborated (no comptime calls).
/// Most signatures use this — e.g., `relu :: Ten T s -> Ten T s`.
///
/// `Deferred`: the return type contains comptime function calls with
/// unresolved arguments, stored as an unevaluated AST expression.
/// Evaluated at each call site after unification resolves the variables.
/// E.g., `matmul :: ... -> Ten T (matmul_shape s1 s2)`.
#[derive(Debug, Clone)]
pub enum ReturnType {
    Concrete(Type),
    Deferred(Arc<crate::ast::Expr>),
}

/// A polymorphic type scheme: `forall bound. params -> ret`.
///
/// `bound` lists the quantified type variable names and their TyVars.
/// At each use site, `TyCtx::instantiate` replaces bound vars with fresh ones.
/// Monomorphic bindings (local variables) have `bound: [], params: []`.
#[derive(Debug, Clone)]
pub struct Scheme {
    pub bound: Vec<(Symbol, TyVar)>,
    pub params: Vec<Type>,
    pub ret: ReturnType,
}

impl Scheme {
    pub fn mono_fn(params: Vec<Type>, ret: Type) -> Self {
        Scheme {
            bound: vec![],
            params,
            ret: ReturnType::Concrete(ret),
        }
    }
}

// ---------------------------------------------------------------------------
// Substitution (immutable cons list)
// ---------------------------------------------------------------------------

/// Immutable substitution mapping TyVars to Types, implemented as a cons list.
///
/// `bind` prepends a new binding (O(1)), `lookup` walks the chain (O(n)).
/// Newer bindings shadow older ones — no `compose` needed. The chain IS
/// the composition.
#[derive(Clone)]
pub enum Substitution {
    Empty,
    Bind {
        var: TyVar,
        ty: Type,
        parent: Arc<Substitution>,
    },
}

impl Substitution {
    pub fn empty() -> Self {
        Substitution::Empty
    }

    pub fn bind(&self, var: TyVar, ty: Type) -> Self {
        Substitution::Bind {
            var,
            ty,
            parent: Arc::new(self.clone()),
        }
    }

    pub fn lookup(&self, v: TyVar) -> Option<&Type> {
        match self {
            Substitution::Empty => None,
            Substitution::Bind { var, ty, parent } => {
                if *var == v {
                    Some(ty)
                } else {
                    parent.lookup(v)
                }
            }
        }
    }

    pub fn apply(&self, ty: &Type) -> Type {
        self.apply_inner(ty, &mut HashSet::new())
    }

    fn apply_inner(&self, ty: &Type, expanding: &mut HashSet<TyVar>) -> Type {
        match ty {
            Type::Var(v) => match self.lookup(*v) {
                Some(t) => {
                    if !expanding.insert(*v) {
                        return ty.clone();
                    }
                    let result = self.apply_inner(t, expanding);
                    expanding.remove(v);
                    result
                }
                None => ty.clone(),
            },
            Type::Tensor { elem, shape } => Type::Tensor {
                elem: Box::new(self.apply_inner(elem, expanding)),
                shape: Box::new(self.apply_inner(shape, expanding)),
            },
            Type::Record { fields } => Type::Record {
                fields: fields
                    .iter()
                    .map(|(name, ty)| (name.clone(), self.apply_inner(ty, expanding)))
                    .collect(),
            },
            Type::Fn { params, ret } => Type::Fn {
                params: params
                    .iter()
                    .map(|p| self.apply_inner(p, expanding))
                    .collect(),
                ret: Box::new(self.apply_inner(ret, expanding)),
            },
            Type::TyCon { name, args } => Type::TyCon {
                name: name.clone(),
                args: args
                    .iter()
                    .map(|a| self.apply_inner(a, expanding))
                    .collect(),
            },
            Type::Val(_) | Type::Scalar(_) => ty.clone(),
        }
    }
}

impl fmt::Debug for Substitution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut entries = vec![];
        let mut current = self;
        loop {
            match current {
                Substitution::Empty => break,
                Substitution::Bind { var, ty, parent } => {
                    entries.push(format!("{var:?} → {ty:?}"));
                    current = parent;
                }
            }
        }
        write!(f, "Subst[{}]", entries.join(", "))
    }
}

// ---------------------------------------------------------------------------
// Type context (threaded, immutable)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct TypeError {
    pub message: String,
}

impl TypeError {
    pub fn new(msg: impl Into<String>) -> Self {
        TypeError {
            message: msg.into(),
        }
    }
}

impl fmt::Display for TypeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

/// Type-checking context, threaded immutably through inference.
///
/// Contains a fresh variable counter and the current substitution.
/// Every inference function takes `TyCtx` by value and returns a new one —
/// no mutation. `fresh_var` allocates a new TyVar, `unify` extends the
/// substitution, `resolve` applies the substitution to chase variable bindings.
#[derive(Clone)]
pub struct TyCtx {
    pub next_var: u32,
    pub subst: Substitution,
}

impl TyCtx {
    pub fn new() -> Self {
        TyCtx {
            next_var: 0,
            subst: Substitution::empty(),
        }
    }

    pub fn fresh_var(self) -> (Type, TyCtx) {
        let v = TyVar(self.next_var);
        (
            Type::Var(v),
            TyCtx {
                next_var: self.next_var + 1,
                subst: self.subst,
            },
        )
    }

    pub fn resolve(&self, ty: &Type) -> Type {
        self.subst.apply(ty)
    }

    pub fn unify(self, a: &Type, b: &Type) -> Result<TyCtx, TypeError> {
        let a = self.resolve(a);
        let b = self.resolve(b);
        let new_subst = unify_inner(&a, &b, &self.subst)?;
        Ok(TyCtx {
            next_var: self.next_var,
            subst: new_subst,
        })
    }

    pub fn instantiate(&mut self, scheme: &Scheme) -> (Vec<Type>, Option<Type>, Vec<(Symbol, TyVar)>) {
        let mut fresh_map = Vec::new();
        let mut subst_pairs = Vec::new();
        for (name, old_var) in &scheme.bound {
            let new_var = TyVar(self.next_var);
            self.next_var += 1;
            fresh_map.push((*old_var, Type::Var(new_var)));
            subst_pairs.push((name.clone(), new_var));
        }
        let rename = fresh_map
            .iter()
            .fold(Substitution::empty(), |s, (v, t)| s.bind(*v, t.clone()));
        let params = scheme.params.iter().map(|p| rename.apply(p)).collect();
        let ret = match &scheme.ret {
            ReturnType::Concrete(ty) => Some(rename.apply(ty)),
            ReturnType::Deferred(_) => None,
        };
        (params, ret, subst_pairs)
    }
}

// ---------------------------------------------------------------------------
// Unification
// ---------------------------------------------------------------------------

fn unify_inner(a: &Type, b: &Type, subst: &Substitution) -> Result<Substitution, TypeError> {
    match (a, b) {
        (Type::Var(v), t) | (t, Type::Var(v)) => {
            if let Type::Var(u) = t {
                if v == u {
                    return Ok(subst.clone());
                }
            }
            if t.free_vars().contains(v) {
                return Err(TypeError::new(format!("occurs check: {v:?} in {t:?}")));
            }
            Ok(subst.bind(*v, t.clone()))
        }

        (Type::Val(a), Type::Val(b)) => {
            if a == b {
                Ok(subst.clone())
            } else {
                Err(TypeError::new(format!("value mismatch: {a:?} vs {b:?}")))
            }
        }

        (Type::Scalar(a), Type::Scalar(b)) => {
            if a == b {
                Ok(subst.clone())
            } else {
                Err(TypeError::new(format!(
                    "scalar type mismatch: {a:?} vs {b:?}"
                )))
            }
        }

        (
            Type::Tensor {
                elem: e1,
                shape: s1,
            },
            Type::Tensor {
                elem: e2,
                shape: s2,
            },
        ) => {
            let subst = unify_inner(e1, e2, subst)?;
            let s1 = subst.apply(s1);
            let s2 = subst.apply(s2);
            unify_inner(&s1, &s2, &subst)
        }

        (Type::Record { fields: f1 }, Type::Record { fields: f2 }) => {
            if f1.len() != f2.len() {
                return Err(TypeError::new("record field count mismatch"));
            }
            let mut subst = subst.clone();
            for ((n1, t1), (n2, t2)) in f1.iter().zip(f2.iter()) {
                if n1 != n2 {
                    return Err(TypeError::new(format!(
                        "record field name mismatch: {n1} vs {n2}"
                    )));
                }
                let t1 = subst.apply(t1);
                let t2 = subst.apply(t2);
                subst = unify_inner(&t1, &t2, &subst)?;
            }
            Ok(subst)
        }

        (
            Type::Fn {
                params: p1,
                ret: r1,
            },
            Type::Fn {
                params: p2,
                ret: r2,
            },
        ) => {
            if p1.len() != p2.len() {
                return Err(TypeError::new(format!(
                    "function arity mismatch: {} vs {}",
                    p1.len(),
                    p2.len()
                )));
            }
            let mut subst = subst.clone();
            for (a, b) in p1.iter().zip(p2.iter()) {
                let a = subst.apply(a);
                let b = subst.apply(b);
                subst = unify_inner(&a, &b, &subst)?;
            }
            let r1 = subst.apply(r1);
            let r2 = subst.apply(r2);
            unify_inner(&r1, &r2, &subst)
        }

        (Type::TyCon { name: n1, args: a1 }, Type::TyCon { name: n2, args: a2 }) => {
            if n1 != n2 {
                return Err(TypeError::new(format!(
                    "type constructor mismatch: {n1} vs {n2}"
                )));
            }
            if a1.len() != a2.len() {
                return Err(TypeError::new(format!(
                    "type argument count mismatch for {n1}"
                )));
            }
            let mut subst = subst.clone();
            for (a, b) in a1.iter().zip(a2.iter()) {
                let a = subst.apply(a);
                let b = subst.apply(b);
                subst = unify_inner(&a, &b, &subst)?;
            }
            Ok(subst)
        }

        _ => Err(TypeError::new(format!("cannot unify {a:?} with {b:?}"))),
    }
}

// ---------------------------------------------------------------------------
// Scope (type checker: Symbol → Binding)
// ---------------------------------------------------------------------------

/// What a name is bound to in the type-checker scope.
///
/// `Value(Scheme)`: a function or variable with a type scheme.
/// `TyCon(TyConDef, Scheme)`: a type constructor definition (e.g., `Linear`)
/// paired with its type scheme. The TyConDef is used for dot-access expansion;
/// the Scheme is used for value-level type inference.
#[derive(Debug, Clone)]
pub enum Binding {
    Value(Scheme),
    TyCon(TyConDef, Scheme),
}

/// A type constructor definition: named parameters + body AST.
/// E.g., `Linear T o i = { w: Ten T [o, i], b: Ten T [o] }` produces
/// `TyConDef { params: [T, o, i], body: <the record Ctor AST> }`.
/// Used during dot access to expand `model.w` when `model` has type
/// `TyCon("Linear", [f32, 10, 5])`.
#[derive(Debug, Clone)]
pub struct TyConDef {
    pub params: Vec<Symbol>,
    pub body: crate::ast::Expr,
}

/// Persistent lexical scope mapping names to `Binding`s (type schemes or TyCon defs).
///
/// Implemented as a parent-chain: each node holds a small set of bindings plus
/// an `Arc` to its parent. `extend` creates a new child scope (O(1)), `lookup`
/// walks the chain (O(depth)).
///
/// Use `lookup_value` to get a name's type scheme (works for both `Value` and
/// `TyCon` bindings). Use `lookup_tycon` to get the TyCon definition for dot
/// access expansion.
pub struct Scope {
    bindings: HashMap<Symbol, Binding>,
    parent: Option<Arc<Scope>>,
}

impl Scope {
    pub fn root(bindings: HashMap<Symbol, Binding>) -> Arc<Self> {
        Arc::new(Scope {
            bindings,
            parent: None,
        })
    }

    pub fn empty() -> Arc<Self> {
        Self::root(HashMap::new())
    }

    pub fn extend(parent: &Arc<Self>, name: Symbol, val: Binding) -> Arc<Self> {
        Arc::new(Scope {
            bindings: HashMap::from_iter([(name, val)]),
            parent: Some(Arc::clone(parent)),
        })
    }

    pub fn extend_many(parent: &Arc<Self>, bindings: HashMap<Symbol, Binding>) -> Arc<Self> {
        Arc::new(Scope {
            bindings,
            parent: Some(Arc::clone(parent)),
        })
    }

    pub fn lookup(&self, name: &Symbol) -> Option<&Binding> {
        self.bindings
            .get(name)
            .or_else(|| self.parent.as_ref().and_then(|p| p.lookup(name)))
    }

    pub fn lookup_value(&self, name: &Symbol) -> Option<&Scheme> {
        match self.lookup(name) {
            Some(Binding::Value(s)) => Some(s),
            Some(Binding::TyCon(_, s)) => Some(s),
            None => None,
        }
    }

    pub fn lookup_tycon(&self, name: &Symbol) -> Option<&TyConDef> {
        match self.lookup(name) {
            Some(Binding::TyCon(d, _)) => Some(d),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_substitution_bind_and_lookup() {
        let s = Substitution::empty();
        let v0 = TyVar(0);
        let s = s.bind(v0, Type::Scalar(ScalarType::F32));
        assert_eq!(s.apply(&Type::Var(v0)), Type::Scalar(ScalarType::F32));
    }

    #[test]
    fn test_substitution_chaining() {
        let v0 = TyVar(0);
        let v1 = TyVar(1);
        let s = Substitution::empty();
        let s = s.bind(v0, Type::Var(v1));
        let s = s.bind(v1, Type::Scalar(ScalarType::I32));
        // v0 → v1 → I32
        assert_eq!(s.apply(&Type::Var(v0)), Type::Scalar(ScalarType::I32));
        assert_eq!(s.apply(&Type::Var(v1)), Type::Scalar(ScalarType::I32));
    }

    #[test]
    fn test_unify_vars() {
        let ctx = TyCtx::new();
        let (a, ctx) = ctx.fresh_var();
        let b = Type::Scalar(ScalarType::F32);
        let ctx = ctx.unify(&a, &b).unwrap();
        assert_eq!(ctx.resolve(&a), Type::Scalar(ScalarType::F32));
    }

    #[test]
    fn test_unify_tensors() {
        let ctx = TyCtx::new();
        let (a, ctx) = ctx.fresh_var();
        let t1 = Type::Tensor {
            elem: Box::new(a.clone()),
            shape: Box::new(Type::Val(Value::int_vec(vec![10]))),
        };
        let t2 = Type::Tensor {
            elem: Box::new(Type::Scalar(ScalarType::F64)),
            shape: Box::new(Type::Val(Value::int_vec(vec![10]))),
        };
        let ctx = ctx.unify(&t1, &t2).unwrap();
        assert_eq!(ctx.resolve(&a), Type::Scalar(ScalarType::F64));
    }

    #[test]
    fn test_unify_occurs_check() {
        let ctx = TyCtx::new();
        let (a, ctx) = ctx.fresh_var();
        let recursive = Type::Tensor {
            elem: Box::new(a.clone()),
            shape: Box::new(a.clone()),
        };
        assert!(ctx.unify(&a, &recursive).is_err());
    }

    #[test]
    fn test_unify_values() {
        let ctx = TyCtx::new();
        let a = Type::Val(Value::int_vec(vec![10, 5]));
        let b = Type::Val(Value::int_vec(vec![10, 5]));
        assert!(ctx.unify(&a, &b).is_ok());

        let ctx = TyCtx::new();
        let a = Type::Val(Value::int_vec(vec![10, 5]));
        let b = Type::Val(Value::int_vec(vec![10, 3]));
        assert!(ctx.unify(&a, &b).is_err());
    }

    #[test]
    fn test_scope_lookup() {
        let dummy_binding =
            || Binding::Value(Scheme::mono_fn(vec![], Type::Scalar(ScalarType::Bool)));

        let root = Scope::root(HashMap::from_iter([(Symbol::from("x"), dummy_binding())]));
        assert!(root.lookup(&Symbol::from("x")).is_some());
        assert!(root.lookup(&Symbol::from("y")).is_none());

        let child = Scope::extend(&root, Symbol::from("y"), dummy_binding());
        assert!(child.lookup(&Symbol::from("x")).is_some());
        assert!(child.lookup(&Symbol::from("y")).is_some());
    }

    #[test]
    fn test_value_constructors() {
        let s = Value::scalar_int(42);
        assert_eq!(s.as_int(), Some(42));
        assert_eq!(s.tensor_shape(), Some([].as_slice()));

        let v = Value::int_vec(vec![10, 5, 3]);
        assert_eq!(v.as_int_slice(), Some([10, 5, 3].as_slice()));
        assert_eq!(v.tensor_shape(), Some([3].as_slice()));

        let b = Value::scalar_bool(true);
        assert_eq!(b.as_bool(), Some(true));
    }
}
