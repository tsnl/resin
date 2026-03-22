use hashbrown::{HashMap, HashSet};

use crate::Symbol;
use crate::vocab::ScalarType;

// ---------------------------------------------------------------------------
// Type variables
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TyVar(pub u32);

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Core type representation.
///
/// Scalars are 0-dimensional tensors: `Tensor { elem, dims: [] }`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Ty {
    /// Unification variable (fresh, to be resolved).
    Var(TyVar),
    /// Tensor with element type and shape dimensions.
    /// A scalar like `f32` is `Tensor { elem: Elem(F32), dims: [] }`.
    /// A vector `[n]f32` is `Tensor { elem: Elem(F32), dims: [Dim(n)] }`.
    Tensor { elem: Box<Ty>, dims: Vec<Ty> },
    /// Record type (struct explosion target).
    Record { fields: Vec<(Symbol, Ty)> },
    /// Function type.
    Fn { params: Vec<Ty>, ret: Box<Ty> },
    /// Named type constructor applied to args: `Linear f32 10 5`.
    App { name: Symbol, args: Vec<Ty> },
    /// Dimension literal (for shape positions).
    Dim(u64),
    /// Scalar element type (e.g. f32, i64, bool).
    Scalar(ScalarType),
}

impl Ty {
    pub fn scalar(st: ScalarType) -> Self {
        Ty::Tensor {
            elem: Box::new(Ty::Scalar(st)),
            dims: vec![],
        }
    }

    /// Collect all free type variables in this type.
    pub fn free_vars(&self) -> HashSet<TyVar> {
        let mut vars = HashSet::new();
        self.collect_free_vars(&mut vars);
        vars
    }

    fn collect_free_vars(&self, out: &mut HashSet<TyVar>) {
        match self {
            Ty::Var(v) => {
                out.insert(*v);
            }
            Ty::Tensor { elem, dims } => {
                elem.collect_free_vars(out);
                for d in dims {
                    d.collect_free_vars(out);
                }
            }
            Ty::Record { fields } => {
                for (_, ty) in fields {
                    ty.collect_free_vars(out);
                }
            }
            Ty::Fn { params, ret } => {
                for p in params {
                    p.collect_free_vars(out);
                }
                ret.collect_free_vars(out);
            }
            Ty::App { args, .. } => {
                for a in args {
                    a.collect_free_vars(out);
                }
            }
            Ty::Dim(_) | Ty::Scalar(_) => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Type schemes
// ---------------------------------------------------------------------------

/// A type scheme: ∀ bound. ty
///
/// Every binding's type is a scheme. Monomorphic bindings have `bound = []`.
#[derive(Debug, Clone)]
pub struct Scheme {
    pub bound: Vec<TyVar>,
    pub ty: Ty,
}

impl Scheme {
    /// A monomorphic scheme (no quantified variables).
    pub fn mono(ty: Ty) -> Self {
        Scheme { bound: vec![], ty }
    }

    /// Collect free type variables (those in ty but not in bound).
    pub fn free_vars(&self) -> HashSet<TyVar> {
        let mut fv = self.ty.free_vars();
        for b in &self.bound {
            fv.remove(b);
        }
        fv
    }
}

// ---------------------------------------------------------------------------
// Substitution
// ---------------------------------------------------------------------------

/// An explicit substitution mapping type variables to types.
#[derive(Debug, Clone, Default)]
pub struct Substitution {
    map: HashMap<TyVar, Ty>,
}

impl Substitution {
    pub fn empty() -> Self {
        Self::default()
    }

    pub fn singleton(var: TyVar, ty: Ty) -> Self {
        let mut map = HashMap::new();
        map.insert(var, ty);
        Substitution { map }
    }

    pub fn from_map(map: HashMap<TyVar, Ty>) -> Self {
        Substitution { map }
    }

    /// Apply this substitution to a type.
    pub fn apply(&self, ty: &Ty) -> Ty {
        self.apply_inner(ty, &mut HashSet::new())
    }

    fn apply_inner(&self, ty: &Ty, expanding: &mut HashSet<TyVar>) -> Ty {
        match ty {
            Ty::Var(v) => match self.map.get(v) {
                Some(t) => {
                    // Guard against infinite recursion from cyclic substitutions
                    if !expanding.insert(*v) {
                        return ty.clone();
                    }
                    let result = self.apply_inner(t, expanding);
                    expanding.remove(v);
                    result
                }
                None => ty.clone(),
            },
            Ty::Tensor { elem, dims } => Ty::Tensor {
                elem: Box::new(self.apply_inner(elem, expanding)),
                dims: dims
                    .iter()
                    .map(|d| self.apply_inner(d, expanding))
                    .collect(),
            },
            Ty::Record { fields } => Ty::Record {
                fields: fields
                    .iter()
                    .map(|(name, ty)| (name.clone(), self.apply_inner(ty, expanding)))
                    .collect(),
            },
            Ty::Fn { params, ret } => Ty::Fn {
                params: params
                    .iter()
                    .map(|p| self.apply_inner(p, expanding))
                    .collect(),
                ret: Box::new(self.apply_inner(ret, expanding)),
            },
            Ty::App { name, args } => Ty::App {
                name: name.clone(),
                args: args
                    .iter()
                    .map(|a| self.apply_inner(a, expanding))
                    .collect(),
            },
            Ty::Dim(_) | Ty::Scalar(_) => ty.clone(),
        }
    }

    /// Apply to a scheme (only substitutes free vars, not bound vars).
    pub fn apply_scheme(&self, scheme: &Scheme) -> Scheme {
        let mut restricted = self.clone();
        for b in &scheme.bound {
            restricted.map.remove(b);
        }
        Scheme {
            bound: scheme.bound.clone(),
            ty: restricted.apply(&scheme.ty),
        }
    }

    /// Compose: `self ∘ other`.
    /// `(self ∘ other)(t) = self(other(t))`
    pub fn compose(&self, other: &Substitution) -> Substitution {
        let mut result: HashMap<TyVar, Ty> =
            other.map.iter().map(|(v, t)| (*v, self.apply(t))).collect();
        for (v, t) in &self.map {
            result.entry(*v).or_insert_with(|| t.clone());
        }
        Substitution { map: result }
    }
}

// ---------------------------------------------------------------------------
// Unification context
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

pub struct TyCtx {
    pub(crate) next_var: u32,
    pub subst: Substitution,
}

impl TyCtx {
    pub fn new() -> Self {
        TyCtx {
            next_var: 0,
            subst: Substitution::empty(),
        }
    }

    /// Allocate a fresh unification variable.
    pub fn fresh_var(&mut self) -> Ty {
        let v = TyVar(self.next_var);
        self.next_var += 1;
        Ty::Var(v)
    }

    /// Instantiate a scheme: replace each bound var with a fresh unification var.
    pub fn instantiate(&mut self, scheme: &Scheme) -> Ty {
        let fresh_subst: HashMap<TyVar, Ty> = scheme
            .bound
            .iter()
            .map(|v| (*v, self.fresh_var()))
            .collect();
        let s = Substitution { map: fresh_subst };
        s.apply(&scheme.ty)
    }

    /// Fully resolve a type under the current substitution.
    pub fn resolve(&self, ty: &Ty) -> Ty {
        self.subst.apply(ty)
    }

    /// Generalize a type into a scheme by quantifying over all free vars
    /// not present in `env_free_vars`.
    pub fn generalize(&self, ty: &Ty, env_free_vars: &HashSet<TyVar>) -> Scheme {
        let resolved = self.resolve(ty);
        let ty_fv = resolved.free_vars();
        let bound: Vec<TyVar> = ty_fv.difference(env_free_vars).copied().collect();
        Scheme {
            bound,
            ty: resolved,
        }
    }

    /// Unify two types, updating the substitution.
    pub fn unify(&mut self, a: &Ty, b: &Ty) -> Result<(), TypeError> {
        let a = self.resolve(a);
        let b = self.resolve(b);
        let s = unify_inner(&a, &b)?;
        self.subst = s.compose(&self.subst);
        Ok(())
    }
}

/// Compute the most general unifier of two types.
fn unify_inner(a: &Ty, b: &Ty) -> Result<Substitution, TypeError> {
    match (a, b) {
        (Ty::Var(v), t) | (t, Ty::Var(v)) => {
            if let Ty::Var(u) = t {
                if v == u {
                    return Ok(Substitution::empty());
                }
            }
            // Occurs check
            if t.free_vars().contains(v) {
                return Err(TypeError::new(format!("occurs check: {v:?} in {t:?}")));
            }
            Ok(Substitution::singleton(*v, t.clone()))
        }

        (Ty::Scalar(a), Ty::Scalar(b)) => {
            if a == b {
                Ok(Substitution::empty())
            } else {
                Err(TypeError::new(format!(
                    "scalar type mismatch: {a:?} vs {b:?}"
                )))
            }
        }

        (Ty::Dim(a), Ty::Dim(b)) => {
            if a == b {
                Ok(Substitution::empty())
            } else {
                Err(TypeError::new(format!("dimension mismatch: {a} vs {b}")))
            }
        }

        (Ty::Tensor { elem: e1, dims: d1 }, Ty::Tensor { elem: e2, dims: d2 }) => {
            if d1.len() != d2.len() {
                return Err(TypeError::new(format!(
                    "rank mismatch: {} vs {}",
                    d1.len(),
                    d2.len()
                )));
            }
            let mut s = unify_inner(e1, e2)?;
            for (a, b) in d1.iter().zip(d2.iter()) {
                let a = s.apply(a);
                let b = s.apply(b);
                let s2 = unify_inner(&a, &b)?;
                s = s2.compose(&s);
            }
            Ok(s)
        }

        (Ty::Record { fields: f1 }, Ty::Record { fields: f2 }) => {
            if f1.len() != f2.len() {
                return Err(TypeError::new("record field count mismatch"));
            }
            let mut s = Substitution::empty();
            for ((n1, t1), (n2, t2)) in f1.iter().zip(f2.iter()) {
                if n1 != n2 {
                    return Err(TypeError::new(format!(
                        "record field name mismatch: {n1} vs {n2}"
                    )));
                }
                let t1 = s.apply(t1);
                let t2 = s.apply(t2);
                let s2 = unify_inner(&t1, &t2)?;
                s = s2.compose(&s);
            }
            Ok(s)
        }

        (
            Ty::Fn {
                params: p1,
                ret: r1,
            },
            Ty::Fn {
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
            let mut s = Substitution::empty();
            for (a, b) in p1.iter().zip(p2.iter()) {
                let a = s.apply(a);
                let b = s.apply(b);
                let s2 = unify_inner(&a, &b)?;
                s = s2.compose(&s);
            }
            let r1 = s.apply(r1);
            let r2 = s.apply(r2);
            let s2 = unify_inner(&r1, &r2)?;
            Ok(s2.compose(&s))
        }

        (Ty::App { name: n1, args: a1 }, Ty::App { name: n2, args: a2 }) => {
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
            let mut s = Substitution::empty();
            for (a, b) in a1.iter().zip(a2.iter()) {
                let a = s.apply(a);
                let b = s.apply(b);
                let s2 = unify_inner(&a, &b)?;
                s = s2.compose(&s);
            }
            Ok(s)
        }

        _ => Err(TypeError::new(format!("cannot unify {a:?} with {b:?}"))),
    }
}

// ---------------------------------------------------------------------------
// Slot shape (derived from resolved types for struct explosion)
// ---------------------------------------------------------------------------

/// How a resolved type decomposes into flat IR slots (tensors).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SlotShape {
    Tensor,
    Record(Vec<(Symbol, SlotShape)>),
}

impl SlotShape {
    /// Total number of leaf tensor slots.
    pub fn total_slots(&self) -> u32 {
        match self {
            SlotShape::Tensor => 1,
            SlotShape::Record(fields) => fields.iter().map(|(_, s)| s.total_slots()).sum(),
        }
    }

    /// Derive slot shape from a resolved type.
    pub fn from_ty(ty: &Ty, ctx: &TyCtx) -> Self {
        let resolved = ctx.resolve(ty);
        match &resolved {
            Ty::Record { fields } => SlotShape::Record(
                fields
                    .iter()
                    .map(|(name, ty)| (name.clone(), SlotShape::from_ty(ty, ctx)))
                    .collect(),
            ),
            _ => SlotShape::Tensor,
        }
    }
}

// ---------------------------------------------------------------------------
// Immutable lexical scopes
// ---------------------------------------------------------------------------

/// Immutable lexical scope with parent chain.
///
/// To add bindings, create a child scope. Lookup walks up the parent chain.
/// Children are stack-allocated during recursive descent — no heap allocation.
pub struct Scope<'a, V> {
    bindings: HashMap<Symbol, V>,
    parent: Option<&'a Scope<'a, V>>,
}

impl<'a, V> Scope<'a, V> {
    /// Create a root scope (no parent).
    pub fn root(bindings: HashMap<Symbol, V>) -> Self {
        Scope {
            bindings,
            parent: None,
        }
    }

    /// Create a child scope stacked on a parent.
    pub fn child(parent: &'a Scope<'a, V>, bindings: HashMap<Symbol, V>) -> Self {
        Scope {
            bindings,
            parent: Some(parent),
        }
    }

    /// Look up a name, walking up the parent chain.
    pub fn lookup(&self, name: &Symbol) -> Option<&V> {
        self.bindings
            .get(name)
            .or_else(|| self.parent.and_then(|p| p.lookup(name)))
    }
}

impl<'a> Scope<'a, Scheme> {
    /// Collect free type variables from all bindings in scope (walks parent chain).
    pub fn free_vars(&self) -> HashSet<TyVar> {
        let mut fv: HashSet<TyVar> = self.bindings.values().flat_map(|s| s.free_vars()).collect();
        if let Some(parent) = self.parent {
            fv.extend(parent.free_vars());
        }
        fv
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_substitution_apply() {
        let v0 = TyVar(0);
        let s = Substitution::singleton(v0, Ty::Scalar(ScalarType::F32));
        let ty = Ty::Var(v0);
        assert_eq!(s.apply(&ty), Ty::Scalar(ScalarType::F32));
    }

    #[test]
    fn test_substitution_compose() {
        let v0 = TyVar(0);
        let v1 = TyVar(1);
        let s1 = Substitution::singleton(v0, Ty::Var(v1));
        let s2 = Substitution::singleton(v1, Ty::Scalar(ScalarType::I32));
        let composed = s2.compose(&s1);
        // composed(v0) = s2(s1(v0)) = s2(v1) = I32
        assert_eq!(composed.apply(&Ty::Var(v0)), Ty::Scalar(ScalarType::I32));
        // composed(v1) = I32
        assert_eq!(composed.apply(&Ty::Var(v1)), Ty::Scalar(ScalarType::I32));
    }

    #[test]
    fn test_unify_vars() {
        let mut ctx = TyCtx::new();
        let a = ctx.fresh_var();
        let b = Ty::Scalar(ScalarType::F32);
        ctx.unify(&a, &b).unwrap();
        assert_eq!(ctx.resolve(&a), Ty::Scalar(ScalarType::F32));
    }

    #[test]
    fn test_unify_tensors() {
        let mut ctx = TyCtx::new();
        let a = ctx.fresh_var();
        let t1 = Ty::Tensor {
            elem: Box::new(a.clone()),
            dims: vec![Ty::Dim(10)],
        };
        let t2 = Ty::Tensor {
            elem: Box::new(Ty::Scalar(ScalarType::F64)),
            dims: vec![Ty::Dim(10)],
        };
        ctx.unify(&t1, &t2).unwrap();
        assert_eq!(ctx.resolve(&a), Ty::Scalar(ScalarType::F64));
    }

    #[test]
    fn test_unify_occurs_check() {
        let mut ctx = TyCtx::new();
        let a = ctx.fresh_var();
        let recursive = Ty::Tensor {
            elem: Box::new(a.clone()),
            dims: vec![a.clone()],
        };
        assert!(ctx.unify(&a, &recursive).is_err());
    }

    #[test]
    fn test_scheme_instantiate() {
        let mut ctx = TyCtx::new();
        let v0 = TyVar(100);
        let scheme = Scheme {
            bound: vec![v0],
            ty: Ty::Fn {
                params: vec![Ty::Var(v0)],
                ret: Box::new(Ty::Var(v0)),
            },
        };
        let instantiated = ctx.instantiate(&scheme);
        // Should have fresh var, not v0
        match &instantiated {
            Ty::Fn { params, ret } => {
                match &params[0] {
                    Ty::Var(v) => assert_ne!(*v, v0),
                    _ => panic!("expected Var"),
                }
                // param and ret should be the same fresh var
                assert_eq!(params[0], **ret);
            }
            _ => panic!("expected Fn"),
        }
    }

    #[test]
    fn test_scope_lookup() {
        let root = Scope::root(HashMap::from_iter([(Symbol::from("x"), 1)]));
        assert_eq!(root.lookup(&Symbol::from("x")), Some(&1));
        assert_eq!(root.lookup(&Symbol::from("y")), None);

        let child = Scope::child(&root, HashMap::from_iter([(Symbol::from("y"), 2)]));
        assert_eq!(child.lookup(&Symbol::from("x")), Some(&1));
        assert_eq!(child.lookup(&Symbol::from("y")), Some(&2));
    }

    #[test]
    fn test_scope_shadowing() {
        let root = Scope::root(HashMap::from_iter([(Symbol::from("x"), 1)]));
        let child = Scope::child(&root, HashMap::from_iter([(Symbol::from("x"), 2)]));
        assert_eq!(child.lookup(&Symbol::from("x")), Some(&2));
        // Parent unchanged
        assert_eq!(root.lookup(&Symbol::from("x")), Some(&1));
    }

    #[test]
    fn test_slot_shape() {
        let ctx = TyCtx::new();
        // Tensor → 1 slot
        let tensor_ty = Ty::scalar(ScalarType::F32);
        assert_eq!(SlotShape::from_ty(&tensor_ty, &ctx), SlotShape::Tensor);
        assert_eq!(SlotShape::Tensor.total_slots(), 1);

        // Record with 2 tensor fields → 2 slots
        let record_ty = Ty::Record {
            fields: vec![
                (Symbol::from("w"), Ty::scalar(ScalarType::F32)),
                (Symbol::from("b"), Ty::scalar(ScalarType::F32)),
            ],
        };
        let shape = SlotShape::from_ty(&record_ty, &ctx);
        assert_eq!(shape.total_slots(), 2);
    }
}
