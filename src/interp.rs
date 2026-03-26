use std::sync::Arc;

use hashbrown::HashMap;

use crate::Symbol;
use crate::ast::{Expr, Pattern, Stmt};
use crate::types::{Value, TensorVal, TensorData};
use crate::vocab;

// ---------------------------------------------------------------------------
// Env (interpreter: Symbol → Value)
// ---------------------------------------------------------------------------

/// Persistent lexical environment mapping names to `Value`s during interpretation.
///
/// Same parent-chain structure as `Scope`, but holds fn_reg values instead of
/// type schemes. Each `extend` creates a new child (O(1)), `lookup` walks the
/// chain (O(depth)).
pub struct Env {
    bindings: HashMap<Symbol, Value>,
    parent: Option<Arc<Env>>,
}

impl Env {
    pub fn root(bindings: HashMap<Symbol, Value>) -> Arc<Self> {
        Arc::new(Env {
            bindings,
            parent: None,
        })
    }

    pub fn empty() -> Arc<Self> {
        Self::root(HashMap::new())
    }

    pub fn extend(parent: &Arc<Self>, name: Symbol, val: Value) -> Arc<Self> {
        Arc::new(Env {
            bindings: HashMap::from_iter([(name, val)]),
            parent: Some(Arc::clone(parent)),
        })
    }

    pub fn lookup(&self, name: &Symbol) -> Option<&Value> {
        self.bindings
            .get(name)
            .or_else(|| self.parent.as_ref().and_then(|p| p.lookup(name)))
    }
}

// ---------------------------------------------------------------------------
// Panic error (fn_reg evaluation failure → type error)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
/// A fn_reg evaluation failure. When a fn_reg function panics (e.g.,
/// `matmul_shape` with incompatible dimensions), this surfaces as a type error
/// during type-checking.
pub struct PanicError {
    pub message: String,
}

impl PanicError {
    pub fn new(msg: impl Into<String>) -> Self {
        PanicError {
            message: msg.into(),
        }
    }
}

fn panic_err(msg: impl Into<String>) -> PanicError {
    PanicError::new(msg)
}

// ---------------------------------------------------------------------------
// FnRegistry: callable functions available for compile-time evaluation
// ---------------------------------------------------------------------------

/// A Rust-implemented builtin function callable at compile time.
pub type BuiltinFn = fn(Vec<Value>) -> Result<Value, PanicError>;

/// A user-defined function compiled for fn_reg interpretation.
/// Stores parameter names and the function body AST.
pub struct CompiledFn {
    pub params: Vec<Symbol>,
    pub body: Arc<Expr>,
}

/// Registry of functions available for compile-time evaluation.
///
/// Contains Rust-implemented builtins (`concat`, `head`, `last`, etc.) and
/// user-defined functions that have been type-checked and registered.
/// Grows monotonically as functions are compiled — each `register` returns
/// a new `FnRegistry`.
///
/// This is NOT a lexical scope — it's a flat, global registry. Lexical
/// variable bindings live in `Env`.
pub struct FnRegistry {
    pub builtins: HashMap<Symbol, BuiltinFn>,
    pub compiled: HashMap<Symbol, Arc<CompiledFn>>,
}

impl FnRegistry {
    pub fn new(builtins: HashMap<Symbol, BuiltinFn>) -> Self {
        FnRegistry {
            builtins,
            compiled: HashMap::new(),
        }
    }

    pub fn call(
        &self,
        func: &Symbol,
        args: Vec<Value>,
    ) -> Result<Value, PanicError> {
        if let Some(builtin) = self.builtins.get(func) {
            return builtin(args);
        }
        if let Some(compiled) = self.compiled.get(func) {
            if args.len() != compiled.params.len() {
                return Err(panic_err(format!(
                    "{}: expected {} args, got {}",
                    func,
                    compiled.params.len(),
                    args.len()
                )));
            }
            let env = compiled
                .params
                .iter()
                .zip(args)
                .fold(Env::empty(), |env, (name, val)| {
                    Env::extend(&env, name.clone(), val)
                });
            return interpret(&compiled.body, &env, self);
        }
        Err(panic_err(format!("unknown function: {func}")))
    }

    pub fn register(&self, name: Symbol, func: CompiledFn) -> Self {
        let mut compiled = self.compiled.clone();
        compiled.insert(name, Arc::new(func));
        FnRegistry {
            builtins: self.builtins.clone(),
            compiled,
        }
    }

    pub fn has(&self, name: &Symbol) -> bool {
        self.builtins.contains_key(name) || self.compiled.contains_key(name)
    }
}

// ---------------------------------------------------------------------------
// Tree-walking interpreter
// ---------------------------------------------------------------------------

/// Tree-walking interpreter for fn_reg evaluation.
///
/// Evaluates a Resin AST expression to a `Value` using the given lexical
/// environment (`env`) and function registry (`fn_reg`). Pure — no mutation.
/// Panics (via `PanicError`) surface as type errors during type-checking.
pub fn interpret(
    expr: &Expr,
    env: &Arc<Env>,
    fn_reg: &FnRegistry,
) -> Result<Value, PanicError> {
    match expr {
        Expr::Literal(lit) => literal_to_value(&lit.val),

        Expr::Name(n) => env
            .lookup(&n.name)
            .cloned()
            .ok_or_else(|| panic_err(format!("undefined variable: {}", n.name))),

        Expr::Apply(app) => {
            let arg_vals: Vec<Value> = app
                .args
                .iter()
                .map(|a| interpret(a, env, fn_reg))
                .collect::<Result<_, _>>()?;
            let func_name = match &app.callee {
                Expr::Name(n) => &n.name,
                _ => return Err(panic_err("higher-order calls not yet supported in fn_reg")),
            };
            fn_reg.call(func_name, arg_vals)
        }

        Expr::If(if_expr) => {
            for (cond, body) in &if_expr.cond_branch_vec {
                let cond_val = interpret(cond, env, fn_reg)?;
                match cond_val.as_bool() {
                    Some(true) => return interpret(body, env, fn_reg),
                    Some(false) => continue,
                    None => return Err(panic_err("if condition is not a boolean")),
                }
            }
            interpret(&if_expr.else_branch, env, fn_reg)
        }

        Expr::Chain(chain) => {
            let mut current_env = Arc::clone(env);
            let mut last_val = None;
            for stmt in &chain.stmt_vec {
                match stmt {
                    Stmt::Let(let_stmt) => {
                        let val = interpret(&let_stmt.init, &current_env, fn_reg)?;
                        let name = match &let_stmt.pattern {
                            Pattern::Name(n) => n.name.clone(),
                            _ => return Err(panic_err("unsupported pattern in fn_reg let")),
                        };
                        current_env = Env::extend(&current_env, name, val);
                    }
                    Stmt::Discard(d) => {
                        last_val = Some(interpret(&d.val, &current_env, fn_reg)?);
                    }
                    _ => return Err(panic_err("unsupported statement in fn_reg")),
                }
            }
            last_val.ok_or_else(|| panic_err("empty chain"))
        }

        Expr::ArrayLit(arr) => {
            let vals: Vec<Value> = arr
                .elements
                .iter()
                .map(|e| interpret(e, env, fn_reg))
                .collect::<Result<_, _>>()?;
            values_to_array(vals)
        }

        Expr::Dot(dot) => {
            let base = interpret(&dot.base, env, fn_reg)?;
            match base {
                Value::Record(fields) => fields
                    .iter()
                    .find(|(name, _)| *name == dot.field)
                    .map(|(_, val)| val.clone())
                    .ok_or_else(|| panic_err(format!("no field '{}' in record", dot.field))),
                _ => Err(panic_err("dot access on non-record")),
            }
        }

        _ => Err(panic_err(format!(
            "unsupported expression in fn_reg: {:?}",
            std::mem::discriminant(expr)
        ))),
    }
}

fn literal_to_value(lit: &vocab::Literal) -> Result<Value, PanicError> {
    match lit {
        vocab::Literal::Number(n) => {
            if n.is_integer() {
                let val: i64 = n
                    .value
                    .to_integer()
                    .try_into()
                    .map_err(|_| panic_err("integer too large"))?;
                Ok(Value::scalar_int(val))
            } else {
                let val = n.value.numer().to_string().parse::<f64>().unwrap_or(0.0)
                    / n.value.denom().to_string().parse::<f64>().unwrap_or(1.0);
                Ok(Value::scalar_float(val))
            }
        }
        vocab::Literal::Bool(b) => Ok(Value::scalar_bool(b.value)),
        vocab::Literal::String(s) => Ok(Value::String(s.content.clone())),
    }
}

fn values_to_array(vals: Vec<Value>) -> Result<Value, PanicError> {
    if vals.is_empty() {
        return Ok(Value::int_vec(vec![]));
    }

    // Check all elements are scalar tensors of the same data type
    let mut ints = vec![];
    let mut floats = vec![];
    let mut bools = vec![];

    for v in &vals {
        match v {
            Value::Tensor(t) if t.shape.is_empty() => match &t.data {
                TensorData::Int(d) => ints.push(d[0]),
                TensorData::Float(d) => floats.push(d[0]),
                TensorData::Bool(d) => bools.push(d[0]),
            },
            _ => return Err(panic_err("array elements must be scalars")),
        }
    }

    let len = vals.len();
    if !ints.is_empty() && floats.is_empty() && bools.is_empty() {
        Ok(Value::Tensor(TensorVal {
            data: TensorData::Int(ints),
            shape: vec![len],
        }))
    } else if !floats.is_empty() && ints.is_empty() && bools.is_empty() {
        Ok(Value::Tensor(TensorVal {
            data: TensorData::Float(floats),
            shape: vec![len],
        }))
    } else if !bools.is_empty() && ints.is_empty() && floats.is_empty() {
        Ok(Value::Tensor(TensorVal {
            data: TensorData::Bool(bools),
            shape: vec![len],
        }))
    } else {
        Err(panic_err("mixed types in array literal"))
    }
}

// ---------------------------------------------------------------------------
// Builtin fn_reg functions
// ---------------------------------------------------------------------------

/// Create the initial function registry with all Rust-implemented builtins.
///
/// Includes array operations (`concat`, `head`, `tail`, `init`, `last`, `len`),
/// arithmetic (`+`, `-`, `*`), comparison (`==`, `!=`, `<=`), and `panic`.
pub fn builtin_fn_registry() -> FnRegistry {
    let mut builtins: HashMap<Symbol, BuiltinFn> = HashMap::new();

    builtins.insert(Symbol::from("concat"), builtin_concat);
    builtins.insert(Symbol::from("head"), builtin_head);
    builtins.insert(Symbol::from("tail"), builtin_tail);
    builtins.insert(Symbol::from("init"), builtin_init);
    builtins.insert(Symbol::from("last"), builtin_last);
    builtins.insert(Symbol::from("len"), builtin_len);
    builtins.insert(Symbol::from("==(_,_)"), builtin_eq);
    builtins.insert(Symbol::from("!=(_,_)"), builtin_ne);
    builtins.insert(Symbol::from("<=(_,_)"), builtin_le);
    builtins.insert(Symbol::from("+(_,_)"), builtin_add);
    builtins.insert(Symbol::from("-(_,_)"), builtin_sub);
    builtins.insert(Symbol::from("*(_,_)"), builtin_mul);
    builtins.insert(Symbol::from("panic"), builtin_panic);

    FnRegistry::new(builtins)
}

fn expect_int_array(v: &Value, name: &str) -> Result<Vec<i64>, PanicError> {
    v.as_int_slice()
        .map(|s| s.to_vec())
        .ok_or_else(|| panic_err(format!("{name}: expected integer array")))
}

fn builtin_concat(args: Vec<Value>) -> Result<Value, PanicError> {
    if args.len() != 2 {
        return Err(panic_err("concat: expected 2 args"));
    }
    let mut a = expect_int_array(&args[0], "concat")?;
    let b = expect_int_array(&args[1], "concat")?;
    a.extend(b);
    Ok(Value::int_vec(a))
}

fn builtin_head(args: Vec<Value>) -> Result<Value, PanicError> {
    if args.len() != 1 {
        return Err(panic_err("head: expected 1 arg"));
    }
    let a = expect_int_array(&args[0], "head")?;
    a.first()
        .copied()
        .map(Value::scalar_int)
        .ok_or_else(|| panic_err("head: empty array"))
}

fn builtin_tail(args: Vec<Value>) -> Result<Value, PanicError> {
    if args.len() != 1 {
        return Err(panic_err("tail: expected 1 arg"));
    }
    let a = expect_int_array(&args[0], "tail")?;
    if a.is_empty() {
        return Err(panic_err("tail: empty array"));
    }
    Ok(Value::int_vec(a[1..].to_vec()))
}

fn builtin_init(args: Vec<Value>) -> Result<Value, PanicError> {
    if args.len() != 1 {
        return Err(panic_err("init: expected 1 arg"));
    }
    let a = expect_int_array(&args[0], "init")?;
    if a.is_empty() {
        return Err(panic_err("init: empty array"));
    }
    Ok(Value::int_vec(a[..a.len() - 1].to_vec()))
}

fn builtin_last(args: Vec<Value>) -> Result<Value, PanicError> {
    if args.len() != 1 {
        return Err(panic_err("last: expected 1 arg"));
    }
    let a = expect_int_array(&args[0], "last")?;
    a.last()
        .copied()
        .map(Value::scalar_int)
        .ok_or_else(|| panic_err("last: empty array"))
}

fn builtin_len(args: Vec<Value>) -> Result<Value, PanicError> {
    if args.len() != 1 {
        return Err(panic_err("len: expected 1 arg"));
    }
    let a = expect_int_array(&args[0], "len")?;
    Ok(Value::scalar_int(a.len() as i64))
}

fn builtin_eq(args: Vec<Value>) -> Result<Value, PanicError> {
    if args.len() != 2 {
        return Err(panic_err("==: expected 2 args"));
    }
    Ok(Value::scalar_bool(args[0] == args[1]))
}

fn builtin_ne(args: Vec<Value>) -> Result<Value, PanicError> {
    if args.len() != 2 {
        return Err(panic_err("!=: expected 2 args"));
    }
    Ok(Value::scalar_bool(args[0] != args[1]))
}

fn builtin_le(args: Vec<Value>) -> Result<Value, PanicError> {
    if args.len() != 2 {
        return Err(panic_err("<=: expected 2 args"));
    }
    match (args[0].as_int(), args[1].as_int()) {
        (Some(a), Some(b)) => Ok(Value::scalar_bool(a <= b)),
        _ => Err(panic_err("<=: expected integers")),
    }
}

fn builtin_add(args: Vec<Value>) -> Result<Value, PanicError> {
    if args.len() != 2 {
        return Err(panic_err("+: expected 2 args"));
    }
    match (args[0].as_int(), args[1].as_int()) {
        (Some(a), Some(b)) => Ok(Value::scalar_int(a + b)),
        _ => Err(panic_err("+: expected integers")),
    }
}

fn builtin_sub(args: Vec<Value>) -> Result<Value, PanicError> {
    if args.len() != 2 {
        return Err(panic_err("-: expected 2 args"));
    }
    match (args[0].as_int(), args[1].as_int()) {
        (Some(a), Some(b)) => Ok(Value::scalar_int(a - b)),
        _ => Err(panic_err("-: expected integers")),
    }
}

fn builtin_mul(args: Vec<Value>) -> Result<Value, PanicError> {
    if args.len() != 2 {
        return Err(panic_err("*: expected 2 args"));
    }
    match (args[0].as_int(), args[1].as_int()) {
        (Some(a), Some(b)) => Ok(Value::scalar_int(a * b)),
        _ => Err(panic_err("*: expected integers")),
    }
}

fn builtin_panic(args: Vec<Value>) -> Result<Value, PanicError> {
    let msg = if args.is_empty() {
        "panic".to_string()
    } else {
        match &args[0] {
            Value::String(s) => format!("panic: {s}"),
            other => format!("panic: {other:?}"),
        }
    };
    Err(panic_err(msg))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_env_lookup() {
        let root = Env::root(HashMap::from_iter([(
            Symbol::from("x"),
            Value::scalar_int(42),
        )]));
        assert_eq!(root.lookup(&Symbol::from("x")).unwrap().as_int(), Some(42));
        assert!(root.lookup(&Symbol::from("y")).is_none());

        let child = Env::extend(&root, Symbol::from("y"), Value::scalar_int(7));
        assert_eq!(child.lookup(&Symbol::from("x")).unwrap().as_int(), Some(42));
        assert_eq!(child.lookup(&Symbol::from("y")).unwrap().as_int(), Some(7));
    }

    #[test]
    fn test_builtin_concat() {
        let a = Value::int_vec(vec![1, 2]);
        let b = Value::int_vec(vec![3, 4, 5]);
        let result = builtin_concat(vec![a, b]).unwrap();
        assert_eq!(result.as_int_slice(), Some([1, 2, 3, 4, 5].as_slice()));
    }

    #[test]
    fn test_builtin_head_tail() {
        let a = Value::int_vec(vec![10, 20, 30]);
        assert_eq!(builtin_head(vec![a.clone()]).unwrap().as_int(), Some(10));
        let tail = builtin_tail(vec![a]).unwrap();
        assert_eq!(tail.as_int_slice(), Some([20, 30].as_slice()));
    }

    #[test]
    fn test_builtin_init_last() {
        let a = Value::int_vec(vec![10, 20, 30]);
        let init = builtin_init(vec![a.clone()]).unwrap();
        assert_eq!(init.as_int_slice(), Some([10, 20].as_slice()));
        assert_eq!(builtin_last(vec![a]).unwrap().as_int(), Some(30));
    }

    #[test]
    fn test_builtin_len() {
        let a = Value::int_vec(vec![1, 2, 3]);
        assert_eq!(builtin_len(vec![a]).unwrap().as_int(), Some(3));
    }

    #[test]
    fn test_builtin_arithmetic() {
        let a = Value::scalar_int(10);
        let b = Value::scalar_int(3);
        assert_eq!(
            builtin_add(vec![a.clone(), b.clone()]).unwrap().as_int(),
            Some(13)
        );
        assert_eq!(
            builtin_sub(vec![a.clone(), b.clone()]).unwrap().as_int(),
            Some(7)
        );
        assert_eq!(
            builtin_mul(vec![a, b]).unwrap().as_int(),
            Some(30)
        );
    }

    #[test]
    fn test_builtin_eq_ne() {
        let a = Value::scalar_int(5);
        let b = Value::scalar_int(5);
        let c = Value::scalar_int(3);
        assert_eq!(builtin_eq(vec![a.clone(), b]).unwrap().as_bool(), Some(true));
        assert_eq!(builtin_eq(vec![a.clone(), c.clone()]).unwrap().as_bool(), Some(false));
        assert_eq!(builtin_ne(vec![a, c]).unwrap().as_bool(), Some(true));
    }

    #[test]
    fn test_builtin_panic() {
        assert!(builtin_panic(vec![]).is_err());
    }

    #[test]
    fn test_interpret_literal() {
        let env = Env::empty();
        let fn_reg = builtin_fn_registry();
        let expr = Expr::new_literal(
            vocab::Literal::Number(vocab::LiteralNumber {
                value: num::BigRational::from_integer(42.into()),
                force_float: false,
            }),
            crate::Span::dummy(),
        );
        let result = interpret(&expr, &env, &fn_reg).unwrap();
        assert_eq!(result.as_int(), Some(42));
    }

    #[test]
    fn test_interpret_name() {
        let env = Env::extend(&Env::empty(), Symbol::from("x"), Value::scalar_int(7));
        let fn_reg = builtin_fn_registry();
        let expr = Expr::new_name(Symbol::from("x"), crate::Span::dummy());
        let result = interpret(&expr, &env, &fn_reg).unwrap();
        assert_eq!(result.as_int(), Some(7));
    }

    #[test]
    fn test_interpret_array_lit() {
        let env = Env::empty();
        let fn_reg = builtin_fn_registry();
        let d = crate::Span::dummy();
        let expr = Expr::new_array_lit(
            vec![
                Expr::new_literal(
                    vocab::Literal::Number(vocab::LiteralNumber {
                        value: num::BigRational::from_integer(10.into()),
                        force_float: false,
                    }),
                    d.clone(),
                ),
                Expr::new_literal(
                    vocab::Literal::Number(vocab::LiteralNumber {
                        value: num::BigRational::from_integer(5.into()),
                        force_float: false,
                    }),
                    d.clone(),
                ),
            ],
            d,
        );
        let result = interpret(&expr, &env, &fn_reg).unwrap();
        assert_eq!(result.as_int_slice(), Some([10, 5].as_slice()));
    }

    #[test]
    fn test_interpret_fn_reg_call() {
        let env = Env::empty();
        let fn_reg = builtin_fn_registry();
        let d = crate::Span::dummy();

        // concat([1, 2], [3])
        let expr = Expr::new_apply(
            Expr::new_name(Symbol::from("concat"), d.clone()),
            vec![
                Expr::new_array_lit(
                    vec![
                        Expr::new_literal(
                            vocab::Literal::Number(vocab::LiteralNumber {
                                value: num::BigRational::from_integer(1.into()),
                                force_float: false,
                            }),
                            d.clone(),
                        ),
                        Expr::new_literal(
                            vocab::Literal::Number(vocab::LiteralNumber {
                                value: num::BigRational::from_integer(2.into()),
                                force_float: false,
                            }),
                            d.clone(),
                        ),
                    ],
                    d.clone(),
                ),
                Expr::new_array_lit(
                    vec![Expr::new_literal(
                        vocab::Literal::Number(vocab::LiteralNumber {
                            value: num::BigRational::from_integer(3.into()),
                            force_float: false,
                        }),
                        d.clone(),
                    )],
                    d.clone(),
                ),
            ],
            d,
        );
        let result = interpret(&expr, &env, &fn_reg).unwrap();
        assert_eq!(result.as_int_slice(), Some([1, 2, 3].as_slice()));
    }

    #[test]
    fn test_compiled_fn_call() {
        let d = crate::Span::dummy();
        let fn_reg = builtin_fn_registry();

        // Define: double x = x + x
        let body = Expr::new_apply(
            Expr::new_name(Symbol::from("+(_,_)"), d.clone()),
            vec![
                Expr::new_name(Symbol::from("x"), d.clone()),
                Expr::new_name(Symbol::from("x"), d.clone()),
            ],
            d.clone(),
        );
        let fn_reg = fn_reg.register(
            Symbol::from("double"),
            CompiledFn {
                params: vec![Symbol::from("x")],
                body: Arc::new(body),
            },
        );

        // Call: double 21
        let expr = Expr::new_apply(
            Expr::new_name(Symbol::from("double"), d.clone()),
            vec![Expr::new_literal(
                vocab::Literal::Number(vocab::LiteralNumber {
                    value: num::BigRational::from_integer(21.into()),
                    force_float: false,
                }),
                d.clone(),
            )],
            d,
        );
        let env = Env::empty();
        let result = interpret(&expr, &env, &fn_reg).unwrap();
        assert_eq!(result.as_int(), Some(42));
    }
}
