//! # Resin-Core IR
//!
//! A minimal, pure dataflow calculus for tensor computation on GPUs.
//! All nodes are pure — no side effects. Evaluation is demand-driven.
//!
//! ## Structure
//!
//! The IR is a set of **functions**. Each function has:
//!
//! ```text
//! Function {
//!     id: FuncId,          // unique integer
//!     params: [ParamDef],  // ordered parameter list
//!     nodes: [Node],       // flat list, DAG within a function
//!     outputs: [Ref],      // return values (zero or more tensors)
//! }
//! ```
//!
//! A program is a collection of functions. One distinguished function
//! is the entry point. Functions cannot be nested — all are top-level
//! peers. The node graph within each function is a DAG; cycles only
//! exist across function boundaries via Call nodes (tail recursion).
//!
//! ## Nodes
//!
//! Most nodes produce exactly **one** output (output index 0).
//! Call produces **M** outputs (one per callee return value) and
//! Cond produces **K** outputs (one per then/else pair).
//!
//! ```text
//! Param(param_idx)                             // 0 inputs, 1 output
//! Const(tensor)                                // 0 inputs, 1 output
//! Elem(op, [Ref])                              // N inputs, 1 output
//! Reduce(op, Ref, dim)                         // 1 input,  1 output
//! Gather(data: Ref, indices: Ref, dim)         // 2 inputs, 1 output
//! Scatter(op, data: Ref, indices: Ref,         // 2 inputs, 1 output
//!         dim, dim_size)
//! Cond(pred: Ref, then: [Ref], else: [Ref])    // 1+2K inputs, K outputs
//! Call(func_id, [Ref])                          // N inputs, M outputs
//! ```
//!
//! ## Refs and Views
//!
//! A **Ref** is a reference to a specific output of a node, viewed
//! through an affine index map.
//!
//! ```text
//! Ref = (NodeId, OutputIndex, View)
//! View = { shape, strides, offset }
//! buffer_position = offset + sum(strides[k] * coord[k])
//! ```
//!
//! For single-output nodes, `OutputIndex` is always 0. For Call
//! and Cond nodes, `OutputIndex` selects which of the multiple
//! outputs to read.
//!
//! Node outputs are always dense (C-contiguous). Views reinterpret
//! the output without copying:
//!
//! | Transform      | Strides | Offset | Shape       |
//! |----------------|---------|--------|-------------|
//! | Identity [M,N] | [N, 1]  | 0      | [M, N]      |
//! | Transpose      | [1, M]  | 0      | [N, M]      |
//! | Broadcast      | [0, 1]  | 0      | [any, N]    |
//! | Slice x[2:5]   | [1]     | 2      | [3]         |
//! | Stride x[::2]  | [2]     | 0      | [ceil(N/2)] |
//! | Reverse        | [-1]    | N-1    | [N]         |
//!
//! Dimension insertion is stride = 0. No NewDim node.
//! Shape compatibility is guaranteed by the type checker.
//!
//! ## Param
//!
//! `Param(param_idx)` references the parameter at position `param_idx`
//! in the enclosing function's parameter list. Since functions cannot
//! be nested, there is no ambiguity — every Param belongs to exactly
//! the function that contains it.
//!
//! ## Elem
//!
//! Pointwise application of a scalar function. All input shapes
//! (after their views) must be identical — that is the output shape.
//!
//! - **Unary**: `neg`, `recip`, `exp`, `log`, `sqrt`, `abs`,
//!   `not`, `cast(dtype)`
//! - **Binary**: `add`, `sub`, `mul`, `div`, `rem`, `pow`,
//!   `max`, `min`, `eq`, `ne`, `lt`, `gt`, `le`, `ge`, `and`, `or`
//! - **Ternary**: `where(cond, on_true, on_false)` — per-element
//!   select; `cond` is boolean
//!
//! ## Reduce
//!
//! Collapse dimension `dim` to length 1 with an associative op.
//! `op` in { `sum`, `prod`, `max`, `min` }.
//! Output shape = input shape with `shape[dim] = 1`.
//!
//! ## Gather
//!
//! Read from `data` at positions given by `indices` along `dim`.
//!
//! ```text
//! output[..., i, ...] = data[..., indices[..., i, ...], ...]
//! ```
//!
//! Shape rules (after views):
//! - `data` and `indices` have the same number of dimensions.
//! - For all `d != dim`: `data.shape[d] = indices.shape[d]`.
//! - Output shape = `indices.shape`.
//! - `indices` values in `[0, data.shape[dim])`.
//!
//! ## Scatter
//!
//! Dual of Gather. Write `data` values to positions given by
//! `indices` along `dim`, resolving collisions with `op`.
//!
//! ```text
//! output[..., indices[..., j, ...], ...]  op=  data[..., j, ...]
//! ```
//!
//! Shape rules (after views):
//! - `data` and `indices` have the same shape.
//! - Output shape = `data.shape` with `shape[dim] = dim_size`.
//! - `indices` values in `[0, dim_size)`.
//! - Unwritten positions receive the identity element for `op`
//!   (`0` for sum, `1` for prod, `-inf` for max, `+inf` for min).
//!
//! ## Cond
//!
//! Lazy conditional with **K outputs**.
//!
//! ```text
//! Cond(pred: Ref, then: [Ref; K], else: [Ref; K])
//! ```
//!
//! `pred` is a scalar boolean. If true, the K outputs are the
//! `then` refs; if false, the `else` refs. Only the taken branch's
//! dependency cone is evaluated. All nodes are pure, so skipping
//! the other branch is safe. This laziness is what makes
//! tail-recursive Call through Cond terminate.
//!
//! Cond is the **only source of branching** in the IR.
//!
//! ## Call
//!
//! Invoke a function by ID, passing arguments positionally.
//!
//! ```text
//! Call(func_id, [Ref])
//! ```
//!
//! The argument list is positional, matching the callee's parameter
//! list in order. The Call node produces M outputs, where M is the
//! number of return values declared by the callee. Downstream nodes
//! reference individual outputs as `(call_node_id, output_idx, view)`.
//!
//! Multiple references to the same `(call_node_id, output_idx)` with
//! different views do not cause re-evaluation — the Call executes
//! once and each output is a distinct tensor.
//!
//! ### Tail recursion
//!
//! A Call whose `func_id` matches the enclosing function is a
//! recursive call. All recursion must be in tail position: the
//! Call's outputs must flow directly through Cond nodes to the
//! function's outputs, with no intervening computation.
//!
//! ```text
//! // sum(acc, i, x, n) -> [result]
//! Function {
//!     id: sum_id,
//!     params: [acc, i, x, n],
//!     nodes: [
//!         acc    = Param(0)
//!         i      = Param(1)
//!         x      = Param(2)
//!         n      = Param(3)
//!         xi     = Gather(x, i, 0)
//!         newacc = Elem(add, [acc, xi])
//!         newi   = Elem(add, [i, Const(1)])
//!         cont   = Elem(lt, [newi, n])
//!         recur  = Call(sum_id, [newacc, newi, x, n])
//!         result = Cond(cont, then: [(recur, 0)], else: [newacc])
//!     ],
//!     outputs: [(result, 0)],
//! }
//! ```
//!
//! The node graph is a DAG — `recur` references `sum_id` (a function),
//! not `result` (a node). The cycle is across the function boundary,
//! broken by Cond's laziness. The compiler lowers tail-recursive
//! Calls to loops.
//!
//! For loops carrying multiple values, a single Cond selects
//! all outputs at once:
//!
//! ```text
//! recur  = Call(f_id, [...])          // 2 outputs
//! result = Cond(cont,
//!     then: [(recur, 0), (recur, 1)],
//!     else: [val_a, val_b])           // 2 outputs
//! outputs: [(result, 0), (result, 1)]
//! ```
//!
//! ## Examples
//!
//! Simple function:
//!
//! ```text
//! // f(x, y) = x + y
//! Function {
//!     id: 0,
//!     params: [x, y],
//!     nodes: [
//!         x  = Param(0)
//!         y  = Param(1)
//!         v0 = Elem(add, [x, y])
//!     ],
//!     outputs: [(v0, 0)],
//! }
//! ```
//!
//! Conditional (scalar):
//!
//! ```text
//! // clamp(x, lo, hi)
//! Function {
//!     id: 1,
//!     params: [x, lo, hi],
//!     nodes: [
//!         x  = Param(0)
//!         lo = Param(1)
//!         hi = Param(2)
//!         v0 = Elem(lt, [x, lo])
//!         v1 = Elem(gt, [x, hi])
//!         v2 = Cond(v1, then: [hi], else: [x])
//!         v3 = Cond(v0, then: [lo], else: [(v2, 0)])
//!     ],
//!     outputs: [(v3, 0)],
//! }
//! ```
//!
//! Convergence loop:
//!
//! ```text
//! // grad_step(model) -> [grad_out]  (separate function)
//! Function { id: 1, params: [model], ..., outputs: [grad_out] }
//!
//! // train(model, lr, tol) -> [result]
//! Function {
//!     id: 2,
//!     params: [model, lr, tol],
//!     nodes: [
//!         model  = Param(0)
//!         lr     = Param(1)
//!         tol    = Param(2)
//!         gstep  = Call(1, [model])         // Call grad_step
//!         grad   = (gstep, 0)               // its single return
//!         step   = Elem(mul, [lr, grad])
//!         model' = Elem(sub, [model, step])
//!         err    = Reduce(max, Elem(abs, [grad]))
//!         cont   = Elem(gt, [err, tol])
//!         recur  = Call(2, [model', lr, tol])  // tail-recursive
//!         result = Cond(cont, then: [(recur, 0)], else: [model'])
//!     ],
//!     outputs: [(result, 0)],
//! }
//! ```
//!
//! ## Compilation pipeline
//!
//! 1. **Translate** user programs into a set of functions.
//!
//! 2. **Inline** all non-tail-position Call nodes via beta reduction
//!    (substitute callee's body, replacing Params with the provided
//!    args). Inlining has a configurable depth limit; exceeding it
//!    is a compilation error. After this phase, the only remaining
//!    Call nodes are self-recursive tail calls guarded by Cond.
//!
//! 3. **Resolve `grad`** higher-order functions. Any Call nodes still
//!    present at this point are tail-recursive and cannot be
//!    differentiated through (would require BPTT). The compiler
//!    panics if `grad` needs to differentiate across a remaining
//!    Call boundary.
//!
//! 4. **Codegen** for the resulting graphs. Tail-recursive Call/Cond
//!    patterns compile to GPU loops.
//!
//! ## Normalization rules
//!
//! **Hoist shared deps**: if a node is in the dependency cone
//! of both Cond branches, move it before the Cond.
//!
//! **Tail call elimination**: a self-recursive Call whose outputs
//! flow directly through Cond to the function's outputs compiles
//! as a loop, not a stack frame.
//!
//! ## Patterns
//!
//! **Counted loop**: `cont = Elem(lt, [i, n])`.
//! Compiler emits a fixed-trip-count loop.
//!
//! **Convergence**: `cont = Elem(gt, [Reduce(max, err), tol])`.
//! Per-element error reduced to a scalar decision.
//!
//! **Scan/accumulate**: carry a pre-allocated `[max_iters, ...]`
//! tensor as a param. Scatter each iteration's value at the
//! counter position.
//!
//! **Passthrough**: when a tail-recursive Call passes a param
//! through unchanged, the value is loop-invariant. Compiler
//! elides the copy.
//!
//! **Compaction copy**: `Gather(data, Const(range(n)), 0)`.
//! Compiler elides when source is already contiguous.
//!
//! **Elem where vs. Cond**: `where` is data-parallel (evaluates
//! both sides, selects per-element). `Cond` is scalar control
//! flow — only one side is evaluated.
//!
//! ## Differentiation
//!
//! The graph is differentiable. Elem, Reduce, Gather, and Scatter
//! all have known derivatives. Cond differentiates through the
//! taken branch. Tail-recursive Call is a differentiation boundary —
//! the compiler does not differentiate through recursion. Users
//! provide custom gradients for recursive functions, or the compiler
//! unrolls fixed-trip-count loops for AD.
//!
//! ## Struct explosion
//!
//! Language-level structs and tuples are erased when lowering to
//! the IR. Each field becomes a separate tensor. Nested structs
//! flatten recursively.
//!
//! ```text
//! // Language level
//! Linear f32 o i = { w: [o][i]f32, b: [o]f32 }
//! step :: Linear f32 o i -> [i]f32 -> Linear f32 o i
//!
//! // IR: Linear explodes to (w, b)
//! // Function: params [w, b, x], outputs [w', b']
//! ```
//!
//! Multi-output functions naturally represent struct-returning
//! functions: each field becomes a separate output.
//!
//! ## Copies and layout
//!
//! Views are free — they reinterpret buffers without copying.
//! Some operations (reshape of non-contiguous data) require
//! actual copies, expressed as `Gather(data, Const(range(n)), 0)`.
//! The compiler elides the copy when the source is already
//! contiguous. No implicit copy-or-not ambiguity.
//!
//! ## GPU mapping
//!
//! The compiler lowers the graph to device-side execution.
//! Tail-recursive Call patterns compile to GPU loops. On CUDA 12.4+,
//! loops use conditional graph nodes with a device-memory flag.
//! For targets without conditional graphs, the compiler falls
//! back to host-side dispatch.

use crate::Symbol;

// ---------------------------------------------------------------------------
// Identifiers
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FuncId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OutputIndex(pub u32);

// ---------------------------------------------------------------------------
// View — affine index map
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct View {
    pub shape: Vec<i64>,
    pub strides: Vec<i64>,
    pub offset: i64,
}

impl View {
    /// Identity view — shape not yet resolved.
    pub fn identity() -> Self {
        View {
            shape: vec![],
            strides: vec![],
            offset: 0,
        }
    }

    pub fn is_identity(&self) -> bool {
        self.shape.is_empty() && self.strides.is_empty() && self.offset == 0
    }
}

// ---------------------------------------------------------------------------
// Ref — edge in the dataflow graph
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct Ref {
    pub node: NodeId,
    pub output: OutputIndex,
    pub view: View,
}

impl Ref {
    /// Shorthand: reference output 0 with identity view.
    pub fn simple(node: NodeId) -> Self {
        Ref {
            node,
            output: OutputIndex(0),
            view: View::identity(),
        }
    }

    /// Reference a specific output index with identity view.
    pub fn output(node: NodeId, idx: u32) -> Self {
        Ref {
            node,
            output: OutputIndex(idx),
            view: View::identity(),
        }
    }
}

// ---------------------------------------------------------------------------
// Operations
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElemOp {
    // Unary
    Neg,
    Recip,
    Exp,
    Log,
    Sqrt,
    Abs,
    Not,
    // Binary
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Pow,
    Max,
    Min,
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
    And,
    Or,
    // Ternary
    Where,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    Sum,
    Prod,
    Max,
    Min,
}

// ---------------------------------------------------------------------------
// Const values
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum ConstVal {
    Int(i64),
    Float(f64),
    Bool(bool),
}

// ---------------------------------------------------------------------------
// Nodes
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum Node {
    Param {
        idx: u32,
    },
    Const {
        val: ConstVal,
    },
    Elem {
        op: ElemOp,
        args: Vec<Ref>,
    },
    Reduce {
        op: ReduceOp,
        input: Ref,
        dim: u32,
    },
    Gather {
        data: Ref,
        indices: Ref,
        dim: u32,
    },
    Scatter {
        op: ReduceOp,
        data: Ref,
        indices: Ref,
        dim: u32,
        dim_size: u32,
    },
    Cond {
        pred: Ref,
        then_refs: Vec<Ref>,
        else_refs: Vec<Ref>,
    },
    Call {
        func: FuncId,
        args: Vec<Ref>,
    },
}

// ---------------------------------------------------------------------------
// Function
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Function {
    pub id: FuncId,
    pub name: Symbol,
    pub params: Vec<Symbol>,
    pub nodes: Vec<Node>,
    pub outputs: Vec<Ref>,
}

// ---------------------------------------------------------------------------
// Program
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Program {
    pub functions: Vec<Function>,
    pub entry: FuncId,
}
