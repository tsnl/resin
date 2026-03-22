//! `ir_rw.rs` = IR rewriter
//!
//! Necessary passes:
//! - Inlining: inline every `Call` that is not a self-recursive tail call.
//!   A configurable depth limit keeps expansion finite.
//! - Grad elimination: replace every `Grad` node with a synthesized gradient
//!   function computed by reverse-mode autodiff, then inline that function back
//!   into its call sites.
//!
//! Nice to have optimization passes:
//! - Constant folding
//! - Common subexpression hoisting
//! - Dead code elimination
//!
//! Out of scope (handled later in a lower level codegen optimizer) (still put
//! this in the ir_rw comment)
//! - Kernel fusion
//! - Fused kernel scheduling
//!
//! Current implementation notes:
//! - Tail recursion is preserved only for self-calls whose outputs flow
//!   directly through `Cond` nodes to function outputs.
//! - Reverse-mode autodiff currently supports elementwise graphs plus `Cond`.
//! - Shape-aware builtin gradients such as `matmul` and `cross_entropy` are
//!   rejected explicitly until the IR carries enough layout information to
//!   express their adjoints.

use std::fmt;

use hashbrown::{HashMap, HashSet};

use crate::Symbol;
use crate::ir::{
    self, BuiltinKind, ConstVal, ElemOp, FuncId, Function, Node, NodeId, OutputIndex, Ref, View,
};

pub const DEFAULT_MAX_INLINE_DEPTH: usize = 64;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, Copy)]
pub struct Options {
    pub max_inline_depth: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            max_inline_depth: DEFAULT_MAX_INLINE_DEPTH,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    MissingFunction {
        func: FuncId,
    },
    MissingRef {
        node: NodeId,
        output: OutputIndex,
    },
    BadCallArity {
        func: FuncId,
        expected: usize,
        got: usize,
    },
    InlineDepthExceeded {
        func: FuncId,
        depth: usize,
        max_depth: usize,
    },
    UnsupportedViewComposition {
        node: NodeId,
        output: OutputIndex,
    },
    GradNeedsSingleOutput {
        func: FuncId,
        outputs: usize,
    },
    GradOutputExceedsParams {
        func: FuncId,
        outputs: usize,
        params: usize,
    },
    NestedGrad {
        func: FuncId,
    },
    UnsupportedGradNode {
        func: FuncId,
        kind: &'static str,
    },
    UnsupportedGradCallBoundary {
        func: FuncId,
        callee: FuncId,
    },
    UnsupportedBuiltinGradient {
        func: FuncId,
        builtin: BuiltinKind,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::MissingFunction { func } => {
                write!(f, "missing function body for function {}", func.0)
            }
            Error::MissingRef { node, output } => {
                write!(f, "missing rewritten ref for {}:{}", node.0, output.0)
            }
            Error::BadCallArity {
                func,
                expected,
                got,
            } => write!(
                f,
                "call to function {} has wrong arity: expected {}, got {}",
                func.0, expected, got
            ),
            Error::InlineDepthExceeded {
                func,
                depth,
                max_depth,
            } => write!(
                f,
                "inlining function {} exceeded max depth {} at depth {}",
                func.0, max_depth, depth
            ),
            Error::UnsupportedViewComposition { node, output } => write!(
                f,
                "rewriting ref {}:{} would require composing non-identity views",
                node.0, output.0
            ),
            Error::GradNeedsSingleOutput { func, outputs } => write!(
                f,
                "grad currently requires the target function to have exactly one output; function {} has {}",
                func.0, outputs
            ),
            Error::GradOutputExceedsParams {
                func,
                outputs,
                params,
            } => write!(
                f,
                "grad for function {} requested {} gradient outputs but the function only has {} params",
                func.0, outputs, params
            ),
            Error::NestedGrad { func } => write!(
                f,
                "grad of function {} depends on another grad node; higher-order differentiation is not implemented",
                func.0
            ),
            Error::UnsupportedGradNode { func, kind } => write!(
                f,
                "grad of function {} needs a '{}' adjoint, which is not implemented yet",
                func.0, kind
            ),
            Error::UnsupportedGradCallBoundary { func, callee } => write!(
                f,
                "grad of function {} crosses call boundary {}",
                func.0, callee.0
            ),
            Error::UnsupportedBuiltinGradient { func, builtin } => write!(
                f,
                "grad of function {} needs builtin gradient for {:?}, which is not implemented yet",
                func.0, builtin
            ),
        }
    }
}

impl std::error::Error for Error {}

pub fn rewrite(program: &ir::Program) -> Result<ir::Program> {
    rewrite_with_options(program, Options::default())
}

pub fn rewrite_with_options(program: &ir::Program, options: Options) -> Result<ir::Program> {
    let inlined = inline_program(program, options)?;
    let grad_free = resolve_grads(&inlined)?;
    inline_program(&grad_free, options)
}

type RefKey = (u32, u32);

fn ref_key(node: NodeId, output: u32) -> RefKey {
    (node.0, output)
}

fn ref_key_of(r: &Ref) -> RefKey {
    ref_key(r.node, r.output.0)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UseSite {
    Output,
    NodeInput,
    CondBranch(NodeId),
}

#[derive(Debug, Clone)]
struct FunctionAnalysis {
    output_counts: Vec<u32>,
    tail_self_calls: HashSet<NodeId>,
}

struct ProgramIndex<'a> {
    functions: HashMap<FuncId, &'a Function>,
    builtins: &'a HashMap<FuncId, BuiltinKind>,
}

impl<'a> ProgramIndex<'a> {
    fn new(program: &'a ir::Program) -> Self {
        Self {
            functions: program
                .functions
                .iter()
                .map(|func| (func.id, func))
                .collect(),
            builtins: &program.builtins,
        }
    }

    fn function(&self, func: FuncId) -> Result<&'a Function> {
        self.functions
            .get(&func)
            .copied()
            .ok_or(Error::MissingFunction { func })
    }

    fn builtin(&self, func: FuncId) -> Option<BuiltinKind> {
        self.builtins.get(&func).copied()
    }
}

fn builtin_output_count(_builtin: BuiltinKind) -> u32 {
    1
}

fn inline_program(program: &ir::Program, options: Options) -> Result<ir::Program> {
    let inliner = Inliner::new(program, options)?;
    let mut functions = Vec::with_capacity(program.functions.len());

    for func in &program.functions {
        let mut sink = NodeSink::default();
        let analysis = inliner.analysis(func.id)?;
        let cloned = inliner.clone_function(
            func,
            &mut sink,
            ParamStrategy::Emit,
            Some(&analysis.tail_self_calls),
            0,
        )?;
        functions.push(Function {
            id: func.id,
            name: func.name.clone(),
            params: func.params.clone(),
            nodes: sink.nodes,
            outputs: cloned.outputs,
        });
    }

    Ok(ir::Program {
        functions,
        builtins: program.builtins.clone(),
    })
}

fn resolve_grads(program: &ir::Program) -> Result<ir::Program> {
    let index = ProgramIndex::new(program);
    let analyses = build_analyses(program, &index)?;
    let mut resolver = GradResolver::new(index, analyses, next_func_id(program));
    let mut functions = Vec::with_capacity(program.functions.len());

    for func in &program.functions {
        let mut nodes = Vec::with_capacity(func.nodes.len());

        for (idx, node) in func.nodes.iter().enumerate() {
            match node {
                Node::Grad { func: target, args } => {
                    let output_count = resolver.analysis(func.id)?.output_counts[idx] as usize;
                    let grad_func = resolver.ensure_gradient_function(*target, output_count)?;
                    nodes.push(Node::Call {
                        func: grad_func,
                        args: args.clone(),
                    });
                }
                _ => nodes.push(node.clone()),
            }
        }

        functions.push(Function {
            id: func.id,
            name: func.name.clone(),
            params: func.params.clone(),
            nodes,
            outputs: func.outputs.clone(),
        });
    }

    functions.extend(resolver.synthesized);

    Ok(ir::Program {
        functions,
        builtins: program.builtins.clone(),
    })
}

fn next_func_id(program: &ir::Program) -> u32 {
    let max_defined = program
        .functions
        .iter()
        .map(|func| func.id.0)
        .max()
        .unwrap_or(0);
    let max_builtin = program.builtins.keys().map(|id| id.0).max().unwrap_or(0);
    max_defined.max(max_builtin).saturating_add(1)
}

fn build_analyses(
    program: &ir::Program,
    index: &ProgramIndex<'_>,
) -> Result<HashMap<FuncId, FunctionAnalysis>> {
    let mut analyses = HashMap::with_capacity(program.functions.len());
    for func in &program.functions {
        analyses.insert(func.id, analyze_function(func, index)?);
    }
    Ok(analyses)
}

fn analyze_function(func: &Function, index: &ProgramIndex<'_>) -> Result<FunctionAnalysis> {
    let mut uses: HashMap<RefKey, Vec<UseSite>> = HashMap::new();
    let mut max_output_index: HashMap<u32, u32> = HashMap::new();

    let mut record_use = |r: &Ref, site: UseSite| {
        uses.entry(ref_key_of(r)).or_default().push(site);
        max_output_index
            .entry(r.node.0)
            .and_modify(|max_seen| *max_seen = (*max_seen).max(r.output.0 + 1))
            .or_insert(r.output.0 + 1);
    };

    for output in &func.outputs {
        record_use(output, UseSite::Output);
    }

    for (idx, node) in func.nodes.iter().enumerate() {
        let node_id = NodeId(idx as u32);
        match node {
            Node::Param { .. } | Node::Const { .. } => {}
            Node::Elem { args, .. } => {
                for arg in args {
                    record_use(arg, UseSite::NodeInput);
                }
            }
            Node::Reduce { input, .. } => record_use(input, UseSite::NodeInput),
            Node::Gather { data, indices, .. } => {
                record_use(data, UseSite::NodeInput);
                record_use(indices, UseSite::NodeInput);
            }
            Node::Scatter { data, indices, .. } => {
                record_use(data, UseSite::NodeInput);
                record_use(indices, UseSite::NodeInput);
            }
            Node::Cond {
                pred,
                then_refs,
                else_refs,
            } => {
                record_use(pred, UseSite::NodeInput);
                for r in then_refs {
                    record_use(r, UseSite::CondBranch(node_id));
                }
                for r in else_refs {
                    record_use(r, UseSite::CondBranch(node_id));
                }
            }
            Node::Call { args, .. } | Node::Grad { args, .. } => {
                for arg in args {
                    record_use(arg, UseSite::NodeInput);
                }
            }
        }
    }

    let mut output_counts = Vec::with_capacity(func.nodes.len());
    for (idx, node) in func.nodes.iter().enumerate() {
        let output_count = match node {
            Node::Param { .. }
            | Node::Const { .. }
            | Node::Elem { .. }
            | Node::Reduce { .. }
            | Node::Gather { .. }
            | Node::Scatter { .. } => 1,
            Node::Cond { then_refs, .. } => then_refs.len() as u32,
            Node::Call { func: callee, .. } => match index.builtin(*callee) {
                Some(builtin) => builtin_output_count(builtin),
                None => index.function(*callee)?.outputs.len() as u32,
            },
            Node::Grad { .. } => max_output_index.get(&(idx as u32)).copied().unwrap_or(0),
        };
        output_counts.push(output_count);
    }

    let tail_self_calls = compute_tail_self_calls(func, &uses, &output_counts);

    Ok(FunctionAnalysis {
        output_counts,
        tail_self_calls,
    })
}

fn compute_tail_self_calls(
    func: &Function,
    uses: &HashMap<RefKey, Vec<UseSite>>,
    output_counts: &[u32],
) -> HashSet<NodeId> {
    fn output_is_tail(
        func: &Function,
        uses: &HashMap<RefKey, Vec<UseSite>>,
        memo: &mut HashMap<u32, bool>,
        node: NodeId,
        output: u32,
    ) -> bool {
        let Some(use_sites) = uses.get(&ref_key(node, output)) else {
            return false;
        };
        !use_sites.is_empty()
            && use_sites.iter().all(|site| match *site {
                UseSite::Output => true,
                UseSite::NodeInput => false,
                UseSite::CondBranch(cond) => cond_is_tail(func, uses, memo, cond),
            })
    }

    fn cond_is_tail(
        func: &Function,
        uses: &HashMap<RefKey, Vec<UseSite>>,
        memo: &mut HashMap<u32, bool>,
        node: NodeId,
    ) -> bool {
        if let Some(value) = memo.get(&node.0) {
            return *value;
        }

        let value = match &func.nodes[node.0 as usize] {
            Node::Cond { then_refs, .. } => (0..then_refs.len() as u32)
                .all(|output| output_is_tail(func, uses, memo, node, output)),
            _ => false,
        };
        memo.insert(node.0, value);
        value
    }

    let mut memo = HashMap::<u32, bool>::new();
    let mut tail_calls = HashSet::new();

    for (idx, node) in func.nodes.iter().enumerate() {
        let node_id = NodeId(idx as u32);
        if let Node::Call { func: callee, .. } = node {
            if *callee != func.id {
                continue;
            }
            let output_count = output_counts[idx];
            if output_count > 0
                && (0..output_count)
                    .all(|output| output_is_tail(func, uses, &mut memo, node_id, output))
            {
                tail_calls.insert(node_id);
            }
        }
    }

    tail_calls
}

#[derive(Default)]
struct NodeSink {
    nodes: Vec<Node>,
}

impl NodeSink {
    fn emit(&mut self, node: Node) -> NodeId {
        let id = NodeId(self.nodes.len() as u32);
        self.nodes.push(node);
        id
    }
}

#[derive(Default)]
struct RefMap {
    refs: HashMap<RefKey, Ref>,
}

impl RefMap {
    fn insert(&mut self, node: NodeId, output: u32, r: Ref) {
        self.refs.insert(ref_key(node, output), r);
    }

    fn get(&self, node: NodeId, output: u32) -> Option<&Ref> {
        self.refs.get(&ref_key(node, output))
    }

    fn rewrite_ref(&self, r: &Ref) -> Result<Ref> {
        let base = self.refs.get(&ref_key_of(r)).ok_or(Error::MissingRef {
            node: r.node,
            output: r.output,
        })?;
        compose_ref(base, &r.view)
    }
}

fn compose_ref(base: &Ref, extra_view: &View) -> Result<Ref> {
    if extra_view.is_identity() {
        return Ok(base.clone());
    }
    if base.view.is_identity() {
        let mut composed = base.clone();
        composed.view = extra_view.clone();
        return Ok(composed);
    }
    Err(Error::UnsupportedViewComposition {
        node: base.node,
        output: base.output,
    })
}

enum ParamStrategy<'a> {
    Emit,
    UseArgs(&'a [Ref]),
}

struct CloneResult {
    outputs: Vec<Ref>,
}

struct Inliner<'a> {
    index: ProgramIndex<'a>,
    analyses: HashMap<FuncId, FunctionAnalysis>,
    options: Options,
}

impl<'a> Inliner<'a> {
    fn new(program: &'a ir::Program, options: Options) -> Result<Self> {
        let index = ProgramIndex::new(program);
        let analyses = build_analyses(program, &index)?;
        Ok(Self {
            index,
            analyses,
            options,
        })
    }

    fn analysis(&self, func: FuncId) -> Result<&FunctionAnalysis> {
        self.analyses
            .get(&func)
            .ok_or(Error::MissingFunction { func })
    }

    fn clone_function(
        &self,
        source: &Function,
        sink: &mut NodeSink,
        params: ParamStrategy<'_>,
        tail_calls_to_preserve: Option<&HashSet<NodeId>>,
        depth: usize,
    ) -> Result<CloneResult> {
        if let ParamStrategy::UseArgs(args) = params {
            if args.len() != source.params.len() {
                return Err(Error::BadCallArity {
                    func: source.id,
                    expected: source.params.len(),
                    got: args.len(),
                });
            }
        }

        let analysis = self.analysis(source.id)?;
        let mut refs = RefMap::default();

        for (idx, node) in source.nodes.iter().enumerate() {
            let node_id = NodeId(idx as u32);

            match node {
                Node::Param { idx } => match params {
                    ParamStrategy::Emit => {
                        let new_id = sink.emit(Node::Param { idx: *idx });
                        refs.insert(node_id, 0, Ref::simple(new_id));
                    }
                    ParamStrategy::UseArgs(args) => {
                        refs.insert(node_id, 0, args[*idx as usize].clone());
                    }
                },
                Node::Const { val } => {
                    let new_id = sink.emit(Node::Const { val: val.clone() });
                    refs.insert(node_id, 0, Ref::simple(new_id));
                }
                Node::Elem { op, args } => {
                    let args = rewrite_ref_slice(&refs, args)?;
                    let new_id = sink.emit(Node::Elem { op: *op, args });
                    refs.insert(node_id, 0, Ref::simple(new_id));
                }
                Node::Reduce { op, input, dim } => {
                    let input = refs.rewrite_ref(input)?;
                    let new_id = sink.emit(Node::Reduce {
                        op: *op,
                        input,
                        dim: *dim,
                    });
                    refs.insert(node_id, 0, Ref::simple(new_id));
                }
                Node::Gather { data, indices, dim } => {
                    let data = refs.rewrite_ref(data)?;
                    let indices = refs.rewrite_ref(indices)?;
                    let new_id = sink.emit(Node::Gather {
                        data,
                        indices,
                        dim: *dim,
                    });
                    refs.insert(node_id, 0, Ref::simple(new_id));
                }
                Node::Scatter {
                    op,
                    data,
                    indices,
                    dim,
                    dim_size,
                } => {
                    let data = refs.rewrite_ref(data)?;
                    let indices = refs.rewrite_ref(indices)?;
                    let new_id = sink.emit(Node::Scatter {
                        op: *op,
                        data,
                        indices,
                        dim: *dim,
                        dim_size: *dim_size,
                    });
                    refs.insert(node_id, 0, Ref::simple(new_id));
                }
                Node::Cond {
                    pred,
                    then_refs,
                    else_refs,
                } => {
                    let pred = refs.rewrite_ref(pred)?;
                    let then_refs = rewrite_ref_slice(&refs, then_refs)?;
                    let else_refs = rewrite_ref_slice(&refs, else_refs)?;
                    let new_id = sink.emit(Node::Cond {
                        pred,
                        then_refs: then_refs.clone(),
                        else_refs,
                    });
                    for output in 0..then_refs.len() as u32 {
                        refs.insert(node_id, output, Ref::output(new_id, output));
                    }
                }
                Node::Call { func: callee, args } => {
                    let rewritten_args = rewrite_ref_slice(&refs, args)?;
                    let preserve = tail_calls_to_preserve
                        .map(|tail_calls| *callee == source.id && tail_calls.contains(&node_id))
                        .unwrap_or(false);

                    if preserve || self.index.builtin(*callee).is_some() {
                        let new_id = sink.emit(Node::Call {
                            func: *callee,
                            args: rewritten_args,
                        });
                        let output_count = analysis.output_counts[idx];
                        for output in 0..output_count {
                            refs.insert(node_id, output, Ref::output(new_id, output));
                        }
                        continue;
                    }

                    let next_depth = depth + 1;
                    if next_depth > self.options.max_inline_depth {
                        return Err(Error::InlineDepthExceeded {
                            func: *callee,
                            depth: next_depth,
                            max_depth: self.options.max_inline_depth,
                        });
                    }

                    let callee_func = self.index.function(*callee)?;
                    let inlined = self.clone_function(
                        callee_func,
                        sink,
                        ParamStrategy::UseArgs(&rewritten_args),
                        None,
                        next_depth,
                    )?;
                    for (output, r) in inlined.outputs.into_iter().enumerate() {
                        refs.insert(node_id, output as u32, r);
                    }
                }
                Node::Grad { func, args } => {
                    let args = rewrite_ref_slice(&refs, args)?;
                    let new_id = sink.emit(Node::Grad { func: *func, args });
                    let output_count = analysis.output_counts[idx];
                    for output in 0..output_count {
                        refs.insert(node_id, output, Ref::output(new_id, output));
                    }
                }
            }
        }

        let outputs = rewrite_ref_slice(&refs, &source.outputs)?;
        Ok(CloneResult { outputs })
    }
}

fn rewrite_ref_slice(refs: &RefMap, values: &[Ref]) -> Result<Vec<Ref>> {
    values.iter().map(|r| refs.rewrite_ref(r)).collect()
}

struct GradResolver<'a> {
    index: ProgramIndex<'a>,
    analyses: HashMap<FuncId, FunctionAnalysis>,
    next_func_id: u32,
    cache: HashMap<(FuncId, usize), FuncId>,
    synthesized: Vec<Function>,
}

impl<'a> GradResolver<'a> {
    fn new(
        index: ProgramIndex<'a>,
        analyses: HashMap<FuncId, FunctionAnalysis>,
        next_func_id: u32,
    ) -> Self {
        Self {
            index,
            analyses,
            next_func_id,
            cache: HashMap::new(),
            synthesized: vec![],
        }
    }

    fn analysis(&self, func: FuncId) -> Result<&FunctionAnalysis> {
        self.analyses
            .get(&func)
            .ok_or(Error::MissingFunction { func })
    }

    fn ensure_gradient_function(&mut self, target: FuncId, grad_outputs: usize) -> Result<FuncId> {
        if let Some(existing) = self.cache.get(&(target, grad_outputs)) {
            return Ok(*existing);
        }

        let id = FuncId(self.next_func_id);
        self.next_func_id += 1;

        let target_func = self.index.function(target)?;
        let grad_func = GradientBuilder::new(target_func, &self.index).build(id, grad_outputs)?;
        self.cache.insert((target, grad_outputs), id);
        self.synthesized.push(grad_func);
        Ok(id)
    }
}

struct GradientBuilder<'a> {
    target: &'a Function,
    index: &'a ProgramIndex<'a>,
    sink: NodeSink,
    primal: RefMap,
    cotangents: HashMap<RefKey, Ref>,
    params_by_idx: HashMap<u32, NodeId>,
}

impl<'a> GradientBuilder<'a> {
    fn new(target: &'a Function, index: &'a ProgramIndex<'a>) -> Self {
        Self {
            target,
            index,
            sink: NodeSink::default(),
            primal: RefMap::default(),
            cotangents: HashMap::new(),
            params_by_idx: HashMap::new(),
        }
    }

    fn build(mut self, new_id: FuncId, grad_outputs: usize) -> Result<Function> {
        if self.target.outputs.len() != 1 {
            return Err(Error::GradNeedsSingleOutput {
                func: self.target.id,
                outputs: self.target.outputs.len(),
            });
        }
        if grad_outputs > self.target.params.len() {
            return Err(Error::GradOutputExceedsParams {
                func: self.target.id,
                outputs: grad_outputs,
                params: self.target.params.len(),
            });
        }

        self.clone_primal()?;

        let seed = self.emit_const(ConstVal::Float(1.0));
        self.add_cotangent_ref(&self.target.outputs[0], seed);

        for idx in (0..self.target.nodes.len()).rev() {
            self.backprop_node(NodeId(idx as u32))?;
        }

        let mut outputs = Vec::with_capacity(grad_outputs);
        for param_idx in 0..grad_outputs as u32 {
            let Some(&param_node) = self.params_by_idx.get(&param_idx) else {
                return Err(Error::GradOutputExceedsParams {
                    func: self.target.id,
                    outputs: grad_outputs,
                    params: self.target.params.len(),
                });
            };

            if let Some(grad) = self.cotangents.get(&ref_key(param_node, 0)).cloned() {
                outputs.push(grad);
            } else {
                let primal_param = self.primal_ref(param_node, 0)?;
                outputs.push(self.zero_like(primal_param));
            }
        }

        Ok(Function {
            id: new_id,
            name: Symbol::from(format!("{}$grad{}", self.target.name.text(), grad_outputs)),
            params: self.target.params.clone(),
            nodes: self.sink.nodes,
            outputs,
        })
    }

    fn clone_primal(&mut self) -> Result<()> {
        for (idx, node) in self.target.nodes.iter().enumerate() {
            let node_id = NodeId(idx as u32);
            match node {
                Node::Param { idx } => {
                    let new_id = self.sink.emit(Node::Param { idx: *idx });
                    self.primal.insert(node_id, 0, Ref::simple(new_id));
                    self.params_by_idx.insert(*idx, node_id);
                }
                Node::Const { val } => {
                    let new_id = self.sink.emit(Node::Const { val: val.clone() });
                    self.primal.insert(node_id, 0, Ref::simple(new_id));
                }
                Node::Elem { op, args } => {
                    let args = rewrite_ref_slice(&self.primal, args)?;
                    let new_id = self.sink.emit(Node::Elem { op: *op, args });
                    self.primal.insert(node_id, 0, Ref::simple(new_id));
                }
                Node::Reduce { op, input, dim } => {
                    let input = self.primal.rewrite_ref(input)?;
                    let new_id = self.sink.emit(Node::Reduce {
                        op: *op,
                        input,
                        dim: *dim,
                    });
                    self.primal.insert(node_id, 0, Ref::simple(new_id));
                }
                Node::Gather { data, indices, dim } => {
                    let data = self.primal.rewrite_ref(data)?;
                    let indices = self.primal.rewrite_ref(indices)?;
                    let new_id = self.sink.emit(Node::Gather {
                        data,
                        indices,
                        dim: *dim,
                    });
                    self.primal.insert(node_id, 0, Ref::simple(new_id));
                }
                Node::Scatter {
                    op,
                    data,
                    indices,
                    dim,
                    dim_size,
                } => {
                    let data = self.primal.rewrite_ref(data)?;
                    let indices = self.primal.rewrite_ref(indices)?;
                    let new_id = self.sink.emit(Node::Scatter {
                        op: *op,
                        data,
                        indices,
                        dim: *dim,
                        dim_size: *dim_size,
                    });
                    self.primal.insert(node_id, 0, Ref::simple(new_id));
                }
                Node::Cond {
                    pred,
                    then_refs,
                    else_refs,
                } => {
                    let pred = self.primal.rewrite_ref(pred)?;
                    let then_refs = rewrite_ref_slice(&self.primal, then_refs)?;
                    let else_refs = rewrite_ref_slice(&self.primal, else_refs)?;
                    let new_id = self.sink.emit(Node::Cond {
                        pred,
                        then_refs: then_refs.clone(),
                        else_refs,
                    });
                    for output in 0..then_refs.len() as u32 {
                        self.primal
                            .insert(node_id, output, Ref::output(new_id, output));
                    }
                }
                Node::Call { func, args } => {
                    let args = rewrite_ref_slice(&self.primal, args)?;
                    let new_id = self.sink.emit(Node::Call { func: *func, args });
                    let output_count = match self.index.builtin(*func) {
                        Some(builtin) => builtin_output_count(builtin),
                        None => self.index.function(*func)?.outputs.len() as u32,
                    };
                    for output in 0..output_count {
                        self.primal
                            .insert(node_id, output, Ref::output(new_id, output));
                    }
                }
                Node::Grad { .. } => {
                    return Err(Error::NestedGrad {
                        func: self.target.id,
                    });
                }
            }
        }
        Ok(())
    }

    fn backprop_node(&mut self, node_id: NodeId) -> Result<()> {
        match &self.target.nodes[node_id.0 as usize] {
            Node::Param { .. } | Node::Const { .. } => Ok(()),
            Node::Elem { op, args } => self.backprop_elem(node_id, *op, args),
            Node::Reduce { .. } => {
                if self.has_cotangent(node_id, 0) {
                    return Err(Error::UnsupportedGradNode {
                        func: self.target.id,
                        kind: "reduce",
                    });
                }
                Ok(())
            }
            Node::Gather { .. } => {
                if self.has_cotangent(node_id, 0) {
                    return Err(Error::UnsupportedGradNode {
                        func: self.target.id,
                        kind: "gather",
                    });
                }
                Ok(())
            }
            Node::Scatter { .. } => {
                if self.has_cotangent(node_id, 0) {
                    return Err(Error::UnsupportedGradNode {
                        func: self.target.id,
                        kind: "scatter",
                    });
                }
                Ok(())
            }
            Node::Cond {
                pred,
                then_refs,
                else_refs,
            } => {
                let pred = self.primal.rewrite_ref(pred)?;
                for output in 0..then_refs.len() {
                    let Some(grad) = self.cotangent(node_id, output as u32) else {
                        continue;
                    };

                    let then_value = self.primal.rewrite_ref(&then_refs[output])?;
                    let else_value = self.primal.rewrite_ref(&else_refs[output])?;
                    let then_zero = self.zero_like(then_value);
                    let else_zero = self.zero_like(else_value);
                    let then_grad = self.emit_cond_single(pred.clone(), grad.clone(), then_zero);
                    let else_grad = self.emit_cond_single(pred.clone(), else_zero, grad);
                    self.add_cotangent_ref(&then_refs[output], then_grad);
                    self.add_cotangent_ref(&else_refs[output], else_grad);
                }
                Ok(())
            }
            Node::Call { func, .. } => {
                let output_count = match self.index.builtin(*func) {
                    Some(builtin) => builtin_output_count(builtin),
                    None => self.index.function(*func)?.outputs.len() as u32,
                };
                for output in 0..output_count {
                    if !self.has_cotangent(node_id, output) {
                        continue;
                    }
                    return match self.index.builtin(*func) {
                        Some(builtin) => Err(Error::UnsupportedBuiltinGradient {
                            func: self.target.id,
                            builtin,
                        }),
                        None => Err(Error::UnsupportedGradCallBoundary {
                            func: self.target.id,
                            callee: *func,
                        }),
                    };
                }
                Ok(())
            }
            Node::Grad { .. } => Err(Error::NestedGrad {
                func: self.target.id,
            }),
        }
    }

    fn backprop_elem(&mut self, node_id: NodeId, op: ElemOp, args: &[Ref]) -> Result<()> {
        let Some(grad) = self.cotangent(node_id, 0) else {
            return Ok(());
        };

        let rewritten_args = rewrite_ref_slice(&self.primal, args)?;

        match op {
            ElemOp::Neg => {
                let contribution = self.neg(grad);
                self.add_cotangent_ref(&args[0], contribution);
            }
            ElemOp::Recip => {
                let x = rewritten_args[0].clone();
                let xx = self.mul(x.clone(), x);
                let frac = self.div(grad, xx);
                let contribution = self.neg(frac);
                self.add_cotangent_ref(&args[0], contribution);
            }
            ElemOp::Exp => {
                let y = self.primal_ref(node_id, 0)?;
                let contribution = self.mul(grad, y);
                self.add_cotangent_ref(&args[0], contribution);
            }
            ElemOp::Log => {
                let x = rewritten_args[0].clone();
                let contribution = self.div(grad, x);
                self.add_cotangent_ref(&args[0], contribution);
            }
            ElemOp::Sqrt => {
                let y = self.primal_ref(node_id, 0)?;
                let two = self.emit_const(ConstVal::Float(2.0));
                let denom = self.mul(two, y);
                let contribution = self.div(grad, denom);
                self.add_cotangent_ref(&args[0], contribution);
            }
            ElemOp::Abs => {
                let x = rewritten_args[0].clone();
                let zero = self.zero_like(x.clone());
                let pred = self.emit_elem(ElemOp::Gt, vec![x, zero]);
                let neg_grad = self.neg(grad.clone());
                let contrib = self.emit_elem(ElemOp::Where, vec![pred, grad, neg_grad]);
                self.add_cotangent_ref(&args[0], contrib);
            }
            ElemOp::Not => {}
            ElemOp::Add => {
                self.add_cotangent_ref(&args[0], grad.clone());
                self.add_cotangent_ref(&args[1], grad);
            }
            ElemOp::Sub => {
                self.add_cotangent_ref(&args[0], grad.clone());
                let contribution = self.neg(grad);
                self.add_cotangent_ref(&args[1], contribution);
            }
            ElemOp::Mul => {
                let lhs = rewritten_args[0].clone();
                let rhs = rewritten_args[1].clone();
                let d_lhs = self.mul(grad.clone(), rhs);
                let d_rhs = self.mul(grad, lhs);
                self.add_cotangent_ref(&args[0], d_lhs);
                self.add_cotangent_ref(&args[1], d_rhs);
            }
            ElemOp::Div => {
                let lhs = rewritten_args[0].clone();
                let rhs = rewritten_args[1].clone();
                let rhs_sq = self.mul(rhs.clone(), rhs.clone());
                let numerator = self.mul(grad.clone(), lhs);
                let d_lhs = self.div(grad, rhs.clone());
                let d_rhs = self.div(numerator, rhs_sq);
                let d_rhs = self.neg(d_rhs);
                self.add_cotangent_ref(&args[0], d_lhs);
                self.add_cotangent_ref(&args[1], d_rhs);
            }
            ElemOp::IntDiv | ElemOp::Rem => {}
            ElemOp::Pow => {
                let lhs = rewritten_args[0].clone();
                let rhs = rewritten_args[1].clone();
                let one = self.emit_const(ConstVal::Float(1.0));
                let exp_minus_one = self.sub(rhs.clone(), one);
                let lhs_pow = self.emit_elem(ElemOp::Pow, vec![lhs.clone(), exp_minus_one]);
                let rhs_times_pow = self.mul(rhs.clone(), lhs_pow);
                let d_lhs = self.mul(grad.clone(), rhs_times_pow);
                let y = self.primal_ref(node_id, 0)?;
                let log_lhs = self.emit_elem(ElemOp::Log, vec![lhs.clone()]);
                let y_log = self.mul(y, log_lhs);
                let d_rhs = self.mul(grad, y_log);
                self.add_cotangent_ref(&args[0], d_lhs);
                self.add_cotangent_ref(&args[1], d_rhs);
            }
            ElemOp::Max => {
                let lhs = rewritten_args[0].clone();
                let rhs = rewritten_args[1].clone();
                let pred = self.emit_elem(ElemOp::Gt, vec![lhs.clone(), rhs]);
                let lhs_zero = self.zero_like(lhs);
                let rhs_zero = self.zero_like(rewritten_args[1].clone());
                let d_lhs =
                    self.emit_elem(ElemOp::Where, vec![pred.clone(), grad.clone(), lhs_zero]);
                let d_rhs = self.emit_elem(ElemOp::Where, vec![pred, rhs_zero, grad]);
                self.add_cotangent_ref(&args[0], d_lhs);
                self.add_cotangent_ref(&args[1], d_rhs);
            }
            ElemOp::Min => {
                let lhs = rewritten_args[0].clone();
                let rhs = rewritten_args[1].clone();
                let pred = self.emit_elem(ElemOp::Lt, vec![lhs.clone(), rhs]);
                let lhs_zero = self.zero_like(lhs);
                let rhs_zero = self.zero_like(rewritten_args[1].clone());
                let d_lhs =
                    self.emit_elem(ElemOp::Where, vec![pred.clone(), grad.clone(), lhs_zero]);
                let d_rhs = self.emit_elem(ElemOp::Where, vec![pred, rhs_zero, grad]);
                self.add_cotangent_ref(&args[0], d_lhs);
                self.add_cotangent_ref(&args[1], d_rhs);
            }
            ElemOp::Eq
            | ElemOp::Ne
            | ElemOp::Lt
            | ElemOp::Gt
            | ElemOp::Le
            | ElemOp::Ge
            | ElemOp::And
            | ElemOp::Or => {}
            ElemOp::Where => {
                let cond = rewritten_args[0].clone();
                let on_true = rewritten_args[1].clone();
                let on_false = rewritten_args[2].clone();
                let true_zero = self.zero_like(on_true);
                let false_zero = self.zero_like(on_false);
                let d_true =
                    self.emit_elem(ElemOp::Where, vec![cond.clone(), grad.clone(), true_zero]);
                let d_false = self.emit_elem(ElemOp::Where, vec![cond, false_zero, grad]);
                self.add_cotangent_ref(&args[1], d_true);
                self.add_cotangent_ref(&args[2], d_false);
            }
        }

        Ok(())
    }

    fn has_cotangent(&self, node: NodeId, output: u32) -> bool {
        self.cotangents.contains_key(&ref_key(node, output))
    }

    fn cotangent(&self, node: NodeId, output: u32) -> Option<Ref> {
        self.cotangents.get(&ref_key(node, output)).cloned()
    }

    fn primal_ref(&self, node: NodeId, output: u32) -> Result<Ref> {
        self.primal
            .get(node, output)
            .cloned()
            .ok_or(Error::MissingRef {
                node,
                output: OutputIndex(output),
            })
    }

    fn add_cotangent_ref(&mut self, r: &Ref, contribution: Ref) {
        self.add_cotangent_key(ref_key_of(r), contribution);
    }

    fn add_cotangent_key(&mut self, key: RefKey, contribution: Ref) {
        if let Some(existing) = self.cotangents.get(&key).cloned() {
            let sum = self.add(existing, contribution);
            self.cotangents.insert(key, sum);
        } else {
            self.cotangents.insert(key, contribution);
        }
    }

    fn emit_const(&mut self, val: ConstVal) -> Ref {
        let node = self.sink.emit(Node::Const { val });
        Ref::simple(node)
    }

    fn emit_elem(&mut self, op: ElemOp, args: Vec<Ref>) -> Ref {
        let node = self.sink.emit(Node::Elem { op, args });
        Ref::simple(node)
    }

    fn emit_cond_single(&mut self, pred: Ref, on_true: Ref, on_false: Ref) -> Ref {
        let node = self.sink.emit(Node::Cond {
            pred,
            then_refs: vec![on_true],
            else_refs: vec![on_false],
        });
        Ref::simple(node)
    }

    fn neg(&mut self, value: Ref) -> Ref {
        self.emit_elem(ElemOp::Neg, vec![value])
    }

    fn add(&mut self, lhs: Ref, rhs: Ref) -> Ref {
        self.emit_elem(ElemOp::Add, vec![lhs, rhs])
    }

    fn sub(&mut self, lhs: Ref, rhs: Ref) -> Ref {
        self.emit_elem(ElemOp::Sub, vec![lhs, rhs])
    }

    fn mul(&mut self, lhs: Ref, rhs: Ref) -> Ref {
        self.emit_elem(ElemOp::Mul, vec![lhs, rhs])
    }

    fn div(&mut self, lhs: Ref, rhs: Ref) -> Ref {
        self.emit_elem(ElemOp::Div, vec![lhs, rhs])
    }

    fn zero_like(&mut self, value: Ref) -> Ref {
        self.sub(value.clone(), value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn func(id: u32, name: &str, params: &[&str], nodes: Vec<Node>, outputs: Vec<Ref>) -> Function {
        Function {
            id: FuncId(id),
            name: Symbol::from(name),
            params: params.iter().map(|name| Symbol::from(*name)).collect(),
            nodes,
            outputs,
        }
    }

    fn program(functions: Vec<Function>) -> ir::Program {
        ir::Program {
            functions,
            builtins: HashMap::new(),
        }
    }

    #[test]
    fn inlines_non_tail_calls() {
        let callee = func(
            0,
            "inc",
            &["x"],
            vec![
                Node::Param { idx: 0 },
                Node::Const {
                    val: ConstVal::Int(1),
                },
                Node::Elem {
                    op: ElemOp::Add,
                    args: vec![Ref::simple(NodeId(0)), Ref::simple(NodeId(1))],
                },
            ],
            vec![Ref::simple(NodeId(2))],
        );
        let caller = func(
            1,
            "use_inc",
            &["y"],
            vec![
                Node::Param { idx: 0 },
                Node::Call {
                    func: FuncId(0),
                    args: vec![Ref::simple(NodeId(0))],
                },
                Node::Const {
                    val: ConstVal::Int(2),
                },
                Node::Elem {
                    op: ElemOp::Add,
                    args: vec![Ref::simple(NodeId(1)), Ref::simple(NodeId(2))],
                },
            ],
            vec![Ref::simple(NodeId(3))],
        );

        let rewritten = rewrite(&program(vec![callee, caller])).unwrap();
        let caller = rewritten
            .functions
            .iter()
            .find(|func| func.id == FuncId(1))
            .unwrap();

        assert!(
            caller
                .nodes
                .iter()
                .all(|node| !matches!(node, Node::Call { .. }))
        );
        assert!(matches!(
            caller.nodes.last(),
            Some(Node::Elem {
                op: ElemOp::Add,
                ..
            })
        ));
    }

    #[test]
    fn preserves_tail_self_calls() {
        let loop_fn = func(
            0,
            "loop",
            &["cond", "x"],
            vec![
                Node::Param { idx: 0 },
                Node::Param { idx: 1 },
                Node::Call {
                    func: FuncId(0),
                    args: vec![Ref::simple(NodeId(0)), Ref::simple(NodeId(1))],
                },
                Node::Cond {
                    pred: Ref::simple(NodeId(0)),
                    then_refs: vec![Ref::simple(NodeId(2))],
                    else_refs: vec![Ref::simple(NodeId(1))],
                },
            ],
            vec![Ref::simple(NodeId(3))],
        );

        let rewritten = rewrite(&program(vec![loop_fn])).unwrap();
        let loop_fn = &rewritten.functions[0];

        assert!(
            loop_fn
                .nodes
                .iter()
                .any(|node| matches!(node, Node::Call { func, .. } if *func == FuncId(0)))
        );
    }

    #[test]
    fn rejects_non_tail_recursion_past_depth_limit() {
        let bad = func(
            0,
            "bad",
            &["x"],
            vec![
                Node::Param { idx: 0 },
                Node::Call {
                    func: FuncId(0),
                    args: vec![Ref::simple(NodeId(0))],
                },
                Node::Elem {
                    op: ElemOp::Add,
                    args: vec![Ref::simple(NodeId(1)), Ref::simple(NodeId(0))],
                },
            ],
            vec![Ref::simple(NodeId(2))],
        );

        let err = rewrite_with_options(
            &program(vec![bad]),
            Options {
                max_inline_depth: 1,
            },
        )
        .unwrap_err();

        assert!(matches!(
            err,
            Error::InlineDepthExceeded {
                func: FuncId(0),
                ..
            }
        ));
    }

    #[test]
    fn lowers_simple_grad_through_mul() {
        let square = func(
            0,
            "square",
            &["x"],
            vec![
                Node::Param { idx: 0 },
                Node::Elem {
                    op: ElemOp::Mul,
                    args: vec![Ref::simple(NodeId(0)), Ref::simple(NodeId(0))],
                },
            ],
            vec![Ref::simple(NodeId(1))],
        );
        let caller = func(
            1,
            "grad_square",
            &["x"],
            vec![
                Node::Param { idx: 0 },
                Node::Grad {
                    func: FuncId(0),
                    args: vec![Ref::simple(NodeId(0))],
                },
            ],
            vec![Ref::simple(NodeId(1))],
        );

        let rewritten = rewrite(&program(vec![square, caller])).unwrap();
        let caller = rewritten
            .functions
            .iter()
            .find(|func| func.id == FuncId(1))
            .unwrap();

        assert!(
            caller
                .nodes
                .iter()
                .all(|node| !matches!(node, Node::Call { .. } | Node::Grad { .. }))
        );
        assert!(caller.nodes.iter().any(|node| matches!(
            node,
            Node::Const {
                val: ConstVal::Float(1.0)
            }
        )));
        assert!(matches!(
            caller.nodes.last(),
            Some(Node::Elem {
                op: ElemOp::Add,
                ..
            })
        ));
    }

    #[test]
    fn rejects_builtin_gradients_until_shape_support_exists() {
        let mut builtins = HashMap::new();
        builtins.insert(FuncId(99), BuiltinKind::Matmul);

        let target = func(
            0,
            "matmul_wrapper",
            &["a", "b"],
            vec![
                Node::Param { idx: 0 },
                Node::Param { idx: 1 },
                Node::Call {
                    func: FuncId(99),
                    args: vec![Ref::simple(NodeId(0)), Ref::simple(NodeId(1))],
                },
            ],
            vec![Ref::simple(NodeId(2))],
        );
        let caller = func(
            1,
            "grad_matmul",
            &["a", "b"],
            vec![
                Node::Param { idx: 0 },
                Node::Param { idx: 1 },
                Node::Grad {
                    func: FuncId(0),
                    args: vec![Ref::simple(NodeId(0)), Ref::simple(NodeId(1))],
                },
            ],
            vec![Ref::simple(NodeId(2))],
        );

        let err = rewrite(&ir::Program {
            functions: vec![target, caller],
            builtins,
        })
        .unwrap_err();

        assert!(matches!(
            err,
            Error::UnsupportedBuiltinGradient {
                func: FuncId(0),
                builtin: BuiltinKind::Matmul,
            }
        ));
    }
}
