use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use resin_core::{Accessor, ElementOperator, ParamTree};
use resin_dsl::{Node, NodeKind, RemapGatherInfo, RemapInfo, RemapScatterInfo, View};

use crate::program::{
    IrBuffer, IrBufferView, IrDispatch, IrElementwiseRpnKernel, IrKernel, IrMatmulKernel,
    IrProgram, IrReductionKernel, IrRemapKernel,
};
use crate::rpn::{ElementRpnExpr, RpnAtom};

/// Lowers a DSL graph (reachable from sinks / registered params) into [`IrProgram`].
#[derive(Default)]
pub struct IrProgramBuilder {
    buffer_memo: HashMap<*const Node, usize>,
    buffer_view_memo: HashMap<(*const Node, AccessorKey), usize>,
    buffers: Vec<IrBuffer>,
    buffer_views: Vec<IrBufferView>,
    queue: Vec<IrDispatch>,
    sinks: Vec<(String, usize)>,
    /// Buffer index → param name for `NodeKind::Param` buffers.
    param_at_buffer: HashMap<usize, String>,
    /// Overrides for param buffer names keyed by node pointer.
    param_name_overrides: HashMap<*const Node, String>,
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct AccessorKey {
    offset: u32,
    shape: Box<[u32]>,
    pitch: Box<[u32]>,
}

impl From<&Accessor> for AccessorKey {
    fn from(a: &Accessor) -> Self {
        Self {
            offset: a.offset,
            shape: a.shape.clone(),
            pitch: a.pitch.clone(),
        }
    }
}

impl IrProgramBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register_param(&mut self, name: impl Into<String>, view: &View) -> Result<(), String> {
        match &view.node.kind {
            NodeKind::Param { .. } => {}
            _ => return Err("register_param expected a param view".into()),
        }
        let name = name.into();
        if self
            .param_name_overrides
            .values()
            .any(|n| n == &name)
            || self.param_at_buffer.values().any(|n| n == &name)
        {
            return Err(format!("duplicate param name: {name:?}"));
        }
        self.param_name_overrides.insert(view.node_key(), name);
        let _ = self.build_view(view);
        Ok(())
    }

    /// Register all leaves of a [`ParamTree`] under dotted paths (optionally prefixed).
    pub fn register_param_tree<M: ParamTree<Leaf = View>>(
        &mut self,
        tree: &M,
        prefix: &str,
    ) -> Result<(), String> {
        for (path, view) in tree.flatten() {
            let name = join_path(prefix, &path);
            self.register_param(name, view)?;
        }
        Ok(())
    }

    pub fn build_sink(&mut self, name: impl Into<String>, view: &View) -> Result<(), String> {
        let name = name.into();
        if self.sinks.iter().any(|(n, _)| n == &name) {
            return Err(format!("sink name conflict: {name}"));
        }
        let view_index = self.build_view(view);
        self.sinks.push((name, view_index));
        Ok(())
    }

    pub fn build_sink_tree<M: ParamTree<Leaf = View>>(
        &mut self,
        prefix: &str,
        tree: &M,
    ) -> Result<(), String> {
        for (path, view) in tree.flatten() {
            let name = join_path(prefix, &path);
            self.build_sink(name, view)?;
        }
        Ok(())
    }

    pub fn finish(self) -> Result<IrProgram, String> {
        let mut param_buffers: Vec<(String, usize)> = self
            .param_at_buffer
            .into_iter()
            .map(|(index, name)| (name, index))
            .collect();
        let mut seen = HashSet::new();
        for (name, _) in &param_buffers {
            if !seen.insert(name.clone()) {
                return Err(format!("duplicate param name: {name:?}"));
            }
        }
        param_buffers.sort_by(|a, b| a.1.cmp(&b.1));

        Ok(IrProgram {
            param_buffers,
            sinks: self.sinks,
            queue: self.queue,
            buffers: self.buffers,
            buffer_views: self.buffer_views,
        })
    }

    fn build_view(&mut self, view: &View) -> usize {
        let key = (view.node_key(), AccessorKey::from(&view.accessor));
        if let Some(&idx) = self.buffer_view_memo.get(&key) {
            return idx;
        }
        let buffer_index = self.build_node(Arc::clone(&view.node));
        let idx = self.buffer_views.len();
        self.buffer_views.push(IrBufferView {
            buffer_index,
            accessor: view.accessor.clone(),
        });
        self.buffer_view_memo.insert(key, idx);
        idx
    }

    fn build_node(&mut self, node: Arc<Node>) -> usize {
        let key = Arc::as_ptr(&node);
        if let Some(&idx) = self.buffer_memo.get(&key) {
            return idx;
        }

        let input_views: Vec<usize> = node.args.iter().map(|v| self.build_view(v)).collect();

        let buffer = allocate_buffer(&node);
        let buffer_index = self.buffers.len();
        self.buffers.push(buffer);
        self.buffer_memo.insert(key, buffer_index);

        if let NodeKind::Param { name } = &node.kind {
            let name = self
                .param_name_overrides
                .get(&key)
                .cloned()
                .unwrap_or_else(|| name.to_string());
            self.param_at_buffer.insert(buffer_index, name);
        }

        if let Some(kernel) = kernel_for_node(&node) {
            self.queue.push(IrDispatch {
                kernel,
                arg_view_indices: input_views,
                output_buffer_index: buffer_index,
            });
        }

        buffer_index
    }
}

fn join_path(prefix: &str, path: &str) -> String {
    if prefix.is_empty() {
        path.to_string()
    } else if path.is_empty() {
        prefix.to_string()
    } else {
        format!("{prefix}.{path}")
    }
}

fn allocate_buffer(node: &Node) -> IrBuffer {
    match &node.kind {
        NodeKind::Const { init } => IrBuffer {
            shape: node.shape.clone(),
            etype: node.etype,
            init: Some(init.clone()),
            readonly: true,
        },
        _ => IrBuffer {
            shape: node.shape.clone(),
            etype: node.etype,
            init: None,
            readonly: false,
        },
    }
}

fn kernel_for_node(node: &Node) -> Option<IrKernel> {
    let arg_accessors: Vec<Accessor> = node.args.iter().map(|v| v.accessor.clone()).collect();
    match &node.kind {
        NodeKind::Const { .. } | NodeKind::Param { .. } => None,
        NodeKind::Elementwise { op } => {
            let arg_etypes: Vec<_> = node.args.iter().map(|v| v.etype()).collect();
            Some(IrKernel::ElementwiseRpn(IrElementwiseRpnKernel {
                arg_accessors,
                arg_etypes,
                etype: node.etype,
                shape: node.shape.clone(),
                rpn_expr: rpn_for_elementwise(*op, node.args.len()),
                clear_output_before_dispatch: false,
            }))
        }
        NodeKind::Matmul => Some(IrKernel::Matmul(IrMatmulKernel {
            arg_accessors,
            etype: node.etype,
            shape: node.shape.clone(),
            clear_output_before_dispatch: false,
        })),
        NodeKind::Reduction { op, axes } => Some(IrKernel::Reduction(IrReductionKernel {
            arg_accessors,
            etype: node.etype,
            shape: node.shape.clone(),
            operator: *op,
            axes: axes.clone(),
            clear_output_before_dispatch: false,
        })),
        NodeKind::Remap { info } => {
            let arg_etypes: Vec<_> = node.args.iter().map(|v| v.etype()).collect();
            let clear_output_before_dispatch = match info {
                RemapInfo::Scatter(RemapScatterInfo { operator, .. }) => operator.is_none(),
                RemapInfo::Gather(RemapGatherInfo { .. }) => true,
            };
            Some(IrKernel::Remap(IrRemapKernel {
                arg_accessors,
                arg_etypes,
                etype: node.etype,
                shape: node.shape.clone(),
                info: info.clone(),
                clear_output_before_dispatch,
            }))
        }
    }
}

fn rpn_for_elementwise(op: ElementOperator, arity: usize) -> ElementRpnExpr {
    let mut atoms = Vec::new();
    for i in 0..arity {
        atoms.push(RpnAtom::Arg(i as u32));
    }
    atoms.push(RpnAtom::Op(op));
    ElementRpnExpr { atoms }
}

#[cfg(test)]
mod tests {
    use super::*;
    use resin_core::F4;
    use resin_dsl::param;

    #[test]
    fn lower_add_of_params() {
        let a = param([2, 3], F4, "a");
        let b = param([2, 3], F4, "b");
        let c = &a + &b;

        let mut builder = IrProgramBuilder::new();
        builder.register_param("a", &a).unwrap();
        builder.register_param("b", &b).unwrap();
        builder.build_sink("out", &c).unwrap();
        let program = builder.finish().unwrap();

        assert_eq!(program.param_buffers.len(), 2);
        assert_eq!(program.sinks.len(), 1);
        assert_eq!(program.queue.len(), 1);
        assert!(matches!(
            program.queue[0].kernel,
            IrKernel::ElementwiseRpn(_)
        ));
    }
}
