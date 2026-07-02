use std::collections::{HashMap, HashSet};

use resin_core::{join_param_path, Accessor, ElementOperator, ElementType, Tree};
use resin_dsl::{NodeKind, NodeRef, RemapInfo, View};
use serde::{Deserialize, Serialize};

use crate::rpn::{ElementRpnExpr, RpnAtom};

/// Intermediate program: also the in-place builder (register params / sinks, lower nodes).
#[derive(Debug, Clone, Default)]
pub struct IrProgram {
    pub param_buffers: Vec<(String, usize)>,
    pub sinks: Vec<(String, usize)>,
    pub queue: Vec<IrDispatch>,
    pub buffers: Vec<IrBuffer>,
    pub buffer_views: Vec<IrBufferView>,

    // Build-time memo (identity-keyed). Extraneous once the program is sealed /
    // lowered; kept on the struct so `IrProgram` is its own builder.
    buffer_memo: HashMap<NodeRef, usize>,
    buffer_view_memo: HashMap<(NodeRef, AccessorKey), usize>,
    param_at_buffer: HashMap<usize, String>,
    param_name_overrides: HashMap<NodeRef, String>,
}

// Manual Serialize/Deserialize without build maps.
impl Serialize for IrProgram {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut st = serializer.serialize_struct("IrProgram", 5)?;
        st.serialize_field("param_buffers", &self.param_buffers)?;
        st.serialize_field("sinks", &self.sinks)?;
        st.serialize_field("queue", &self.queue)?;
        st.serialize_field("buffers", &self.buffers)?;
        st.serialize_field("buffer_views", &self.buffer_views)?;
        st.end()
    }
}

impl<'de> Deserialize<'de> for IrProgram {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct Wire {
            param_buffers: Vec<(String, usize)>,
            sinks: Vec<(String, usize)>,
            queue: Vec<IrDispatch>,
            buffers: Vec<IrBuffer>,
            buffer_views: Vec<IrBufferView>,
        }
        let w = Wire::deserialize(deserializer)?;
        Ok(Self {
            param_buffers: w.param_buffers,
            sinks: w.sinks,
            queue: w.queue,
            buffers: w.buffers,
            buffer_views: w.buffer_views,
            ..Self::default()
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
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

impl IrProgram {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register_param(&mut self, name: impl Into<String>, view: &View) -> Result<(), String> {
        match &view.node_ref.kind {
            NodeKind::Param => {}
            _ => return Err("register_param expected a param view".into()),
        }
        let name = name.into();
        if self.param_name_overrides.values().any(|n| n == &name)
            || self.param_at_buffer.values().any(|n| n == &name)
        {
            return Err(format!("duplicate param name: {name:?}"));
        }
        self.param_name_overrides
            .insert(view.node_ref.clone(), name);
        let _ = self.build_view(view);
        Ok(())
    }

    pub fn register_tree<M: Tree<Leaf = View>>(
        &mut self,
        tree: &M,
        prefix: &str,
    ) -> Result<(), String> {
        for (path, view) in tree.flatten() {
            self.register_param(join_param_path(prefix, &path), view)?;
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

    pub fn build_sink_tree<M: Tree<Leaf = View>>(
        &mut self,
        prefix: &str,
        tree: &M,
    ) -> Result<(), String> {
        for (path, view) in tree.flatten() {
            self.build_sink(join_param_path(prefix, &path), view)?;
        }
        Ok(())
    }

    /// Freeze param name table from build-time maps (idempotent).
    pub fn seal_params(&mut self) -> Result<(), String> {
        let mut param_buffers: Vec<(String, usize)> = self
            .param_at_buffer
            .iter()
            .map(|(&index, name)| (name.clone(), index))
            .collect();
        let mut seen = HashSet::new();
        for (name, _) in &param_buffers {
            if !seen.insert(name.clone()) {
                return Err(format!("duplicate param name: {name:?}"));
            }
        }
        param_buffers.sort_by_key(|a| a.1);
        self.param_buffers = param_buffers;
        Ok(())
    }

    fn build_view(&mut self, view: &View) -> usize {
        let key = (view.node_ref.clone(), AccessorKey::from(&view.accessor));
        if let Some(&idx) = self.buffer_view_memo.get(&key) {
            return idx;
        }
        let buffer_index = self.build_node(view.node_ref.clone());
        let idx = self.buffer_views.len();
        self.buffer_views.push(IrBufferView {
            buffer_index,
            accessor: view.accessor.clone(),
        });
        self.buffer_view_memo.insert(key, idx);
        idx
    }

    fn build_node(&mut self, node: NodeRef) -> usize {
        if let Some(&idx) = self.buffer_memo.get(&node) {
            return idx;
        }

        let input_views: Vec<usize> = node.args.iter().map(|v| self.build_view(v)).collect();
        let buffer = allocate_buffer(&node);
        let buffer_index = self.buffers.len();
        self.buffers.push(buffer);
        self.buffer_memo.insert(node.clone(), buffer_index);

        if let NodeKind::Param = &node.kind {
            let name = self
                .param_name_overrides
                .get(&node)
                .expect("param must be registered with a buffer name before lowering")
                .clone();
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

fn allocate_buffer(node: &resin_dsl::Node) -> IrBuffer {
    match &node.kind {
        NodeKind::Const(c) => IrBuffer {
            shape: node.shape.clone(),
            etype: node.element_type,
            init: Some(c.init.clone()),
            readonly: true,
        },
        _ => IrBuffer {
            shape: node.shape.clone(),
            etype: node.element_type,
            init: None,
            readonly: false,
        },
    }
}

fn kernel_for_node(node: &resin_dsl::Node) -> Option<IrKernel> {
    let arg_accessors: Vec<Accessor> = node.args.iter().map(|v| v.accessor.clone()).collect();
    match &node.kind {
        NodeKind::Const(_) | NodeKind::Param => None,
        NodeKind::Elementwise(k) => {
            let arg_etypes: Vec<_> = node.args.iter().map(|v| v.element_type()).collect();
            Some(IrKernel::ElementwiseRpn(IrElementwiseRpnKernel {
                arg_accessors,
                arg_etypes,
                etype: node.element_type,
                shape: node.shape.clone(),
                rpn_expr: rpn_for_elementwise(k.op, node.args.len()),
                clear_output_before_dispatch: false,
            }))
        }
        NodeKind::Matmul(_) => Some(IrKernel::Matmul(IrMatmulKernel {
            arg_accessors,
            etype: node.element_type,
            shape: node.shape.clone(),
            clear_output_before_dispatch: false,
        })),
        NodeKind::Reduction(k) => Some(IrKernel::Reduction(IrReductionKernel {
            arg_accessors,
            etype: node.element_type,
            shape: node.shape.clone(),
            operator: k.op,
            axes: k.axes.clone(),
            clear_output_before_dispatch: false,
        })),
        NodeKind::Remap(k) => {
            let arg_etypes: Vec<_> = node.args.iter().map(|v| v.element_type()).collect();
            // Always clear: overwrite scatter and gather need a clean buffer, and
            // scatter-add (atomic accumulate) must start from zeros every run or
            // the second `Interp::run` double-counts stale output (adjoint paths).
            Some(IrKernel::Remap(IrRemapKernel {
                arg_accessors,
                arg_etypes,
                etype: node.element_type,
                shape: node.shape.clone(),
                info: k.info.clone(),
                clear_output_before_dispatch: true,
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IrBuffer {
    pub shape: Box<[u32]>,
    pub etype: ElementType,
    pub init: Option<Box<[u8]>>,
    pub readonly: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IrBufferView {
    pub buffer_index: usize,
    pub accessor: Accessor,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IrDispatch {
    pub kernel: IrKernel,
    pub arg_view_indices: Vec<usize>,
    pub output_buffer_index: usize,
}

/// Kernel payload variants — shared metadata (`shape`, `etype`, …) lives on each struct,
/// same pattern as [`resin_dsl::NodeKind`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum IrKernel {
    ElementwiseRpn(IrElementwiseRpnKernel),
    Matmul(IrMatmulKernel),
    Reduction(IrReductionKernel),
    Remap(IrRemapKernel),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IrElementwiseRpnKernel {
    pub arg_accessors: Vec<Accessor>,
    pub arg_etypes: Vec<ElementType>,
    pub etype: ElementType,
    pub shape: Box<[u32]>,
    pub rpn_expr: ElementRpnExpr,
    pub clear_output_before_dispatch: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IrMatmulKernel {
    pub arg_accessors: Vec<Accessor>,
    pub etype: ElementType,
    pub shape: Box<[u32]>,
    pub clear_output_before_dispatch: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IrReductionKernel {
    pub arg_accessors: Vec<Accessor>,
    pub etype: ElementType,
    pub shape: Box<[u32]>,
    pub operator: ElementOperator,
    pub axes: Box<[u32]>,
    pub clear_output_before_dispatch: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IrRemapKernel {
    pub arg_accessors: Vec<Accessor>,
    pub arg_etypes: Vec<ElementType>,
    pub etype: ElementType,
    pub shape: Box<[u32]>,
    pub info: RemapInfo,
    pub clear_output_before_dispatch: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use resin_core::F4;
    use resin_dsl::param;

    #[test]
    fn lower_add_of_params() {
        let a = param([2, 3], F4);
        let b = param([2, 3], F4);
        let c = &a + &b;

        let mut program = IrProgram::new();
        program.register_param("a", &a).unwrap();
        program.register_param("b", &b).unwrap();
        program.build_sink("out", &c).unwrap();
        program.seal_params().unwrap();

        assert_eq!(program.param_buffers.len(), 2);
        assert_eq!(program.sinks.len(), 1);
        assert_eq!(program.queue.len(), 1);
        assert!(matches!(
            program.queue[0].kernel,
            IrKernel::ElementwiseRpn(_)
        ));
    }
}
