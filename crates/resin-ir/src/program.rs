use resin_core::{
    Accessor, BinaryAssocElementOperator, ElementType, Tree,
};

use crate::refs::{validate_tree_indices, BufferRef, BufferViewRef};
use crate::remap::{RemapGatherInfo, RemapInfo, RemapScatterInfo};
use crate::rpn::ElementRpnExpr;

/// Compiled middle-end program.
///
/// Param leaves are [`BufferRef`]s into [`IrProgram::buffers`]; sink leaves are
/// [`BufferViewRef`]s into [`IrProgram::buffer_views`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IrProgram<P = BufferRef, S = BufferViewRef>
where
    P: Tree<BufferRef>,
    S: Tree<BufferViewRef>,
{
    pub params: P,
    pub sinks: S,
    pub queue: Vec<IrDispatch>,
    pub buffers: Vec<IrBuffer>,
    pub buffer_views: Vec<IrBufferView>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IrBuffer {
    pub shape: Box<[u32]>,
    pub etype: ElementType,
    pub init: Option<Box<[u8]>>,
    pub readonly: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IrBufferView {
    pub buffer_index: BufferRef,
    pub accessor: Accessor,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IrDispatch {
    pub kernel: IrKernel,
    pub arg_view_indices: Vec<BufferViewRef>,
    pub output_buffer_index: BufferRef,
}

/// Middle-end kernel. `etype` is the output buffer element type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IrKernel {
    ElementwiseRpn(IrElementwiseRpnKernel),
    Matmul(IrMatmulKernel),
    Reduction(IrReductionKernel),
    Remap(IrRemapKernel),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IrElementwiseRpnKernel {
    pub arg_accessors: Vec<Accessor>,
    pub arg_etypes: Vec<ElementType>,
    pub etype: ElementType,
    pub shape: Box<[u32]>,
    pub rpn_expr: ElementRpnExpr,
    pub clear_output_before_dispatch: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IrMatmulKernel {
    pub arg_accessors: Vec<Accessor>,
    pub etype: ElementType,
    pub shape: Box<[u32]>,
    pub clear_output_before_dispatch: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IrReductionKernel {
    pub arg_accessors: Vec<Accessor>,
    pub etype: ElementType,
    pub shape: Box<[u32]>,
    pub operator: BinaryAssocElementOperator,
    pub axes: Box<[u32]>,
    pub clear_output_before_dispatch: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IrRemapKernel {
    pub arg_accessors: Vec<Accessor>,
    pub arg_etypes: Vec<ElementType>,
    pub etype: ElementType,
    pub shape: Box<[u32]>,
    pub info: RemapInfo,
    pub clear_output_before_dispatch: bool,
}

impl IrKernel {
    pub fn arg_accessors(&self) -> &[Accessor] {
        match self {
            IrKernel::ElementwiseRpn(k) => &k.arg_accessors,
            IrKernel::Matmul(k) => &k.arg_accessors,
            IrKernel::Reduction(k) => &k.arg_accessors,
            IrKernel::Remap(k) => &k.arg_accessors,
        }
    }

    pub fn etype(&self) -> ElementType {
        match self {
            IrKernel::ElementwiseRpn(k) => k.etype,
            IrKernel::Matmul(k) => k.etype,
            IrKernel::Reduction(k) => k.etype,
            IrKernel::Remap(k) => k.etype,
        }
    }

    pub fn shape(&self) -> &[u32] {
        match self {
            IrKernel::ElementwiseRpn(k) => &k.shape,
            IrKernel::Matmul(k) => &k.shape,
            IrKernel::Reduction(k) => &k.shape,
            IrKernel::Remap(k) => &k.shape,
        }
    }

    pub fn clear_output_before_dispatch(&self) -> bool {
        match self {
            IrKernel::ElementwiseRpn(k) => k.clear_output_before_dispatch,
            IrKernel::Matmul(k) => k.clear_output_before_dispatch,
            IrKernel::Reduction(k) => k.clear_output_before_dispatch,
            IrKernel::Remap(k) => k.clear_output_before_dispatch,
        }
    }

    /// Element type of each argument buffer (defaults to output etype when unspecified).
    pub fn operand_etypes(&self) -> Vec<ElementType> {
        match self {
            IrKernel::ElementwiseRpn(k) => k.arg_etypes.clone(),
            IrKernel::Remap(k) => k.arg_etypes.clone(),
            IrKernel::Matmul(k) => vec![k.etype; k.arg_accessors.len()],
            IrKernel::Reduction(k) => vec![k.etype; k.arg_accessors.len()],
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        match self {
            IrKernel::ElementwiseRpn(k) => k.validate(),
            IrKernel::Matmul(k) => k.validate(),
            IrKernel::Reduction(k) => k.validate(),
            IrKernel::Remap(k) => k.validate(),
        }
    }
}

impl IrElementwiseRpnKernel {
    pub fn validate(&self) -> Result<(), String> {
        for accessor in &self.arg_accessors {
            if accessor.shape.as_ref() != self.shape.as_ref() {
                return Err(format!(
                    "elementwise arg shape {:?} != kernel shape {:?}",
                    accessor.shape, self.shape
                ));
            }
        }
        if self.arg_etypes.len() != self.arg_accessors.len() {
            return Err("elementwise arg_etypes length mismatch".into());
        }
        Ok(())
    }
}

impl IrMatmulKernel {
    pub fn k(&self) -> u32 {
        self.arg_accessors[0].shape[self.arg_accessors[0].shape.len() - 1]
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.arg_accessors.len() != 2 {
            return Err("matmul requires exactly two arguments".into());
        }
        let a0 = &self.arg_accessors[0];
        let a1 = &self.arg_accessors[1];
        if a0.shape.len() < 2 || a1.shape.len() < 2 {
            return Err("matmul arguments must be at least rank 2".into());
        }
        if a0.shape[a0.shape.len() - 1] != a1.shape[a1.shape.len() - 2] {
            return Err("matmul inner dimension mismatch".into());
        }
        if a0.shape[..a0.shape.len() - 2] != a1.shape[..a1.shape.len() - 2] {
            return Err("matmul batch dimensions mismatch".into());
        }
        if a0.shape.len() != a1.shape.len() || a0.shape.len() != self.shape.len() {
            return Err("matmul rank mismatch".into());
        }
        let expected: Box<[u32]> = a0.shape[..a0.shape.len() - 1]
            .iter()
            .chain(std::iter::once(&a1.shape[a1.shape.len() - 1]))
            .copied()
            .collect();
        if self.shape.as_ref() != expected.as_ref() {
            return Err(format!(
                "matmul output shape {:?} != expected {:?}",
                self.shape, expected
            ));
        }
        Ok(())
    }
}

impl IrReductionKernel {
    pub fn input_shape(&self) -> &[u32] {
        &self.arg_accessors[0].shape
    }

    pub fn reduced_count(&self) -> u32 {
        self.axes
            .iter()
            .map(|&axis| self.input_shape()[axis as usize])
            .product()
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.arg_accessors.len() != 1 {
            return Err("reduction requires exactly one argument".into());
        }
        let input_shape = self.input_shape();
        if input_shape.len() != self.shape.len() {
            return Err("reduction input/output rank mismatch".into());
        }
        for &axis in &self.axes {
            if axis as usize >= self.shape.len() {
                return Err(format!("reduction axis {axis} out of range"));
            }
            if self.shape[axis as usize] != 1 {
                return Err(format!(
                    "reduction output axis {axis} must be size 1, got {}",
                    self.shape[axis as usize]
                ));
            }
        }
        for (i, (&out_dim, &in_dim)) in self.shape.iter().zip(input_shape.iter()).enumerate() {
            if self.axes.contains(&(i as u32)) {
                continue;
            }
            if out_dim != in_dim {
                return Err(format!(
                    "reduction preserved axis {i}: output {out_dim} != input {in_dim}"
                ));
            }
        }
        Ok(())
    }
}

impl IrRemapKernel {
    pub fn validate(&self) -> Result<(), String> {
        if self.arg_etypes.len() != self.arg_accessors.len() {
            return Err("remap arg_etypes length mismatch".into());
        }
        match &self.info {
            RemapInfo::Scatter(RemapScatterInfo { accessor, .. }) => {
                if let Some(accessor) = accessor {
                    if self.arg_accessors.len() != 1 {
                        return Err("scatter with accessor requires one argument".into());
                    }
                    if accessor.shape.as_ref() != self.arg_accessors[0].shape.as_ref() {
                        return Err("scatter accessor shape mismatch".into());
                    }
                } else if self.arg_accessors.len() != 2 {
                    return Err("scatter without accessor requires two arguments".into());
                } else {
                    let source_accessor = &self.arg_accessors[0];
                    let indices_accessor = &self.arg_accessors[1];
                    if indices_accessor.shape[indices_accessor.shape.len() - 1]
                        != self.shape.len() as u32
                    {
                        return Err("scatter indices trailing dim must match output rank".into());
                    }
                    if indices_accessor.shape[..indices_accessor.shape.len() - 1]
                        != source_accessor.shape[..]
                    {
                        return Err("scatter indices/source batch shape mismatch".into());
                    }
                }
            }
            RemapInfo::Gather(RemapGatherInfo {
                accessor,
                source_shape,
            }) => {
                if accessor.is_none() && source_shape.is_none() {
                    if self.arg_accessors.len() != 1 {
                        return Err("gather with implicit source requires one argument".into());
                    }
                } else if let (Some(accessor), Some(source_shape)) = (accessor, source_shape) {
                    if self.arg_accessors.len() != 2 {
                        return Err("gather with accessor requires two arguments".into());
                    }
                    let source_accessor = &self.arg_accessors[0];
                    let indices_accessor = &self.arg_accessors[1];
                    if indices_accessor.shape[indices_accessor.shape.len() - 1] != accessor.rank() as u32
                    {
                        return Err("gather indices trailing dim must match accessor rank".into());
                    }
                    if indices_accessor.shape[..indices_accessor.shape.len() - 1]
                        != source_accessor.shape[..]
                    {
                        return Err("gather indices/source batch shape mismatch".into());
                    }
                    if accessor.shape.as_ref() != source_shape.as_ref() {
                        return Err("gather accessor/source_shape mismatch".into());
                    }
                } else {
                    return Err(format!("invalid RemapGatherInfo: {accessor:?}, {source_shape:?}"));
                }
            }
        }
        Ok(())
    }
}

impl<P: Tree<BufferRef>, S: Tree<BufferViewRef>> IrProgram<P, S> {
    pub fn validate(&self) -> Result<(), String> {
        for dispatch in &self.queue {
            dispatch.kernel.validate()?;
            for view_ref in &dispatch.arg_view_indices {
                if view_ref.index() >= self.buffer_views.len() {
                    return Err(format!(
                        "dispatch arg view index {} out of range",
                        view_ref.index()
                    ));
                }
            }
            if dispatch.output_buffer_index.index() >= self.buffers.len() {
                return Err(format!(
                    "dispatch output buffer index {} out of range",
                    dispatch.output_buffer_index.index()
                ));
            }
        }
        validate_tree_indices(&self.params, self.buffers.len(), "param buffer")?;
        validate_tree_indices(&self.sinks, self.buffer_views.len(), "sink view")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use resin_core::{
        BinaryElementOperator, ElementOperator, F4, UnaryElementOperator,
    };

    use crate::rpn::{ElementRpnExpr, RpnAtom};

    #[test]
    fn matmul_kernel_validates() {
        let kernel = IrKernel::Matmul(IrMatmulKernel {
            arg_accessors: vec![
                Accessor::dense([2, 4], 0),
                Accessor::dense([4, 3], 0),
            ],
            etype: F4,
            shape: Box::from([2, 3]),
            clear_output_before_dispatch: false,
        });
        kernel.validate().unwrap();
        assert_eq!(kernel.shape(), &[2, 3]);
    }

    #[test]
    fn elementwise_rpn_kernel_validates() {
        let shape: Box<[u32]> = Box::from([2, 3]);
        let kernel = IrKernel::ElementwiseRpn(IrElementwiseRpnKernel {
            arg_accessors: vec![Accessor::dense(shape.clone(), 0), Accessor::dense(shape.clone(), 0)],
            arg_etypes: vec![F4, F4],
            etype: F4,
            shape,
            rpn_expr: ElementRpnExpr {
                atoms: vec![
                    RpnAtom::Arg(0),
                    RpnAtom::Arg(1),
                    RpnAtom::Op(ElementOperator::Binary(BinaryElementOperator::Assoc(
                        BinaryAssocElementOperator::Add,
                    ))),
                ],
            },
            clear_output_before_dispatch: false,
        });
        kernel.validate().unwrap();
    }

    #[test]
    fn reduction_kernel_validates() {
        let kernel = IrKernel::Reduction(IrReductionKernel {
            arg_accessors: vec![Accessor::dense([2, 3, 4], 0)],
            etype: F4,
            shape: Box::from([2, 1, 4]),
            operator: BinaryAssocElementOperator::Add,
            axes: Box::from([1]),
            clear_output_before_dispatch: false,
        });
        kernel.validate().unwrap();
        assert_eq!(
            kernel
                .as_reduction()
                .unwrap()
                .reduced_count(),
            3
        );
    }

    #[test]
    fn program_validates_tree_io() {
        use resin_macros::Tree;

        #[derive(Debug, Clone, PartialEq, Eq, Tree)]
        struct ParamTree<T> {
            weight: T,
            bias: T,
        }

        #[derive(Debug, Clone, PartialEq, Eq, Tree)]
        struct SinkTree<T> {
            loss: T,
            grads: ParamTree<T>,
        }

        let program = IrProgram::<ParamTree<BufferRef>, SinkTree<BufferViewRef>> {
            params: ParamTree {
                weight: BufferRef::new(0),
                bias: BufferRef::new(1),
            },
            sinks: SinkTree {
                loss: BufferViewRef::new(0),
                grads: ParamTree {
                    weight: BufferViewRef::new(1),
                    bias: BufferViewRef::new(2),
                },
            },
            queue: vec![],
            buffers: vec![
                IrBuffer {
                    shape: Box::from([4, 4]),
                    etype: F4,
                    init: None,
                    readonly: false,
                },
                IrBuffer {
                    shape: Box::from([4]),
                    etype: F4,
                    init: None,
                    readonly: false,
                },
            ],
            buffer_views: vec![
                IrBufferView {
                    buffer_index: BufferRef::new(0),
                    accessor: Accessor::dense([4, 4], 0),
                },
                IrBufferView {
                    buffer_index: BufferRef::new(1),
                    accessor: Accessor::dense([4], 0),
                },
                IrBufferView {
                    buffer_index: BufferRef::new(0),
                    accessor: Accessor::dense([4, 4], 0),
                },
            ],
        };
        program.validate().unwrap();
    }

    #[test]
    fn unary_elementwise_rpn() {
        let shape: Box<[u32]> = Box::from([4]);
        let kernel = IrKernel::ElementwiseRpn(IrElementwiseRpnKernel {
            arg_accessors: vec![Accessor::dense(shape.clone(), 0)],
            arg_etypes: vec![F4],
            etype: F4,
            shape,
            rpn_expr: ElementRpnExpr {
                atoms: vec![
                    RpnAtom::Arg(0),
                    RpnAtom::Op(ElementOperator::Unary(UnaryElementOperator::Neg)),
                ],
            },
            clear_output_before_dispatch: false,
        });
        kernel.validate().unwrap();
    }
}

impl IrKernel {
    pub fn as_matmul(&self) -> Option<&IrMatmulKernel> {
        match self {
            IrKernel::Matmul(k) => Some(k),
            _ => None,
        }
    }

    pub fn as_reduction(&self) -> Option<&IrReductionKernel> {
        match self {
            IrKernel::Reduction(k) => Some(k),
            _ => None,
        }
    }
}