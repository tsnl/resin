use resin_core::{
    Accessor, BinaryAssocElementOperator, ElementType, Tree,
};
use serde::{Deserialize, Serialize};

use crate::error::IrError;
use crate::refs::{validate_tree_indices, BufferRef, BufferViewRef};
use crate::remap::{RemapGatherInfo, RemapInfo, RemapScatterInfo};
use crate::rpn::ElementRpnExpr;

/// Compiled middle-end program.
///
/// Param leaves are [`BufferRef`]s into [`IrProgram::buffers`]; sink leaves are
/// [`BufferViewRef`]s into [`IrProgram::buffer_views`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IrBuffer {
    pub shape: Box<[u32]>,
    pub element_type: ElementType,
    pub init: Option<Box<[u8]>>,
    pub readonly: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IrBufferView {
    pub buffer_index: BufferRef,
    pub accessor: Accessor,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IrDispatch {
    pub kernel: IrKernel,
    pub arg_view_indices: Vec<BufferViewRef>,
    pub output_buffer_index: BufferRef,
}

/// Middle-end kernel. `element_type` is the output buffer element type.
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
    pub arg_element_types: Vec<ElementType>,
    pub element_type: ElementType,
    pub shape: Box<[u32]>,
    pub rpn_expr: ElementRpnExpr,
    pub clear_output_before_dispatch: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IrMatmulKernel {
    pub arg_accessors: Vec<Accessor>,
    pub element_type: ElementType,
    pub shape: Box<[u32]>,
    pub clear_output_before_dispatch: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IrReductionKernel {
    pub arg_accessors: Vec<Accessor>,
    pub element_type: ElementType,
    pub shape: Box<[u32]>,
    pub operator: BinaryAssocElementOperator,
    pub axes: Box<[u32]>,
    pub clear_output_before_dispatch: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IrRemapKernel {
    pub arg_accessors: Vec<Accessor>,
    pub arg_element_types: Vec<ElementType>,
    pub element_type: ElementType,
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

    pub fn element_type(&self) -> ElementType {
        match self {
            IrKernel::ElementwiseRpn(k) => k.element_type,
            IrKernel::Matmul(k) => k.element_type,
            IrKernel::Reduction(k) => k.element_type,
            IrKernel::Remap(k) => k.element_type,
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

    /// Element type of each argument buffer (defaults to output element type when unspecified).
    pub fn operand_element_types(&self) -> Vec<ElementType> {
        match self {
            IrKernel::ElementwiseRpn(k) => k.arg_element_types.clone(),
            IrKernel::Remap(k) => k.arg_element_types.clone(),
            IrKernel::Matmul(k) => vec![k.element_type; k.arg_accessors.len()],
            IrKernel::Reduction(k) => vec![k.element_type; k.arg_accessors.len()],
        }
    }

    pub fn validate(&self) -> Result<(), IrError> {
        match self {
            IrKernel::ElementwiseRpn(k) => k.validate(),
            IrKernel::Matmul(k) => k.validate(),
            IrKernel::Reduction(k) => k.validate(),
            IrKernel::Remap(k) => k.validate(),
        }
    }

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

impl IrElementwiseRpnKernel {
    pub fn validate(&self) -> Result<(), IrError> {
        for accessor in &self.arg_accessors {
            if accessor.shape.as_ref() != self.shape.as_ref() {
                return Err(IrError::ElementwiseArgShape {
                    arg: accessor.shape.clone(),
                    kernel: self.shape.clone(),
                });
            }
        }
        if self.arg_element_types.len() != self.arg_accessors.len() {
            return Err(IrError::ElementwiseArgElementTypesLen {
                element_types: self.arg_element_types.len(),
                accessors: self.arg_accessors.len(),
            });
        }
        Ok(())
    }
}

impl IrMatmulKernel {
    pub fn k(&self) -> u32 {
        self.arg_accessors[0].shape[self.arg_accessors[0].shape.len() - 1]
    }

    pub fn validate(&self) -> Result<(), IrError> {
        if self.arg_accessors.len() != 2 {
            return Err(IrError::MatmulArgCount {
                got: self.arg_accessors.len(),
            });
        }
        let a0 = &self.arg_accessors[0];
        let a1 = &self.arg_accessors[1];
        if a0.shape.len() < 2 || a1.shape.len() < 2 {
            return Err(IrError::MatmulRankTooLow);
        }
        if a0.shape[a0.shape.len() - 1] != a1.shape[a1.shape.len() - 2] {
            return Err(IrError::MatmulInnerDim {
                lhs: a0.shape[a0.shape.len() - 1],
                rhs: a1.shape[a1.shape.len() - 2],
            });
        }
        if a0.shape[..a0.shape.len() - 2] != a1.shape[..a1.shape.len() - 2] {
            return Err(IrError::MatmulBatchDims);
        }
        if a0.shape.len() != a1.shape.len() || a0.shape.len() != self.shape.len() {
            return Err(IrError::MatmulRankMismatch);
        }
        let expected: Box<[u32]> = a0.shape[..a0.shape.len() - 1]
            .iter()
            .chain(std::iter::once(&a1.shape[a1.shape.len() - 1]))
            .copied()
            .collect();
        if self.shape.as_ref() != expected.as_ref() {
            return Err(IrError::MatmulOutputShape {
                output: self.shape.clone(),
                expected,
            });
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

    pub fn validate(&self) -> Result<(), IrError> {
        if self.arg_accessors.len() != 1 {
            return Err(IrError::ReductionArgCount {
                got: self.arg_accessors.len(),
            });
        }
        let input_shape = self.input_shape();
        if input_shape.len() != self.shape.len() {
            return Err(IrError::ReductionRankMismatch {
                input: input_shape.len(),
                output: self.shape.len(),
            });
        }
        for &axis in &self.axes {
            if axis as usize >= self.shape.len() {
                return Err(IrError::ReductionAxisOutOfRange {
                    axis,
                    rank: self.shape.len(),
                });
            }
            if self.shape[axis as usize] != 1 {
                return Err(IrError::ReductionAxisNotUnit {
                    axis,
                    size: self.shape[axis as usize],
                });
            }
        }
        for (i, (&out_dim, &in_dim)) in self.shape.iter().zip(input_shape.iter()).enumerate() {
            if self.axes.contains(&(i as u32)) {
                continue;
            }
            if out_dim != in_dim {
                return Err(IrError::ReductionPreservedAxis {
                    axis: i,
                    output: out_dim,
                    input: in_dim,
                });
            }
        }
        Ok(())
    }
}

impl IrRemapKernel {
    pub fn validate(&self) -> Result<(), IrError> {
        if self.arg_element_types.len() != self.arg_accessors.len() {
            return Err(IrError::RemapArgElementTypesLen {
                element_types: self.arg_element_types.len(),
                accessors: self.arg_accessors.len(),
            });
        }
        match &self.info {
            RemapInfo::Scatter(RemapScatterInfo { accessor, .. }) => {
                if let Some(accessor) = accessor {
                    if self.arg_accessors.len() != 1 {
                        return Err(IrError::ScatterAccessorArgCount {
                            got: self.arg_accessors.len(),
                        });
                    }
                    if accessor.shape.as_ref() != self.arg_accessors[0].shape.as_ref() {
                        return Err(IrError::ScatterAccessorShape {
                            accessor: accessor.shape.clone(),
                            source_shape: self.arg_accessors[0].shape.clone(),
                        });
                    }
                } else if self.arg_accessors.len() != 2 {
                    return Err(IrError::ScatterArgCount {
                        got: self.arg_accessors.len(),
                    });
                } else {
                    let source_accessor = &self.arg_accessors[0];
                    let indices_accessor = &self.arg_accessors[1];
                    if indices_accessor.shape[indices_accessor.shape.len() - 1]
                        != self.shape.len() as u32
                    {
                        return Err(IrError::ScatterIndicesTrailing {
                            trailing: indices_accessor.shape[indices_accessor.shape.len() - 1],
                            output_rank: self.shape.len(),
                        });
                    }
                    if indices_accessor.shape[..indices_accessor.shape.len() - 1]
                        != source_accessor.shape[..]
                    {
                        return Err(IrError::ScatterBatchShape);
                    }
                }
            }
            RemapInfo::Gather(RemapGatherInfo {
                accessor,
                source_shape,
            }) => {
                if accessor.is_none() && source_shape.is_none() {
                    if self.arg_accessors.len() != 1 {
                        return Err(IrError::GatherImplicitArgCount {
                            got: self.arg_accessors.len(),
                        });
                    }
                } else if let (Some(accessor), Some(source_shape)) = (accessor, source_shape) {
                    if self.arg_accessors.len() != 2 {
                        return Err(IrError::GatherAccessorArgCount {
                            got: self.arg_accessors.len(),
                        });
                    }
                    let source_accessor = &self.arg_accessors[0];
                    let indices_accessor = &self.arg_accessors[1];
                    if indices_accessor.shape[indices_accessor.shape.len() - 1] != accessor.rank() as u32
                    {
                        return Err(IrError::GatherIndicesTrailing {
                            trailing: indices_accessor.shape[indices_accessor.shape.len() - 1],
                            accessor_rank: accessor.rank(),
                        });
                    }
                    if indices_accessor.shape[..indices_accessor.shape.len() - 1]
                        != source_accessor.shape[..]
                    {
                        return Err(IrError::GatherBatchShape);
                    }
                    if accessor.shape.as_ref() != source_shape.as_ref() {
                        return Err(IrError::GatherAccessorSourceShape {
                            accessor: accessor.shape.clone(),
                            source_shape: source_shape.clone(),
                        });
                    }
                } else {
                    return Err(IrError::InvalidGatherInfo {
                        has_accessor: accessor.is_some(),
                        has_source_shape: source_shape.is_some(),
                    });
                }
            }
        }
        Ok(())
    }
}

impl<P: Tree<BufferRef>, S: Tree<BufferViewRef>> IrProgram<P, S> {
    pub fn validate(&self) -> Result<(), IrError> {
        for dispatch in &self.queue {
            dispatch.kernel.validate()?;
            for view_ref in &dispatch.arg_view_indices {
                if view_ref.index() >= self.buffer_views.len() {
                    return Err(IrError::DispatchArgViewOutOfRange {
                        index: view_ref.index(),
                        len: self.buffer_views.len(),
                    });
                }
            }
            if dispatch.output_buffer_index.index() >= self.buffers.len() {
                return Err(IrError::DispatchOutputBufferOutOfRange {
                    index: dispatch.output_buffer_index.index(),
                    len: self.buffers.len(),
                });
            }
        }
        for (view, buffer_view) in self.buffer_views.iter().enumerate() {
            if buffer_view.buffer_index.index() >= self.buffers.len() {
                return Err(IrError::BufferViewBufferOutOfRange {
                    view,
                    index: buffer_view.buffer_index.index(),
                    len: self.buffers.len(),
                });
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
            element_type: F4,
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
            arg_element_types: vec![F4, F4],
            element_type: F4,
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
            element_type: F4,
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
                    element_type: F4,
                    init: None,
                    readonly: false,
                },
                IrBuffer {
                    shape: Box::from([4]),
                    element_type: F4,
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
    fn program_rejects_invalid_buffer_view_buffer_index() {
        let program = IrProgram::<BufferRef, BufferViewRef> {
            params: BufferRef::new(0),
            sinks: BufferViewRef::new(0),
            queue: vec![],
            buffers: vec![IrBuffer {
                shape: Box::from([4]),
                element_type: F4,
                init: None,
                readonly: false,
            }],
            buffer_views: vec![IrBufferView {
                buffer_index: BufferRef::new(1),
                accessor: Accessor::dense([4], 0),
            }],
        };
        let err = program.validate().unwrap_err();
        assert!(matches!(
            err,
            IrError::BufferViewBufferOutOfRange {
                view: 0,
                index: 1,
                len: 1,
            }
        ));
    }

    #[test]
    fn unary_elementwise_rpn() {
        let shape: Box<[u32]> = Box::from([4]);
        let kernel = IrKernel::ElementwiseRpn(IrElementwiseRpnKernel {
            arg_accessors: vec![Accessor::dense(shape.clone(), 0)],
            arg_element_types: vec![F4],
            element_type: F4,
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

    #[test]
    fn program_serde_round_trip() {
        let shape: Box<[u32]> = Box::from([2, 3]);
        let program = IrProgram::<BufferRef, BufferViewRef> {
            params: BufferRef::new(0),
            sinks: BufferViewRef::new(0),
            queue: vec![IrDispatch {
                kernel: IrKernel::ElementwiseRpn(IrElementwiseRpnKernel {
                    arg_accessors: vec![Accessor::dense(shape.clone(), 0)],
                    arg_element_types: vec![F4],
                    element_type: F4,
                    shape: shape.clone(),
                    rpn_expr: ElementRpnExpr {
                        atoms: vec![
                            RpnAtom::Arg(0),
                            RpnAtom::Op(ElementOperator::Unary(UnaryElementOperator::Neg)),
                        ],
                    },
                    clear_output_before_dispatch: false,
                }),
                arg_view_indices: vec![BufferViewRef::new(0)],
                output_buffer_index: BufferRef::new(0),
            }],
            buffers: vec![IrBuffer {
                shape: shape.clone(),
                element_type: F4,
                init: Some(Box::from([1u8, 2, 3, 4])),
                readonly: true,
            }],
            buffer_views: vec![IrBufferView {
                buffer_index: BufferRef::new(0),
                accessor: Accessor::dense(shape, 0),
            }],
        };
        program.validate().unwrap();

        let json = serde_json::to_string(&program).unwrap();
        let restored: IrProgram = serde_json::from_str(&json).unwrap();
        assert_eq!(program, restored);
    }

    #[test]
    fn program_tree_serde_round_trip() {
        use resin_macros::Tree;
        use serde::{Deserialize, Serialize};

        #[derive(Debug, Clone, PartialEq, Eq, Tree, Serialize, Deserialize)]
        struct ParamTree<T> {
            weight: T,
            bias: T,
        }

        #[derive(Debug, Clone, PartialEq, Eq, Tree, Serialize, Deserialize)]
        struct SinkTree<T> {
            loss: T,
            grads: ParamTree<T>,
        }

        let shape: Box<[u32]> = Box::from([2, 3]);
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
                    shape: shape.clone(),
                    element_type: F4,
                    init: None,
                    readonly: false,
                },
                IrBuffer {
                    shape: Box::from([3]),
                    element_type: F4,
                    init: None,
                    readonly: false,
                },
            ],
            buffer_views: vec![
                IrBufferView {
                    buffer_index: BufferRef::new(0),
                    accessor: Accessor::dense(shape.clone(), 0),
                },
                IrBufferView {
                    buffer_index: BufferRef::new(0),
                    accessor: Accessor::dense(shape.clone(), 0),
                },
                IrBufferView {
                    buffer_index: BufferRef::new(1),
                    accessor: Accessor::dense([3], 0),
                },
            ],
        };
        program.validate().unwrap();

        let json = serde_json::to_string(&program).unwrap();
        let restored: IrProgram<ParamTree<BufferRef>, SinkTree<BufferViewRef>> =
            serde_json::from_str(&json).unwrap();
        assert_eq!(program, restored);
    }
}