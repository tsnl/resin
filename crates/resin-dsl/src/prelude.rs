//! Common constructors for the DSL frontend.

use std::sync::Arc;

use resin_core::{ElementType, F4, U4};

use crate::node::{ConstNodeKind, Node, NodeKind, ParamNodeKind};
use crate::node_ref::NodeRef;
use crate::view::View;

/// Named parameter leaf (host-writable buffer).
pub fn param(
    shape: impl Into<Box<[u32]>>,
    element_type: ElementType,
    name: impl Into<Arc<str>>,
) -> View {
    let shape = shape.into();
    View::identity(NodeRef::new(Node {
        shape,
        element_type,
        args: vec![],
        kind: NodeKind::Param(ParamNodeKind { name: name.into() }),
    }))
}

/// Constant buffer from raw little-endian bytes.
pub fn const_bytes(
    shape: impl Into<Box<[u32]>>,
    element_type: ElementType,
    init: impl Into<Box<[u8]>>,
) -> View {
    let shape = shape.into();
    let init = init.into();
    let expected =
        shape.iter().map(|&d| u64::from(d)).product::<u64>() * u64::from(element_type.nbytes());
    assert_eq!(
        init.len() as u64,
        expected,
        "const init byte length mismatch"
    );
    View::identity(NodeRef::new(Node {
        shape,
        element_type,
        args: vec![],
        kind: NodeKind::Const(ConstNodeKind { init }),
    }))
}

/// Host scalar that can be packed into a const buffer; etype is inferred from `Self`.
pub trait ConstData: Copy {
    const ELEMENT_TYPE: ElementType;
    fn write_le(self, out: &mut Vec<u8>);
}

impl ConstData for f32 {
    const ELEMENT_TYPE: ElementType = F4;
    fn write_le(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_le_bytes());
    }
}

impl ConstData for u32 {
    const ELEMENT_TYPE: ElementType = U4;
    fn write_le(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_le_bytes());
    }
}

/// Constant from an explicit shape and **row-major** values.
///
/// Element type is inferred from the slice: `&[f32]` → F4, `&[u32]` → U4.
/// Python analogue: `const([[…], …])` without nested-list inference — pass the
/// flattened buffer and `shape` yourself.
///
/// # Panics
/// If `data.len()` is not the product of `shape`.
pub fn constant<T: ConstData>(shape: impl Into<Box<[u32]>>, data: &[T]) -> View {
    let shape = shape.into();
    let n: usize = shape.iter().map(|&d| d as usize).product();
    assert_eq!(
        data.len(),
        n,
        "constant data length {} != shape product {n} (shape={shape:?})",
        data.len()
    );
    let mut bytes = Vec::with_capacity(n * size_of::<T>());
    for &x in data {
        x.write_le(&mut bytes);
    }
    const_bytes(shape, T::ELEMENT_TYPE, bytes.into_boxed_slice())
}

fn broadcast_scalar(shape: Box<[u32]>, leaf: View) -> View {
    if shape.is_empty() {
        leaf
    } else {
        leaf.broadcast(&shape)
    }
}

/// Fill `shape` with an F4 scalar (Python `full` for floats).
pub fn full(shape: impl Into<Box<[u32]>>, value: f32, element_type: ElementType) -> View {
    assert_eq!(
        element_type, F4,
        "full() packs f32 as F4; use full_u32 for U4 (got {element_type:?})"
    );
    broadcast_scalar(shape.into(), value.into())
}

/// Fill `shape` with a U4 scalar (`value.into()` then broadcast).
pub fn full_u32(shape: impl Into<Box<[u32]>>, value: u32) -> View {
    broadcast_scalar(shape.into(), value.into())
}

/// Ones in F4 or U4 (`1.0` / `1u32`).
pub fn ones(shape: impl Into<Box<[u32]>>, element_type: ElementType) -> View {
    match element_type {
        F4 => full(shape, 1.0, F4),
        U4 => full_u32(shape, 1),
        other => panic!("ones() supports F4 and U4 only (got {other:?})"),
    }
}

/// Zeros in F4 or U4.
pub fn zeros(shape: impl Into<Box<[u32]>>, element_type: ElementType) -> View {
    match element_type {
        F4 => full(shape, 0.0, F4),
        U4 => full_u32(shape, 0),
        other => panic!("zeros() supports F4 and U4 only (got {other:?})"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use resin_core::F4;

    #[test]
    fn constant_shape_and_bytes() {
        let v = constant([2, 2], &[1.0f32, 2.0, 3.0, 4.0]);
        assert_eq!(v.shape(), &[2, 2]);
        assert_eq!(v.element_type(), F4);
        assert!(v.is_identity());
        match &v.node_ref.kind {
            crate::node::NodeKind::Const(c) => {
                assert_eq!(c.init.len(), 16);
                assert_eq!(&c.init[..4], &1.0f32.to_le_bytes());
            }
            _ => panic!("expected const"),
        }

        let u = constant([3], &[1u32, 2, 3]);
        assert_eq!(u.element_type(), U4);
        match &u.node_ref.kind {
            crate::node::NodeKind::Const(c) => {
                assert_eq!(&c.init[..4], &1u32.to_le_bytes());
            }
            _ => panic!("expected const"),
        }
    }

    #[test]
    fn full_broadcasts_scalar() {
        let v = full([2, 3], 7.0, F4);
        assert_eq!(v.shape(), &[2, 3]);
        assert!(!v.is_identity()); // broadcast pitch 0
        assert_eq!(v.offset(), 0);
        assert_eq!(v.pitch(), &[0, 0]);
    }

    #[test]
    fn ones_zeros() {
        assert_eq!(ones([4], F4).shape(), &[4]);
        assert_eq!(zeros([], F4).shape(), &[] as &[u32]);
    }

    #[test]
    fn f32_into_rank0_const() {
        let v: View = 42.0f32.into();
        assert_eq!(v.shape(), &[] as &[u32]);
        assert!(v.is_identity());
        match &v.node_ref.kind {
            crate::node::NodeKind::Const(c) => {
                assert_eq!(&*c.init, &42.0f32.to_le_bytes());
            }
            _ => panic!("expected const"),
        }
    }

    #[test]
    fn u32_into_rank0_const() {
        let v: View = 7u32.into();
        assert_eq!(v.element_type(), U4);
        match &v.node_ref.kind {
            crate::node::NodeKind::Const(c) => {
                assert_eq!(&*c.init, &7u32.to_le_bytes());
            }
            _ => panic!("expected const"),
        }
        let filled = full_u32([2, 2], 3);
        assert_eq!(filled.shape(), &[2, 2]);
        assert_eq!(filled.element_type(), U4);
    }
}
