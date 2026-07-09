//! Reverse-mode automatic differentiation over traced [`Tensor`] graphs.

use std::collections::HashMap;

use super::{Remap, ScatterOp, Tensor, TensorKind};
use crate::ops::{AssocOp, BinaryOp, Op, UnaryOp};
use crate::tree::Tree;

#[derive(Debug, thiserror::Error)]
pub enum GradError {
    #[error("expected a scalar loss, got shape {0:?}")]
    NonScalarLoss(Box<[usize]>),

    #[error("no differentiation rule for {0}")]
    Undifferentiable(&'static str),
}

/// Gradients of a traced scalar `loss` with respect to each leaf of `params`.
/// Leaves the loss does not depend on get zero gradients.
pub fn grad_wrt<T: Tree<Tensor>>(
    loss: &Tensor,
    params: &T,
) -> Result<T::Mapped<Tensor>, GradError> {
    if !loss.shape().is_empty() {
        return Err(GradError::NonScalarLoss(loss.shape().into()));
    }
    let grads = grad_by_node(loss)?;
    Ok(params.map(|param| grads.get(param).cloned().unwrap_or_else(|| param.zeros_like())))
}

/// JAX-style transform: `grad(f)` maps params to `(loss, gradients)`.
pub fn grad<T>(f: impl Fn(T) -> Tensor) -> impl Fn(T) -> Result<(Tensor, T), GradError>
where
    T: Clone + Tree<Tensor, Mapped<Tensor> = T>,
{
    move |params: T| {
        let loss = f(params.clone());
        let grads = grad_wrt(&loss, &params)?;
        Ok((loss, grads))
    }
}

type GradMap = HashMap<Tensor, Tensor>;

/// Adjoints of every node the scalar loss depends on, keyed by node identity.
fn grad_by_node(loss: &Tensor) -> Result<GradMap, GradError> {
    let mut grads = GradMap::new();
    grads.insert(loss.clone(), loss.ones_like());
    for node in loss.toposort().into_iter().rev() {
        let Some(adjoint) = grads.get(&node).cloned() else {
            continue; // Not on any path from the loss.
        };
        backward(&mut grads, &node, &adjoint)?;
    }
    Ok(grads)
}

fn accumulate(grads: &mut GradMap, tensor: &Tensor, adjoint: Tensor) {
    // Only real-valued primals carry adjoints. Integers (indices, masks-as-u32,
    // bit ops) are discrete — see [`ElementType::is_float`]. When f16/f64 are
    // added they become floats and flow through here without a new special case.
    if !tensor.element_type().is_float() {
        return;
    }
    debug_assert_eq!(
        adjoint.element_type(),
        tensor.element_type(),
        "adjoint etype must match the primal (seeded by loss.ones_like())"
    );
    // Broadcasting in a forward op fans one input element out to many output
    // elements, so the adjoint sums back down to the input's shape.
    let adjoint = sum_to_shape(&adjoint, tensor.shape());
    match grads.remove(tensor) {
        Some(existing) => grads.insert(tensor.clone(), existing + adjoint),
        None => grads.insert(tensor.clone(), adjoint),
    };
}

/// Sum `adjoint` down to `shape` (inverse of NumPy trailing broadcast).
fn sum_to_shape(adjoint: &Tensor, shape: &[usize]) -> Tensor {
    if adjoint.shape() == shape {
        return adjoint.clone();
    }
    let lead = adjoint.shape().len() - shape.len();
    let axes: Vec<usize> = (0..adjoint.shape().len())
        .filter(|&axis| axis < lead || (shape[axis - lead] == 1 && adjoint.shape()[axis] != 1))
        .collect();
    let mut result = adjoint.sum_axes(&axes);
    if lead > 0 {
        result = result.squeeze(&(0..lead).collect::<Vec<_>>());
    }
    result
}

/// Apply this node's VJP: propagate the node's adjoint into its arguments.
fn backward(grads: &mut GradMap, node: &Tensor, adjoint: &Tensor) -> Result<(), GradError> {
    match node.kind() {
        TensorKind::Constant { .. } | TensorKind::Parameter => {}

        TensorKind::Elementwise { op, args } => {
            backward_elementwise(grads, node, *op, args, adjoint)?;
        }

        TensorKind::Matmul { lhs, rhs } => {
            accumulate(grads, lhs, adjoint.matmul(&rhs.transpose()));
            accumulate(grads, rhs, lhs.transpose().matmul(adjoint));
        }

        TensorKind::Reduction { op, arg, .. } => {
            // Reductions are keepdims: the adjoint already has the arg's rank,
            // with unit sizes on reduced axes. Expand by identity broadcast.
            let identity: Vec<usize> = (0..arg.shape().len()).collect();
            let expanded = adjoint.broadcast_to(arg.shape(), &identity);
            match op {
                AssocOp::Add => accumulate(grads, arg, expanded),
                AssocOp::Mul => accumulate(grads, arg, expanded * arg.clone() / node.clone()),
                AssocOp::Max | AssocOp::Min => {
                    return Err(GradError::Undifferentiable("max/min reduction"));
                }
            }
        }

        TensorKind::Broadcast { arg, axes } => {
            // Sum over output axes that were expanded, squeeze the unmapped ones.
            let out_rank = adjoint.shape().len();
            let mapped: Vec<Option<usize>> = {
                let mut m = vec![None; out_rank];
                for (arg_axis, &out_axis) in axes.iter().enumerate() {
                    m[out_axis] = Some(arg_axis);
                }
                m
            };
            let sum_axes: Vec<usize> = (0..out_rank)
                .filter(|&out_axis| match mapped[out_axis] {
                    None => true,
                    Some(arg_axis) => {
                        arg.shape()[arg_axis] == 1 && adjoint.shape()[out_axis] > 1
                    }
                })
                .collect();
            let unmapped: Vec<usize> = (0..out_rank).filter(|&a| mapped[a].is_none()).collect();
            let mut g = adjoint.clone();
            if !sum_axes.is_empty() {
                g = g.sum_axes(&sum_axes);
            }
            if !unmapped.is_empty() {
                g = g.squeeze(&unmapped);
            }
            accumulate(grads, arg, g);
        }

        TensorKind::Transpose { arg } => accumulate(grads, arg, adjoint.transpose()),

        TensorKind::Squeeze { arg, axes } => {
            // Re-insert the squeezed size-1 axes via broadcast.
            let mapping: Vec<usize> =
                (0..arg.shape().len()).filter(|a| !axes.contains(a)).collect();
            accumulate(grads, arg, adjoint.broadcast_to(arg.shape(), &mapping));
        }

        TensorKind::Index { arg, key } => {
            accumulate(grads, arg, adjoint.scatter_index(arg.shape(), key));
        }

        TensorKind::Remap(remap) => match remap {
            Remap::GatherRows { source, indices } => {
                // Gather adjoint is scatter-add through the same indices.
                let grad_source =
                    adjoint.scatter_rows(indices, source.shape()[0], ScatterOp::Add);
                accumulate(grads, source, grad_source);
            }
            Remap::ScatterRows { source, indices, .. } => {
                // Scatter adjoint is gather through the same indices.
                accumulate(grads, source, adjoint.gather_rows(indices));
            }
            Remap::ScatterView { source, key, .. } => {
                accumulate(grads, source, Tensor::new_index(adjoint.clone(), key));
            }
        },
    }
    Ok(())
}

fn backward_elementwise(
    grads: &mut GradMap,
    node: &Tensor,
    op: Op,
    args: &[Tensor],
    adjoint: &Tensor,
) -> Result<(), GradError> {
    let g = || adjoint.clone();
    let n = || node.clone();

    match op {
        Op::Unary(op) => {
            let x = &args[0];
            match op {
                // Piecewise-constant / type-change: zero gradient.
                UnaryOp::Floor
                | UnaryOp::Ceil
                | UnaryOp::Cast { .. }
                | UnaryOp::Bitcast { .. } => {}
                UnaryOp::Neg => accumulate(grads, x, -g()),
                UnaryOp::Exp => accumulate(grads, x, g() * n()),
                UnaryOp::Log => accumulate(grads, x, g() / x.clone()),
                UnaryOp::Abs => {
                    let eps = Tensor::scalar(1e-12);
                    accumulate(grads, x, g() * (x.clone() / (x.abs() + eps)));
                }
                UnaryOp::Relu => {
                    let eps = Tensor::scalar(1e-12);
                    accumulate(grads, x, g() * (x.relu() / (x.abs() + eps)));
                }
                UnaryOp::Sqrt => accumulate(grads, x, g() / (Tensor::scalar(2.0) * n())),
                UnaryOp::Sin => accumulate(grads, x, g() * x.cos()),
                UnaryOp::Cos => accumulate(grads, x, -(g() * x.sin())),
            }
        }
        Op::Binary(op) => {
            let (a, b) = (&args[0], &args[1]);
            match op {
                BinaryOp::Assoc(AssocOp::Add) => {
                    accumulate(grads, a, g());
                    accumulate(grads, b, g());
                }
                BinaryOp::Sub => {
                    accumulate(grads, a, g());
                    accumulate(grads, b, -g());
                }
                BinaryOp::Assoc(AssocOp::Mul) => {
                    accumulate(grads, a, g() * b.clone());
                    accumulate(grads, b, g() * a.clone());
                }
                BinaryOp::Div => {
                    accumulate(grads, a, g() / b.clone());
                    accumulate(grads, b, g() * -n() / b.clone());
                }
                BinaryOp::Pow => {
                    accumulate(grads, a, g() * b.clone() * n() / a.clone());
                    accumulate(grads, b, g() * a.log() * n());
                }
                BinaryOp::Assoc(AssocOp::Min) => {
                    // Subgradient: lhs wins on ties (<=).
                    let lhs_wins = a.cmp_le(b);
                    accumulate(grads, a, g() * lhs_wins);
                    let rhs_wins = a.cmp_gt(b);
                    accumulate(grads, b, g() * rhs_wins);
                }
                BinaryOp::Assoc(AssocOp::Max) => {
                    let lhs_wins = a.cmp_ge(b);
                    accumulate(grads, a, g() * lhs_wins);
                    let rhs_wins = a.cmp_lt(b);
                    accumulate(grads, b, g() * rhs_wins);
                }
                // Integer / piecewise-constant: zero gradient.
                BinaryOp::CmpEq
                | BinaryOp::CmpNe
                | BinaryOp::CmpLt
                | BinaryOp::CmpLe
                | BinaryOp::CmpGt
                | BinaryOp::CmpGe
                | BinaryOp::Band
                | BinaryOp::Bor
                | BinaryOp::Bxor
                | BinaryOp::Shl
                | BinaryOp::Shr => {}
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsl::debug::{debug_str, dedent};

    #[test]
    fn grad_rejects_non_scalar_loss() {
        let a = Tensor::parameter(&[2]);
        assert!(matches!(
            grad_wrt(&a.clone(), &a),
            Err(GradError::NonScalarLoss(_))
        ));
    }

    #[test]
    fn grad_add() {
        let grad_fn = grad(|params: Vec<Tensor>| params[0].clone() + params[1].clone());
        let (_, grads) = grad_fn(vec![Tensor::parameter(&[]), Tensor::parameter(&[])]).unwrap();
        assert_eq!(debug_str(&grads[0]), "constant(value=1) :: f32()");
        assert_eq!(debug_str(&grads[1]), "constant(value=1) :: f32()");
    }

    #[test]
    fn grad_shared_operand_accumulates() {
        let grad_fn = grad(|params: Vec<Tensor>| params[0].clone() + params[0].clone());
        let (_, grads) = grad_fn(vec![Tensor::parameter(&[])]).unwrap();
        let expected = dedent(
            r#"
            elementwise(op='add') :: f32()
            ├ %0
            └ %0
            %0 := constant(value=1) :: f32()
            "#,
        );
        assert_eq!(debug_str(&grads[0]), expected);
    }

    #[test]
    fn grad_mul_swaps_operands() {
        let grad_fn = grad(|params: Vec<Tensor>| params[0].clone() * params[1].clone());
        let (_, grads) = grad_fn(vec![Tensor::parameter(&[]), Tensor::parameter(&[])]).unwrap();
        assert!(debug_str(&grads[0]).contains("elementwise(op='mul')"));
        assert!(debug_str(&grads[1]).contains("elementwise(op='mul')"));
    }

    #[test]
    fn grad_of_implicit_broadcast_sums_down() {
        // loss = mean(m * s) where s is a scalar: grad(s) must be a scalar.
        let m = Tensor::parameter(&[2, 3]);
        let s = Tensor::parameter(&[]);
        let loss = (m.clone() * s.clone()).mean_all();
        let grads = grad_wrt(&loss, &vec![m, s.clone()]).unwrap();
        assert!(grads[1].shape().is_empty(), "got {:?}", grads[1].shape());
    }

    #[test]
    fn grad_unused_param_is_zero() {
        let used = Tensor::parameter(&[]);
        let unused = Tensor::parameter(&[3]);
        let loss = used.clone() * used.clone();
        let grads = grad_wrt(&loss, &vec![used, unused.clone()]).unwrap();
        assert_eq!(grads[1].shape(), unused.shape());
        assert_eq!(debug_str(&grads[1]), "constant(value=[0.0, 0.0, 0.0]) :: f32(3,)");
    }
}
