use super::*;

use crate::Tree;
use crate::tensor::{ElementOperator, IndexKeyElement, ScatterOp, Tensor, TensorKind};

/// grad computes the gradient of a tree of tensors.
///
/// The function `f` is called once to trace the computation graph of the value
/// function, and the gradients are computed using reverse-mode automatic
/// differentiation.
/// Gradients of a traced scalar `loss` with respect to each leaf of `params`.
pub fn grad_wrt<T: Tree<Tensor>>(loss: &Tensor, params: &T) -> Result<T::Mapped<Tensor>, GradError> {
    if !loss.shape().is_empty() {
        return Err(GradError::NonScalarOutput(Box::from(loss.shape())));
    }
    let grad_map = grad_by_node(loss)?;
    Ok(params.map(|param| {
        grad_map
            .get(param)
            .cloned()
            .unwrap_or_else(|| Tensor::zeros_like(param))
    }))
}

pub fn grad<T: Clone + Tree<Tensor, Mapped<Tensor> = T>>(
    f: impl Fn(T) -> Tensor,
) -> impl Fn(T) -> Result<(Tensor, T), GradError> {
    move |params: T| -> Result<(Tensor, T), GradError> {
        // Trace the function:
        let scalar = f(params.clone());

        // Validate the traced graph:
        if !scalar.shape().is_empty() {
            return Err(GradError::NonScalarOutput(Box::from(scalar.shape())));
        }

        // Compute the gradients:
        let grad_map = grad_by_node(&scalar)?;

        // Gather the gradients for each input parameter, default to zero (no gradient
        // flow):
        let params_grads = params.map(|param| {
            grad_map
                .get(param)
                .cloned()
                .unwrap_or_else(|| Tensor::zeros_like(&param))
        });
        Ok((scalar, params_grads))
    }
}

#[derive(Debug, thiserror::Error)]
pub enum GradError {
    #[error("Expected a scalar tensor, but got a tensor with shape {0:?}")]
    NonScalarOutput(Box<[usize]>),

    #[error("Detected cyclic dependencies in the computation graph")]
    CyclicDependencies,

    #[error("Operator {operator:?} does not support reverse-mode differentiation")]
    Undifferentiable { operator: ElementOperator },
}

type GradMap = HashMap<Tensor, Tensor>;

/// Gradients of a scalar loss, keyed by tensor identity.
fn grad_by_node(loss: &Tensor) -> Result<GradMap, GradError> {
    let mut grad_map = GradMap::new();

    accumulate(&mut grad_map, loss, Tensor::ones_like(loss));

    let order = loss.toposort().map_err(|_| GradError::CyclicDependencies)?;
    for node in order.into_iter().rev() {
        let Some(df_dout) = grad_map.get(&node).cloned() else {
            continue;
        };
        backward(&mut grad_map, &node, &df_dout)?;
    }

    Ok(grad_map)
}

fn accumulate(grad_map: &mut GradMap, tensor: &Tensor, adjoint: Tensor) {
    match grad_map.entry(tensor.clone()) {
        hash_map::Entry::Occupied(mut entry) => {
            *entry.get_mut() = entry.get().clone() + adjoint;
        }
        hash_map::Entry::Vacant(entry) => {
            entry.insert(adjoint);
        }
    }
}

/// Apply this node's VJP: propagate the adjoint at `node` into its operands.
fn backward(grad_map: &mut GradMap, node: &Tensor, df_dout: &Tensor) -> Result<(), GradError> {
    match node.kind() {
        TensorKind::Constant { .. } => backward_constant(),
        TensorKind::Parameter => backward_parameter(),
        TensorKind::Elementwise { operator, args } => {
            backward_elementwise(grad_map, node, *operator, args.as_slice(), df_dout)
        }
        TensorKind::Reduction {
            operator,
            axes,
            arg,
        } => backward_reduction(grad_map, node, *operator, axes, arg, df_dout),
        TensorKind::Index { arg, key } => backward_index(grad_map, arg, key, df_dout),
        TensorKind::Gather { source, indices } => {
            backward_gather(grad_map, source, indices, df_dout)
        }
        TensorKind::Scatter {
            source, indices, ..
        } => backward_scatter(grad_map, source, indices, df_dout),
        TensorKind::Broadcast { arg, axes, .. } => backward_broadcast(grad_map, arg, axes, df_dout),
        TensorKind::ScatterIndex { source, key, .. } => {
            backward_scatter_index(grad_map, source, key, df_dout)
        }
        TensorKind::Transpose { arg } => backward_transpose(grad_map, arg, df_dout),
        TensorKind::Squeeze { arg, axes } => backward_squeeze(grad_map, arg, axes, df_dout),
    }
}

fn backward_constant() -> Result<(), GradError> {
    Ok(())
}

fn backward_parameter() -> Result<(), GradError> {
    Ok(())
}

fn backward_index(
    grad_map: &mut GradMap,
    arg: &Tensor,
    key: &[IndexKeyElement],
    df_dout: &Tensor,
) -> Result<(), GradError> {
    accumulate(grad_map, arg, df_dout.scatter_index(arg.shape(), key));
    Ok(())
}

fn backward_broadcast(
    grad_map: &mut GradMap,
    arg: &Tensor,
    axes: &[usize],
    df_dout: &Tensor,
) -> Result<(), GradError> {
    // `axes[i]` maps arg axis `i` → output axis. Sum over output axes that were
    // expanded (unmapped, or mapped from a size-1 arg dim), then squeeze only
    // the unmapped axes so the result matches `arg.shape()`.
    let out_rank = df_dout.shape().len();
    let arg_shape = arg.shape();
    let mut out_to_arg = vec![None; out_rank];
    for (arg_axis, &out_axis) in axes.iter().enumerate() {
        out_to_arg[out_axis] = Some(arg_axis);
    }

    let mut sum_axes = Vec::new();
    let mut unmapped = Vec::new();
    for out_axis in 0..out_rank {
        match out_to_arg[out_axis] {
            None => {
                sum_axes.push(out_axis);
                unmapped.push(out_axis);
            }
            Some(arg_axis)
                if arg_shape[arg_axis] == 1 && df_dout.shape()[out_axis] > 1 =>
            {
                sum_axes.push(out_axis);
            }
            Some(_) => {}
        }
    }

    let mut g = df_dout.clone();
    if !sum_axes.is_empty() {
        g = g.sum_axes(&sum_axes);
    }
    unmapped.sort_unstable();
    for &axis in unmapped.iter().rev() {
        g = g.squeeze(&[axis]);
    }
    accumulate(grad_map, arg, g);
    Ok(())
}

fn backward_squeeze(
    grad_map: &mut GradMap,
    arg: &Tensor,
    axes: &[usize],
    df_dout: &Tensor,
) -> Result<(), GradError> {
    // Inverse of squeeze: re-insert size-1 axes via broadcast.
    let squeezed: HashSet<usize> = axes.iter().copied().collect();
    let mapping: Vec<usize> = (0..arg.shape().len())
        .filter(|a| !squeezed.contains(a))
        .collect();
    accumulate(
        grad_map,
        arg,
        df_dout.broadcast_to(arg.shape(), &mapping),
    );
    Ok(())
}

fn backward_scatter_index(
    grad_map: &mut GradMap,
    source: &Tensor,
    key: &[IndexKeyElement],
    df_dout: &Tensor,
) -> Result<(), GradError> {
    accumulate(grad_map, source, Tensor::new_index(df_dout.clone(), key));
    Ok(())
}

fn backward_transpose(
    grad_map: &mut GradMap,
    arg: &Tensor,
    df_dout: &Tensor,
) -> Result<(), GradError> {
    accumulate(grad_map, arg, df_dout.transpose());
    Ok(())
}

fn backward_elementwise(
    grad_map: &mut GradMap,
    node: &Tensor,
    operator: ElementOperator,
    args: &[Tensor],
    df_dout: &Tensor,
) -> Result<(), GradError> {
    let n = || node.clone();

    match operator {
        ElementOperator::Neg => accumulate(grad_map, &args[0], -df_dout.clone()),
        ElementOperator::Log => {
            accumulate(grad_map, &args[0], df_dout.clone() / args[0].clone());
        }
        ElementOperator::Exp => accumulate(grad_map, &args[0], df_dout.clone() * n()),
        ElementOperator::Relu => {
            let eps = args[0].full_like(1e-12);
            let mask = args[0].clone().relu() / (args[0].clone().abs() + eps);
            accumulate(grad_map, &args[0], df_dout.clone() * mask);
        }
        ElementOperator::Abs => {
            let eps = args[0].full_like(1e-12);
            let sign = args[0].clone() / (args[0].clone().abs() + eps);
            accumulate(grad_map, &args[0], df_dout.clone() * sign);
        }
        ElementOperator::Sqrt => {
            let half = args[0].full_like(0.5);
            accumulate(grad_map, &args[0], df_dout.clone() * half / n());
        }
        ElementOperator::Min => {
            let lhs_wins = args[0].cmp_le(&args[1]);
            accumulate(grad_map, &args[0], df_dout.clone() * lhs_wins.clone());
            let rhs_wins = args[0].cmp_gt(&args[1]);
            accumulate(grad_map, &args[1], df_dout.clone() * rhs_wins);
        }
        ElementOperator::Max => {
            let lhs_wins = args[0].cmp_ge(&args[1]);
            accumulate(grad_map, &args[0], df_dout.clone() * lhs_wins.clone());
            let rhs_wins = args[0].cmp_lt(&args[1]);
            accumulate(grad_map, &args[1], df_dout.clone() * rhs_wins);
        }
        // Piecewise-constant and integer ops: zero gradient, no propagation.
        ElementOperator::Floor
        | ElementOperator::Cast
        | ElementOperator::Bitcast
        | ElementOperator::CmpEq
        | ElementOperator::CmpNe
        | ElementOperator::CmpLt
        | ElementOperator::CmpLe
        | ElementOperator::CmpGt
        | ElementOperator::CmpGe
        | ElementOperator::BitAnd
        | ElementOperator::BitOr
        | ElementOperator::BitXor
        | ElementOperator::Shl
        | ElementOperator::Shr => {}
        ElementOperator::Add => {
            accumulate(grad_map, &args[0], df_dout.clone());
            accumulate(grad_map, &args[1], df_dout.clone());
        }
        ElementOperator::Sub => {
            accumulate(grad_map, &args[0], df_dout.clone());
            accumulate(grad_map, &args[1], -df_dout.clone());
        }
        ElementOperator::Mul => {
            accumulate(grad_map, &args[0], df_dout.clone() * args[1].clone());
            accumulate(grad_map, &args[1], df_dout.clone() * args[0].clone());
        }
        ElementOperator::Div => {
            accumulate(grad_map, &args[0], df_dout.clone() / args[1].clone());
            accumulate(grad_map, &args[1], df_dout.clone() * -n() / args[1].clone());
        }
        ElementOperator::Rem => {
            return Err(GradError::Undifferentiable { operator });
        }
        ElementOperator::Pow => {
            accumulate(
                grad_map,
                &args[0],
                df_dout.clone() * args[1].clone() * n() / args[0].clone(),
            );
            accumulate(
                grad_map,
                &args[1],
                df_dout.clone() * args[0].clone().log() * n(),
            );
        }
        ElementOperator::Matmul => {
            accumulate(
                grad_map,
                &args[0],
                df_dout.clone().matmul(&args[1].clone().transpose()),
            );
            accumulate(
                grad_map,
                &args[1],
                args[0].clone().transpose().matmul(&df_dout.clone()),
            );
        }
    }
    Ok(())
}

fn backward_reduction(
    grad_map: &mut GradMap,
    node: &Tensor,
    operator: ElementOperator,
    axes: &[usize],
    arg: &Tensor,
    df_dout: &Tensor,
) -> Result<(), GradError> {
    let _ = axes;
    // Reductions are keepdims: `df_dout` already has the same rank as `arg`
    // (unit sizes on reduced axes). Expand those units by identity broadcast.
    let identity: Vec<usize> = (0..arg.shape().len()).collect();
    let g = df_dout.broadcast_to(arg.shape(), &identity);
    let grad_arg = match operator {
        ElementOperator::Add => g,
        ElementOperator::Mul => g * arg.clone() / node.clone(),
        _ => {
            return Err(GradError::Undifferentiable { operator });
        }
    };
    accumulate(grad_map, arg, grad_arg);
    Ok(())
}

/// Gather adjoint: rows read multiple times accumulate their contributions,
/// so d/dsource is a scatter-add of `df_dout` through the same indices.
/// Indices are integer data — no gradient flows into them.
fn backward_gather(
    grad_map: &mut GradMap,
    source: &Tensor,
    indices: &Tensor,
    df_dout: &Tensor,
) -> Result<(), GradError> {
    let grad_source = df_dout.scatter_rows(indices, source.shape()[0], ScatterOp::Add);
    accumulate(grad_map, source, grad_source);
    Ok(())
}

/// Scatter adjoint: each source row lands at `indices[i]`, so d/dsource is a
/// gather of `df_dout` through the same indices. For `ScatterOp::Write` with
/// duplicate indices this is a subgradient (all writers receive the adjoint of
/// the surviving row).
fn backward_scatter(
    grad_map: &mut GradMap,
    source: &Tensor,
    indices: &Tensor,
    df_dout: &Tensor,
) -> Result<(), GradError> {
    let grad_source = df_dout.gather_rows(indices);
    accumulate(grad_map, source, grad_source);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::debug_print::{debug_str, dedent};
    use crate::tensor::{ElementType, Tensor};

    #[test]
    fn grad_shared_operand_succeeds() {
        let a = Tensor::parameter(&[], ElementType::F32);
        let grad_fn = grad(|params: Vec<Tensor>| params[0].clone() + params[0].clone());
        assert!(grad_fn(vec![a]).is_ok());
    }

    #[test]
    fn grad_rejects_non_scalar_output() {
        let a = Tensor::parameter(&[2], ElementType::F32);
        let grad_fn = grad(|params: Vec<Tensor>| params[0].clone());
        assert!(matches!(
            grad_fn(vec![a]),
            Err(GradError::NonScalarOutput(_))
        ));
    }

    #[test]
    fn grad_rejects_undifferentiable_rem() {
        let a = Tensor::parameter(&[], ElementType::F32);
        let b = Tensor::parameter(&[], ElementType::F32);
        let grad_fn = grad(|params: Vec<Tensor>| params[0].clone() % params[1].clone());
        assert!(matches!(
            grad_fn(vec![a, b]),
            Err(GradError::Undifferentiable {
                operator: ElementOperator::Rem
            })
        ));
    }

    #[test]
    fn grad_add_expect() {
        let a = Tensor::parameter(&[], ElementType::F32);
        let b = Tensor::parameter(&[], ElementType::F32);
        let grad_fn = grad(|params: Vec<Tensor>| params[0].clone() + params[1].clone());
        let (_, grads) = grad_fn(vec![a, b]).unwrap();
        assert_eq!(debug_str(&grads[0]), "constant(value=1) :: f32()");
        assert_eq!(debug_str(&grads[1]), "constant(value=1) :: f32()");
    }

    #[test]
    fn grad_shared_operand_expect() {
        let a = Tensor::parameter(&[], ElementType::F32);
        let grad_fn = grad(|params: Vec<Tensor>| params[0].clone() + params[0].clone());
        let (_, grads) = grad_fn(vec![a]).unwrap();
        let expected = dedent(
            r#"
            elementwise(operator='add') :: f32()
            ├ %0
            └ %0
            %0 := constant(value=1) :: f32()
            "#,
        );
        assert_eq!(debug_str(&grads[0]), expected);
    }

    #[test]
    fn grad_mul_expect() {
        let a = Tensor::parameter(&[], ElementType::F32);
        let b = Tensor::parameter(&[], ElementType::F32);
        let grad_fn = grad(|params: Vec<Tensor>| params[0].clone() * params[1].clone());
        let (_, grads) = grad_fn(vec![a, b]).unwrap();
        let grad_a = debug_str(&grads[0]);
        let grad_b = debug_str(&grads[1]);
        assert!(grad_a.contains("elementwise(operator='mul')"), "{grad_a}");
        assert!(grad_b.contains("elementwise(operator='mul')"), "{grad_b}");
    }
}
