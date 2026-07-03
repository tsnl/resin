//! Reverse-mode autodiff on DSL graphs.

use std::collections::HashMap;

use crate::dsl::{NodeKind, NodeRef, RemapGatherInfo, RemapInfo, RemapScatterInfo, View};
use resin_core::{Accessor, ElementOperator, ElementType, Tree};
use thiserror::Error;

#[derive(Debug, Error)]
#[error("not differentiable: {0}")]
pub struct NotDifferentiableError(pub String);

/// Push gradient `g` (w.r.t. `view`'s logical values) into the node's dense output space.
pub fn accessor_adjoint(view: &View, g: &View) -> Result<View, String> {
    let broadcast_axes: Vec<usize> = view
        .pitch()
        .iter()
        .enumerate()
        .filter_map(|(i, &p)| if p == 0 { Some(i) } else { None })
        .collect();
    let x = if broadcast_axes.is_empty() {
        g.clone()
    } else {
        g.reduce(&broadcast_axes, ElementOperator::Add)?
    };

    if view.is_identity() {
        return Ok(x);
    }

    Ok(View::remap(
        &x,
        RemapInfo::Scatter(RemapScatterInfo {
            accessor: Some(Accessor::new(
                view.offset(),
                x.shape().to_vec(),
                view.pitch().to_vec(),
            )),
            operator: Some(ElementOperator::Add),
        }),
        view.node_ref.shape.clone(),
        view.element_type(),
        [],
    ))
}

fn identity_view(node: &NodeRef) -> View {
    View::identity(node.clone())
}

/// Given ∂f/∂(node's dense output), return one entry per operand in `node.args`.
///
/// Entries may be `None` when that operand is not in the differentiable path —
/// most often **remap indices** (`args[1]`): multi-indices are discrete addresses,
/// not continuous values, so no adjoint flows through them. Forward remap still
/// has a source adjoint (`Some`) for the data operand.
pub fn df_do(node: &NodeRef, df_dout: &View) -> Result<Vec<Option<View>>, NotDifferentiableError> {
    if node.args.is_empty() {
        return Ok(vec![]);
    }
    match &node.kind {
        NodeKind::Const(_) | NodeKind::Param => Ok(vec![]),
        NodeKind::Elementwise(k) => df_do_elementwise(node, k.op, df_dout),
        NodeKind::Reduction(k) => df_do_reduction(node, k.op, &k.axes, df_dout),
        NodeKind::Matmul(_) => {
            let a = &node.args[0];
            let b = &node.args[1];
            let bt = b.transpose().map_err(NotDifferentiableError)?;
            let at = a.transpose().map_err(NotDifferentiableError)?;
            let da = df_dout.matmul(&bt).map_err(NotDifferentiableError)?;
            let db = at.matmul(df_dout).map_err(NotDifferentiableError)?;
            Ok(vec![Some(da), Some(db)])
        }
        NodeKind::Remap(k) => df_do_remap(node, &k.info, df_dout),
    }
}

fn df_do_elementwise(
    node: &NodeRef,
    op: ElementOperator,
    df_dout: &View,
) -> Result<Vec<Option<View>>, NotDifferentiableError> {
    let n = identity_view(node);
    let map_err = NotDifferentiableError;
    match op {
        ElementOperator::Relu => {
            let zero = const_scalar(0.0, node.args[0].element_type());
            Ok(vec![Some(
                df_dout * &node.args[0].gt(&zero).map_err(map_err)?,
            )])
        }
        ElementOperator::Neg => Ok(vec![Some(-df_dout)]),
        ElementOperator::Exp => Ok(vec![Some(df_dout * &n)]),
        ElementOperator::Log => Ok(vec![Some(df_dout / &node.args[0])]),
        ElementOperator::Sqrt => {
            let two = const_scalar(2.0, n.element_type());
            Ok(vec![Some(df_dout / &(&two * &n))])
        }
        ElementOperator::Sin => Ok(vec![Some(df_dout * &node.args[0].cos())]),
        ElementOperator::Cos => Ok(vec![Some(df_dout * &(-node.args[0].sin()))]),
        ElementOperator::Mul => Ok(vec![
            Some(df_dout * &node.args[1]),
            Some(df_dout * &node.args[0]),
        ]),
        ElementOperator::Div => {
            let neg_n = -&n;
            Ok(vec![
                Some(df_dout / &node.args[1]),
                Some(&(df_dout * &neg_n) / &node.args[1]),
            ])
        }
        ElementOperator::Add => Ok(vec![Some(df_dout.clone()), Some(df_dout.clone())]),
        ElementOperator::Sub => Ok(vec![Some(df_dout.clone()), Some(-df_dout)]),
        ElementOperator::Max => Ok(vec![
            Some(df_dout * &node.args[0].gt(&node.args[1]).map_err(map_err)?),
            Some(df_dout * &node.args[1].gt(&node.args[0]).map_err(map_err)?),
        ]),
        ElementOperator::Min => Ok(vec![
            Some(df_dout * &node.args[0].lt(&node.args[1]).map_err(map_err)?),
            Some(df_dout * &node.args[1].lt(&node.args[0]).map_err(map_err)?),
        ]),
        ElementOperator::Pow => {
            let d0 = &(&(df_dout * &node.args[1]) * &n) / &node.args[0];
            let d1 = &(df_dout * &node.args[0].log()) * &n;
            Ok(vec![Some(d0), Some(d1)])
        }
        _ => Err(NotDifferentiableError(format!("elementwise {op:?}"))),
    }
}

fn df_do_reduction(
    node: &NodeRef,
    op: ElementOperator,
    axes: &[u32],
    df_dout: &View,
) -> Result<Vec<Option<View>>, NotDifferentiableError> {
    let operand = &node.args[0];
    let pitch: Box<[u32]> = (0..operand.rank())
        .map(|i| {
            if axes.iter().any(|&a| a as usize == i) {
                0
            } else {
                df_dout.pitch()[i]
            }
        })
        .collect();
    let g = View {
        node_ref: df_dout.node_ref.clone(),
        accessor: Accessor::new(df_dout.offset(), operand.shape().to_vec(), pitch),
    };
    let n = identity_view(node);
    match op {
        ElementOperator::Add => Ok(vec![Some(g)]),
        ElementOperator::Mul => Ok(vec![Some(&g * &(&n / operand))]),
        ElementOperator::Max | ElementOperator::Min => {
            let cond = operand.eq(&n).map_err(NotDifferentiableError)?;
            Ok(vec![Some(&g * &cond)])
        }
        _ => Err(NotDifferentiableError(format!("reduction {op:?}"))),
    }
}

fn df_do_remap(
    node: &NodeRef,
    info: &RemapInfo,
    df_dout: &View,
) -> Result<Vec<Option<View>>, NotDifferentiableError> {
    let source = &node.args[0];
    let dense = df_dout.copy(None);
    match info {
        RemapInfo::Gather(RemapGatherInfo {
            accessor: None,
            source_shape: None,
        }) => Ok(vec![Some(View {
            node_ref: dense.node_ref.clone(),
            accessor: Accessor::new_c_contiguous(source.shape().to_vec(), 0),
        })]),
        RemapInfo::Scatter(RemapScatterInfo {
            accessor: Some(accessor),
            ..
        }) => Ok(vec![Some(View {
            node_ref: dense.node_ref.clone(),
            accessor: accessor.clone(),
        })]),
        RemapInfo::Scatter(RemapScatterInfo { accessor: None, .. }) => Ok(vec![
            Some(View::remap(
                &dense,
                RemapInfo::Gather(RemapGatherInfo {
                    accessor: Some(Accessor::new_c_contiguous(node.shape.clone(), 0)),
                    source_shape: Some(node.shape.clone()),
                }),
                source.shape().to_vec(),
                source.element_type(),
                [node.args[1].clone()],
            )),
            None,
        ]),
        RemapInfo::Gather(RemapGatherInfo {
            accessor: Some(_),
            source_shape: Some(source_shape),
        }) => Ok(vec![
            Some(View::remap(
                &dense,
                RemapInfo::Scatter(RemapScatterInfo {
                    accessor: None,
                    operator: Some(ElementOperator::Add),
                }),
                source_shape.clone(),
                source.element_type(),
                [node.args[1].clone()],
            )),
            None,
        ]),
        _ => Err(NotDifferentiableError(format!("remap {info:?}"))),
    }
}

fn const_scalar(value: f32, etype: ElementType) -> View {
    assert_eq!(etype, ElementType::F4);
    value.into()
}

fn ones_scalar(etype: ElementType) -> View {
    const_scalar(1.0, etype)
}

/// Gradients of a scalar sink, keyed by node identity.
pub fn grad_by_node(f: &View) -> Result<HashMap<NodeRef, View>, String> {
    if !f.shape().is_empty() {
        return Err("output graph must be a scalar (shape=[])".into());
    }

    let mut grad_node: HashMap<NodeRef, View> = HashMap::new();

    fn accumulate(
        grad_node: &mut HashMap<NodeRef, View>,
        view: &View,
        g: &View,
    ) -> Result<(), String> {
        let contrib = accessor_adjoint(view, g)?;
        let key = view.node_ref.clone();
        let next = match grad_node.remove(&key) {
            Some(existing) => existing + contrib,
            None => contrib,
        };
        grad_node.insert(key, next);
        Ok(())
    }

    accumulate(&mut grad_node, f, &ones_scalar(f.element_type()))?;

    let order = View::toposort(std::slice::from_ref(f));
    for node in order.into_iter().rev() {
        let Some(df_dn) = grad_node.get(&node).cloned() else {
            continue;
        };
        let df_dos = df_do(&node, &df_dn).map_err(|e| e.to_string())?;
        assert_eq!(df_dos.len(), node.args.len());
        for (operand, df_do_i) in node.args.iter().zip(df_dos) {
            if let Some(g) = df_do_i {
                accumulate(&mut grad_node, operand, &g)?;
            }
        }
    }

    Ok(grad_node)
}

pub fn grad_wrt<M: Tree<Leaf = View>>(f: &View, wrt: &M) -> Result<M::Map<View>, String> {
    use resin_core::format_param_path;

    let by_node = grad_by_node(f)?;
    // Check before `map` so we can name the missing leaf path (map only sees Views).
    for (path, v) in wrt.flatten() {
        if !by_node.contains_key(&v.node_ref) {
            return Err(format!(
                "no gradient for param {}",
                format_param_path(&path)
            ));
        }
    }
    Ok(wrt.map(|_path, v| {
        by_node
            .get(&v.node_ref)
            .cloned()
            .expect("grad_wrt: leaf checked in flatten pass")
    }))
}

pub fn grad_view(f: &View, wrt: &View) -> Result<View, String> {
    let by_node = grad_by_node(f)?;
    by_node
        .get(&wrt.node_ref)
        .cloned()
        .ok_or_else(|| "no gradient for view".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsl::param;
    use resin_core::F4;

    #[test]
    fn grad_of_sum_of_param() {
        let x = param([2], F4);
        let loss = x.sum(None).unwrap().squeeze(&[0]).unwrap();
        assert!(loss.shape().is_empty());
        let g = grad_view(&loss, &x).unwrap();
        assert_eq!(g.shape(), &[2]);
    }

    /// Single-leaf tree so `grad_wrt` can name the path without `resin-front::nn`.
    struct OneParam<T>(T);

    impl<T> Tree for OneParam<T> {
        type Leaf = T;
        type Map<U> = OneParam<U>;

        fn flatten(&self) -> impl Iterator<Item = (resin_core::TreePath, &T)> + '_ {
            std::iter::once((resin_core::path_name("orphan"), &self.0))
        }

        fn consume_unflatten<I>(stream: &mut resin_core::TreeLeaves<T, I>) -> Self
        where
            I: Iterator<Item = (resin_core::TreePath, T)>,
        {
            let (path, value) = stream
                .pop()
                .expect("Tree::unflatten for OneParam: missing leaf");
            assert_eq!(path, resin_core::path_name("orphan"));
            OneParam(value)
        }
    }

    #[test]
    fn grad_wrt_missing_param_returns_err() {
        let x = param([2], F4);
        // Leaf in `wrt` is not an ancestor of `loss`.
        let orphan = OneParam(param([2], F4));
        let loss = x.sum(None).unwrap().squeeze(&[0]).unwrap();
        let err = match grad_wrt(&loss, &orphan) {
            Ok(_) => panic!("expected missing-param error"),
            Err(e) => e,
        };
        assert!(
            err.contains("no gradient for param") && err.contains("orphan"),
            "err={err}"
        );
    }
}
