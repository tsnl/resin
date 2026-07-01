//! Reverse-mode autodiff on DSL graphs.

use std::collections::HashMap;
use std::sync::Arc;

use resin_core::{
    Accessor, BinaryAssocElementOperator, BinaryElementOperator, ElementOperator, ElementType,
    ParamTree, UnaryElementOperator,
};
use resin_dsl::{
    Node, NodeKind, RemapGatherInfo, RemapInfo, RemapScatterInfo, View,
};

#[derive(Debug)]
pub struct NotDifferentiableError(pub String);

impl std::fmt::Display for NotDifferentiableError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "not differentiable: {}", self.0)
    }
}

impl std::error::Error for NotDifferentiableError {}

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
        g.reduce(&broadcast_axes, BinaryAssocElementOperator::Add)?
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
            operator: Some(BinaryAssocElementOperator::Add),
        }),
        view.node.shape.clone(),
        view.etype(),
        [],
    ))
}

fn identity_view(node: &Arc<Node>) -> View {
    View::identity(Arc::clone(node))
}

/// Given ∂f/∂(node's dense output), return one entry per operand in `node.args`.
pub fn df_do(
    node: &Arc<Node>,
    df_dout: &View,
) -> Result<Vec<Option<View>>, NotDifferentiableError> {
    if node.args.is_empty() {
        return Ok(vec![]);
    }
    match &node.kind {
        NodeKind::Const { .. } | NodeKind::Param { .. } => Ok(vec![]),
        NodeKind::Elementwise { op } => df_do_elementwise(node, *op, df_dout),
        NodeKind::Reduction { op, axes } => df_do_reduction(node, *op, axes, df_dout),
        NodeKind::Matmul => {
            let a = &node.args[0];
            let b = &node.args[1];
            let bt = b.transpose().map_err(NotDifferentiableError)?;
            let at = a.transpose().map_err(NotDifferentiableError)?;
            let da = df_dout.matmul(&bt).map_err(NotDifferentiableError)?;
            let db = at.matmul(df_dout).map_err(NotDifferentiableError)?;
            Ok(vec![Some(da), Some(db)])
        }
        NodeKind::Remap { info } => df_do_remap(node, info, df_dout),
    }
}

fn df_do_elementwise(
    node: &Arc<Node>,
    op: ElementOperator,
    df_dout: &View,
) -> Result<Vec<Option<View>>, NotDifferentiableError> {
    let n = identity_view(node);
    let map_err = NotDifferentiableError;
    match op {
        ElementOperator::Unary(UnaryElementOperator::Neg) => Ok(vec![Some(-df_dout)]),
        ElementOperator::Unary(UnaryElementOperator::Exp) => Ok(vec![Some(df_dout * &n)]),
        ElementOperator::Unary(UnaryElementOperator::Log) => {
            Ok(vec![Some(df_dout / &node.args[0])])
        }
        ElementOperator::Unary(UnaryElementOperator::Sqrt) => {
            let two = const_scalar(2.0, n.etype());
            Ok(vec![Some(df_dout / &(&two * &n))])
        }
        ElementOperator::Unary(UnaryElementOperator::Sin) => {
            Ok(vec![Some(df_dout * &node.args[0].cos())])
        }
        ElementOperator::Unary(UnaryElementOperator::Cos) => {
            Ok(vec![Some(df_dout * &(-node.args[0].sin()))])
        }
        ElementOperator::Binary(BinaryElementOperator::Assoc(BinaryAssocElementOperator::Mul)) => {
            Ok(vec![
                Some(df_dout * &node.args[1]),
                Some(df_dout * &node.args[0]),
            ])
        }
        ElementOperator::Binary(BinaryElementOperator::Div) => {
            let neg_n = -&n;
            Ok(vec![
                Some(df_dout / &node.args[1]),
                Some(&(df_dout * &neg_n) / &node.args[1]),
            ])
        }
        ElementOperator::Binary(BinaryElementOperator::Assoc(BinaryAssocElementOperator::Add)) => {
            Ok(vec![Some(df_dout.clone()), Some(df_dout.clone())])
        }
        ElementOperator::Binary(BinaryElementOperator::Sub) => {
            Ok(vec![Some(df_dout.clone()), Some(-df_dout)])
        }
        ElementOperator::Binary(BinaryElementOperator::Assoc(BinaryAssocElementOperator::Max)) => {
            Ok(vec![
                Some(
                    df_dout
                        * &node.args[0]
                            .gt(&node.args[1])
                            .map_err(map_err)?,
                ),
                Some(
                    df_dout
                        * &node.args[1]
                            .gt(&node.args[0])
                            .map_err(map_err)?,
                ),
            ])
        }
        ElementOperator::Binary(BinaryElementOperator::Assoc(BinaryAssocElementOperator::Min)) => {
            Ok(vec![
                Some(
                    df_dout
                        * &node.args[0]
                            .lt(&node.args[1])
                            .map_err(map_err)?,
                ),
                Some(
                    df_dout
                        * &node.args[1]
                            .lt(&node.args[0])
                            .map_err(map_err)?,
                ),
            ])
        }
        ElementOperator::Binary(BinaryElementOperator::Pow) => {
            let d0 = &(&(df_dout * &node.args[1]) * &n) / &node.args[0];
            let d1 = &(df_dout * &node.args[0].log()) * &n;
            Ok(vec![Some(d0), Some(d1)])
        }
        _ => Err(NotDifferentiableError(format!("elementwise {op:?}"))),
    }
}

fn df_do_reduction(
    node: &Arc<Node>,
    op: BinaryAssocElementOperator,
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
        node: Arc::clone(&df_dout.node),
        accessor: Accessor::new(df_dout.offset(), operand.shape().to_vec(), pitch),
    };
    let n = identity_view(node);
    match op {
        BinaryAssocElementOperator::Add => Ok(vec![Some(g)]),
        BinaryAssocElementOperator::Mul => Ok(vec![Some(&g * &(&n / operand))]),
        BinaryAssocElementOperator::Max | BinaryAssocElementOperator::Min => {
            let cond = operand.eq(&n).map_err(NotDifferentiableError)?;
            Ok(vec![Some(&g * &cond)])
        }
    }
}

fn df_do_remap(
    node: &Arc<Node>,
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
            node: Arc::clone(&dense.node),
            accessor: Accessor::dense(source.shape().to_vec(), 0),
        })]),
        RemapInfo::Scatter(RemapScatterInfo {
            accessor: Some(accessor),
            ..
        }) => Ok(vec![Some(View {
            node: Arc::clone(&dense.node),
            accessor: accessor.clone(),
        })]),
        RemapInfo::Scatter(RemapScatterInfo {
            accessor: None,
            ..
        }) => Ok(vec![
            Some(View::remap(
                &dense,
                RemapInfo::Gather(RemapGatherInfo {
                    accessor: Some(Accessor::dense(node.shape.clone(), 0)),
                    source_shape: Some(node.shape.clone()),
                }),
                source.shape().to_vec(),
                source.etype(),
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
                    operator: Some(BinaryAssocElementOperator::Add),
                }),
                source_shape.clone(),
                source.etype(),
                [node.args[1].clone()],
            )),
            None,
        ]),
        _ => Err(NotDifferentiableError(format!("remap {info:?}"))),
    }
}

fn const_scalar(value: f32, etype: ElementType) -> View {
    assert_eq!(etype, ElementType::F4);
    resin_dsl::const_bytes([], etype, value.to_le_bytes())
}

fn ones_scalar(etype: ElementType) -> View {
    const_scalar(1.0, etype)
}

/// Gradients of a scalar sink, keyed by node identity.
pub fn grad_by_node(f: &View) -> Result<HashMap<*const Node, View>, String> {
    if !f.shape().is_empty() {
        return Err("output graph must be a scalar (shape=[])".into());
    }

    let mut grad_node: HashMap<*const Node, View> = HashMap::new();

    fn accumulate(
        grad_node: &mut HashMap<*const Node, View>,
        view: &View,
        g: &View,
    ) -> Result<(), String> {
        let contrib = accessor_adjoint(view, g)?;
        let key = view.node_key();
        let next = match grad_node.remove(&key) {
            Some(existing) => existing + contrib,
            None => contrib,
        };
        grad_node.insert(key, next);
        Ok(())
    }

    accumulate(&mut grad_node, f, &ones_scalar(f.etype()))?;

    let order = View::toposort(std::slice::from_ref(f));
    for node in order.into_iter().rev() {
        let key = Arc::as_ptr(&node);
        let Some(df_dn) = grad_node.get(&key).cloned() else {
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

/// Gradient of scalar `f` w.r.t. a module of `View` leaves (same shape as `wrt`).
pub fn grad_wrt<M: ParamTree<Leaf = View>>(
    f: &View,
    wrt: M,
) -> Result<M::Map<View>, String> {
    let by_node = grad_by_node(f)?;
    Ok(wrt.map(|v| {
        by_node
            .get(&v.node_key())
            .cloned()
            .unwrap_or_else(|| panic!("no gradient for param"))
    }))
}

/// Gradient w.r.t. a single view.
pub fn grad_view(f: &View, wrt: &View) -> Result<View, String> {
    let by_node = grad_by_node(f)?;
    by_node
        .get(&wrt.node_key())
        .cloned()
        .ok_or_else(|| "no gradient for view".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use resin_core::F4;
    use resin_dsl::param;

    #[test]
    fn grad_of_sum_of_param() {
        let x = param([2], F4, "x");
        let loss = x.sum(None).unwrap().squeeze(&[0]).unwrap();
        assert!(loss.shape().is_empty());
        let g = grad_view(&loss, &x).unwrap();
        assert_eq!(g.shape(), &[2]);
    }
}
