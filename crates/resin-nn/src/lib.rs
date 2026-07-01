//! Neural-net modules and losses (`ParamTree` impls).

use resin_core::{BinaryAssocElementOperator, ElementType, ParamTree, F4};
use resin_dsl::{const_bytes, param, View};

#[derive(Debug, Clone)]
pub struct Linear<T> {
    pub weight: T,
    pub bias: Option<T>,
}

impl Linear<View> {
    pub fn new(m: u32, n: u32, bias: bool) -> Self {
        Self {
            weight: param([n, m], F4, "weight"),
            bias: bias.then(|| param([n], F4, "bias")),
        }
    }

    pub fn forward(&self, x: &View) -> Result<View, String> {
        let mut out = x.matmul(&self.weight.transpose()?)?;
        if let Some(b) = &self.bias {
            out = &out + b;
        }
        Ok(out)
    }
}

impl<T> ParamTree for Linear<T> {
    type Leaf = T;
    type Map<U> = Linear<U>;

    fn map<U>(self, mut f: impl FnMut(T) -> U) -> Linear<U> {
        Linear {
            weight: f(self.weight),
            bias: self.bias.map(&mut f),
        }
    }

    fn flatten(&self) -> impl Iterator<Item = (String, &T)> + '_ {
        std::iter::once(("weight".to_string(), &self.weight)).chain(
            self.bias.iter().map(|b| ("bias".to_string(), b)),
        )
    }

    fn zip_with<U, O>(self, other: Linear<U>, mut f: impl FnMut(T, U) -> O) -> Linear<O> {
        Linear {
            weight: f(self.weight, other.weight),
            bias: match (self.bias, other.bias) {
                (Some(a), Some(b)) => Some(f(a, b)),
                (None, None) => None,
                _ => panic!("bias presence mismatch"),
            },
        }
    }
}

#[derive(Debug, Clone)]
pub struct Mlp<T> {
    pub layers: Vec<Linear<T>>,
}

impl Mlp<View> {
    pub fn new(in_dim: u32, out_dim: u32, n_hidden: usize, hidden_dim: u32, bias: bool) -> Self {
        let mut layers = Vec::new();
        layers.push(Linear::new(in_dim, hidden_dim, bias));
        for _ in 0..n_hidden.saturating_sub(1) {
            layers.push(Linear::new(hidden_dim, hidden_dim, bias));
        }
        layers.push(Linear::new(hidden_dim, out_dim, bias));
        Self { layers }
    }

    pub fn forward(&self, mut x: View) -> Result<View, String> {
        let last = self.layers.len() - 1;
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x)?;
            if i != last {
                x = relu(&x)?;
            }
        }
        Ok(x)
    }
}

impl<T> ParamTree for Mlp<T> {
    type Leaf = T;
    type Map<U> = Mlp<U>;

    fn map<U>(self, mut f: impl FnMut(T) -> U) -> Mlp<U> {
        Mlp {
            layers: self.layers.into_iter().map(|l| l.map(&mut f)).collect(),
        }
    }

    fn flatten(&self) -> impl Iterator<Item = (String, &T)> + '_ {
        self.layers.iter().enumerate().flat_map(|(i, layer)| {
            layer.flatten().map(move |(p, v)| {
                let path = if p.is_empty() {
                    i.to_string()
                } else {
                    format!("{i}.{p}")
                };
                // Match Python Mlp(layers=list): path `layers.0.weight`
                (format!("layers.{path}"), v)
            })
        })
    }

    fn zip_with<U, O>(self, other: Mlp<U>, mut f: impl FnMut(T, U) -> O) -> Mlp<O> {
        assert_eq!(self.layers.len(), other.layers.len());
        Mlp {
            layers: self
                .layers
                .into_iter()
                .zip(other.layers)
                .map(|(a, b)| a.zip_with(b, &mut f))
                .collect(),
        }
    }
}

pub fn relu(x: &View) -> Result<View, String> {
    let zero = const_bytes([], x.etype(), 0f32.to_le_bytes());
    x.max_elem(&zero)
}

pub fn softmax(x: &View, axes: &[usize]) -> Result<View, String> {
    let exp_x = x.exp();
    let sum_exp = exp_x.reduce(axes, BinaryAssocElementOperator::Add)?;
    Ok(&exp_x / &sum_exp)
}

pub fn cross_entropy(y_hat: &View, y: &View) -> Result<View, String> {
    assert_eq!(y_hat.shape(), y.shape());
    let axis = y_hat.rank() - 1;
    let prod = y * &y_hat.log();
    let summed = prod.reduce(&[axis], BinaryAssocElementOperator::Add)?;
    Ok(-summed.squeeze(&[axis])?)
}

pub fn mean(n: &View) -> Result<View, String> {
    let count = n.shape().iter().map(|&d| u64::from(d)).product::<u64>() as f32;
    let mut reduced = n.sum(None)?;
    for axis in (0..reduced.rank()).rev() {
        reduced = reduced.squeeze(&[axis])?;
    }
    let c = const_bytes([], n.etype(), count.to_le_bytes());
    Ok(&reduced / &c)
}

/// Graph-level SGD: builds `p - lr * g` views (same module shape as `params`).
/// For training loops that write buffer bytes, update leaves on the host instead.
pub fn sgd_tree<M: ParamTree<Leaf = View>>(
    params: M,
    grads: M::Map<View>,
    lr: f32,
) -> M::Map<View> {
    let lr_v = const_bytes([], ElementType::F4, lr.to_le_bytes());
    params.zip_with(grads, |p, g| &p - &(&g * &lr_v))
}
