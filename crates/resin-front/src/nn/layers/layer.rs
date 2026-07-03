use crate::dsl::View;
use resin_core::{Tree, TreeLeaves, TreePath};

use super::Linear;

/// One step in a layer stack (linear, relu, …).
#[derive(Debug, Clone)]
pub enum Layer<T> {
    Linear(Linear<T>),
    Relu,
}

enum LayerFlatten<'a, T, I>
where
    I: Iterator<Item = (TreePath, &'a T)>,
{
    Linear(I),
    Relu(std::iter::Empty<(TreePath, &'a T)>),
}

impl<'a, T, I> Iterator for LayerFlatten<'a, T, I>
where
    I: Iterator<Item = (TreePath, &'a T)>,
{
    type Item = (TreePath, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Linear(it) => it.next(),
            Self::Relu(it) => it.next(),
        }
    }
}

impl<T> Tree for Layer<T> {
    type Leaf = T;
    type Map<U> = Layer<U>;

    fn flatten(&self) -> impl Iterator<Item = (TreePath, &Self::Leaf)> + '_ {
        match self {
            Self::Linear(l) => LayerFlatten::Linear(l.flatten()),
            Self::Relu => LayerFlatten::Relu(std::iter::empty()),
        }
    }

    fn consume_unflatten<I>(stream: &mut TreeLeaves<Self::Leaf, I>) -> Self
    where
        I: Iterator<Item = (TreePath, Self::Leaf)>,
    {
        if stream.is_empty() {
            Self::Relu
        } else {
            Self::Linear(Linear::consume_unflatten(stream))
        }
    }
}

impl Layer<View> {
    pub fn linear(m: u32, n: u32, bias: bool) -> Self {
        Self::Linear(Linear::new(m, n, bias))
    }

    pub fn forward(&self, x: &View) -> Result<View, String> {
        match self {
            Self::Linear(l) => l.forward(x),
            Self::Relu => Ok(x.relu()),
        }
    }
}
