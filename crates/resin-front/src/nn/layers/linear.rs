use crate::dsl::{param, View};
use resin_core::{Tree, F4};

#[derive(Debug, Clone, Tree)]
pub struct Linear<T> {
    pub weight: T,
    pub bias: Option<T>,
}

impl Linear<View> {
    pub fn new(m: u32, n: u32, bias: bool) -> Self {
        Self {
            weight: param([n, m], F4),
            bias: bias.then(|| param([n], F4)),
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
