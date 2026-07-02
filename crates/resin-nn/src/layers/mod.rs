//! Common [`Tree`]-shaped layers (`Linear`, [`Layer`]) and functional forward helpers.

mod layer;
mod linear;

pub use layer::Layer;
pub use linear::Linear;

use resin_dsl::View;

use crate::loss;

/// Run `layers` left-to-right; first layer sees `x`, each subsequent layer sees the prior output.
pub fn forward(layers: &[Layer<View>], x: &View) -> Result<View, String> {
    layers
        .iter()
        .try_fold(None, |state: Option<View>, layer| {
            Ok(Some(match &state {
                None => layer.forward(x)?,
                Some(v) => layer.forward(v)?,
            }))
        })
        .and_then(|o| o.ok_or_else(|| "forward: empty layer stack".to_string()))
}

/// [`forward`] then stable softmax over axis 1.
pub fn forward_probs(layers: &[Layer<View>], x: &View) -> Result<View, String> {
    loss::softmax(&forward(layers, x)?, &[1])
}

/// Classifier stack: linear layers with ReLU between; apply [`forward_probs`] for probabilities.
pub fn classifier_layers(
    in_dim: u32,
    out_dim: u32,
    n_hidden: usize,
    hidden_dim: u32,
    bias: bool,
) -> Vec<Layer<View>> {
    let mut layers = vec![Layer::linear(in_dim, hidden_dim, bias)];
    for _ in 0..n_hidden.saturating_sub(1) {
        layers.push(Layer::Relu);
        layers.push(Layer::linear(hidden_dim, hidden_dim, bias));
    }
    layers.push(Layer::Relu);
    layers.push(Layer::linear(hidden_dim, out_dim, bias));
    layers
}
