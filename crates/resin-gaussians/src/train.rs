//! Training: `grad()` on the forward function — nothing else.
//!
//! There is no backward implementation anywhere in this crate. The forward
//! renderer is composed from differentiable ops, so a training step is
//! literally:
//!
//! ```ignore
//! let step = jit.jit(move |raw: &RawGaussianCloud<Tensor>| {
//!     let loss_fn = |raw: RawGaussianCloud<Tensor>| {
//!         let image = render(&activate(&raw), &camera);
//!         mse(&image, &target)
//!     };
//!     let (loss, grads) = grad(loss_fn)(raw.clone()).unwrap();
//!     TrainStep { loss, grads }
//! });
//! ```
//!
//! The only "care taken in the forward" is parameterization:
//! [`RawGaussianCloud`] stores checkpoint-convention raw values (log scales,
//! logit opacities/colors) and [`activate`] maps them into valid ranges with
//! smooth bijections, so unconstrained gradient steps stay physical.

use resin_dsl::{ElementType, Tensor};
use resin_jit::ConcreteTensor;
use resin_macros::Tree;

use crate::cloud::{CloudData, GaussianCloud};

/// `1 / (1 + e^{-x})`, composed from existing ops.
pub fn sigmoid(x: &Tensor) -> Tensor {
    let one = Tensor::full(&[], 1.0, ElementType::F32);
    one.clone() / (one + (-x.clone()).exp())
}

/// Pre-activation gaussian parameters (the checkpoint convention):
/// unconstrained values an optimizer can step freely.
#[derive(Debug, Clone, Tree)]
pub struct RawGaussianCloud<T> {
    /// `[N, 3]` positions (unconstrained).
    pub means: T,
    /// `[N, 3]` log scales → `exp` keeps scales positive.
    pub log_scales: T,
    /// `[N, 4]` unnormalized quaternions (renderer normalizes in-graph).
    pub quats: T,
    /// `[N, 3]` color logits → `sigmoid` keeps RGB in (0, 1).
    pub color_logits: T,
    /// `[N]` opacity logits → `sigmoid` keeps opacity in (0, 1).
    pub opacity_logits: T,
}

/// Map raw parameters to renderable attributes (all smooth bijections).
pub fn activate(raw: &RawGaussianCloud<Tensor>) -> GaussianCloud<Tensor> {
    GaussianCloud {
        means: raw.means.clone(),
        scales: raw.log_scales.exp(),
        quats: raw.quats.clone(),
        colors: sigmoid(&raw.color_logits),
        opacities: sigmoid(&raw.opacity_logits),
    }
}

/// Loss + gradients of one training step (a JIT output tree).
#[derive(Debug, Clone, Tree)]
pub struct TrainStep<T> {
    pub loss: T,
    pub grads: RawGaussianCloud<T>,
}

impl RawGaussianCloud<Vec<f32>> {
    /// Invert the activations of host cloud data (clamped away from the
    /// sigmoid asymptotes so logits stay finite).
    pub fn from_cloud_data(data: &CloudData) -> Self {
        let logit = |v: f32| {
            let v = v.clamp(1e-4, 1.0 - 1e-4);
            (v / (1.0 - v)).ln()
        };
        Self {
            means: data.means_flat(),
            log_scales: data.scales_flat().iter().map(|s| s.max(1e-8).ln()).collect(),
            quats: data.quats_flat(),
            color_logits: data.colors_flat().iter().map(|&c| logit(c)).collect(),
            opacity_logits: data.opacities.iter().map(|&o| logit(o)).collect(),
        }
    }

    /// Upload to backend tensors (`n` gaussians).
    pub fn to_tensors<T: ConcreteTensor>(&self, n: usize) -> RawGaussianCloud<T> {
        RawGaussianCloud {
            means: T::from_f32(&[n, 3], &self.means),
            log_scales: T::from_f32(&[n, 3], &self.log_scales),
            quats: T::from_f32(&[n, 4], &self.quats),
            color_logits: T::from_f32(&[n, 3], &self.color_logits),
            opacity_logits: T::from_f32(&[n], &self.opacity_logits),
        }
    }
}

/// Per-attribute learning rates (3DGS convention: positions move far more
/// gently than logits).
#[derive(Debug, Clone, Copy)]
pub struct SgdRates {
    pub means: f32,
    pub log_scales: f32,
    pub quats: f32,
    pub color_logits: f32,
    pub opacity_logits: f32,
}

impl SgdRates {
    pub fn uniform(lr: f32) -> Self {
        Self {
            means: lr,
            log_scales: lr,
            quats: lr,
            color_logits: lr,
            opacity_logits: lr,
        }
    }
}

/// One in-place SGD step over a raw parameter tree (host-side; the GPU step
/// is the compiled loss+grad program).
pub fn sgd_step<T: ConcreteTensor>(
    params: &mut RawGaussianCloud<T>,
    grads: &RawGaussianCloud<T>,
    rates: &SgdRates,
) {
    fn axpy<T: ConcreteTensor>(param: &mut T, grad: &T, lr: f32) {
        let shape = param.shape().to_vec();
        let updated: Vec<f32> = param
            .to_f32()
            .iter()
            .zip(grad.to_f32())
            .map(|(p, g)| p - lr * g)
            .collect();
        *param = T::from_f32(&shape, &updated);
    }
    axpy(&mut params.means, &grads.means, rates.means);
    axpy(&mut params.log_scales, &grads.log_scales, rates.log_scales);
    axpy(&mut params.quats, &grads.quats, rates.quats);
    axpy(&mut params.color_logits, &grads.color_logits, rates.color_logits);
    axpy(
        &mut params.opacity_logits,
        &grads.opacity_logits,
        rates.opacity_logits,
    );
}

/// Mean squared error between an image tensor and a host target.
pub fn mse(image: &Tensor, target: &[f32]) -> Tensor {
    let shape = image.shape().to_vec();
    let count: usize = shape.iter().product();
    assert_eq!(count, target.len(), "mse: target size mismatch");
    let target = Tensor::constant_f32(&shape, target);
    let err = image.clone() - target;
    let axes: Vec<usize> = (0..shape.len()).collect();
    (err.clone() * err).sum_axes(&axes).squeeze_all()
        * Tensor::full(&[], 1.0 / count as f32, ElementType::F32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cloud::gnomen_cloud;

    #[test]
    fn raw_round_trips_through_activation() {
        let data = gnomen_cloud();
        let raw = RawGaussianCloud::<Vec<f32>>::from_cloud_data(&data);
        // Host-side check of the inverse transforms.
        let sig = |x: f32| 1.0 / (1.0 + (-x).exp());
        for (i, &o) in data.opacities.iter().enumerate() {
            assert!((sig(raw.opacity_logits[i]) - o).abs() < 1e-4);
        }
        for (i, s) in data.scales_flat().iter().enumerate() {
            assert!((raw.log_scales[i].exp() - s).abs() < 1e-6);
        }
    }

    #[test]
    fn sigmoid_shape_and_type() {
        let x = Tensor::parameter(&[5], ElementType::F32);
        let y = sigmoid(&x);
        assert_eq!(y.shape(), &[5]);
        assert_eq!(y.element_type(), ElementType::F32);
    }
}
