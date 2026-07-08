use resin_dsl::ElementType;

/// Concrete runtime tensor for a [`crate::Jit`] backend.
pub trait ConcreteTensor: Clone + Send + Sync {
    fn shape(&self) -> &[usize];
    fn element_type(&self) -> ElementType;
    fn zeros(shape: &[usize], element_type: ElementType) -> Self;

    /// Build a tensor from host `f32` values (row-major).
    fn from_f32(shape: &[usize], values: &[f32]) -> Self;

    /// Read all elements as host `f32` (row-major).
    fn to_f32(&self) -> Vec<f32>;

    /// Read a rank-0 tensor as a single `f32`.
    fn scalar_f32(&self) -> f32 {
        assert!(
            self.shape().is_empty(),
            "scalar_f32 requires rank-0 tensor, got {:?}",
            self.shape()
        );
        let values = self.to_f32();
        assert_eq!(values.len(), 1);
        values[0]
    }
}