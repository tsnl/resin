//! [`resin_dataset::MinibatchWriter`] adapter for [`crate::session::Session`].

use resin_dataset::MinibatchWriter;

use crate::pipeline::PipelineError;
use crate::session::Session;

/// Writes host `f32` batch tensors into session buffers.
pub struct SessionWriter<'a, S: Session> {
    session: &'a S,
}

impl<'a, S: Session> SessionWriter<'a, S> {
    pub fn new(session: &'a S) -> Self {
        Self { session }
    }
}

impl<'a, S: Session> MinibatchWriter<S::Buffer> for SessionWriter<'a, S> {
    type Error = PipelineError;

    fn write_xs(&self, slot: &S::Buffer, features: &[f32]) -> Result<(), PipelineError> {
        self.session.write_buffer(slot, 0, &f32_to_bytes(features))
    }

    fn write_ys(&self, slot: &S::Buffer, labels: &[f32]) -> Result<(), PipelineError> {
        self.session.write_buffer(slot, 0, &f32_to_bytes(labels))
    }
}

fn f32_to_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
