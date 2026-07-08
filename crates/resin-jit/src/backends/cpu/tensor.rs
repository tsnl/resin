use resin_dsl::ElementType;

use crate::tensor::ConcreteTensor;

/// Host-side concrete tensor for the CPU backend.
#[doc(hidden)]
#[derive(Debug, Clone, PartialEq)]
pub struct CpuTensor {
    shape: Box<[usize]>,
    element_type: ElementType,
    bytes: Box<[u8]>,
}

impl ConcreteTensor for CpuTensor {
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn element_type(&self) -> ElementType {
        self.element_type
    }

    fn zeros(shape: &[usize], element_type: ElementType) -> Self {
        let nbytes = shape.iter().product::<usize>() * element_type.nbytes();
        Self {
            shape: shape.into(),
            element_type,
            bytes: vec![0u8; nbytes].into_boxed_slice(),
        }
    }

    fn from_f32(shape: &[usize], values: &[f32]) -> Self {
        assert_eq!(shape.iter().product::<usize>(), values.len());
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        Self {
            shape: shape.into(),
            element_type: ElementType::F32,
            bytes: bytes.into_boxed_slice(),
        }
    }

    fn to_f32(&self) -> Vec<f32> {
        self.bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect()
    }
}

impl CpuTensor {
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn bytes_mut(&mut self) -> &mut [u8] {
        &mut self.bytes
    }
}