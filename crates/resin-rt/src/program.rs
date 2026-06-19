use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WgpuProgram {
    #[serde(default = "default_schema_version")]
    pub schema_version: u32,
    pub param_buffer_ids: BTreeMap<u64, usize>,
    pub sinks: BTreeMap<String, usize>,
    pub queue: Vec<WgpuDispatch>,
    pub buffers: Vec<WgpuBufferSpec>,
    pub buffer_views: Vec<WgpuBufferViewSpec>,
    pub pipelines: Vec<WgpuPipelineSpec>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WgpuDispatch {
    pub pipeline_index: usize,
    pub arg_buffer_view_indices: Vec<usize>,
    pub output_buffer_index: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WgpuBufferSpec {
    pub shape: Vec<u32>,
    pub dtype: DType,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub init: Option<Vec<u8>>,
    pub readonly: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WgpuBufferViewSpec {
    pub buffer_index: usize,
    pub accessor: WgpuAccessorSpec,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WgpuAccessorSpec {
    pub offset: u32,
    pub shape: Vec<u32>,
    pub pitch: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum WgpuPipelineSpec {
    #[serde(rename = "compute")]
    Compute(WgpuComputePipelineSpec),
    #[serde(rename = "copy")]
    Copy(WgpuCopyPipelineSpec),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct WgpuCopyPipelineSpec {}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WgpuComputePipelineSpec {
    pub wgsl: String,
    #[serde(default = "default_entry_point")]
    pub entry_point: String,
    pub dispatch_size: [u32; 3],
    pub num_arg_bindings: u32,
    #[serde(default)]
    pub clear_output_before_dispatch: bool,
}

fn default_entry_point() -> String {
    "main".to_string()
}

fn default_schema_version() -> u32 {
    SCHEMA_VERSION
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DType {
    #[serde(rename = "f4")]
    F4,
    #[serde(rename = "f2")]
    F2,
    #[serde(rename = "u4")]
    U4,
}

impl DType {
    pub fn nbytes(self) -> u32 {
        match self {
            DType::F4 => 4,
            DType::F2 => 2,
            DType::U4 => 4,
        }
    }
}

impl WgpuBufferSpec {
    pub fn byte_len(&self) -> u64 {
        let count: u64 = self.shape.iter().map(|&d| d as u64).product();
        count * self.dtype.nbytes() as u64
    }
}

impl WgpuProgram {
    pub fn to_msgpack(&self) -> Result<Vec<u8>, rmp_serde::encode::Error> {
        rmp_serde::to_vec_named(self)
    }

    pub fn from_msgpack(bytes: &[u8]) -> Result<Self, rmp_serde::decode::Error> {
        rmp_serde::from_slice(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn msgpack_round_trip() {
        let program = WgpuProgram {
            schema_version: SCHEMA_VERSION,
            param_buffer_ids: BTreeMap::new(),
            sinks: BTreeMap::from([("out".to_string(), 0)]),
            queue: vec![],
            buffers: vec![WgpuBufferSpec {
                shape: vec![3],
                dtype: DType::F4,
                init: Some(vec![0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 64, 64]),
                readonly: true,
            }],
            buffer_views: vec![WgpuBufferViewSpec {
                buffer_index: 0,
                accessor: WgpuAccessorSpec {
                    offset: 0,
                    shape: vec![3],
                    pitch: vec![1],
                },
            }],
            pipelines: vec![],
        };

        let bytes = program.to_msgpack().expect("encode");
        let restored = WgpuProgram::from_msgpack(&bytes).expect("decode");
        assert_eq!(program, restored);
    }
}
