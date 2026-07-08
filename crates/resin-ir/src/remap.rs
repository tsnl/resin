use resin_core::{Accessor, BinaryAssocElementOperator};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RemapInfo {
    Scatter(RemapScatterInfo),
    Gather(RemapGatherInfo),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RemapScatterInfo {
    pub accessor: Option<Accessor>,
    pub operator: Option<BinaryAssocElementOperator>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RemapGatherInfo {
    pub accessor: Option<Accessor>,
    pub source_shape: Option<Box<[u32]>>,
}